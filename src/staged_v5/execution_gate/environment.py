from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
from staged_v5.execution_gate.features import TickProxyStore
from universe import PIP_SIZES


@dataclass(slots=True)
class ExecutionSimulationResult:
    action: str
    filled: bool
    fill_price: float | None
    reward: float
    execution_edge_vs_mid: float
    slippage_penalty: float
    mfe: float
    mae: float
    no_fill: bool
    near_miss: bool = False


def _trade_through_fill(direction: int, limit_price: float, low: float, high: float, pip_size: float) -> bool:
    if direction == 1:
        return low <= (limit_price - pip_size)
    return high >= (limit_price + pip_size)


def _near_miss_limit(direction: int, limit_price: float, low: np.ndarray, high: np.ndarray, pip_size: float) -> bool:
    if direction == 1:
        return bool(np.any(low <= (limit_price + pip_size)))
    return bool(np.any(high >= (limit_price - pip_size)))


def _limit_price_from_action(
    action: NeuralGateAction,
    *,
    event: GateEvent,
    first_mid: float,
    pip_size: float,
    reference_price_mode: str,
) -> float:
    if action == NeuralGateAction.LIMIT_2TICK:
        return float(first_mid - event.direction * 2.0 * pip_size)
    if action == NeuralGateAction.LIMIT_3TICK:
        return float(first_mid - event.direction * 3.0 * pip_size)
    if action == NeuralGateAction.LIMIT_HALF_ATR:
        return float(first_mid - event.direction * 0.5 * float(event.atr))
    if action == NeuralGateAction.PASSIVE_LIMIT_1TICK:
        return float(first_mid - event.direction * pip_size)
    raise ValueError(f"Unsupported limit action: {action}")


def _reward_components(
    direction: int,
    fill_price: float,
    first_mid: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    mfe_weight: float = 0.25,
    mae_weight: float = 0.10,
) -> tuple[float, float, float, float, float]:
    """Return (edge, slippage, reward, raw_mfe, raw_mae)."""
    edge = float((first_mid - fill_price) * direction)
    slippage = float(abs(fill_price - first_mid))
    terminal_close = float(close[-1]) if len(close) > 0 else fill_price
    terminal_pnl = float((terminal_close - fill_price) * direction)
    if direction == 1:
        mfe = float(np.max(high - fill_price))
        mae = float(np.max(fill_price - low))
    else:
        mfe = float(np.max(fill_price - low))
        mae = float(np.max(high - fill_price))
    reward = terminal_pnl + max(0.0, mfe) * mfe_weight - max(0.0, mae) * mae_weight
    return edge, slippage, reward, mfe, mae


def simulate_action(
    event: GateEvent,
    action: NeuralGateAction | int,
    tick_store: TickProxyStore,
    *,
    reference_price_mode: str = "poc",
    reject_wait_penalty: float = -5e-5,
    entry_slippage_atr: float = 0.01,
    market_order_slippage_ticks: float = 1.0,
    limit_at_poc_near_miss_reward: float = 5e-5,
    limit_at_poc_no_fill_penalty: float = -1e-5,
    passive_limit_no_fill_penalty: float = -2.5e-5,
    horizon_bars: int = 10,
) -> ExecutionSimulationResult:
    if isinstance(action, NeuralGateAction):
        resolved_action = action
    else:
        action_value = action
        if isinstance(action_value, np.ndarray):
            if action_value.size != 1:
                raise TypeError(f"simulate_action expected a scalar action, got array with shape {action_value.shape}")
            action_value = action_value.reshape(-1)[0]
        while isinstance(action_value, (list, tuple)):
            if len(action_value) != 1:
                raise TypeError(f"simulate_action expected a scalar action, got sequence with length {len(action_value)}")
            action_value = action_value[0]
        resolved_action = NeuralGateAction(int(action_value))
    action = resolved_action
    if action == NeuralGateAction.REJECT_WAIT:
        return ExecutionSimulationResult(
            action.name.lower(),
            False,
            None,
            float(reject_wait_penalty),
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            False,
        )

    window = tick_store.get_execution_window(event.pair_name, event.timestamp, horizon_bars)
    if len(window) == 0:
        return ExecutionSimulationResult(action.name.lower(), False, None, 0.0, 0.0, 0.0, 0.0, 0.0, True, False)

    first_mid = float(window["o"].iloc[0])
    direction = int(event.direction)
    pip_size = float(PIP_SIZES.get(event.pair_name, 1e-4))
    limit_price = None
    fill_price = None
    fill_start = 0

    if action == NeuralGateAction.MARKET_NOW:
        atr_slippage = entry_slippage_atr * max(float(event.atr), 1e-8)
        tick_slippage = market_order_slippage_ticks * pip_size
        effective_slippage = max(float(atr_slippage), float(tick_slippage))
        fill_price = float(first_mid + direction * effective_slippage)
    else:
        limit_price = _limit_price_from_action(
            action,
            event=event,
            first_mid=first_mid,
            pip_size=pip_size,
            reference_price_mode=reference_price_mode,
        )
        for offset, (_, row) in enumerate(window.iterrows()):
            if _trade_through_fill(direction, float(limit_price), float(row["l"]), float(row["h"]), pip_size):
                fill_price = float(limit_price)
                fill_start = offset
                break
        if fill_price is None:
            low = window["l"].to_numpy(dtype=np.float32)
            high = window["h"].to_numpy(dtype=np.float32)
            near_miss = _near_miss_limit(direction, float(limit_price), low, high, pip_size)
            reward = float(passive_limit_no_fill_penalty)
            return ExecutionSimulationResult(action.name.lower(), False, None, reward, 0.0, 0.0, 0.0, 0.0, True, near_miss)

    high = window["h"].to_numpy(dtype=np.float32)[fill_start:]
    low = window["l"].to_numpy(dtype=np.float32)[fill_start:]
    close_arr = window["c"].to_numpy(dtype=np.float32)[fill_start:]
    edge, slippage, reward, raw_mfe, raw_mae = _reward_components(direction, float(fill_price), first_mid, high, low, close_arr)
    return ExecutionSimulationResult(action.name.lower(), True, float(fill_price), float(reward), edge, slippage, raw_mfe, raw_mae, False, False)
