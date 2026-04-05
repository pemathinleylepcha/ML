from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_dataset import compute_atr_series
from staged_v4.config import BacktestConfig
from staged_v4.utils.critical_fixes import (
    adaptive_entry_threshold,
    enforce_correlation_exposure,
    probability_spread_mask,
)


@dataclass(slots=True)
class _OpenPosition:
    node_idx: int
    pair_name: str
    direction: int
    signal_bar: int
    entry_bar: int
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252.0 * 24.0 * 12.0))


def _compute_atr_matrix(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    atr = np.zeros_like(close, dtype=np.float64)
    for node_idx in range(close.shape[1]):
        atr[:, node_idx] = compute_atr_series(
            high[:, node_idx].astype(np.float32),
            low[:, node_idx].astype(np.float32),
            close[:, node_idx].astype(np.float32),
            period=period,
        )
    return atr


def _realized_return(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        return float((exit_price - entry_price) / max(abs(entry_price), 1e-8))
    return float((entry_price - exit_price) / max(abs(entry_price), 1e-8))


def _trade_confidence(prob_buy: float, direction: int, entry_prob: float | None = None) -> float:
    directional_conf = prob_buy if direction == 1 else (1.0 - prob_buy)
    if entry_prob is None:
        return float(directional_conf)
    entry_conf = entry_prob if direction == 1 else (1.0 - entry_prob)
    return float(min(directional_conf, entry_conf))


def _build_threshold_diagnostics(confidences: list[float], returns: list[float]) -> list[dict[str, float | int]]:
    if not confidences:
        return []
    confidence_arr = np.asarray(confidences, dtype=np.float64)
    returns_arr = np.asarray(returns, dtype=np.float64)
    buckets = ((0.50, 0.57), (0.57, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01))
    diagnostics: list[dict[str, float | int]] = []
    for lo, hi in buckets:
        mask = (confidence_arr >= lo) & (confidence_arr < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            diagnostics.append({"min_confidence": lo, "max_confidence": hi, "count": 0, "win_rate": 0.0, "avg_return": 0.0})
            continue
        bucket_returns = returns_arr[mask]
        diagnostics.append(
            {
                "min_confidence": lo,
                "max_confidence": hi,
                "count": count,
                "win_rate": float(np.mean(bucket_returns > 0.0)),
                "avg_return": float(np.mean(bucket_returns)),
            }
        )
    return diagnostics


def backtest_probabilities(
    prob_buy: np.ndarray,
    prob_entry: np.ndarray | None,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volatility: np.ndarray,
    session_codes: np.ndarray,
    pair_names: tuple[str, ...],
    cfg: BacktestConfig,
) -> dict:
    n_steps, n_nodes = prob_buy.shape
    atr = _compute_atr_matrix(high, low, close)

    active_cooldown = np.zeros(n_nodes, dtype=np.int32)
    open_positions: list[_OpenPosition] = []
    trade_returns: list[float] = []
    trade_confidences: list[float] = []
    bar_returns = np.zeros(n_steps, dtype=np.float64)
    trade_count = 0
    wins = 0
    blocked_by_exposure = 0
    exit_reason_counts: dict[str, int] = {}
    max_open_positions = 0
    hold_bars: list[int] = []

    def _close_position(position: _OpenPosition, current_bar: int, exit_price: float, reason: str) -> None:
        nonlocal trade_count, wins
        realized = _realized_return(position.direction, position.entry_price, exit_price)
        trade_returns.append(realized)
        trade_confidences.append(position.confidence)
        bar_returns[current_bar] += realized
        trade_count += 1
        if realized > 0.0:
            wins += 1
        else:
            active_cooldown[position.node_idx] = max(active_cooldown[position.node_idx], cfg.cooldown_bars)
        hold_bars.append(max(current_bar - position.entry_bar, 0))
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    for t in range(max(1, cfg.latency_bars), n_steps):
        active_cooldown = np.maximum(active_cooldown - 1, 0)
        still_open: list[_OpenPosition] = []
        for position in open_positions:
            if t <= position.entry_bar:
                still_open.append(position)
                continue

            hi = float(high[t, position.node_idx])
            lo = float(low[t, position.node_idx])
            cls = float(close[t, position.node_idx])

            if position.direction == 1:
                hit_tp = hi >= position.tp_price
                hit_sl = lo <= position.sl_price
                signal_exit = float(prob_buy[t, position.node_idx]) < cfg.exit_threshold
            else:
                hit_tp = lo <= position.tp_price
                hit_sl = hi >= position.sl_price
                signal_exit = float(prob_buy[t, position.node_idx]) > (1.0 - cfg.exit_threshold)

            exit_price = None
            exit_reason = None
            if hit_tp and hit_sl:
                favorable = cls >= position.entry_price if position.direction == 1 else cls <= position.entry_price
                if favorable:
                    exit_price = position.tp_price
                    exit_reason = "tp_sl_same_bar_tp"
                else:
                    exit_price = position.sl_price
                    exit_reason = "tp_sl_same_bar_sl"
            elif hit_tp:
                exit_price = position.tp_price
                exit_reason = "take_profit"
            elif hit_sl:
                exit_price = position.sl_price
                exit_reason = "stop_loss"
            elif (t - position.entry_bar) >= cfg.max_hold_bars:
                exit_price = cls
                exit_reason = "horizon_exit"
            elif signal_exit:
                exit_price = cls
                exit_reason = "signal_exit"

            if exit_price is None:
                still_open.append(position)
                continue
            _close_position(position, t, exit_price, exit_reason)

        open_positions = still_open
        max_open_positions = max(max_open_positions, len(open_positions))
        if t + cfg.latency_bars >= n_steps:
            continue
        if session_codes[t] == 0 or len(open_positions) >= cfg.max_positions:
            continue

        open_nodes = {position.node_idx for position in open_positions}
        active_pairs = [position.pair_name for position in open_positions]
        ranked = np.argsort(-np.abs(prob_buy[t] - 0.5))
        spread_ok = probability_spread_mask(prob_buy[t], 1.0 - prob_buy[t], cfg.probability_spread_threshold)
        for node_idx in ranked:
            if len(open_positions) >= cfg.max_positions:
                break
            if node_idx in open_nodes or active_cooldown[node_idx] > 0:
                continue

            threshold = adaptive_entry_threshold(cfg.base_entry_threshold, float(volatility[t, node_idx]), cfg.threshold_volatility_coeff)
            p = float(prob_buy[t, node_idx])
            directional_confidence = max(p, 1.0 - p)
            if directional_confidence > cfg.max_confidence_threshold:
                continue
            direction = 0
            if spread_ok[node_idx] and p >= threshold:
                direction = 1
            elif spread_ok[node_idx] and p <= 1.0 - threshold:
                direction = -1
            if direction == 0:
                continue

            entry_probability = float(prob_entry[t, node_idx]) if prob_entry is not None else None
            if entry_probability is not None:
                gate_conf = entry_probability if direction == 1 else (1.0 - entry_probability)
                if gate_conf < cfg.entry_gate_threshold:
                    continue

            proposed = active_pairs + [pair_names[node_idx]]
            capped = enforce_correlation_exposure(proposed, active_pairs, cfg.max_group_exposure)
            if len(capped) < len(proposed):
                blocked_by_exposure += 1
                continue

            entry_bar = t + cfg.latency_bars
            entry_atr = float(max(atr[entry_bar, node_idx], 1e-8))
            if cfg.use_limit_entries:
                limit_price = float(close[t, node_idx]) - direction * cfg.limit_offset_atr * entry_atr
                if direction == 1:
                    if float(low[entry_bar, node_idx]) > limit_price:
                        continue
                else:
                    if float(high[entry_bar, node_idx]) < limit_price:
                        continue
                entry_price = limit_price
            else:
                entry_price = float(close[entry_bar, node_idx])

            if direction == 1:
                tp_price = entry_price + cfg.take_profit_atr * entry_atr
                sl_price = entry_price - cfg.stop_loss_atr * entry_atr
            else:
                tp_price = entry_price - cfg.take_profit_atr * entry_atr
                sl_price = entry_price + cfg.stop_loss_atr * entry_atr

            open_positions.append(
                _OpenPosition(
                    node_idx=node_idx,
                    pair_name=pair_names[node_idx],
                    direction=direction,
                    signal_bar=t,
                    entry_bar=entry_bar,
                    entry_price=entry_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    confidence=_trade_confidence(p, direction, entry_probability),
                )
            )
            active_pairs.append(pair_names[node_idx])
            open_nodes.add(node_idx)
            max_open_positions = max(max_open_positions, len(open_positions))

    if open_positions:
        final_bar = n_steps - 1
        for position in open_positions:
            _close_position(position, final_bar, float(close[final_bar, position.node_idx]), "end_of_test")

    return {
        "trade_count": int(trade_count),
        "win_rate": float(wins / trade_count) if trade_count else 0.0,
        "strategy_sharpe": _sharpe(bar_returns),
        "net_return": float(bar_returns.sum()),
        "confidence_hit_rate": float(np.mean(trade_confidences)) if trade_confidences else 0.0,
        "blocked_by_exposure": int(blocked_by_exposure),
        "exit_reason_counts": exit_reason_counts,
        "max_open_positions": int(max_open_positions),
        "avg_hold_bars": float(np.mean(hold_bars)) if hold_bars else 0.0,
        "threshold_diagnostics": _build_threshold_diagnostics(trade_confidences, trade_returns),
        "bar_returns": bar_returns,
    }
