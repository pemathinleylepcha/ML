from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BacktestMetrics:
    sharpe: float
    trade_count: int
    turnover: float
    mean_confidence: float
    confidence_hit_rate: float
    exposure_mean: float
    win_rate: float
    avg_trade_return: float
    net_return: float


class BinaryHysteresisFilter:
    def __init__(self, entry_threshold: float = 0.60, exit_threshold: float = 0.52, confidence_threshold: float = 0.10):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.confidence_threshold = confidence_threshold
        self.position = 0

    def step(self, p_sell: float, p_buy: float) -> int:
        edge = abs(p_buy - p_sell)
        if self.position == 0:
            if edge >= self.confidence_threshold and p_buy >= self.entry_threshold:
                self.position = 1
            elif edge >= self.confidence_threshold and p_sell >= self.entry_threshold:
                self.position = -1
        elif self.position == 1:
            if p_buy < self.exit_threshold or edge < self.confidence_threshold:
                self.position = 0
        else:
            if p_sell < self.exit_threshold or edge < self.confidence_threshold:
                self.position = 0
        return self.position


def regime_scale(regime_code: float) -> float:
    if regime_code >= 4.0:
        return 0.35
    if regime_code >= 3.0:
        return 0.55
    if regime_code >= 2.0:
        return 0.75
    return 1.0


def _compute_point_steps(
    forward_returns: np.ndarray,
    point_lookback: int,
    min_point_step: float,
) -> np.ndarray:
    alpha = 2.0 / (max(point_lookback, 1) + 1.0)
    steps = np.zeros(len(forward_returns), dtype=np.float64)
    smoothed = max(float(min_point_step), 0.0)
    for idx, raw_ret in enumerate(np.abs(np.nan_to_num(forward_returns, nan=0.0)).astype(np.float64, copy=False)):
        if idx == 0:
            smoothed = max(raw_ret, min_point_step)
        else:
            smoothed = alpha * raw_ret + (1.0 - alpha) * smoothed
            smoothed = max(smoothed, min_point_step)
        steps[idx] = smoothed
    return steps


def _signal_persists(
    position: int,
    p_sell: float,
    p_buy: float,
    edge: float,
    confidence_threshold: float,
    persistence_threshold: float,
) -> bool:
    if edge < confidence_threshold:
        return False
    if position > 0:
        return p_buy >= persistence_threshold and p_buy > p_sell
    if position < 0:
        return p_sell >= persistence_threshold and p_sell > p_buy
    return False


def run_probability_backtest(
    p_buy: np.ndarray,
    forward_returns: np.ndarray,
    regime_codes: np.ndarray,
    session_codes: np.ndarray,
    p_sell: np.ndarray | None = None,
    entry_threshold: float = 0.60,
    exit_threshold: float = 0.52,
    confidence_threshold: float = 0.10,
    persistence_threshold: float = 0.64,
    tp_points: float = 2.0,
    sl_points: float = 1.0,
    trail_points: float = 1.0,
    point_lookback: int = 24,
    min_point_step: float = 1e-5,
) -> BacktestMetrics:
    if p_sell is None:
        p_sell = 1.0 - p_buy

    filt = BinaryHysteresisFilter(entry_threshold, exit_threshold, confidence_threshold)
    point_steps = _compute_point_steps(forward_returns, point_lookback=point_lookback, min_point_step=min_point_step)
    strategy_returns = np.zeros(len(p_buy), dtype=np.float64)
    confidences = np.zeros(len(p_buy), dtype=np.float64)
    exposures = np.zeros(len(p_buy), dtype=np.float64)
    turnover = 0.0
    hit_mask = np.zeros(len(p_buy), dtype=np.bool_)
    closed_trade_returns: list[float] = []
    current_position = 0
    current_trade_pnl = 0.0
    current_trade_raw = 0.0
    current_tp = 0.0
    current_sl = 0.0
    current_step = 0.0

    prev_position = 0
    
    def close_trade() -> None:
        nonlocal current_position, current_trade_pnl, current_trade_raw, current_tp, current_sl, current_step
        if current_position != 0:
            closed_trade_returns.append(float(current_trade_pnl))
        current_position = 0
        current_trade_pnl = 0.0
        current_trade_raw = 0.0
        current_tp = 0.0
        current_sl = 0.0
        current_step = 0.0
        filt.position = 0

    for idx in range(len(p_buy)):
        p_buy_i = float(p_buy[idx])
        p_sell_i = float(p_sell[idx])
        edge = abs(p_buy_i - p_sell_i)
        confidence = edge
        confidences[idx] = confidence

        if int(session_codes[idx]) == 0:
            if current_position != 0:
                close_trade()
            turnover += abs(prev_position)
            prev_position = 0
            continue

        desired_position = filt.step(p_sell_i, p_buy_i)
        if current_position != 0 and desired_position != current_position:
            close_trade()
        if current_position == 0 and desired_position != 0:
            current_position = int(desired_position)
            current_step = float(point_steps[idx])
            current_tp = max(tp_points, 0.0) * current_step
            current_sl = -max(sl_points, 0.0) * current_step

        position = current_position
        turnover += abs(position - prev_position)
        prev_position = position

        if position == 0:
            continue

        size = confidence * regime_scale(float(regime_codes[idx]))
        strategy_returns[idx] = position * size * float(forward_returns[idx])
        confidences[idx] = confidence
        exposures[idx] = abs(position * size)
        current_trade_pnl += float(strategy_returns[idx])
        current_trade_raw += position * float(forward_returns[idx])
        if position != 0 and np.sign(strategy_returns[idx]) == np.sign(position):
            hit_mask[idx] = True

        if current_trade_raw <= current_sl:
            close_trade()
            prev_position = 0
            continue

        persists = _signal_persists(
            position=position,
            p_sell=p_sell_i,
            p_buy=p_buy_i,
            edge=edge,
            confidence_threshold=confidence_threshold,
            persistence_threshold=max(entry_threshold, persistence_threshold),
        )
        trail_step = max(trail_points, 0.0) * current_step
        if trail_step > 0.0:
            while current_position != 0 and current_trade_raw >= current_tp:
                if persists:
                    prev_tp = current_tp
                    current_tp = prev_tp + trail_step
                    current_sl = max(current_sl, prev_tp - trail_step)
                else:
                    close_trade()
                    prev_position = 0
                    break
        elif current_trade_raw >= current_tp:
            close_trade()
            prev_position = 0
            continue

    if current_position != 0:
        close_trade()

    std = float(np.std(strategy_returns)) + 1e-8
    sharpe = float(np.mean(strategy_returns) / std * np.sqrt(max(len(strategy_returns), 1)))
    active_mask = exposures > 0
    confidence_hit_rate = float(hit_mask[active_mask].mean()) if active_mask.any() else 0.0
    trade_returns = np.asarray(closed_trade_returns, dtype=np.float64)
    win_rate = float((trade_returns > 0.0).mean()) if len(trade_returns) > 0 else 0.0
    avg_trade_return = float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0

    return BacktestMetrics(
        sharpe=sharpe,
        trade_count=int(len(trade_returns)),
        turnover=float(turnover / max(len(p_buy), 1)),
        mean_confidence=float(confidences.mean()),
        confidence_hit_rate=confidence_hit_rate,
        exposure_mean=float(exposures.mean()),
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        net_return=float(strategy_returns.sum()),
    )
