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


def run_probability_backtest(
    p_buy: np.ndarray,
    forward_returns: np.ndarray,
    regime_codes: np.ndarray,
    session_codes: np.ndarray,
    p_sell: np.ndarray | None = None,
    entry_threshold: float = 0.60,
    exit_threshold: float = 0.52,
    confidence_threshold: float = 0.10,
) -> BacktestMetrics:
    if p_sell is None:
        p_sell = 1.0 - p_buy

    filt = BinaryHysteresisFilter(entry_threshold, exit_threshold, confidence_threshold)
    strategy_returns = np.zeros(len(p_buy), dtype=np.float64)
    confidences = np.zeros(len(p_buy), dtype=np.float64)
    exposures = np.zeros(len(p_buy), dtype=np.float64)
    trade_count = 0
    turnover = 0.0
    hit_mask = np.zeros(len(p_buy), dtype=np.bool_)

    prev_position = 0
    for idx in range(len(p_buy)):
        if int(session_codes[idx]) == 0:
            position = 0
        else:
            position = filt.step(float(p_sell[idx]), float(p_buy[idx]))
        if position != prev_position and position != 0:
            trade_count += 1
        turnover += abs(position - prev_position)
        prev_position = position

        confidence = abs(float(p_buy[idx]) - float(p_sell[idx]))
        size = confidence * regime_scale(float(regime_codes[idx]))
        strategy_returns[idx] = position * size * float(forward_returns[idx])
        confidences[idx] = confidence
        exposures[idx] = abs(position * size)
        if position != 0 and np.sign(strategy_returns[idx]) == np.sign(position):
            hit_mask[idx] = True

    std = float(np.std(strategy_returns)) + 1e-8
    sharpe = float(np.mean(strategy_returns) / std * np.sqrt(max(len(strategy_returns), 1)))
    active_mask = exposures > 0
    confidence_hit_rate = float(hit_mask[active_mask].mean()) if active_mask.any() else 0.0

    return BacktestMetrics(
        sharpe=sharpe,
        trade_count=int(trade_count),
        turnover=float(turnover / max(len(p_buy), 1)),
        mean_confidence=float(confidences.mean()),
        confidence_hit_rate=confidence_hit_rate,
        exposure_mean=float(exposures.mean()),
    )
