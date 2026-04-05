from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class KellyConfig:
    fractional_scale: float = 0.25
    max_fraction_per_trade: float = 0.05
    portfolio_cap: float = 0.25
    min_fraction: float = 0.0


@dataclass(frozen=True)
class KellyBacktestResult:
    sharpe: float
    trade_count: int
    win_rate: float
    avg_fraction: float
    confidence_hit_rate: float
    mean_return: float


def raw_kelly_fraction(win_probability: float, payoff_ratio: float) -> float:
    p = float(np.clip(win_probability, 0.0, 1.0))
    b = max(float(payoff_ratio), 1e-6)
    q = 1.0 - p
    return (b * p - q) / b


def fractional_kelly_fraction(
    win_probability: float,
    payoff_ratio: float,
    config: KellyConfig | None = None,
) -> float:
    cfg = config or KellyConfig()
    raw = raw_kelly_fraction(win_probability, payoff_ratio)
    scaled = raw * cfg.fractional_scale
    clipped = float(np.clip(scaled, cfg.min_fraction, cfg.max_fraction_per_trade))
    return clipped


def normalize_kelly_allocations(
    fractions: Iterable[float],
    config: KellyConfig | None = None,
) -> np.ndarray:
    cfg = config or KellyConfig()
    values = np.maximum(np.asarray(list(fractions), dtype=np.float32), cfg.min_fraction)
    total = float(values.sum())
    if total <= cfg.portfolio_cap:
        return values
    if total <= 0.0:
        return values
    return values * (cfg.portfolio_cap / total)


def run_fractional_kelly_backtest(
    probabilities: np.ndarray,
    labels: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float = 0.08,
    payoff_ratio: float = 1.0,
    config: KellyConfig | None = None,
    trade_session_mask: np.ndarray | None = None,
    open_ready: np.ndarray | None = None,
    momentum_score: np.ndarray | None = None,
    momentum_floor: float = 0.0,
    volatility_ratio: np.ndarray | None = None,
    volatility_ratio_floor: float = 0.0,
    volatility_ratio_cap: float | None = None,
    breakout_signed: np.ndarray | None = None,
    breakout_floor: float = 0.0,
    scalp_hard_gate: bool = False,
    lap_neighbor_momentum: np.ndarray | None = None,
    lap_laggard_score: np.ndarray | None = None,
    lap_laggard_floor: float = 0.0,
    lap_allocation_scale: float = 0.0,
    lap_alignment_scale: float = 0.35,
    lap_hard_gate: bool = False,
) -> KellyBacktestResult:
    cfg = config or KellyConfig()
    prob = np.asarray(probabilities, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    fwd = np.asarray(forward_returns, dtype=np.float32)
    side = np.where(prob >= 0.5 + threshold, 1, np.where(prob <= 0.5 - threshold, -1, 0))

    if trade_session_mask is not None:
        session_gate = np.asarray(trade_session_mask, dtype=bool)
        side = np.where(session_gate, side, 0)
    if open_ready is not None:
        ready_gate = np.asarray(open_ready, dtype=bool)
        side = np.where(ready_gate, side, 0)
    scalp_multipliers: list[np.ndarray] = []
    if momentum_score is not None and momentum_floor > 0.0:
        momentum_values = np.asarray(momentum_score, dtype=np.float32)
        if scalp_hard_gate:
            momentum_gate = momentum_values >= float(momentum_floor)
            side = np.where(momentum_gate, side, 0)
        momentum_ratio = momentum_values / max(float(momentum_floor), 1e-6)
        momentum_multiplier = np.where(
            side != 0,
            np.clip(0.35 + (0.65 * np.clip(momentum_ratio, 0.0, 1.5)), 0.25, 1.25),
            1.0,
        ).astype(np.float32, copy=False)
        scalp_multipliers.append(momentum_multiplier)
    if volatility_ratio is not None:
        vol_values = np.asarray(volatility_ratio, dtype=np.float32)
        vol_gate = np.ones_like(side, dtype=bool)
        if volatility_ratio_floor > 0.0:
            vol_gate &= vol_values >= float(volatility_ratio_floor)
        if volatility_ratio_cap is not None:
            vol_gate &= vol_values <= float(volatility_ratio_cap)
        if scalp_hard_gate:
            side = np.where(vol_gate, side, 0)
        vol_multiplier = np.ones_like(prob, dtype=np.float32)
        if volatility_ratio_floor > 0.0:
            low_scale = np.clip(vol_values / max(float(volatility_ratio_floor), 1e-6), 0.25, 1.0)
            vol_multiplier = np.minimum(vol_multiplier, low_scale.astype(np.float32, copy=False))
        if volatility_ratio_cap is not None:
            high_scale = np.clip(float(volatility_ratio_cap) / np.maximum(vol_values, 1e-6), 0.25, 1.0)
            vol_multiplier = np.minimum(vol_multiplier, high_scale.astype(np.float32, copy=False))
        vol_multiplier = np.where(side != 0, vol_multiplier, 1.0).astype(np.float32, copy=False)
        scalp_multipliers.append(vol_multiplier)
    if breakout_signed is not None and breakout_floor > 0.0:
        breakout = np.asarray(breakout_signed, dtype=np.float32)
        aligned_breakout = np.where(side > 0, breakout, np.where(side < 0, -breakout, 0.0))
        sign_gate = np.where(side != 0, aligned_breakout > 0.0, True)
        side = np.where(sign_gate, side, 0)
        if scalp_hard_gate:
            breakout_gate = np.where(side != 0, aligned_breakout >= float(breakout_floor), True)
            side = np.where(breakout_gate, side, 0)
        breakout_ratio = np.clip(aligned_breakout / max(float(breakout_floor), 1e-6), 0.0, 1.5)
        breakout_multiplier = np.where(
            side != 0,
            np.clip(0.35 + (0.65 * breakout_ratio), 0.25, 1.25),
            1.0,
        ).astype(np.float32, copy=False)
        scalp_multipliers.append(breakout_multiplier)
    scalp_multiplier = np.ones_like(prob, dtype=np.float32)
    if scalp_multipliers:
        scalp_multiplier = np.mean(np.stack(scalp_multipliers, axis=0), axis=0).astype(np.float32, copy=False)
    alignment_multiplier = np.ones_like(prob, dtype=np.float32)
    if lap_neighbor_momentum is not None:
        neighbor_pressure = np.asarray(lap_neighbor_momentum, dtype=np.float32)
        if lap_hard_gate:
            laggard_align = np.ones_like(side, dtype=bool)
            laggard_align &= np.where(side > 0, neighbor_pressure > 0.0, True)
            laggard_align &= np.where(side < 0, neighbor_pressure < 0.0, True)
            side = np.where(laggard_align, side, 0)
        signed_alignment = side.astype(np.float32) * neighbor_pressure
        alignment_multiplier = np.where(
            side != 0,
            np.clip(1.0 + (float(lap_alignment_scale) * np.tanh(signed_alignment)), 0.5, 1.5),
            1.0,
        ).astype(np.float32, copy=False)
    laggard_score_values = None
    laggard_multiplier = np.ones_like(prob, dtype=np.float32)
    if lap_laggard_score is not None:
        laggard_score_values = np.asarray(lap_laggard_score, dtype=np.float32)
        if lap_laggard_floor > 0.0 and lap_hard_gate:
            laggard_gate = np.where(side != 0, laggard_score_values >= float(lap_laggard_floor), True)
            side = np.where(laggard_gate, side, 0)
        if lap_laggard_floor > 0.0:
            readiness = np.clip(laggard_score_values / max(float(lap_laggard_floor), 1e-6), 0.0, 1.0)
        else:
            readiness = np.clip(laggard_score_values, 0.0, 1.0)
        laggard_multiplier = np.where(
            side != 0,
            np.clip(0.65 + (0.35 * readiness), 0.5, 1.0),
            1.0,
        ).astype(np.float32, copy=False)

    win_prob = np.where(side > 0, prob, 1.0 - prob)
    fractions = np.asarray(
        [
            fractional_kelly_fraction(float(p), payoff_ratio, cfg) if s != 0 else 0.0
            for p, s in zip(win_prob, side, strict=False)
        ],
        dtype=np.float32,
    )
    fractions = np.where(side != 0, fractions * scalp_multiplier * alignment_multiplier * laggard_multiplier, fractions)
    if laggard_score_values is not None and lap_allocation_scale > 0.0:
        boost = 1.0 + (float(lap_allocation_scale) * np.clip(laggard_score_values, 0.0, 1.0))
        fractions = np.where(side != 0, fractions * boost.astype(np.float32, copy=False), fractions)
        fractions = np.clip(fractions, cfg.min_fraction, cfg.max_fraction_per_trade)
    signed_returns = side.astype(np.float32) * fwd * fractions
    traded = side != 0
    trade_returns = signed_returns[traded]
    if len(trade_returns) == 0:
        return KellyBacktestResult(
            sharpe=0.0,
            trade_count=0,
            win_rate=0.0,
            avg_fraction=0.0,
            confidence_hit_rate=0.0,
            mean_return=0.0,
        )
    std = float(trade_returns.std(ddof=0))
    sharpe = 0.0 if std < 1e-8 else float(trade_returns.mean() / std) * np.sqrt(252.0 * 78.0)
    predictions = (prob >= 0.5).astype(np.int32)
    confidence_hits = predictions[traded] == y[traded]
    return KellyBacktestResult(
        sharpe=sharpe,
        trade_count=int(traded.sum()),
        win_rate=float((trade_returns > 0.0).mean()),
        avg_fraction=float(fractions[traded].mean()),
        confidence_hit_rate=float(confidence_hits.mean()) if len(confidence_hits) > 0 else 0.0,
        mean_return=float(trade_returns.mean()),
    )
