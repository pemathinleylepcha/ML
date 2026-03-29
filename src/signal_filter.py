"""
signal_filter.py -- Hysteresis signal filter for Algo C2 v2

Prevents signal whipsawing by requiring higher model confidence to enter a
position than to remain in it (entry_threshold > exit_threshold).

Used by:
  - train_catboost_v2.py: strategy Sharpe calculation in walk-forward CV
  - live_engine.py (future): per-bar signal emission

Class order assumed: SELL=0, HOLD=1, BUY=2  (matches train_catboost_v2.py)
"""

from __future__ import annotations

import numpy as np

SELL_IDX = 0
BUY_IDX  = 2


class HysteresisFilter:
    """
    Stateful hysteresis gate for a single instrument.

    State machine:
      FLAT  -> BUY   when P(BUY)  >= entry_threshold
      FLAT  -> SELL  when P(SELL) >= entry_threshold
      BUY   -> FLAT  when P(BUY)  <  exit_threshold
      SELL  -> FLAT  when P(SELL) <  exit_threshold
      BUY  and SELL never switch directly; must pass through FLAT.

    Args:
        entry_threshold: minimum P(direction) to open a position (default 0.45)
        exit_threshold:  minimum P(direction) to keep a position  (default 0.35)
    """

    def __init__(self, entry_threshold: float = 0.45, exit_threshold: float = 0.35):
        if exit_threshold >= entry_threshold:
            raise ValueError(
                f"exit_threshold ({exit_threshold}) must be < entry_threshold ({entry_threshold})"
            )
        self.entry = entry_threshold
        self.exit  = exit_threshold
        self.position: int = 0   # -1=SHORT, 0=FLAT, +1=LONG

    def step(self, p_sell: float, p_buy: float) -> int:
        """
        Process one bar's probabilities, update position, return signal.

        Returns:
            +1 (BUY), -1 (SELL), or 0 (FLAT)
        """
        if self.position == 0:
            # Enter on sufficient conviction
            if p_buy >= self.entry:
                self.position = 1
            elif p_sell >= self.entry:
                self.position = -1
        elif self.position == 1:
            # Exit long if BUY conviction drops below exit threshold
            if p_buy < self.exit:
                self.position = 0
        else:  # position == -1
            # Exit short if SELL conviction drops below exit threshold
            if p_sell < self.exit:
                self.position = 0
        return self.position

    def reset(self) -> None:
        """Reset to flat (call between CV folds)."""
        self.position = 0


def apply_hysteresis(
    proba: np.ndarray,
    entry_threshold: float = 0.45,
    exit_threshold: float = 0.35,
) -> np.ndarray:
    """
    Apply hysteresis filter to a (N, 3) probability array.

    Returns (N,) int32 array of signals: +1=BUY, -1=SELL, 0=FLAT.

    The filter is stateful — applied sequentially from bar 0 to bar N-1.
    Resets automatically at the start of each call (fresh state per fold/pair).
    """
    filt = HysteresisFilter(entry_threshold, exit_threshold)
    n = len(proba)
    signals = np.zeros(n, dtype=np.int32)
    for i in range(n):
        signals[i] = filt.step(float(proba[i, SELL_IDX]), float(proba[i, BUY_IDX]))
    return signals


_REGIME_SCALE: dict[str, float] = {
    "LOW_VOL":     1.0,
    "NORMAL":      1.0,
    "TRANSITIONAL": 0.7,
    "HIGH_STRESS": 0.5,
    "FRAGMENTED":  0.3,
}


class PositionSizer:
    """
    Dynamic position sizing: base_risk * confidence_factor * regime_scale.

    confidence_factor = clip((P(direction) - entry_threshold) / (1 - entry_threshold), 0, 1)
    regime_scale      = lookup in _REGIME_SCALE (default 1.0 = NORMAL)

    Designed to work alongside HysteresisFilter:
      1. HysteresisFilter decides direction (-1, 0, +1)
      2. PositionSizer decides magnitude

    For CV evaluation, regime is not available per-bar so regime_scale defaults
    to 1.0.  For live inference, pass math_state.regime each bar.
    """

    def __init__(self, base_risk: float = 1.0, entry_threshold: float = 0.45):
        self.base_risk = base_risk
        self.entry     = entry_threshold

    def size(self, p_direction: float, regime: str = "NORMAL") -> float:
        """
        Compute fractional position size ∈ [0, base_risk].

        Args:
            p_direction: model probability in the signal direction (P(BUY) or P(SELL))
            regime:      current market regime string (from MathState.regime)

        Returns:
            Signed magnitude in [0, base_risk].
        """
        conf = (p_direction - self.entry) / max(1.0 - self.entry, 1e-8)
        conf = float(np.clip(conf, 0.0, 1.0))
        r_scale = _REGIME_SCALE.get(regime, 1.0)
        return self.base_risk * conf * r_scale

    def size_from_proba(self, proba_row: np.ndarray, direction: int,
                        regime: str = "NORMAL") -> float:
        """
        Convenience wrapper: extract P(direction) from a (3,) proba vector.

        Args:
            proba_row: (3,) array [P(SELL), P(HOLD), P(BUY)]
            direction: +1 (BUY) or -1 (SELL); 0 returns 0.0
            regime:    market regime string
        """
        if direction == 0:
            return 0.0
        p_dir = float(proba_row[BUY_IDX] if direction == 1 else proba_row[SELL_IDX])
        return self.size(p_dir, regime)


def apply_sized_signals(
    proba: np.ndarray,
    entry_threshold: float = 0.45,
    exit_threshold: float = 0.35,
    base_risk: float = 1.0,
    regimes: list[str] | None = None,
) -> np.ndarray:
    """
    Apply hysteresis filter + dynamic position sizing to (N, 3) proba array.

    Returns (N,) float64 array of signed position sizes.
      Positive = long, Negative = short, Zero = flat.

    regimes: optional list of N regime strings; defaults to all "NORMAL".
    """
    filt  = HysteresisFilter(entry_threshold, exit_threshold)
    sizer = PositionSizer(base_risk, entry_threshold)
    n     = len(proba)
    sizes = np.zeros(n, dtype=np.float64)
    _reg  = regimes if regimes is not None else ["NORMAL"] * n

    for i in range(n):
        direction = filt.step(float(proba[i, SELL_IDX]), float(proba[i, BUY_IDX]))
        if direction != 0:
            s = sizer.size_from_proba(proba[i], direction, _reg[i])
            sizes[i] = float(direction) * s

    return sizes


def strategy_sharpe_hysteresis(
    proba: np.ndarray,
    log_rets: np.ndarray,
    entry_threshold: float = 0.45,
    exit_threshold: float = 0.35,
) -> float:
    """
    Compute strategy Sharpe using hysteresis-filtered signals.

    signal[t] ∈ {-1, 0, +1}
    strategy_ret[t] = signal[t] * log_ret[t]
    sharpe = mean/std * sqrt(n)

    More realistic than continuous-signal Sharpe: zero signal bars
    contribute zero return (no position, no cost).
    """
    if len(proba) == 0:
        return 0.0
    signals = apply_hysteresis(proba, entry_threshold, exit_threshold).astype(np.float64)
    lr = log_rets[: len(signals)]
    strat = signals * lr
    std = float(np.std(strat)) + 1e-8
    return float(np.mean(strat) / std * (len(strat) ** 0.5))


def strategy_sharpe_sized(
    proba: np.ndarray,
    log_rets: np.ndarray,
    entry_threshold: float = 0.45,
    exit_threshold: float = 0.35,
    base_risk: float = 1.0,
    regimes: list[str] | None = None,
) -> float:
    """
    Compute strategy Sharpe using hysteresis + dynamic position sizing.

    size[t] = base_risk * confidence_factor[t] * regime_scale[t]  (signed)
    strategy_ret[t] = size[t] * log_ret[t]
    sharpe = mean/std * sqrt(n)

    More realistic than unit-size Sharpe: high-confidence bars get larger
    positions, flat bars contribute nothing, fragmented-regime bars are
    scaled down.
    """
    if len(proba) == 0:
        return 0.0
    sizes = apply_sized_signals(proba, entry_threshold, exit_threshold, base_risk, regimes)
    lr    = log_rets[: len(sizes)]
    strat = sizes * lr
    std   = float(np.std(strat)) + 1e-8
    return float(np.mean(strat) / std * (len(strat) ** 0.5))
