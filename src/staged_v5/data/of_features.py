from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling sum using cumulative sums."""
    n = len(arr)
    csum = np.concatenate(([0.0], np.cumsum(arr.astype(np.float64))))
    idx = np.arange(n)
    start = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[start]).astype(np.float32)


def _imbalance_streak(imbalance: np.ndarray) -> np.ndarray:
    """Count consecutive same-sign imbalance bars, normalized to [0, 1]."""
    n = len(imbalance)
    streak = np.zeros(n, dtype=np.float32)
    if n == 0:
        return streak
    sign = np.sign(imbalance)
    streak[0] = 1.0 if sign[0] != 0 else 0.0
    for i in range(1, n):
        if sign[i] == 0:
            streak[i] = 0.0
        elif sign[i] == sign[i - 1]:
            streak[i] = streak[i - 1] + 1.0
        else:
            streak[i] = 1.0
    return np.clip(streak / 30.0, 0.0, 1.0)


def _spread_compression(spread_z: np.ndarray, window: int = 60) -> np.ndarray:
    """Turn narrower recent spread into a positive compression signal."""
    n = len(spread_z)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    csum = np.concatenate(([0.0], np.cumsum(spread_z.astype(np.float64))))
    idx = np.arange(n)
    start = np.maximum(0, idx - window + 1)
    count = (idx - start + 1).astype(np.float64)
    rolling_mean = (csum[idx + 1] - csum[start]) / count
    return (-rolling_mean).astype(np.float32)


def _absorption_ratio(price_velocity: np.ndarray, tick_volume: np.ndarray, window: int = 60) -> np.ndarray:
    """High ratio means volume is being absorbed without much price movement."""
    n = len(price_velocity)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    abs_pv = np.abs(price_velocity).astype(np.float64)
    vol = tick_volume.astype(np.float64)
    rolling_abs_pv = _rolling_sum(abs_pv, window).astype(np.float64)
    rolling_vol = _rolling_sum(vol, window).astype(np.float64)
    ratio = np.where(rolling_abs_pv > 1e-12, rolling_vol / rolling_abs_pv, 0.0)
    positive = ratio[ratio > 0]
    median_ratio = float(np.median(positive)) if positive.size else 1.0
    if median_ratio < 1e-12:
        median_ratio = 1.0
    return np.clip((ratio / median_ratio).astype(np.float32), 0.0, 10.0)


def compute_of_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute 6-dim order-flow features from 1000ms bars."""
    tick_velocity = bars["tick_velocity"].to_numpy(dtype=np.float32)
    spread_z = bars["spread_z"].to_numpy(dtype=np.float32)
    bid_ask_imbalance = bars["bid_ask_imbalance"].to_numpy(dtype=np.float32)
    price_velocity = bars["price_velocity"].to_numpy(dtype=np.float32)
    tick_volume = bars["tk"].to_numpy(dtype=np.float32)

    of_delta = tick_velocity
    of_delta_cum_60 = _rolling_sum(tick_velocity, 60)
    of_delta_cum_300 = _rolling_sum(tick_velocity, 300)
    of_absorption = _absorption_ratio(price_velocity, tick_volume, window=60)
    of_streak = _imbalance_streak(bid_ask_imbalance)
    of_spread_comp = _spread_compression(spread_z, window=60)

    return np.column_stack(
        [
            of_delta,
            of_delta_cum_60,
            of_delta_cum_300,
            of_absorption,
            of_streak,
            of_spread_comp,
        ]
    ).astype(np.float32)
