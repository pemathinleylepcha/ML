from __future__ import annotations

import numpy as np
import pandas as pd


_VP_WINDOWS = (300, 900, 3600)  # 5min, 15min, 1h
_N_PRICE_BINS = 50
_VALUE_AREA_PCT = 0.70


def _volume_profile_single(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    current_price: float,
    atr: float,
) -> tuple[float, float, float]:
    """Compute a compact set of VP metrics for a single rolling window."""
    if len(close) < 2 or atr < 1e-10:
        return 0.0, 0.0, 0.0

    price_min = float(np.min(low))
    price_max = float(np.max(high))
    if price_max - price_min < 1e-12:
        return 0.0, 0.0, 0.0

    bin_edges = np.linspace(price_min, price_max, _N_PRICE_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.clip(np.digitize(close, bin_edges) - 1, 0, _N_PRICE_BINS - 1)
    profile = np.bincount(bin_indices, weights=volume, minlength=_N_PRICE_BINS).astype(np.float64)

    total_vol = float(profile.sum())
    if total_vol < 1e-10:
        return 0.0, 0.0, 0.0

    poc_idx = int(np.argmax(profile))
    poc_price = float(bin_centers[poc_idx])
    poc_distance_atr = float((current_price - poc_price) / atr)

    sorted_idx = np.argsort(profile)[::-1]
    cumvol = np.cumsum(profile[sorted_idx])
    va_count = int(np.searchsorted(cumvol, total_vol * _VALUE_AREA_PCT) + 1)
    va_indices = sorted(sorted_idx[:va_count])
    va_low = float(bin_edges[va_indices[0]])
    va_high = float(bin_edges[min(va_indices[-1] + 1, _N_PRICE_BINS)])
    va_width_atr = float((va_high - va_low) / atr)

    mean_vol = float(profile.mean())
    std_vol = float(profile.std())
    threshold = mean_vol + std_vol
    hv_nodes = int(np.sum(profile > threshold))
    hv_nodes_norm = float(hv_nodes / _N_PRICE_BINS)

    return poc_distance_atr, va_width_atr, hv_nodes_norm


def compute_vp_features(
    bars: pd.DataFrame,
    atr_series: np.ndarray,
    windows: tuple[int, ...] = _VP_WINDOWS,
) -> np.ndarray:
    """Compute 4-dim volume-profile features over rolling tick windows."""
    close = bars["c"].to_numpy(dtype=np.float64)
    high = bars["h"].to_numpy(dtype=np.float64)
    low = bars["l"].to_numpy(dtype=np.float64)
    volume = bars["tk"].to_numpy(dtype=np.float64)
    n = len(close)
    atr = np.maximum(np.asarray(atr_series, dtype=np.float64), 1e-10)

    poc_dist_sum = np.zeros(n, dtype=np.float64)
    va_width_sum = np.zeros(n, dtype=np.float64)
    hv_nodes_sum = np.zeros(n, dtype=np.float64)
    poc_prices_short = np.zeros(n, dtype=np.float64)

    for w_idx, window in enumerate(windows):
        for i in range(n):
            start = max(0, i - window + 1)
            w_close = close[start : i + 1]
            w_high = high[start : i + 1]
            w_low = low[start : i + 1]
            w_vol = volume[start : i + 1]
            poc_d, va_w, hv_n = _volume_profile_single(
                w_close,
                w_high,
                w_low,
                w_vol,
                float(close[i]),
                float(atr[i]),
            )
            poc_dist_sum[i] += poc_d
            va_width_sum[i] += va_w
            hv_nodes_sum[i] += hv_n

            if w_idx == 0 and len(w_close) >= 2:
                price_min = float(np.min(w_low))
                price_max = float(np.max(w_high))
                if price_max - price_min > 1e-12:
                    bin_edges = np.linspace(price_min, price_max, _N_PRICE_BINS + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    bin_indices = np.clip(np.digitize(w_close, bin_edges) - 1, 0, _N_PRICE_BINS - 1)
                    profile = np.bincount(bin_indices, weights=w_vol, minlength=_N_PRICE_BINS).astype(np.float64)
                    poc_prices_short[i] = bin_centers[int(np.argmax(profile))]

    n_windows = float(len(windows))
    poc_dist_avg = (poc_dist_sum / n_windows).astype(np.float32)
    va_width_avg = (va_width_sum / n_windows).astype(np.float32)
    hv_nodes_avg = (hv_nodes_sum / n_windows).astype(np.float32)

    poc_slope = np.zeros(n, dtype=np.float32)
    slope_lookback = 60
    for i in range(slope_lookback, n):
        if atr[i] > 1e-10 and poc_prices_short[i] != 0.0 and poc_prices_short[i - slope_lookback] != 0.0:
            poc_slope[i] = float((poc_prices_short[i] - poc_prices_short[i - slope_lookback]) / (atr[i] * slope_lookback))

    return np.column_stack(
        [
            np.clip(poc_dist_avg, -50.0, 50.0),
            np.clip(va_width_avg, 0.0, 50.0),
            hv_nodes_avg,
            np.clip(poc_slope, -10.0, 10.0),
        ]
    ).astype(np.float32)
