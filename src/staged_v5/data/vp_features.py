from __future__ import annotations

import numpy as np
import pandas as pd


_VP_WINDOWS = (300, 900, 3600)  # 5min, 15min, 1h
_N_PRICE_BINS = 50
_VALUE_AREA_PCT = 0.70

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _njit(*args, **kwargs):
        """No-op decorator when numba is unavailable."""
        def wrapper(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrapper


@_njit(cache=True)
def _vp_kernel(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    window: int,
    n_bins: int,
    va_pct: float,
) -> tuple:
    """Inner VP kernel — accelerated by numba when available."""
    n = len(close)
    poc_dist = np.zeros(n, dtype=np.float64)
    va_width = np.zeros(n, dtype=np.float64)
    hv_nodes = np.zeros(n, dtype=np.float64)
    poc_prices = np.zeros(n, dtype=np.float64)

    for i in range(n):
        a = atr[i]
        if a < 1e-10:
            continue
        w_start = max(0, i - window + 1)
        w_len = i - w_start + 1
        if w_len < 2:
            continue

        price_min = low[w_start]
        price_max = high[w_start]
        for j in range(w_start + 1, i + 1):
            if low[j] < price_min:
                price_min = low[j]
            if high[j] > price_max:
                price_max = high[j]

        span = price_max - price_min
        if span < 1e-12:
            continue

        bin_width = span / n_bins
        profile = np.zeros(n_bins, dtype=np.float64)
        for j in range(w_start, i + 1):
            idx = int((close[j] - price_min) / bin_width)
            if idx >= n_bins:
                idx = n_bins - 1
            if idx < 0:
                idx = 0
            profile[idx] += volume[j]

        total_vol = 0.0
        poc_idx = 0
        max_vol = -1.0
        for b in range(n_bins):
            total_vol += profile[b]
            if profile[b] > max_vol:
                max_vol = profile[b]
                poc_idx = b

        if total_vol < 1e-10:
            continue

        poc_price = price_min + (poc_idx + 0.5) * bin_width
        poc_prices[i] = poc_price
        poc_dist[i] = (close[i] - poc_price) / a

        # Value area: sort profile descending, accumulate to VA threshold
        sorted_idx = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            sorted_idx[b] = b
        # Simple insertion sort (n_bins=50, fast enough)
        for b in range(1, n_bins):
            key_idx = sorted_idx[b]
            key_val = profile[key_idx]
            k = b - 1
            while k >= 0 and profile[sorted_idx[k]] < key_val:
                sorted_idx[k + 1] = sorted_idx[k]
                k -= 1
            sorted_idx[k + 1] = key_idx

        cumvol = 0.0
        va_min_idx = n_bins
        va_max_idx = 0
        threshold = total_vol * va_pct
        for b in range(n_bins):
            cumvol += profile[sorted_idx[b]]
            idx_b = sorted_idx[b]
            if idx_b < va_min_idx:
                va_min_idx = idx_b
            if idx_b > va_max_idx:
                va_max_idx = idx_b
            if cumvol >= threshold:
                break

        va_low = price_min + va_min_idx * bin_width
        va_high = price_min + min(va_max_idx + 1, n_bins) * bin_width
        va_width[i] = (va_high - va_low) / a

        # High-volume nodes
        mean_vol = total_vol / n_bins
        sum_sq = 0.0
        for b in range(n_bins):
            diff = profile[b] - mean_vol
            sum_sq += diff * diff
        std_vol = (sum_sq / n_bins) ** 0.5
        hv_threshold = mean_vol + std_vol
        hv_count = 0
        for b in range(n_bins):
            if profile[b] > hv_threshold:
                hv_count += 1
        hv_nodes[i] = float(hv_count) / n_bins

    return poc_dist, va_width, hv_nodes, poc_prices


def compute_vp_features(
    bars: pd.DataFrame,
    atr_series: np.ndarray,
    windows: tuple[int, ...] = _VP_WINDOWS,
) -> np.ndarray:
    """Compute 4-dim volume-profile features over rolling tick windows.

    When numba is installed, the inner kernel runs ~50x faster — recommended
    for RunPod or any GPU cloud environment with large tick datasets.
    """
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
        poc_d, va_w, hv_n, poc_p = _vp_kernel(
            close, high, low, volume, atr, window,
            _N_PRICE_BINS, _VALUE_AREA_PCT,
        )
        poc_dist_sum += poc_d
        va_width_sum += va_w
        hv_nodes_sum += hv_n
        if w_idx == 0:
            poc_prices_short = poc_p

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
