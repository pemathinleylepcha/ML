from __future__ import annotations

import logging

import numpy as np

from staged_v5.config import ATR_MIN_THRESHOLD, TPO_FEATURE_NAMES
from tpo_normal_layer import compute_tpo_memory_state, is_degenerate_atr

_MAX_TPO_BARS = 500_000
_LOGGER = logging.getLogger(__name__)


def compute_rolling_volatility(close: np.ndarray, window: int = 24) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if len(close) == 0:
        return np.zeros(0, dtype=np.float32)
    log_ret = np.zeros(len(close), dtype=np.float64)
    if len(close) > 1:
        log_ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    csum = np.concatenate(([0.0], np.cumsum(log_ret, dtype=np.float64)))
    csum2 = np.concatenate(([0.0], np.cumsum(log_ret * log_ret, dtype=np.float64)))
    idx = np.arange(len(close), dtype=np.int64)
    start = np.maximum(0, idx - window + 1)
    count = (idx - start + 1).astype(np.float64)
    sum1 = csum[idx + 1] - csum[start]
    sum2 = csum2[idx + 1] - csum2[start]
    mean = sum1 / count
    var = np.maximum((sum2 / count) - (mean * mean), 0.0)
    return np.sqrt(var).astype(np.float32)


def compute_tpo_feature_panel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lower_high: np.ndarray | None = None,
    lower_low: np.ndarray | None = None,
    lower_close: np.ndarray | None = None,
    lower_lookup: np.ndarray | None = None,
    lookbacks: tuple[int, ...] = (24, 48, 96, 192),
) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n_bars = len(close)
    features = np.zeros((n_bars, len(TPO_FEATURE_NAMES)), dtype=np.float32)
    if n_bars == 0:
        return features, np.zeros(0, dtype=np.float32)
    if n_bars > _MAX_TPO_BARS:
        return features, np.zeros(n_bars, dtype=np.float32)
    volatility = compute_rolling_volatility(close, window=24)

    if lower_high is None or lower_low is None or lower_close is None or lower_lookup is None:
        lower_high = high
        lower_low = low
        lower_close = close
        lower_lookup = np.arange(n_bars, dtype=np.int32)
    else:
        lower_high = np.asarray(lower_high, dtype=np.float64)
        lower_low = np.asarray(lower_low, dtype=np.float64)
        lower_close = np.asarray(lower_close, dtype=np.float64)
        lower_lookup = np.asarray(lower_lookup, dtype=np.int32)

    atr_proxy_raw = np.maximum(np.abs(high - low), np.abs(close - np.roll(close, 1)))
    atr_proxy_raw[0] = float(high[0] - low[0])
    atr_csum = np.concatenate(([0.0], np.cumsum(atr_proxy_raw, dtype=np.float64)))
    atr_idx = np.arange(n_bars, dtype=np.int64)
    atr_start = np.maximum(0, atr_idx - 13)
    atr_count = (atr_idx - atr_start + 1).astype(np.float64)
    atr_mean = (atr_csum[atr_idx + 1] - atr_csum[atr_start]) / atr_count
    lower_lookup = np.clip(lower_lookup, 0, len(lower_close) - 1)
    degenerate_skips = 0
    considered_bars = 0

    for idx in range(n_bars):
        lower_end = int(lower_lookup[min(idx, len(lower_lookup) - 1)])
        start = max(0, lower_end - max(lookbacks) + 1)
        sub_high = lower_high[start : lower_end + 1]
        sub_low = lower_low[start : lower_end + 1]
        sub_close = lower_close[start : lower_end + 1]
        if len(sub_close) < 8:
            continue
        considered_bars += 1
        atr_price = float(atr_mean[idx])
        if is_degenerate_atr(atr_price):
            degenerate_skips += 1
            continue
        state = compute_tpo_memory_state(
            high=sub_high,
            low=sub_low,
            close=sub_close,
            atr_price=atr_price,
            lookbacks=lookbacks,
        )
        profile = state.composite_profile
        features[idx] = np.asarray(
            [
                profile.distance_to_poc_atr,
                profile.value_area_width_atr,
                profile.balance_score,
                state.support_score,
                state.resistance_score,
                state.rejection_score,
                state.poc_drift_atr,
                state.value_area_overlap,
            ],
            dtype=np.float32,
        )
    if considered_bars > 0:
        skip_ratio = float(degenerate_skips / considered_bars)
        if skip_ratio > 0.05:
            _LOGGER.info(
                "state=tpo_degenerate_atr_skips skipped=%s considered=%s ratio=%.4f threshold=%.1e",
                degenerate_skips,
                considered_bars,
                skip_ratio,
                ATR_MIN_THRESHOLD,
            )
    return features, volatility
