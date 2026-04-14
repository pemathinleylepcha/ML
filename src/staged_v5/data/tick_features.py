from __future__ import annotations

import numpy as np
import pandas as pd

from staged_v5.config import RAW_FEATURE_NAMES


_ATR_WINDOW = 30  # 30 bars = 30 seconds at 1000ms resolution


def _rolling_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = _ATR_WINDOW) -> np.ndarray:
    """Compute a simple rolling ATR from OHLC arrays."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    csum = np.concatenate(([0.0], np.cumsum(tr)))
    idx = np.arange(n)
    start = np.maximum(0, idx - window + 1)
    count = (idx - start + 1).astype(np.float64)
    atr = (csum[idx + 1] - csum[start]) / count
    return np.maximum(atr, 1e-10).astype(np.float32)


def build_tick_raw_features(
    frame: pd.DataFrame,
    session_code: int = 0,
    session_transition: int = 0,
    regime_signal: float = 0.0,
    lap_residual: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 14-dim staged_v5 raw feature vector from 1000ms bars."""
    o = frame["o"].to_numpy(dtype=np.float64)
    h = frame["h"].to_numpy(dtype=np.float64)
    l = frame["l"].to_numpy(dtype=np.float64)
    c = frame["c"].to_numpy(dtype=np.float64)
    sp = frame["sp"].to_numpy(dtype=np.float32)
    tk = frame["tk"].to_numpy(dtype=np.float32)
    n = len(c)

    ret_1 = np.zeros(n, dtype=np.float32)
    if n > 1:
        ret_1[1:] = (c[1:] / np.maximum(c[:-1], 1e-12) - 1.0).astype(np.float32)

    ret_3 = np.zeros(n, dtype=np.float32)
    if n > 3:
        ret_3[3:] = (c[3:] / np.maximum(c[:-3], 1e-12) - 1.0).astype(np.float32)

    atr = _rolling_atr(h, l, c, window=_ATR_WINDOW)
    bar_range = (h - l).astype(np.float32)
    atr_norm = np.clip(bar_range / np.maximum(atr, 1e-10), 0.0, 50.0)

    hl_range = np.maximum((h - l).astype(np.float32), 1e-10)
    range_norm = ((c - l) / hl_range).astype(np.float32)

    features = np.column_stack(
        [
            o.astype(np.float32),
            h.astype(np.float32),
            l.astype(np.float32),
            c.astype(np.float32),
            sp,
            tk,
            ret_1,
            ret_3,
            atr_norm,
            range_norm,
            np.full(n, lap_residual, dtype=np.float32),
            np.full(n, regime_signal, dtype=np.float32),
            np.full(n, session_code, dtype=np.float32),
            np.full(n, session_transition, dtype=np.float32),
        ]
    ).astype(np.float32)

    expected_shape = (n, len(RAW_FEATURE_NAMES))
    if features.shape != expected_shape:
        raise ValueError(f"Feature shape mismatch: {features.shape} vs expected {expected_shape}")

    session_codes_arr = np.full(n, int(session_code), dtype=np.int32)
    return features, session_codes_arr
