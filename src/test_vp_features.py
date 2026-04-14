from __future__ import annotations

import numpy as np
import pandas as pd


def _make_tick_bars(n_bars: int = 600) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2026-01-05 08:00:00", periods=n_bars, freq="1s")
    close = 1.1000 + np.cumsum(rng.randn(n_bars) * 0.0001)
    high = close + rng.uniform(0.00005, 0.0002, n_bars)
    low = close - rng.uniform(0.00005, 0.0002, n_bars)
    tick_count = rng.randint(1, 20, n_bars).astype(float)
    return pd.DataFrame(
        {
            "dt": timestamps,
            "o": close + rng.randn(n_bars) * 0.00005,
            "h": high,
            "l": low,
            "c": close,
            "sp": rng.uniform(0.5, 3.0, n_bars),
            "tk": tick_count,
        }
    ).set_index("dt")


def test_compute_vp_features_shape():
    from staged_v5.data.vp_features import compute_vp_features

    bars = _make_tick_bars(600)
    vp = compute_vp_features(bars, atr_series=np.full(600, 0.001, dtype=np.float32))
    assert vp.shape == (600, 4), f"Expected (600, 4), got {vp.shape}"
    assert vp.dtype == np.float32
    print("PASS test_compute_vp_features_shape")


def test_compute_vp_features_no_nans():
    from staged_v5.data.vp_features import compute_vp_features

    bars = _make_tick_bars(600)
    vp = compute_vp_features(bars, atr_series=np.full(600, 0.001, dtype=np.float32))
    assert not np.any(np.isnan(vp)), "VP features contain NaN"
    print("PASS test_compute_vp_features_no_nans")


def test_compute_vp_features_short_window():
    from staged_v5.data.vp_features import compute_vp_features

    bars = _make_tick_bars(10)
    vp = compute_vp_features(bars, atr_series=np.full(10, 0.001, dtype=np.float32))
    assert vp.shape == (10, 4)
    assert not np.any(np.isnan(vp))
    print("PASS test_compute_vp_features_short_window")


if __name__ == "__main__":
    test_compute_vp_features_shape()
    test_compute_vp_features_no_nans()
    test_compute_vp_features_short_window()
