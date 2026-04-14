from __future__ import annotations

import numpy as np
import pandas as pd


def _make_tick_bars(n_bars: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2026-01-05 08:00:00", periods=n_bars, freq="1s")
    close = 1.1000 + np.cumsum(rng.randn(n_bars) * 0.0001)
    return pd.DataFrame(
        {
            "dt": timestamps,
            "o": close + rng.randn(n_bars) * 0.00005,
            "h": close + rng.uniform(0.00005, 0.0002, n_bars),
            "l": close - rng.uniform(0.00005, 0.0002, n_bars),
            "c": close,
            "sp": rng.uniform(0.5, 3.0, n_bars),
            "tk": rng.randint(1, 20, n_bars).astype(float),
            "tick_velocity": rng.randn(n_bars) * 0.5,
            "spread_z": rng.randn(n_bars),
            "bid_ask_imbalance": rng.randn(n_bars) * 0.3,
            "price_velocity": rng.randn(n_bars) * 0.001,
        }
    ).set_index("dt")


def test_compute_of_features_shape():
    from staged_v5.data.of_features import compute_of_features

    bars = _make_tick_bars(300)
    of = compute_of_features(bars)
    assert of.shape == (300, 6), f"Expected (300, 6), got {of.shape}"
    assert of.dtype == np.float32
    print("PASS test_compute_of_features_shape")


def test_compute_of_features_no_nans():
    from staged_v5.data.of_features import compute_of_features

    bars = _make_tick_bars(300)
    of = compute_of_features(bars)
    assert not np.any(np.isnan(of)), "OF features contain NaN"
    print("PASS test_compute_of_features_no_nans")


def test_compute_of_features_delta_sign():
    from staged_v5.data.of_features import compute_of_features

    bars = _make_tick_bars(300)
    of = compute_of_features(bars)
    np.testing.assert_allclose(of[:, 0], bars["tick_velocity"].values.astype(np.float32), atol=1e-6)
    print("PASS test_compute_of_features_delta_sign")


def test_compute_of_features_short():
    from staged_v5.data.of_features import compute_of_features

    bars = _make_tick_bars(5)
    of = compute_of_features(bars)
    assert of.shape == (5, 6)
    assert not np.any(np.isnan(of))
    print("PASS test_compute_of_features_short")


if __name__ == "__main__":
    test_compute_of_features_shape()
    test_compute_of_features_no_nans()
    test_compute_of_features_delta_sign()
    test_compute_of_features_short()
