from __future__ import annotations

import numpy as np
import pandas as pd


def _make_tick_frame(n_bars: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2026-01-05 08:00:00", periods=n_bars, freq="1s")
    close = 1.1000 + np.cumsum(rng.randn(n_bars) * 0.0001)
    high = close + rng.uniform(0.00005, 0.0002, n_bars)
    low = close - rng.uniform(0.00005, 0.0002, n_bars)
    open_ = close + rng.randn(n_bars) * 0.00005
    spread = rng.uniform(0.5, 3.0, n_bars)
    tick_count = rng.randint(1, 20, n_bars).astype(float)
    tick_velocity = rng.randn(n_bars) * 0.5
    spread_z = rng.randn(n_bars)
    bid_ask_imbalance = rng.randn(n_bars) * 0.3
    price_velocity = rng.randn(n_bars) * 0.001
    return pd.DataFrame(
        {
            "dt": timestamps,
            "o": open_,
            "h": high,
            "l": low,
            "c": close,
            "sp": spread,
            "tk": tick_count,
            "tick_velocity": tick_velocity,
            "spread_z": spread_z,
            "bid_ask_imbalance": bid_ask_imbalance,
            "price_velocity": price_velocity,
        }
    )


def test_build_tick_raw_features_shape():
    from staged_v5.data.tick_features import build_tick_raw_features

    frame = _make_tick_frame(200)
    features, session_codes = build_tick_raw_features(frame, session_code=1)
    assert features.shape == (200, 14), f"Expected (200, 14), got {features.shape}"
    assert features.dtype == np.float32
    assert session_codes.shape == (200,)
    print("PASS test_build_tick_raw_features_shape")


def test_build_tick_raw_features_ohlc_passthrough():
    from staged_v5.data.tick_features import build_tick_raw_features

    frame = _make_tick_frame(50)
    features, _ = build_tick_raw_features(frame, session_code=1)
    np.testing.assert_allclose(features[:, 0], frame["o"].values, atol=1e-6)
    np.testing.assert_allclose(features[:, 1], frame["h"].values, atol=1e-6)
    np.testing.assert_allclose(features[:, 2], frame["l"].values, atol=1e-6)
    np.testing.assert_allclose(features[:, 3], frame["c"].values, atol=1e-6)
    print("PASS test_build_tick_raw_features_ohlc_passthrough")


def test_build_tick_raw_features_returns():
    from staged_v5.data.tick_features import build_tick_raw_features

    frame = _make_tick_frame(100)
    features, _ = build_tick_raw_features(frame, session_code=1)
    close = frame["c"].values
    expected_ret_1 = np.zeros(100, dtype=np.float32)
    expected_ret_1[1:] = close[1:] / close[:-1] - 1.0
    np.testing.assert_allclose(features[:, 6], expected_ret_1, atol=1e-5)
    print("PASS test_build_tick_raw_features_returns")


def test_build_tick_raw_features_no_nans():
    from staged_v5.data.tick_features import build_tick_raw_features

    frame = _make_tick_frame(200)
    features, _ = build_tick_raw_features(frame, session_code=1)
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"
    print("PASS test_build_tick_raw_features_no_nans")


if __name__ == "__main__":
    test_build_tick_raw_features_shape()
    test_build_tick_raw_features_ohlc_passthrough()
    test_build_tick_raw_features_returns()
    test_build_tick_raw_features_no_nans()
