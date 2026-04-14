from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _write_tick_csv(root: str, symbol: str, start: str, periods: int) -> Path:
    out = Path(root) / f"{symbol}_1000ms.csv"
    dt = pd.date_range(start=start, periods=periods, freq="1s")
    frame = pd.DataFrame(
        {
            "dt": dt,
            "o": np.linspace(1.0, 1.1, periods),
            "h": np.linspace(1.0, 1.1, periods) + 0.001,
            "l": np.linspace(1.0, 1.1, periods) - 0.001,
            "c": np.linspace(1.0, 1.1, periods),
            "sp": np.ones(periods, dtype=float),
            "tk": np.ones(periods, dtype=float),
        }
    )
    frame.to_csv(out, index=False)
    return out


def test_tick_preflight_complete():
    from staged_v5.data.tick_preflight import inspect_tick_root_preflight

    with tempfile.TemporaryDirectory() as tmp:
        _write_tick_csv(tmp, "EURUSD", "2025-10-01 00:00:00", 10)
        _write_tick_csv(tmp, "GBPUSD", "2025-10-01 00:00:00", 10)
        result = inspect_tick_root_preflight(
            tick_root=tmp,
            symbols=("EURUSD", "GBPUSD"),
            start="2025-10-01 00:00:00",
            end="2025-10-01 00:00:09",
        )
        assert result.complete
        print("PASS test_tick_preflight_complete")


def test_tick_preflight_missing_symbol():
    from staged_v5.data.tick_preflight import inspect_tick_root_preflight

    with tempfile.TemporaryDirectory() as tmp:
        _write_tick_csv(tmp, "EURUSD", "2025-10-01 00:00:00", 10)
        result = inspect_tick_root_preflight(
            tick_root=tmp,
            symbols=("EURUSD", "GBPUSD"),
            start="2025-10-01 00:00:00",
            end="2025-10-01 00:00:09",
        )
        assert not result.complete
        assert result.missing_symbols == ("GBPUSD",)
        print("PASS test_tick_preflight_missing_symbol")


def test_tick_preflight_insufficient_window():
    from staged_v5.data.tick_preflight import inspect_tick_root_preflight

    with tempfile.TemporaryDirectory() as tmp:
        _write_tick_csv(tmp, "EURUSD", "2025-10-01 00:00:05", 5)
        result = inspect_tick_root_preflight(
            tick_root=tmp,
            symbols=("EURUSD",),
            start="2025-10-01 00:00:00",
            end="2025-10-01 00:00:09",
        )
        assert not result.complete
        assert result.insufficient_symbols == ("EURUSD",)
        print("PASS test_tick_preflight_insufficient_window")


if __name__ == "__main__":
    test_tick_preflight_complete()
    test_tick_preflight_missing_symbol()
    test_tick_preflight_insufficient_window()
