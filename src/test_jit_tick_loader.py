from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def test_get_available_ram_mb():
    from staged_v5.data.memory_budget import get_available_ram_mb

    ram_mb = get_available_ram_mb()
    assert isinstance(ram_mb, float)
    assert ram_mb > 0.0
    print(f"PASS test_get_available_ram_mb: {ram_mb:.0f} MB")


def test_get_available_vram_mb_cpu():
    from staged_v5.data.memory_budget import get_available_vram_mb

    vram_mb = get_available_vram_mb(device_type="cpu")
    assert vram_mb == 0.0
    print("PASS test_get_available_vram_mb_cpu")


def test_compute_tick_chunk_size():
    from staged_v5.data.memory_budget import compute_tick_chunk_size

    chunk_bars = compute_tick_chunk_size(
        available_ram_mb=8000.0,
        n_nodes=42,
        n_features=22,
        budget_fraction=0.25,
        min_chunk_bars=1000,
        max_chunk_bars=600_000,
    )
    assert isinstance(chunk_bars, int)
    assert chunk_bars >= 1000
    assert chunk_bars <= 600_000
    print(f"PASS test_compute_tick_chunk_size: {chunk_bars} bars")


def test_compute_tick_chunk_size_low_memory():
    from staged_v5.data.memory_budget import compute_tick_chunk_size

    chunk_bars = compute_tick_chunk_size(
        available_ram_mb=1.0,
        n_nodes=42,
        n_features=22,
        budget_fraction=0.25,
        min_chunk_bars=1000,
        max_chunk_bars=600_000,
    )
    assert chunk_bars == 1000
    print(f"PASS test_compute_tick_chunk_size_low_memory: {chunk_bars} bars")


def _write_synthetic_tick_csv(tmp_dir: str, pair_name: str, n_bars: int = 2000) -> Path:
    rng = np.random.RandomState(hash(pair_name) % 2**31)
    timestamps = pd.date_range("2026-01-05 08:00:00", periods=n_bars, freq="1s")
    close = 1.1000 + np.cumsum(rng.randn(n_bars) * 0.0001)
    frame = pd.DataFrame(
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
    )
    path = Path(tmp_dir) / f"{pair_name.upper()}_1000ms.csv"
    frame.to_csv(path, index=False)
    return path


def test_jit_tick_loader_basic():
    from staged_v5.data.jit_tick_loader import JITTickLoader

    with tempfile.TemporaryDirectory() as tmp:
        pairs = ("EURUSD", "GBPUSD")
        for pair in pairs:
            _write_synthetic_tick_csv(tmp, pair, n_bars=500)
        loader = JITTickLoader(
            tick_root=tmp,
            node_names=pairs,
            chunk_bars=200,
            tick_seq_len=60,
        )
        anchor_ts = pd.Timestamp("2026-01-05 08:01:40")
        batch = loader.get_tick_sequence_batch(
            anchor_timestamps=np.array([anchor_ts]),
            device_type="cpu",
        )
        assert batch.node_features.shape == (1, 60, 2, 14), f"Got {batch.node_features.shape}"
        assert batch.tpo_features.shape == (1, 60, 2, 8)
        assert batch.session_codes.shape == (1, 60)
        print("PASS test_jit_tick_loader_basic")


def test_jit_tick_loader_chunk_reuse():
    from staged_v5.data.jit_tick_loader import JITTickLoader

    with tempfile.TemporaryDirectory() as tmp:
        pairs = ("EURUSD",)
        _write_synthetic_tick_csv(tmp, "EURUSD", n_bars=1000)
        loader = JITTickLoader(
            tick_root=tmp,
            node_names=pairs,
            chunk_bars=500,
            tick_seq_len=60,
        )
        ts1 = pd.Timestamp("2026-01-05 08:02:00")
        ts2 = pd.Timestamp("2026-01-05 08:03:00")
        loader.get_tick_sequence_batch(np.array([ts1]), "cpu")
        loads_before = loader._chunk_load_count
        loader.get_tick_sequence_batch(np.array([ts2]), "cpu")
        loads_after = loader._chunk_load_count
        assert loads_after == loads_before, "Chunk was reloaded unnecessarily"
        print("PASS test_jit_tick_loader_chunk_reuse")


if __name__ == "__main__":
    test_get_available_ram_mb()
    test_get_available_vram_mb_cpu()
    test_compute_tick_chunk_size()
    test_compute_tick_chunk_size_low_memory()
    test_jit_tick_loader_basic()
    test_jit_tick_loader_chunk_reuse()
