from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_bridge import BridgeContextEncoder
from research_dataset import (
    BUY_CLASS,
    CanonicalResearchDataset,
    SESSION_CODES,
    build_split_metadata,
    load_canonical_research_dataset,
    make_triple_barrier_labels,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class ResearchStackTests(unittest.TestCase):
    def test_triple_barrier_binary_labels(self):
        close = np.array([1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06], dtype=np.float32)
        high = close + 0.005
        low = close - 0.002
        spread = np.full(len(close), 0.0001, dtype=np.float32)
        labels, valid = make_triple_barrier_labels(
            close, high, low, spread, "EURUSD", horizon=3, atr_period=3, binary=True
        )
        self.assertTrue(valid.any())
        self.assertIn(BUY_CLASS, labels[valid])

    def test_split_metadata_outer_holdout(self):
        quarter_ids = np.array(["2025-Q1", "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"], dtype=object)
        meta = build_split_metadata(quarter_ids, ("2025-Q4", "2026-Q1"), n_inner_folds=3)
        self.assertEqual(meta["outer_holdout_quarters"], ["2025-Q4", "2026-Q1"])
        self.assertTrue(all("train_quarters" in fold for fold in meta["inner_folds"]))

    def test_bridge_context_shape(self):
        timestamps = np.array(pd.date_range("2026-01-01", periods=12, freq="5min"), dtype="datetime64[ns]")
        frame = {
            "dt": timestamps,
            "o": np.linspace(1.0, 1.1, len(timestamps), dtype=np.float32),
            "h": np.linspace(1.01, 1.11, len(timestamps), dtype=np.float32),
            "l": np.linspace(0.99, 1.09, len(timestamps), dtype=np.float32),
            "c": np.linspace(1.0, 1.1, len(timestamps), dtype=np.float32),
            "sp": np.full(len(timestamps), 0.0001, dtype=np.float32),
            "tk": np.full(len(timestamps), 10.0, dtype=np.float32),
            "real": np.ones(len(timestamps), dtype=np.bool_),
        }
        tf_data = {"M5": {}, "M15": {}, "H1": {}, "H4": {}}
        for tf_name in tf_data:
            tf_data[tf_name]["BTCUSD"] = frame
            tf_data[tf_name]["EURUSD"] = frame
            tf_data[tf_name]["GBPUSD"] = frame
            tf_data[tf_name]["US30"] = frame
            tf_data[tf_name]["XAUUSD"] = frame
            tf_data[tf_name]["XBRUSD"] = frame
        dataset = CanonicalResearchDataset(
            base_timeframe="M5",
            timeframes=("M5", "M15", "H1", "H4"),
            tf_data=tf_data,
            base_timestamps=timestamps,
            session_codes=np.full(len(timestamps), SESSION_CODES["london"], dtype=np.int8),
            session_names=np.array(["london"] * len(timestamps), dtype=object),
            quarter_ids=np.array(["2026-Q1"] * len(timestamps), dtype=object),
            tf_index_for_base={tf: np.arange(len(timestamps), dtype=np.int32) for tf in tf_data},
            outer_holdout_quarters=("2026-Q1",),
        )
        encoder = BridgeContextEncoder()
        ctx = encoder.encode(dataset, 5)
        self.assertEqual(len(ctx.features), 72)

    def test_load_canonical_research_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            ts = pd.date_range("2026-01-01 00:05:00", periods=24, freq="5min")
            for idx, dt in enumerate(ts):
                rows.append({
                    "bar_time": dt.strftime("%Y.%m.%d %H:%M:%S"),
                    "open": 1.0 + idx * 0.001,
                    "high": 1.002 + idx * 0.001,
                    "low": 0.998 + idx * 0.001,
                    "close": 1.001 + idx * 0.001,
                    "tick_volume": 10 + idx,
                    "spread": 0.0001,
                    "real_volume": 0,
                })
            for symbol in ("BTCUSD", "EURUSD"):
                _write_csv(root / "2026" / "Q1" / symbol / "candles_M5.csv", rows)

            dataset = load_canonical_research_dataset(str(root), symbols=["BTCUSD", "EURUSD"])
            self.assertEqual(dataset.base_timeframe, "M5")
            self.assertGreater(dataset.n_bars, 0)
            self.assertIn("EURUSD", dataset.tf_data["M15"])


if __name__ == "__main__":
    unittest.main()
