from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from export_research_stgnn_pack import FEATURE_ORDER, build_tensor_pack
from research_bridge import BridgeContextEncoder
from research_dataset import (
    BUY_CLASS,
    CanonicalResearchDataset,
    FILL_POLICY_CARRY,
    FILL_POLICY_MASK,
    SESSION_CODES,
    build_split_metadata,
    load_canonical_research_dataset,
    make_triple_barrier_labels,
)
from train_research_compact import _quarter_based_splits


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
            fill_policy=FILL_POLICY_CARRY,
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
        matrix = encoder.build_feature_matrix(dataset)
        self.assertEqual(matrix.shape, (len(timestamps), 72))

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
            self.assertEqual(dataset.fill_policy, FILL_POLICY_CARRY)

    def test_load_canonical_dataset_keeps_common_timeline_for_staggered_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            btc_rows = []
            btc_ts = pd.date_range("2026-01-01 00:05:00", periods=24, freq="5min")
            for idx, dt in enumerate(btc_ts):
                btc_rows.append({
                    "bar_time": dt.strftime("%Y.%m.%d %H:%M:%S"),
                    "open": 1.0 + idx * 0.001,
                    "high": 1.002 + idx * 0.001,
                    "low": 0.998 + idx * 0.001,
                    "close": 1.001 + idx * 0.001,
                    "tick_volume": 10 + idx,
                    "spread": 0.0001,
                    "real_volume": 0,
                })

            eur_rows = []
            eur_ts = btc_ts[4:]
            for idx, dt in enumerate(eur_ts):
                eur_rows.append({
                    "bar_time": dt.strftime("%Y.%m.%d %H:%M:%S"),
                    "open": 1.2 + idx * 0.001,
                    "high": 1.202 + idx * 0.001,
                    "low": 1.198 + idx * 0.001,
                    "close": 1.201 + idx * 0.001,
                    "tick_volume": 20 + idx,
                    "spread": 0.0001,
                    "real_volume": 0,
                })

            _write_csv(root / "2026" / "Q1" / "BTCUSD" / "candles_M5.csv", btc_rows)
            _write_csv(root / "2026" / "Q1" / "EURUSD" / "candles_M5.csv", eur_rows)

            dataset = load_canonical_research_dataset(str(root), symbols=["BTCUSD", "EURUSD"])
            self.assertEqual(dataset.n_bars, len(btc_ts))
            self.assertEqual(len(dataset.tf_data["M5"]["BTCUSD"]["c"]), len(btc_ts))
            self.assertEqual(len(dataset.tf_data["M5"]["EURUSD"]["c"]), len(btc_ts))
            self.assertFalse(dataset.tf_data["M5"]["EURUSD"]["real"][:4].any())
            self.assertTrue(dataset.tf_data["M5"]["EURUSD"]["real"][4:].all())

    def test_mask_fill_policy_preserves_missing_bars_and_tensor_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            btc_rows = []
            btc_ts = pd.date_range("2026-01-01 00:05:00", periods=24, freq="5min")
            for idx, dt in enumerate(btc_ts):
                btc_rows.append({
                    "bar_time": dt.strftime("%Y.%m.%d %H:%M:%S"),
                    "open": 1.0 + idx * 0.001,
                    "high": 1.002 + idx * 0.001,
                    "low": 0.998 + idx * 0.001,
                    "close": 1.001 + idx * 0.001,
                    "tick_volume": 10 + idx,
                    "spread": 0.0001,
                    "real_volume": 0,
                })

            eur_rows = []
            eur_ts = btc_ts[4:]
            for idx, dt in enumerate(eur_ts):
                eur_rows.append({
                    "bar_time": dt.strftime("%Y.%m.%d %H:%M:%S"),
                    "open": 1.2 + idx * 0.001,
                    "high": 1.202 + idx * 0.001,
                    "low": 1.198 + idx * 0.001,
                    "close": 1.201 + idx * 0.001,
                    "tick_volume": 20 + idx,
                    "spread": 0.0001,
                    "real_volume": 0,
                })

            _write_csv(root / "2026" / "Q1" / "BTCUSD" / "candles_M5.csv", btc_rows)
            _write_csv(root / "2026" / "Q1" / "EURUSD" / "candles_M5.csv", eur_rows)

            dataset = load_canonical_research_dataset(
                str(root),
                symbols=["BTCUSD", "EURUSD"],
                fill_policy=FILL_POLICY_MASK,
            )
            self.assertEqual(dataset.fill_policy, FILL_POLICY_MASK)
            self.assertTrue(np.isnan(dataset.tf_data["M5"]["EURUSD"]["c"][:4]).all())
            self.assertFalse(dataset.tf_data["M5"]["EURUSD"]["real"][:4].any())

            pack, timestamps = build_tensor_pack(dataset)
            m5_tensor = pack["M5"]
            eur_idx = 1
            close_idx = FEATURE_ORDER.index("c")
            valid_idx = FEATURE_ORDER.index("validity")
            session_idx = FEATURE_ORDER.index("session_code")
            transition_idx = FEATURE_ORDER.index("session_transition")
            regime_idx = FEATURE_ORDER.index("regime_signal")

            self.assertEqual(m5_tensor.shape[-1], len(FEATURE_ORDER))
            self.assertTrue(np.isnan(m5_tensor[:4, eur_idx, close_idx]).all())
            self.assertTrue(np.allclose(m5_tensor[:4, eur_idx, valid_idx], 0.0))
            self.assertTrue(np.isfinite(m5_tensor[:, eur_idx, session_idx]).all())
            self.assertTrue(np.isfinite(m5_tensor[:, eur_idx, transition_idx]).all())
            self.assertTrue(np.isfinite(m5_tensor[:, eur_idx, regime_idx]).all())
            self.assertEqual(len(timestamps["M5"]), m5_tensor.shape[0])

    def test_quarter_splits_accept_datetime64_timestamps(self):
        timestamps = np.array(pd.date_range("2025-01-01", periods=12, freq="MS"), dtype="datetime64[ns]")
        quarter_ids = np.array(
            ["2025-Q1", "2025-Q1", "2025-Q1", "2025-Q2", "2025-Q2", "2025-Q2",
             "2025-Q3", "2025-Q3", "2025-Q3", "2025-Q4", "2025-Q4", "2025-Q4"],
            dtype=object,
        )
        splits, holdout_mask = _quarter_based_splits(quarter_ids, timestamps, ("2025-Q4",), purge_bars=1)
        self.assertTrue(len(splits) >= 1)
        self.assertIsNotNone(holdout_mask)
        self.assertTrue(holdout_mask[-1])


if __name__ == "__main__":
    unittest.main()
