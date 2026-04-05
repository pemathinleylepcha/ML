from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_dataset import CanonicalResearchDataset, SESSION_CODES
from train_research_stgnn_dual import build_overlap_day_splits, build_panel_targets


def _make_frame(timestamps: np.ndarray, start: float) -> dict[str, np.ndarray]:
    values = np.linspace(start, start + 0.2, len(timestamps), dtype=np.float32)
    return {
        "dt": timestamps,
        "o": values,
        "h": values + 0.01,
        "l": values - 0.01,
        "c": values + 0.005,
        "sp": np.full(len(timestamps), 0.0001, dtype=np.float32),
        "tk": np.full(len(timestamps), 50.0, dtype=np.float32),
        "real": np.ones(len(timestamps), dtype=np.bool_),
    }


class ResearchStgnnDualTests(unittest.TestCase):
    def test_build_panel_targets_marks_valid_rows(self):
        timestamps = np.array(pd.date_range("2025-01-01 00:05:00", periods=80, freq="5min"), dtype="datetime64[ns]")
        tf_data = {tf: {} for tf in ("M5", "M15", "H1", "H4")}
        for tf_name in tf_data:
            tf_data[tf_name]["BTCUSD"] = _make_frame(timestamps, 1.0)
            tf_data[tf_name]["EURUSD"] = _make_frame(timestamps, 1.2)
            tf_data[tf_name]["GBPUSD"] = _make_frame(timestamps, 1.4)

        dataset = CanonicalResearchDataset(
            base_timeframe="M5",
            timeframes=("M5", "M15", "H1", "H4"),
            fill_policy="mask",
            tf_data=tf_data,
            base_timestamps=timestamps,
            session_codes=np.full(len(timestamps), SESSION_CODES["london"], dtype=np.int8),
            session_names=np.array(["london"] * len(timestamps), dtype=object),
            quarter_ids=np.array(["2025-Q1"] * len(timestamps), dtype=object),
            tf_index_for_base={tf: np.arange(len(timestamps), dtype=np.int32) for tf in tf_data},
            outer_holdout_quarters=("2025-Q1",),
        )

        panel = build_panel_targets(dataset)
        self.assertEqual(panel["labels"].shape[0], len(timestamps))
        self.assertEqual(panel["labels"].shape[1], 28)
        self.assertTrue(panel["valid"][:, 0].any())
        self.assertTrue(np.isfinite(panel["forward_returns"][panel["valid"][:, 0], 0]).all())

    def test_build_overlap_day_splits_creates_multiple_folds(self):
        timestamps = np.array(pd.date_range("2025-01-06 13:00:00", periods=10, freq="B"), dtype="datetime64[ns]")
        quarter_ids = np.array(["2025-Q1"] * len(timestamps), dtype=object)
        session_codes = np.full(len(timestamps), SESSION_CODES["overlap"], dtype=np.int8)
        valid_panel = np.ones((len(timestamps), 28), dtype=np.bool_)

        splits, holdout_mask = build_overlap_day_splits(
            timestamps,
            quarter_ids,
            session_codes,
            valid_panel,
            outer_holdout_quarters=tuple(),
            overlap_fold_days=2,
            min_train_blocks=1,
            purge_bars=1,
        )

        self.assertGreaterEqual(len(splits), 2)
        self.assertFalse(holdout_mask.any())
        train_idx, val_idx = splits[0]
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)


if __name__ == "__main__":
    unittest.main()
