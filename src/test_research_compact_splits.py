from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_dataset import SESSION_CODES
from train_research_compact import _build_split_plan


class ResearchCompactSplitTests(unittest.TestCase):
    def test_overlap_split_plan_creates_multiple_folds(self):
        timestamps = np.array(pd.date_range("2025-01-06 13:00:00", periods=30, freq="B"), dtype="datetime64[ns]")
        quarter_ids = np.array(["2025-Q1"] * len(timestamps), dtype=object)
        sessions = np.full(len(timestamps), SESSION_CODES["overlap"], dtype=np.int8)

        splits, meta = _build_split_plan(
            quarter_ids=quarter_ids,
            timestamps=timestamps,
            sessions=sessions,
            outer_holdout_quarters=tuple(),
            split_mode="overlap",
            overlap_fold_days=5,
            min_train_blocks=2,
            purge_bars=1,
        )

        self.assertGreaterEqual(len(splits), 2)
        self.assertEqual(meta["mode"], "overlap")
        self.assertEqual(len(meta["inner_folds"]), len(splits))

    def test_month_split_plan_creates_multiple_folds(self):
        timestamps = np.array(pd.date_range("2025-01-01", periods=210, freq="D"), dtype="datetime64[ns]")
        quarter_ids = np.array(["2025-Q1"] * 90 + ["2025-Q2"] * 120, dtype=object)
        sessions = np.full(len(timestamps), SESSION_CODES["london"], dtype=np.int8)

        splits, meta = _build_split_plan(
            quarter_ids=quarter_ids,
            timestamps=timestamps,
            sessions=sessions,
            outer_holdout_quarters=tuple(),
            split_mode="month",
            overlap_fold_days=10,
            min_train_blocks=2,
            purge_bars=1,
        )

        self.assertGreaterEqual(len(splits), 2)
        self.assertEqual(meta["mode"], "month")
        self.assertEqual(len(meta["inner_folds"]), len(splits))


if __name__ == "__main__":
    unittest.main()
