from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from build_tick_range_root import _discover_files, _match_instrument


class BuildTickRangeRootDiscoveryTests(unittest.TestCase):
    def test_match_instrument_from_filename_stem(self):
        path = Path(r"D:\raw\EURUSD_20251001000000_20251001235959.csv")
        self.assertEqual(_match_instrument(path, ("EURUSD", "GBPUSD")), "EURUSD")

    def test_match_instrument_from_parent_directory(self):
        path = Path(r"D:\COLLECT-TICK-MT5\EACollectorTier1_2Y\2025\Q4\NZDUSD\ticks.csv")
        self.assertEqual(_match_instrument(path, ("EURUSD", "NZDUSD")), "NZDUSD")

    def test_discover_files_filters_irrelevant_quarters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q4 = root / "2025" / "Q4" / "EURUSD"
            q1 = root / "2026" / "Q1" / "EURUSD"
            q4.mkdir(parents=True)
            q1.mkdir(parents=True)
            (q4 / "ticks.csv").write_text("dt,o,h,l,c,sp,tk\n", encoding="utf-8")
            (q1 / "ticks.csv").write_text("dt,o,h,l,c,sp,tk\n", encoding="utf-8")
            discovered = _discover_files(
                root,
                ("EURUSD",),
                start_ts=pd.Timestamp("2025-10-01"),
                end_ts=pd.Timestamp("2025-12-31"),
            )
            self.assertEqual(discovered["EURUSD"], [q4 / "ticks.csv"])


if __name__ == "__main__":
    unittest.main()
