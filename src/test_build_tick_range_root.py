from __future__ import annotations

import unittest
from pathlib import Path

from build_tick_range_root import _match_instrument


class BuildTickRangeRootDiscoveryTests(unittest.TestCase):
    def test_match_instrument_from_filename_stem(self):
        path = Path(r"D:\raw\EURUSD_20251001000000_20251001235959.csv")
        self.assertEqual(_match_instrument(path, ("EURUSD", "GBPUSD")), "EURUSD")

    def test_match_instrument_from_parent_directory(self):
        path = Path(r"D:\COLLECT-TICK-MT5\EACollectorTier1_2Y\2025\Q4\NZDUSD\ticks.csv")
        self.assertEqual(_match_instrument(path, ("EURUSD", "NZDUSD")), "NZDUSD")


if __name__ == "__main__":
    unittest.main()
