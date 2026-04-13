from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tick_resampler import parse_tick_csv, stream_resample_tick_csv


_CSV_HEADER = "<DATE>,<TIME>,<BID>,<ASK>,<LAST>,<VOLUME>,<FLAGS>\n"
_CSV_ROWS = [
    "2025.10.01,00:00:00.000,0.61000,0.61020,,,6\n",
    "2025.10.01,00:00:00.500,0.61010,0.61030,,,6\n",
    "2025.10.01,00:00:01.000,0.61020,0.61040,,,6\n",
]
_TSV_HEADER = "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n"
_TSV_ROWS = [
    "2025.10.01\t00:00:00.000\t0.72000\t0.72020\t\t\t6\n",
    "2025.10.01\t00:00:00.250\t0.72010\t0.72030\t\t\t6\n",
    "2025.10.01\t00:00:01.000\t0.72020\t0.72040\t\t\t6\n",
]
_COLLECTOR_HEADER = "time_msc,time_sec,bid,ask,last,volume,volume_real,flags\n"
_COLLECTOR_ROWS = [
    "1759277127130,2025.10.01 00:05:27,1.17321,1.17375,0.00000,0,0.00000000,134\n",
    "1759277149585,2025.10.01 00:05:49,1.17322,1.17376,0.00000,0,0.00000000,134\n",
    "1759277187126,2025.10.01 00:06:27,1.17323,1.17377,0.00000,0,0.00000000,134\n",
]


class TickResamplerParsingTests(unittest.TestCase):
    def _write_text(self, path: Path, text: str, encoding: str = "utf-8") -> Path:
        path.write_text(text, encoding=encoding)
        return path

    def test_parse_tick_csv_accepts_comma_mt5_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_text(Path(tmp) / "NZDUSD_ticks.csv", _CSV_HEADER + "".join(_CSV_ROWS))
            frame = parse_tick_csv(path)
        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(float(frame.iloc[0]["mid"]), 0.61010, places=6)
        self.assertEqual(int(frame.iloc[-1]["FLAGS"]), 6)

    def test_parse_tick_csv_accepts_utf16_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_text(Path(tmp) / "EURUSD_ticks.csv", _CSV_HEADER + "".join(_CSV_ROWS), encoding="utf-16")
            frame = parse_tick_csv(path)
        self.assertEqual(len(frame), 3)
        self.assertTrue(frame["datetime"].is_monotonic_increasing)

    def test_stream_resample_tick_csv_accepts_tab_mt5_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_text(Path(tmp) / "AUDUSD_ticks.csv", _TSV_HEADER + "".join(_TSV_ROWS))
            bars = stream_resample_tick_csv(path, instrument="AUDUSD", chunksize=2)
        self.assertEqual(list(bars.columns), ["dt", "o", "h", "l", "c", "sp", "tk", "tick_velocity", "spread_z", "bid_ask_imbalance", "price_velocity"])
        self.assertGreaterEqual(len(bars), 2)
        self.assertEqual(int(bars.iloc[0]["tk"]), 2)

    def test_parse_tick_csv_accepts_collector_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_text(Path(tmp) / "EURUSD_ticks.csv", _COLLECTOR_HEADER + "".join(_COLLECTOR_ROWS))
            frame = parse_tick_csv(path)
        self.assertEqual(len(frame), 3)
        self.assertEqual(int(frame.iloc[0]["FLAGS"]), 134)
        self.assertTrue(frame["datetime"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
