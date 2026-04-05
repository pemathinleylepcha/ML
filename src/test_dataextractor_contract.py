from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audit_year_dataset import audit_year_dataset
from build_clean_year_dataset import build_clean_year_dataset
from dataextractor_contract import resolve_candle_path


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "bar_time,open,high,low,close,tick_volume,spread,real_volume",
                "2025.01.02 00:05:00,1,1,1,1,10,1,0",
                "2025.01.02 00:10:00,1,1,1,1,10,1,0",
            ]
        ),
        encoding="utf-8",
    )


class DataExtractorContractTests(unittest.TestCase):
    def test_resolve_candle_path_uses_alias_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            quarter_dir = Path(tmp) / "2025" / "Q1"
            _write_csv(quarter_dir / "X" / "candles_M5.csv")
            path, actual = resolve_candle_path(quarter_dir, "XTIUSD", "M5")
            self.assertIsNotNone(path)
            self.assertEqual(actual, "X")

    def test_audit_year_dataset_reports_alias_and_missing_quarters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                _write_csv(root / "2025" / quarter / "EURUSD" / "candles_M5.csv")
            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                _write_csv(root / "2025" / quarter / "X" / "candles_M5.csv")

            report = audit_year_dataset(root=root, year="2025", timeframes=("M5",), detect_gaps=True)
            alias_usage = report["year_summary"]["alias_usage"]
            self.assertIn("XTIUSD", alias_usage)
            self.assertEqual(alias_usage["XTIUSD"], ["X"])
            self.assertEqual(report["year_summary"]["missing_required_by_symbol"].get("EURUSD"), None)
            self.assertTrue(any(item["symbol"] == "XTIUSD" for item in report["year_summary"]["fetch_plan"]))

    def test_build_clean_year_dataset_copies_alias_into_canonical_dest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                _write_csv(source / "2025" / quarter / "EURUSD" / "candles_M5.csv")
                _write_csv(source / "2025" / quarter / "X" / "candles_M5.csv")

            result = build_clean_year_dataset(
                source_root=source,
                dest_root=dest,
                year="2025",
                copy_timeframes=("M5",),
                fetch_missing_m5=False,
                chunk_days=31,
            )
            self.assertTrue((dest / "2025" / "Q1" / "XTIUSD" / "candles_M5.csv").exists())
            self.assertNotIn("XTIUSD", result["final_missing_required_by_symbol"])


if __name__ == "__main__":
    unittest.main()
