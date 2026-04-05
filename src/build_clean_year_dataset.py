from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from audit_year_dataset import audit_year_dataset, write_csv_report
from dataextractor_contract import RAW_TIMEFRAMES, canonical_symbols, resolve_candle_path


def _copy_existing_year(
    source_root: Path,
    dest_root: Path,
    year: str,
    timeframes: tuple[str, ...],
) -> dict:
    copied_files = 0
    copied_symbols: set[str] = set()
    copied_timeframes: set[str] = set()
    missing_sources: list[dict] = []

    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        source_quarter_dir = source_root / year / quarter
        for symbol in canonical_symbols():
            for timeframe in timeframes:
                candle_path, actual_symbol = resolve_candle_path(source_quarter_dir, symbol, timeframe)
                if candle_path is None:
                    missing_sources.append({
                        "year": year,
                        "quarter": quarter,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    })
                    continue
                dest_path = dest_root / year / quarter / symbol / f"candles_{timeframe}.csv"
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candle_path, dest_path)
                copied_files += 1
                copied_symbols.add(symbol)
                copied_timeframes.add(timeframe)

    return {
        "copied_files": copied_files,
        "copied_symbols": sorted(copied_symbols),
        "copied_timeframes": sorted(copied_timeframes),
        "missing_source_files": missing_sources,
    }


def _save_report(report: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)


def build_clean_year_dataset(
    source_root: Path,
    dest_root: Path,
    year: str,
    copy_timeframes: tuple[str, ...],
    fetch_missing_m5: bool,
    chunk_days: int,
    report_prefix: str | None = None,
) -> dict:
    copy_summary = _copy_existing_year(source_root, dest_root, year, copy_timeframes)

    initial_report = audit_year_dataset(dest_root, year=year, timeframes=("M5",), detect_gaps=True)
    initial_missing = initial_report["year_summary"]["missing_required_by_symbol"]

    fetch_result: dict = {
        "attempted": False,
        "symbols": [],
        "total_saved_bars": 0,
    }
    final_report = initial_report

    if fetch_missing_m5 and initial_missing:
        from mt5_m5_download import download_instruments

        fetch_symbols = sorted(initial_missing.keys())
        fetch_result["attempted"] = True
        fetch_result["symbols"] = fetch_symbols
        fetch_result["total_saved_bars"] = download_instruments(
            instruments=fetch_symbols,
            out_dir=dest_root,
            start_dt=datetime.strptime(f"{year}-01-01", "%Y-%m-%d"),
            end_dt=datetime.strptime(f"{year}-12-31", "%Y-%m-%d").replace(hour=23, minute=59, second=59),
            chunk_days=chunk_days,
        )
        final_report = audit_year_dataset(dest_root, year=year, timeframes=("M5",), detect_gaps=True)

    prefix = report_prefix or f"clean_{year}"
    json_path = dest_root / f"{prefix}_audit.json"
    csv_path = dest_root / f"{prefix}_audit.csv"
    _save_report(final_report, json_path, csv_path)

    return {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "year": year,
        "copy_summary": copy_summary,
        "initial_missing_symbols": sorted(initial_missing.keys()),
        "fetch_result": fetch_result,
        "final_report_json": str(json_path),
        "final_report_csv": str(csv_path),
        "final_fully_covered_symbols": final_report["year_summary"]["fully_covered_symbols"],
        "final_missing_required_by_symbol": final_report["year_summary"]["missing_required_by_symbol"],
    }


def _resolve_timeframes(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return RAW_TIMEFRAMES
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(items) if items else ("M5",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean one-year DataExtractor slice.")
    parser.add_argument("--source-root", default="data/DataExtractor", help="Existing DataExtractor root")
    parser.add_argument("--dest-root", required=True, help="Destination root for the clean dataset")
    parser.add_argument("--year", default="2025", help="Year to rebuild")
    parser.add_argument("--copy-timeframes", default="all", help="Comma-separated raw timeframes to copy or 'all'")
    parser.add_argument("--fetch-missing-m5", action="store_true", help="Fetch missing M5 symbols from MT5 into dest root")
    parser.add_argument("--chunk-days", type=int, default=31, help="MT5 chunk size in days")
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    result = build_clean_year_dataset(
        source_root=Path(args.source_root),
        dest_root=Path(args.dest_root),
        year=args.year,
        copy_timeframes=_resolve_timeframes(args.copy_timeframes),
        fetch_missing_m5=args.fetch_missing_m5,
        chunk_days=args.chunk_days,
    )

    summary_out = Path(args.summary_out) if args.summary_out else Path(args.dest_root) / f"clean_{args.year}_build_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved build summary to {summary_out}")
    print(f"Final fully covered symbols: {len(result['final_fully_covered_symbols'])}/{len(canonical_symbols())}")
    print(f"Final missing symbols: {sorted(result['final_missing_required_by_symbol'].keys())}")


if __name__ == "__main__":
    main()
