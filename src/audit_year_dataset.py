from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from dataextractor_contract import (
    RAW_TIMEFRAMES,
    RESEARCH_REQUIRED_RAW_TIMEFRAMES,
    canonical_symbols,
    resolve_candle_path,
    symbol_dir_candidates,
)
from universe import TIMEFRAME_MINUTES

QUARTERS = ("Q1", "Q2", "Q3", "Q4")


def _parse_ts(value: str) -> datetime | None:
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _scan_csv(path: Path, timeframe: str, detect_gaps: bool) -> dict:
    first_ts: str | None = None
    last_ts: str | None = None
    row_count = 0
    max_gap_minutes = 0.0
    significant_gap_count = 0
    prev_dt: datetime | None = None
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 5)
    significant_gap_threshold = tf_minutes * 3

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        ts_idx = 0
        if header:
            lowered = [str(col).strip().lower().strip("<>") for col in header]
            ts_idx = next(
                (idx for idx, col in enumerate(lowered) if col in {"bar_time", "datetime", "date", "time", "timestamp"}),
                0,
            )

        for parts in reader:
            if not parts or len(parts) <= ts_idx:
                continue
            ts = parts[ts_idx].strip().strip('"')
            if not ts:
                continue
            row_count += 1
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            if detect_gaps:
                dt = _parse_ts(ts)
                if dt is None:
                    continue
                if prev_dt is not None:
                    gap_minutes = (dt - prev_dt).total_seconds() / 60.0
                    max_gap_minutes = max(max_gap_minutes, gap_minutes)
                    if gap_minutes > significant_gap_threshold:
                        significant_gap_count += 1
                prev_dt = dt

    result = {
        "rows": row_count,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }
    if detect_gaps:
        result["max_gap_minutes"] = round(max_gap_minutes, 2)
        result["significant_gap_count"] = significant_gap_count
    return result


def _resolve_timeframes(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return RAW_TIMEFRAMES
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return RESEARCH_REQUIRED_RAW_TIMEFRAMES
    return tuple(items)


def audit_year_dataset(
    root: Path,
    year: str,
    timeframes: tuple[str, ...],
    detect_gaps: bool,
) -> dict:
    year_dir = root / year
    if not year_dir.is_dir():
        raise ValueError(f"Year directory not found: {year_dir}")

    symbols = canonical_symbols()
    per_quarter: dict[str, list[dict]] = {}
    alias_usage: dict[str, set[str]] = {}
    missing_required_by_symbol: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    missing_required_by_quarter: dict[str, list[str]] = {quarter: [] for quarter in QUARTERS}
    fully_covered_symbols: list[str] = []

    for quarter in QUARTERS:
        quarter_dir = year_dir / quarter
        quarter_records: list[dict] = []
        actual_dirs = sorted([item.name for item in quarter_dir.iterdir() if item.is_dir()]) if quarter_dir.is_dir() else []
        expected_dir_names = {candidate for symbol in symbols for candidate in symbol_dir_candidates(symbol)}
        unexpected_dirs = sorted(set(actual_dirs) - expected_dir_names)

        for symbol in symbols:
            record = {
                "year": year,
                "quarter": quarter,
                "symbol": symbol,
                "timeframes": {},
                "actual_symbol_dir": None,
                "alias_used": False,
                "missing_required": False,
            }
            actual_symbol_dir: str | None = None

            for timeframe in timeframes:
                candle_path, resolved_symbol = resolve_candle_path(quarter_dir, symbol, timeframe)
                if resolved_symbol is not None and actual_symbol_dir is None:
                    actual_symbol_dir = resolved_symbol
                if candle_path is None:
                    record["timeframes"][timeframe] = {"exists": False}
                    continue

                stats = _scan_csv(candle_path, timeframe, detect_gaps=detect_gaps and timeframe == "M5")
                tf_record = {"exists": True, "path": str(candle_path), **stats}
                record["timeframes"][timeframe] = tf_record

            record["actual_symbol_dir"] = actual_symbol_dir
            record["alias_used"] = actual_symbol_dir not in (None, symbol)
            if record["alias_used"] and actual_symbol_dir is not None:
                alias_usage.setdefault(symbol, set()).add(actual_symbol_dir)

            missing_required = [
                timeframe for timeframe in RESEARCH_REQUIRED_RAW_TIMEFRAMES
                if not record["timeframes"].get(timeframe, {}).get("exists", False)
            ]
            record["missing_required"] = bool(missing_required)
            record["missing_required_timeframes"] = missing_required
            if missing_required:
                missing_required_by_symbol[symbol].append(quarter)
                missing_required_by_quarter[quarter].append(symbol)

            quarter_records.append(record)

        quarter_summary = {
            "quarter": quarter,
            "records": quarter_records,
            "unexpected_dirs": unexpected_dirs,
        }
        per_quarter[quarter] = quarter_summary

    for symbol in symbols:
        if not missing_required_by_symbol[symbol]:
            fully_covered_symbols.append(symbol)

    fetch_plan = []
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    for symbol in symbols:
        reasons: list[str] = []
        if missing_required_by_symbol[symbol]:
            reasons.append("missing_M5_quarters=" + ",".join(missing_required_by_symbol[symbol]))
        if symbol in alias_usage:
            reasons.append("alias_dirs=" + ",".join(sorted(alias_usage[symbol])))
        if reasons:
            fetch_plan.append({
                "symbol": symbol,
                "start": start_date,
                "end": end_date,
                "reasons": reasons,
            })

    fetch_instruments = ",".join(item["symbol"] for item in fetch_plan)
    fetch_command = None
    if fetch_plan:
        fetch_command = (
            f"python src/mt5_m5_download.py --out <clean_output_root> "
            f"--start {start_date} --end {end_date} --instruments {fetch_instruments}"
        )

    return {
        "root": str(root),
        "year": year,
        "timeframes": list(timeframes),
        "required_raw_timeframes": list(RESEARCH_REQUIRED_RAW_TIMEFRAMES),
        "symbol_count": len(symbols),
        "quarter_summaries": per_quarter,
        "year_summary": {
            "fully_covered_symbols": fully_covered_symbols,
            "missing_required_by_symbol": {k: v for k, v in missing_required_by_symbol.items() if v},
            "missing_required_by_quarter": {k: v for k, v in missing_required_by_quarter.items() if v},
            "alias_usage": {k: sorted(v) for k, v in alias_usage.items()},
            "fetch_plan": fetch_plan,
            "fetch_command": fetch_command,
        },
    }


def write_csv_report(report: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for quarter_summary in report["quarter_summaries"].values():
        for record in quarter_summary["records"]:
            base = {
                "year": record["year"],
                "quarter": record["quarter"],
                "symbol": record["symbol"],
                "actual_symbol_dir": record["actual_symbol_dir"],
                "alias_used": record["alias_used"],
                "missing_required": record["missing_required"],
                "missing_required_timeframes": ",".join(record["missing_required_timeframes"]),
            }
            for timeframe, tf_record in record["timeframes"].items():
                row = dict(base)
                row["timeframe"] = timeframe
                row["exists"] = tf_record.get("exists", False)
                row["rows"] = tf_record.get("rows")
                row["first_ts"] = tf_record.get("first_ts")
                row["last_ts"] = tf_record.get("last_ts")
                row["max_gap_minutes"] = tf_record.get("max_gap_minutes")
                row["significant_gap_count"] = tf_record.get("significant_gap_count")
                row["path"] = tf_record.get("path")
                rows.append(row)

    fieldnames = [
        "year",
        "quarter",
        "symbol",
        "actual_symbol_dir",
        "alias_used",
        "missing_required",
        "missing_required_timeframes",
        "timeframe",
        "exists",
        "rows",
        "first_ts",
        "last_ts",
        "max_gap_minutes",
        "significant_gap_count",
        "path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one-year DataExtractor coverage for the research dataset contract.")
    parser.add_argument("--root", default="data/DataExtractor", help="DataExtractor root")
    parser.add_argument("--year", default="2025", help="Target year to audit")
    parser.add_argument("--timeframes", default="M5", help="Comma-separated timeframes or 'all'")
    parser.add_argument("--detect-gaps", action="store_true", help="Scan M5 files for timestamp gaps")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    report = audit_year_dataset(
        root=Path(args.root),
        year=args.year,
        timeframes=_resolve_timeframes(args.timeframes),
        detect_gaps=args.detect_gaps,
    )

    json_out = Path(args.json_out) if args.json_out else Path(f"data/year_audit_{args.year}.json")
    csv_out = Path(args.csv_out) if args.csv_out else Path(f"data/year_audit_{args.year}.csv")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_out)

    summary = report["year_summary"]
    print(f"Saved JSON report to {json_out}")
    print(f"Saved CSV report to {csv_out}")
    print(f"Fully covered symbols: {len(summary['fully_covered_symbols'])}/{report['symbol_count']}")
    print(f"Fetch-plan symbols: {len(summary['fetch_plan'])}")
    if summary["fetch_command"]:
        print(summary["fetch_command"])


if __name__ == "__main__":
    main()
