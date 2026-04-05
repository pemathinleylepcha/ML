from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dataextractor_contract import canonical_symbols, resolve_candle_path

@dataclass(slots=True)
class QuarterWindow:
    start: pd.Timestamp
    end: pd.Timestamp


def _quarter_window(year: int, quarter: str) -> QuarterWindow:
    if quarter == "Q1":
        return QuarterWindow(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=3, day=31, hour=23, minute=55))
    if quarter == "Q2":
        return QuarterWindow(pd.Timestamp(year=year, month=4, day=1), pd.Timestamp(year=year, month=6, day=30, hour=23, minute=55))
    if quarter == "Q3":
        return QuarterWindow(pd.Timestamp(year=year, month=7, day=1), pd.Timestamp(year=year, month=9, day=30, hour=23, minute=55))
    return QuarterWindow(pd.Timestamp(year=year, month=10, day=1), pd.Timestamp(year=year, month=12, day=31, hour=23, minute=55))


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.normalize() - pd.Timedelta(days=ts.weekday())


def _slot_id(ts: pd.Timestamp) -> int:
    return int(ts.weekday()) * 288 + int(ts.hour) * 12 + int(ts.minute // 5)


def _load_timestamps(csv_path: Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(csv_path, usecols=["bar_time"])
    ts = pd.to_datetime(frame["bar_time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    ts = ts.dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(ts)


def _infer_daily_reference_counts(timestamps: pd.DatetimeIndex, window: QuarterWindow) -> dict[int, int]:
    if len(timestamps) == 0:
        return {}

    ts_frame = pd.DataFrame({"dt": timestamps})
    ts_frame["date"] = ts_frame["dt"].dt.normalize()
    daily_counts = ts_frame.groupby("date").size()

    reference: dict[int, int] = {}
    all_dates = pd.date_range(window.start.normalize(), window.end.normalize(), freq="D")
    total_by_weekday = {weekday: 0 for weekday in range(7)}
    for date in all_dates:
        total_by_weekday[int(date.weekday())] += 1

    for weekday in range(7):
        weekday_dates = [date for date in daily_counts.index if int(date.weekday()) == weekday]
        if not weekday_dates:
            continue
        active_ratio = len(weekday_dates) / max(total_by_weekday.get(weekday, 1), 1)
        if active_ratio < 0.35:
            continue
        counts = daily_counts.loc[weekday_dates].astype(float)
        reference[weekday] = max(1, int(round(counts.quantile(0.9))))
    return reference


def audit_m5_completeness(root: Path, year: str, min_completeness_pct: float) -> dict:
    rows: list[dict] = []
    failing: list[dict] = []
    year_int = int(year)

    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        quarter_dir = root / year / quarter
        window = _quarter_window(year_int, quarter)
        for symbol in canonical_symbols():
            csv_path, actual_symbol = resolve_candle_path(quarter_dir, symbol, "M5")
            if csv_path is None:
                rows.append(
                    {
                        "year": year,
                        "quarter": quarter,
                        "symbol": symbol,
                        "actual_symbol_dir": actual_symbol,
                        "exists": False,
                        "actual_rows": 0,
                        "expected_rows": 0,
                        "completeness_pct": 0.0,
                        "max_gap_minutes": None,
                        "significant_gap_count": None,
                    }
                )
                failing.append(rows[-1])
                continue

            timestamps = _load_timestamps(csv_path)
            daily_reference = _infer_daily_reference_counts(timestamps, window)
            expected_rows = 0
            for date in pd.date_range(window.start.normalize(), window.end.normalize(), freq="D"):
                expected_rows += daily_reference.get(int(date.weekday()), 0)

            actual_rows = int(len(timestamps))
            completeness_pct = float(round((actual_rows / expected_rows) * 100.0, 2)) if expected_rows > 0 else 0.0

            ts_frame = pd.DataFrame({"dt": timestamps})
            ts_frame["date"] = ts_frame["dt"].dt.normalize()
            intra_day_gaps: list[float] = []
            intra_day_missing_bars = 0
            for _, group in ts_frame.groupby("date", sort=True):
                diffs = group["dt"].diff().dropna()
                minutes = diffs.dt.total_seconds().div(60.0)
                for gap in minutes:
                    gap_value = float(gap)
                    if gap_value > 5.0:
                        intra_day_gaps.append(gap_value)
                        intra_day_missing_bars += max(0, int(round(gap_value / 5.0)) - 1)

            max_gap_minutes = float(round(max(intra_day_gaps), 2)) if intra_day_gaps else 0.0
            significant_gap_count = int(intra_day_missing_bars)

            row = {
                "year": year,
                "quarter": quarter,
                "symbol": symbol,
                "actual_symbol_dir": actual_symbol or symbol,
                "exists": True,
                "actual_rows": actual_rows,
                "expected_rows": expected_rows,
                "completeness_pct": completeness_pct,
                "max_gap_minutes": max_gap_minutes,
                "significant_gap_count": significant_gap_count,
            }
            rows.append(row)
            if completeness_pct < min_completeness_pct:
                failing.append(row)

    overall_actual = sum(row["actual_rows"] for row in rows)
    overall_expected = sum(row["expected_rows"] for row in rows)
    overall_pct = round((overall_actual / overall_expected) * 100.0, 2) if overall_expected > 0 else 0.0
    return {
        "root": str(root),
        "year": year,
        "min_completeness_pct": min_completeness_pct,
        "overall_actual_rows": overall_actual,
        "overall_expected_rows": overall_expected,
        "overall_completeness_pct": overall_pct,
        "rows": rows,
        "failing": failing,
    }


def write_csv(report: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "quarter",
        "symbol",
        "actual_symbol_dir",
        "exists",
        "actual_rows",
        "expected_rows",
        "completeness_pct",
        "max_gap_minutes",
        "significant_gap_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit M5 candle completeness for a one-year dataset.")
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--year", default="2025", help="Target year")
    parser.add_argument("--min-completeness-pct", type=float, default=95.0, help="Threshold for failures")
    parser.add_argument("--json-out", default=None, help="Optional JSON output")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")
    args = parser.parse_args()

    report = audit_m5_completeness(Path(args.root), args.year, args.min_completeness_pct)
    json_out = Path(args.json_out) if args.json_out else Path(f"data/m5_completeness_{args.year}.json")
    csv_out = Path(args.csv_out) if args.csv_out else Path(f"data/m5_completeness_{args.year}.csv")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report, csv_out)
    print(f"Saved JSON report to {json_out}")
    print(f"Saved CSV report to {csv_out}")
    print(f"Overall completeness: {report['overall_completeness_pct']}%")
    print(f"Failing symbol-quarters: {len(report['failing'])}")


if __name__ == "__main__":
    main()
