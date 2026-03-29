"""
Dataset Audit Script — Algo C2 v2
Drills into D:\\dataset-ml\\DataExtractor and reports data validity
for all years (2018-2026), quarters, symbols, and 10 timeframes.

Run on remote machine:
    python audit_dataset.py [--root D:\\dataset-ml\\DataExtractor] [--out audit_report.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

YEARS    = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]

# Minutes per bar for each timeframe
TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "H12": 720,
    "D1": 1440, "W1": 10080, "MN1": 43200,
}

# Expected trading minutes per week (FX: 120h/week Mon-Fri; 24/7 for BTC/indices/metals/energy)
CONTINUOUS_SYMBOLS = {"BTCUSD", "ETHUSD", "XAUUSD", "XAGUSD", "XBRUSD", "XTIUSD",
                      "AUS200", "US30", "GER40", "UK100", "NAS100", "EUSTX50",
                      "JPN225", "SPX500"}
FX_WEEK_MINUTES  = 120 * 60   # 7200 min/week
C24_WEEK_MINUTES = 168 * 60   # 10080 min/week

# Quarter date ranges (start inclusive, used for expected-bar calc)
QUARTER_DATES = {
    "Q1": ("01-01", "03-31"),
    "Q2": ("04-01", "06-30"),
    "Q3": ("07-01", "09-30"),
    "Q4": ("10-01", "12-31"),
}

# Gap alert thresholds (hours) — flag if any single gap exceeds this
GAP_WARN = {
    "M1": 4, "M5": 8, "M15": 12, "M30": 24,
    "H1": 48, "H4": 72, "H12": 120, "D1": 168, "W1": 336, "MN1": 720,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fast_count_and_bounds(filepath: Path) -> tuple[int, Optional[str], Optional[str]]:
    """
    Efficiently count data rows and return first/last timestamp strings.
    Reads only the first ~1000 bytes and last ~512 bytes for large files,
    then does a full line-count pass (no pandas required).
    """
    try:
        size = filepath.stat().st_size
    except OSError:
        return 0, None, None

    if size == 0:
        return 0, None, None

    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    row_count = 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip()
            cols = [c.strip().lower() for c in header.split(",")]
            ts_col = next((i for i, c in enumerate(cols)
                           if c in ("bar_time", "datetime", "date", "time", "timestamp")), 0)

            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row_count += 1
                parts = line.split(",")
                if parts and len(parts) > ts_col:
                    ts = parts[ts_col].strip().strip('"')
                    if row_count == 1:
                        first_ts = ts
                    last_ts = ts

    except Exception as e:
        return 0, None, None

    return row_count, first_ts, last_ts


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse MT5 bar_time string '2022.01.03 00:00:00' or ISO variants."""
    if not ts_str:
        return None
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _expected_bars(symbol: str, tf: str, first_dt: Optional[datetime],
                   last_dt: Optional[datetime]) -> Optional[int]:
    """
    Estimate expected bar count between first and last timestamp.
    Uses the actual span in the file (not quarter boundaries) so that
    partial files are assessed fairly.
    """
    if first_dt is None or last_dt is None:
        return None
    if tf not in TF_MINUTES:
        return None

    span_minutes = (last_dt - first_dt).total_seconds() / 60.0
    if span_minutes <= 0:
        return None

    tf_min = TF_MINUTES[tf]
    is_continuous = symbol in CONTINUOUS_SYMBOLS

    if tf in ("D1", "W1", "MN1"):
        # Calendar-based: just divide span by tf
        return max(1, int(span_minutes / tf_min))

    # Intraday: scale by trading-time ratio
    week_minutes = C24_WEEK_MINUTES if is_continuous else FX_WEEK_MINUTES
    trading_ratio = week_minutes / C24_WEEK_MINUTES  # fraction of 168h that's trading
    expected = (span_minutes * trading_ratio) / tf_min
    return max(1, int(expected))


def _completeness(actual: int, expected: Optional[int]) -> Optional[float]:
    if expected is None or expected == 0:
        return None
    return min(100.0, round(actual / expected * 100, 1))


def _detect_gaps(filepath: Path, tf: str) -> tuple[Optional[float], int]:
    """
    Read all timestamps, find the maximum gap and count of significant gaps.
    Returns (max_gap_hours, n_significant_gaps).
    Only runs for files <= 50 MB to stay fast.
    """
    try:
        size = filepath.stat().st_size
    except OSError:
        return None, 0

    if size > 50 * 1024 * 1024:   # skip gap analysis for very large files
        return None, 0

    tf_min = TF_MINUTES.get(tf, 1)
    warn_h = GAP_WARN.get(tf, 24)
    warn_min = warn_h * 60
    # A gap is significant if it's > 3× the expected bar interval
    sig_threshold_min = tf_min * 3

    prev_dt: Optional[datetime] = None
    max_gap_min = 0.0
    n_sig_gaps = 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip()
            cols = [c.strip().lower() for c in header.split(",")]
            ts_col = next((i for i, c in enumerate(cols)
                           if c in ("bar_time", "datetime", "date", "time", "timestamp")), 0)

            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if not parts or len(parts) <= ts_col:
                    continue
                ts_str = parts[ts_col].strip().strip('"')
                dt = _parse_ts(ts_str)
                if dt is None:
                    continue
                if prev_dt is not None:
                    gap_min = (dt - prev_dt).total_seconds() / 60.0
                    if gap_min > max_gap_min:
                        max_gap_min = gap_min
                    if gap_min > sig_threshold_min:
                        n_sig_gaps += 1
                prev_dt = dt

    except Exception:
        return None, 0

    max_gap_h = round(max_gap_min / 60, 2) if max_gap_min > 0 else 0.0
    return max_gap_h, n_sig_gaps


# ── Main audit ───────────────────────────────────────────────────────────────

def audit(root: Path, out_csv: Path, gap_analysis: bool = True) -> None:
    print(f"\nAudit root : {root}")
    print(f"Output CSV : {out_csv}")
    print(f"Gap analysis : {'yes (skips files >50MB)' if gap_analysis else 'disabled'}\n")

    rows: list[dict] = []

    # Collect all unique symbols across all year/quarter dirs (for summary)
    all_symbols: set[str] = set()

    # ── First pass: discover symbols ─────────────────────────────────────────
    for year in YEARS:
        for q in QUARTERS:
            qdir = root / year / q
            if not qdir.is_dir():
                continue
            for sym_dir in qdir.iterdir():
                if sym_dir.is_dir() and sym_dir.name != "X":
                    all_symbols.add(sym_dir.name)

    symbols_sorted = sorted(all_symbols)
    print(f"Symbols found : {len(symbols_sorted)}")
    print(f"Symbols       : {', '.join(symbols_sorted)}\n")
    print("Scanning files...", flush=True)

    total_files = 0
    present_files = 0
    missing_files = 0

    # ── Second pass: per-file audit ───────────────────────────────────────────
    for year in YEARS:
        for q in QUARTERS:
            qdir = root / year / q
            if not qdir.is_dir():
                # Entire quarter dir missing — record all as missing
                for sym in symbols_sorted:
                    for tf in TIMEFRAMES:
                        rows.append({
                            "year": year, "quarter": q, "symbol": sym, "timeframe": tf,
                            "status": "dir_missing",
                            "file_size_kb": 0, "row_count": 0,
                            "first_bar": "", "last_bar": "",
                            "expected_bars": "", "completeness_pct": "",
                            "max_gap_h": "", "n_sig_gaps": "",
                            "notes": f"Quarter dir missing: {qdir}",
                        })
                        missing_files += 1
                        total_files += 1
                continue

            for sym in symbols_sorted:
                sym_dir = qdir / sym
                for tf in TIMEFRAMES:
                    total_files += 1
                    csv_path = sym_dir / f"candles_{tf}.csv"

                    row: dict = {
                        "year": year, "quarter": q, "symbol": sym, "timeframe": tf,
                        "status": "", "file_size_kb": 0, "row_count": 0,
                        "first_bar": "", "last_bar": "",
                        "expected_bars": "", "completeness_pct": "",
                        "max_gap_h": "", "n_sig_gaps": "",
                        "notes": "",
                    }

                    if not sym_dir.is_dir():
                        row["status"] = "symbol_missing"
                        row["notes"] = f"Symbol dir not found: {sym_dir.name}"
                        missing_files += 1
                        rows.append(row)
                        continue

                    if not csv_path.exists():
                        row["status"] = "file_missing"
                        row["notes"] = f"candles_{tf}.csv not found"
                        missing_files += 1
                        rows.append(row)
                        continue

                    # File exists
                    fsize_kb = round(csv_path.stat().st_size / 1024, 1)
                    row["file_size_kb"] = fsize_kb

                    if fsize_kb < 0.1:
                        row["status"] = "empty"
                        row["notes"] = "File is 0 or near-empty"
                        missing_files += 1
                        rows.append(row)
                        continue

                    # Count rows + bounds
                    n_rows, first_ts, last_ts = _fast_count_and_bounds(csv_path)
                    row["row_count"] = n_rows
                    row["first_bar"] = first_ts or ""
                    row["last_bar"]  = last_ts or ""

                    if n_rows == 0:
                        row["status"] = "empty"
                        row["notes"] = "Parsed 0 data rows"
                        missing_files += 1
                        rows.append(row)
                        continue

                    first_dt = _parse_ts(first_ts)
                    last_dt  = _parse_ts(last_ts)
                    exp = _expected_bars(sym, tf, first_dt, last_dt)
                    comp = _completeness(n_rows, exp)

                    row["expected_bars"]    = exp if exp is not None else ""
                    row["completeness_pct"] = comp if comp is not None else ""

                    # Gap analysis
                    if gap_analysis:
                        max_gap_h, n_sig = _detect_gaps(csv_path, tf)
                        row["max_gap_h"]  = max_gap_h if max_gap_h is not None else "skipped(>50MB)"
                        row["n_sig_gaps"] = n_sig
                    else:
                        row["max_gap_h"] = ""; row["n_sig_gaps"] = ""

                    # Status classification
                    if comp is None:
                        row["status"] = "present_unverified"
                    elif comp >= 90:
                        row["status"] = "ok"
                    elif comp >= 70:
                        row["status"] = "partial"
                    elif comp >= 30:
                        row["status"] = "sparse"
                    else:
                        row["status"] = "very_sparse"

                    notes = []
                    if comp is not None and comp < 90:
                        notes.append(f"completeness={comp}%")
                    if isinstance(row["max_gap_h"], float) and row["max_gap_h"] > GAP_WARN.get(tf, 24):
                        notes.append(f"max_gap={row['max_gap_h']}h")
                    if isinstance(row["n_sig_gaps"], int) and row["n_sig_gaps"] > 10:
                        notes.append(f"sig_gaps={row['n_sig_gaps']}")
                    row["notes"] = "; ".join(notes)

                    present_files += 1
                    rows.append(row)

            # Progress indicator
            sys.stdout.write(f"\r  {year}/{q} done ({len(rows)} entries so far)   ")
            sys.stdout.flush()

    print(f"\n\nScan complete. Total entries: {len(rows)}")
    print(f"  Present  : {present_files}")
    print(f"  Missing  : {missing_files}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["year","quarter","symbol","timeframe","status",
                  "file_size_kb","row_count","expected_bars","completeness_pct",
                  "first_bar","last_bar","max_gap_h","n_sig_gaps","notes"]

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {out_csv}")

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(rows, symbols_sorted)


def _print_summary(rows: list[dict], symbols: list[str]) -> None:
    print("\n" + "="*80)
    print("SUMMARY BY TIMEFRAME × YEAR")
    print("="*80)

    # Grid: year × TF → (ok, partial, sparse, missing)
    from collections import defaultdict
    grid: dict[tuple, dict] = defaultdict(lambda: {"ok":0,"partial":0,"sparse":0,"missing":0,"total":0})

    for r in rows:
        key = (r["year"], r["timeframe"])
        grid[key]["total"] += 1
        s = r["status"]
        if s in ("ok", "present_unverified"):
            grid[key]["ok"] += 1
        elif s in ("partial",):
            grid[key]["partial"] += 1
        elif s in ("sparse","very_sparse","empty"):
            grid[key]["sparse"] += 1
        else:
            grid[key]["missing"] += 1

    # Header
    tf_order = ["M1","M5","M15","M30","H1","H4","H12","D1","W1","MN1"]
    yr_order  = ["2018","2019","2020","2021","2022","2023","2024","2025","2026"]
    col_w = 12

    header = f"{'YEAR':<6}" + "".join(f"{tf:>{col_w}}" for tf in tf_order)
    print(header)
    print("-" * len(header))

    for yr in yr_order:
        row_str = f"{yr:<6}"
        for tf in tf_order:
            g = grid.get((yr, tf), {"ok":0,"partial":0,"sparse":0,"missing":0,"total":0})
            tot = g["total"]
            if tot == 0:
                cell = "---"
            elif g["missing"] == tot:
                cell = "MISSING"
            elif g["ok"] == tot:
                cell = f"OK({tot})"
            elif g["ok"] + g["partial"] == tot:
                cell = f"PART({g['partial']})"
            else:
                pct = round((g["ok"]+g["partial"])/tot*100) if tot else 0
                cell = f"{pct}% ({tot})"
            row_str += f"{cell:>{col_w}}"
        print(row_str)

    print("\n" + "="*80)
    print("MISSING FILES BY TIMEFRAME (all years)")
    print("="*80)
    for tf in tf_order:
        missing = [r for r in rows if r["timeframe"] == tf and
                   r["status"] in ("file_missing","symbol_missing","dir_missing","empty","very_sparse")]
        if missing:
            years_missing = sorted(set(r["year"] for r in missing))
            syms_missing  = sorted(set(r["symbol"] for r in missing))
            print(f"  {tf:>4}: {len(missing):>4} missing/sparse  "
                  f"years={years_missing}  "
                  f"n_syms={len(syms_missing)}")
        else:
            print(f"  {tf:>4}: all present")

    print("\n" + "="*80)
    print("COMPLETENESS ISSUES (completeness < 90%)")
    print("="*80)
    issues = [r for r in rows
              if r["completeness_pct"] != "" and isinstance(r["completeness_pct"], (int, float))
              and r["completeness_pct"] < 90]
    issues.sort(key=lambda r: (r["timeframe"], r["year"], r["symbol"]))
    if not issues:
        print("  None — all present files meet 90% threshold.")
    else:
        print(f"  {'TF':<5} {'Year':<6} {'Q':<3} {'Symbol':<10} {'Rows':>8} {'Exp':>8} {'Comp%':>7}  {'Notes'}")
        print(f"  {'-'*5} {'-'*6} {'-'*3} {'-'*10} {'-'*8} {'-'*8} {'-'*7}  {'-'*30}")
        for r in issues:
            print(f"  {r['timeframe']:<5} {r['year']:<6} {r['quarter']:<3} {r['symbol']:<10} "
                  f"{str(r['row_count']):>8} {str(r['expected_bars']):>8} "
                  f"{str(r['completeness_pct']):>7}  {r['notes']}")

    print("\n" + "="*80)
    print("GAP ALERTS (max gap > threshold)")
    print("="*80)
    gap_alerts = [r for r in rows
                  if isinstance(r.get("max_gap_h"), float)
                  and r["max_gap_h"] > GAP_WARN.get(r["timeframe"], 24)]
    gap_alerts.sort(key=lambda r: -r["max_gap_h"])
    if not gap_alerts:
        print("  None.")
    else:
        print(f"  {'TF':<5} {'Year':<6} {'Q':<3} {'Symbol':<10} {'MaxGap(h)':>10} {'SigGaps':>8}")
        print(f"  {'-'*5} {'-'*6} {'-'*3} {'-'*10} {'-'*10} {'-'*8}")
        for r in gap_alerts[:50]:   # cap at 50 lines
            print(f"  {r['timeframe']:<5} {r['year']:<6} {r['quarter']:<3} {r['symbol']:<10} "
                  f"{r['max_gap_h']:>10} {str(r['n_sig_gaps']):>8}")
        if len(gap_alerts) > 50:
            print(f"  ... ({len(gap_alerts)-50} more — see CSV)")

    print("\n" + "="*80)
    print("SYMBOL COVERAGE (H1 across 2018-2026)")
    print("="*80)
    for sym in symbols:
        h1_rows = [r for r in rows if r["symbol"] == sym and r["timeframe"] == "H1"]
        ok_yrs   = sorted(r["year"] for r in h1_rows if r["status"] in ("ok","present_unverified"))
        miss_yrs = sorted(r["year"] for r in h1_rows if r["status"] not in ("ok","present_unverified","partial"))
        total_rows = sum(r["row_count"] for r in h1_rows if isinstance(r["row_count"], int))
        print(f"  {sym:<10}  H1 present={len(ok_yrs)} yrs  missing={miss_yrs or 'none'}  total_h1_bars={total_rows:,}")

    print("\n" + "="*80)
    print("TRAINING READINESS ASSESSMENT")
    print("="*80)
    phases = [
        ("Phase 1 Macro (H1, 2018-2020)",    "H1",  ["2018","2019","2020"]),
        ("Phase 2 Intraday (M15, 2022-2023)", "M15", ["2022","2023"]),
        ("Phase 3 Recent (M5, 2024-2025)",    "M5",  ["2024","2025"]),
        ("Phase 3b Recent (M1, 2025 Q4+2026)","M1",  ["2025","2026"]),
    ]
    for label, tf, yrs in phases:
        phase_rows = [r for r in rows if r["timeframe"] == tf and r["year"] in yrs]
        ok = sum(1 for r in phase_rows if r["status"] in ("ok","present_unverified","partial"))
        total = len(phase_rows)
        avg_comp_vals = [r["completeness_pct"] for r in phase_rows
                         if isinstance(r.get("completeness_pct"), (int,float))]
        avg_comp = round(sum(avg_comp_vals)/len(avg_comp_vals), 1) if avg_comp_vals else None
        ready = "READY" if ok >= total * 0.85 else ("PARTIAL" if ok >= total * 0.5 else "NOT READY")
        print(f"  {label}")
        print(f"    Files: {ok}/{total} present  avg_completeness={avg_comp}%  -> {ready}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Algo C2 dataset validity")
    parser.add_argument("--root", default=r"D:\dataset-ml\DataExtractor",
                        help="Root DataExtractor directory")
    parser.add_argument("--out",  default=r"D:\dataset-ml\DataExtractor\full_audit.csv",
                        help="Output CSV path")
    parser.add_argument("--no-gaps", action="store_true",
                        help="Skip gap analysis (faster)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    audit(root=root, out_csv=Path(args.out), gap_analysis=not args.no_gaps)
