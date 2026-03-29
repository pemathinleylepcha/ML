"""
Tick Data Audit Script — Algo C2 v2
Audits all tick CSVs in D:\\dataset-ml\\2025-March-2026-March

Reports: date range, row count, duplicate files, missing pairs,
gap analysis, bid/ask validity, spread stats.

Usage:
    python audit_ticks.py [--root D:\\dataset-ml\\2025-March-2026-March]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Expected symbols (43-node universe) ──────────────────────────────────────

FX_TRADEABLE = [
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADCHF","CADJPY","CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","EURUSD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPNZD","GBPUSD",
    "NZDCAD","NZDCHF","NZDJPY","NZDUSD",
    "USDCAD","USDCHF","USDJPY",
]
SIGNAL_ONLY = [
    "BTCUSD",
    "AUS200","GER40","UK100","NAS100","EUSTX50","JPN225","SPX500","US30",
    "XTIUSD","XBRUSD",
    "XAUUSD","XAGUSD",
    "USDMXN","USDZAR",
]
ALL_EXPECTED = FX_TRADEABLE + SIGNAL_ONLY  # 43


def _count_rows_and_bounds(fpath: Path) -> tuple[int, Optional[str], Optional[str]]:
    """Fast row count + first/last timestamp. Reads line-by-line."""
    n = 0
    first_ts = last_ts = None
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            _hdr = fh.readline()  # skip header
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                # Tab-separated: DATE \t TIME \t BID \t ASK ...
                parts = line.split("\t")
                if len(parts) >= 2:
                    ts = parts[0].strip() + " " + parts[1].strip()
                    if n == 1:
                        first_ts = ts
                    last_ts = ts
    except Exception:
        pass
    return n, first_ts, last_ts


def _parse_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    # "2025.04.09 00:05:10.496" — strip subsecond
    ts_clean = ts.split(".")[0] if ts.count(".") > 2 else ts
    ts_clean = ts_clean.replace(".", "-", 2)  # 2025-04-09 00:05:10
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue
    return None


def _sample_spreads(fpath: Path, n_sample: int = 5000) -> dict:
    """Read up to n_sample rows, compute bid/ask/spread stats."""
    spreads = []
    bids = []
    asks = []
    zero_spreads = 0
    neg_spreads = 0
    rows_read = 0
    flags_counter: dict[str, int] = defaultdict(int)

    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            hdr = fh.readline().lower()
            for line in fh:
                if rows_read >= n_sample:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                try:
                    bid = float(parts[2]) if parts[2].strip() else None
                    ask = float(parts[3]) if parts[3].strip() else None
                    flag = parts[6].strip() if len(parts) > 6 else ""
                    flags_counter[flag] += 1
                    if bid is not None and ask is not None:
                        sp = ask - bid
                        spreads.append(sp)
                        bids.append(bid)
                        asks.append(ask)
                        if sp == 0:
                            zero_spreads += 1
                        elif sp < 0:
                            neg_spreads += 1
                    rows_read += 1
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass

    if not spreads:
        return {}

    spreads_s = sorted(spreads)
    n = len(spreads_s)
    return {
        "n_sampled": rows_read,
        "avg_bid": round(sum(bids) / len(bids), 6),
        "avg_ask": round(sum(asks) / len(asks), 6),
        "min_spread": round(min(spreads_s), 6),
        "max_spread": round(max(spreads_s), 6),
        "avg_spread": round(sum(spreads_s) / n, 6),
        "median_spread": round(spreads_s[n // 2], 6),
        "zero_spreads": zero_spreads,
        "neg_spreads": neg_spreads,
        "flags": dict(sorted(flags_counter.items(), key=lambda x: -x[1])[:5]),
    }


def _detect_major_gaps(fpath: Path, gap_hours: float = 12.0) -> list[tuple[str, str, float]]:
    """Find gaps > gap_hours between consecutive ticks. Returns list of (from, to, hours)."""
    gaps = []
    prev_dt = None
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            fh.readline()  # header
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                ts = parts[0].strip() + " " + parts[1].strip()
                dt = _parse_dt(ts)
                if dt is None:
                    continue
                if prev_dt is not None:
                    diff_h = (dt - prev_dt).total_seconds() / 3600.0
                    if diff_h > gap_hours:
                        gaps.append((prev_dt.strftime("%Y-%m-%d %H:%M"),
                                     dt.strftime("%Y-%m-%d %H:%M"),
                                     round(diff_h, 1)))
                prev_dt = dt
    except Exception:
        pass
    return gaps


def audit_ticks(root: Path) -> None:
    print(f"\nTick Audit root: {root}\n")

    # Collect all CSV files recursively
    all_files: list[Path] = sorted(root.rglob("*.csv"))
    print(f"Total CSV files found: {len(all_files)}\n")

    # Map symbol → list of files
    sym_files: dict[str, list[Path]] = defaultdict(list)
    for f in all_files:
        # Extract symbol from filename: SYMBOL_STARTDATE_ENDDATE.csv
        stem = f.stem  # e.g. EURUSD_202504090005_202603242359
        sym = stem.split("_")[0]
        sym_files[sym].append(f)

    found_symbols = sorted(sym_files.keys())
    print(f"Symbols found    : {len(found_symbols)}")
    print(f"Symbols          : {', '.join(found_symbols)}\n")

    # Missing from universe
    missing_syms = [s for s in ALL_EXPECTED if s not in sym_files]
    extra_syms   = [s for s in found_symbols if s not in ALL_EXPECTED]
    print(f"Missing from 43-node spec : {missing_syms}")
    print(f"Extra (not in spec)       : {extra_syms}\n")

    # Duplicates
    dups = {s: fs for s, fs in sym_files.items() if len(fs) > 1}
    if dups:
        print("="*70)
        print("DUPLICATE FILES (same symbol, multiple files)")
        print("="*70)
        for sym, files in sorted(dups.items()):
            print(f"  {sym}:")
            for f in files:
                size_mb = round(f.stat().st_size / 1024 / 1024, 1)
                print(f"    [{size_mb:>8.1f} MB]  {f.name}")
        print()

    # Per-symbol detailed report
    print("="*70)
    print("PER-SYMBOL TICK REPORT")
    print("="*70)
    fmt = "  {:<12} {:>10}  {:>19}  {:>19}  {:>8}  {:>10}  {}"
    print(fmt.format("SYMBOL", "ROWS", "FIRST TICK", "LAST TICK",
                     "SIZE MB", "AVG_SPREAD", "NOTES"))
    print("  " + "-"*10 + "  " + "-"*10 + "  " + "-"*19 + "  " + "-"*19 +
          "  " + "-"*8 + "  " + "-"*10 + "  " + "-"*20)

    all_results: list[dict] = []

    for sym in sorted(ALL_EXPECTED):
        files = sym_files.get(sym, [])
        if not files:
            print(fmt.format(sym, "MISSING", "---", "---", "---", "---", "NOT IN FOLDER"))
            all_results.append({"symbol": sym, "status": "missing"})
            continue

        # Use largest file if duplicates (most complete)
        fpath = max(files, key=lambda f: f.stat().st_size)
        size_mb = round(fpath.stat().st_size / 1024 / 1024, 1)

        n_rows, first_ts, last_ts = _count_rows_and_bounds(fpath)

        # Sample spread stats
        stats = _sample_spreads(fpath, n_sample=3000)
        avg_sp = stats.get("avg_spread", "N/A")
        neg_sp = stats.get("neg_spreads", 0)
        zero_sp = stats.get("zero_spreads", 0)
        flags = stats.get("flags", {})

        notes = []
        if len(files) > 1:
            notes.append(f"DUPLICATE x{len(files)}")
        if neg_sp > 0:
            notes.append(f"neg_spread={neg_sp}")
        if zero_sp > 100:
            notes.append(f"zero_spread={zero_sp}")

        # Date coverage from filename
        stem = fpath.stem
        parts = stem.split("_")
        start_str = parts[1] if len(parts) > 1 else "?"
        end_str   = parts[2] if len(parts) > 2 else "?"

        # Compute span
        first_dt = _parse_dt(first_ts)
        last_dt  = _parse_dt(last_ts)
        if first_dt and last_dt:
            span_days = (last_dt - first_dt).days
        else:
            span_days = 0

        note_str = "; ".join(notes)
        avg_sp_str = f"{avg_sp:.6f}" if isinstance(avg_sp, float) else str(avg_sp)
        first_disp = (first_ts or "")[:19]
        last_disp  = (last_ts or "")[:19]

        print(fmt.format(sym, f"{n_rows:,}", first_disp, last_disp,
                         str(size_mb), avg_sp_str, note_str))

        all_results.append({
            "symbol": sym, "status": "present", "n_rows": n_rows,
            "first_ts": first_ts, "last_ts": last_ts,
            "size_mb": size_mb, "span_days": span_days,
            "avg_spread": avg_sp, "neg_spreads": neg_sp,
            "zero_spreads": zero_sp, "flags": flags,
            "n_files": len(files),
        })

    # Gap analysis for key FX pairs and all present symbols < 100 MB
    print("\n" + "="*70)
    print("GAP ANALYSIS (gaps > 12h; files <= 500 MB sampled)")
    print("="*70)

    gap_threshold = 12.0
    for sym in sorted(ALL_EXPECTED):
        files = sym_files.get(sym, [])
        if not files:
            continue
        fpath = max(files, key=lambda f: f.stat().st_size)
        if fpath.stat().st_size > 500 * 1024 * 1024:
            print(f"  {sym:<12} skipped (>{500}MB)")
            continue

        gaps = _detect_major_gaps(fpath, gap_hours=gap_threshold)
        weekend_gaps = [g for g in gaps if g[2] <= 56]  # ~Fri close to Mon open
        big_gaps = [g for g in gaps if g[2] > 56]       # abnormal

        if not gaps:
            print(f"  {sym:<12} no gaps > {gap_threshold}h  OK")
        else:
            print(f"  {sym:<12} {len(gaps)} gaps > {gap_threshold}h  "
                  f"(weekend-like: {len(weekend_gaps)}, abnormal >56h: {len(big_gaps)})")
            for g in big_gaps[:3]:
                print(f"             GAP: {g[0]}  ->  {g[1]}  ({g[2]}h)")

    # Coverage summary
    print("\n" + "="*70)
    print("COVERAGE SUMMARY")
    print("="*70)
    present = [r for r in all_results if r["status"] == "present"]
    missing = [r for r in all_results if r["status"] == "missing"]

    print(f"\n  Total expected  : {len(ALL_EXPECTED)}")
    print(f"  Present         : {len(present)}")
    print(f"  Missing         : {len(missing)}")
    print(f"  Missing symbols : {[r['symbol'] for r in missing]}\n")

    if present:
        print(f"  {'Symbol':<12} {'Rows':>12} {'SpanDays':>9} {'SizeMB':>8} {'AvgSpread':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*9} {'-'*8} {'-'*12}")
        for r in sorted(present, key=lambda x: x["symbol"]):
            print(f"  {r['symbol']:<12} {r['n_rows']:>12,} {r['span_days']:>9} "
                  f"{r['size_mb']:>8.1f} {str(r['avg_spread']):>12}")

    # Date range overview
    print("\n" + "="*70)
    print("DATE RANGE OVERVIEW")
    print("="*70)
    print(f"\n  {'Symbol':<12} {'First Tick':<22} {'Last Tick':<22} {'Days':>6}")
    print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*6}")
    for r in sorted(present, key=lambda x: x.get("first_ts") or ""):
        ft = (r.get("first_ts") or "")[:19]
        lt = (r.get("last_ts") or "")[:19]
        print(f"  {r['symbol']:<12} {ft:<22} {lt:<22} {r['span_days']:>6}")

    # Identify pairs with very short history (< 90 days)
    short = [r for r in present if r.get("span_days", 999) < 90]
    if short:
        print(f"\n  WARNING — Pairs with < 90 days tick history:")
        for r in sorted(short, key=lambda x: x.get("span_days", 0)):
            print(f"    {r['symbol']:<12}  {r['span_days']} days  "
                  f"({(r.get('first_ts') or '')[:10]} to {(r.get('last_ts') or '')[:10]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\dataset-ml\2025-March-2026-March")
    args = parser.parse_args()
    audit_ticks(Path(args.root))
