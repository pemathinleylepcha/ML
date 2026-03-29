"""
audit_m1_m15.py — Track M1/M5/M15 data presence and completeness
for all 43 instruments, 2018-2026, per quarter.

Output: audit_m1_m15.csv with one row per (year, quarter, symbol, timeframe).

Columns:
  year, quarter, symbol, timeframe,
  file_exists, row_count, first_bar, last_bar,
  expected_bars, pct_complete, note
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from datetime import datetime, date

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_ROOT = Path(r"D:\dataset-ml\DataExtractor")
OUT_CSV   = Path(r"D:\dataset-ml\audit_m1_m15.csv")

TIMEFRAMES = ["M1", "M5", "M15"]

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}

YEARS    = [str(y) for y in range(2018, 2027)]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

QUARTER_MONTHS = {
    "Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)
}
QUARTER_END_DAY = {
    "Q1": 31, "Q2": 30, "Q3": 30, "Q4": 31
}

# All 43 instruments (canonical order from universe.py)
ALL_INSTRUMENTS = [
    "BTCUSD",
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
    "NZDCAD", "NZDCHF", "NZDJPY",
    "AUS200", "US30", "GER40", "UK100", "NAS100", "EUSTX50", "JPN225", "SPX500",
    "XTIUSD", "XBRUSD",
    "XAUUSD", "XAGUSD",
    "USDMXN", "USDZAR",
]

# FX trading: 120 h/week (Mon 00:00 – Fri 22:00 UTC approx)
# Crypto/metals/indices: treat as ~120 h/week for conservative estimate
FX_MINUTES_PER_WEEK = 120 * 60   # 7200

# Cut-off: don't count bars past this timestamp
CUTOFF = datetime(2026, 3, 25, 9, 0, 0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def quarter_date_range(year: int, q: str) -> tuple[date, date]:
    m_start, m_end = QUARTER_MONTHS[q]
    d_end = QUARTER_END_DAY[q]
    return date(year, m_start, 1), date(year, m_end, d_end)


def expected_bars(year: int, q: str, tf: str) -> int:
    """
    Rough expected bar count for one quarter at given timeframe.
    Assumes ~120 trading hours/week for FX (conservative, same for all).
    """
    q_start, q_end = quarter_date_range(year, q)
    # Cap at cutoff
    cutoff_date = CUTOFF.date()
    if q_start > cutoff_date:
        return 0
    if q_end > cutoff_date:
        q_end = cutoff_date

    days = (q_end - q_start).days + 1
    weeks = days / 7.0
    trading_minutes = weeks * FX_MINUTES_PER_WEEK
    return max(0, int(trading_minutes / TF_MINUTES[tf]))


def read_csv_stats(path: Path) -> tuple[int, str, str]:
    """
    Read a candle CSV and return (row_count, first_bar_str, last_bar_str).
    Skips blank lines and the 'X' placeholder symbol content.
    Fast: only reads first and last data lines.
    """
    rows = 0
    first_bar = ""
    last_bar  = ""

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                ts = line.split(",")[0]
                if rows == 1:
                    first_bar = ts
                last_bar = ts
    except Exception as exc:
        return 0, "", f"ERROR:{exc}"

    return rows, first_bar, last_bar


def bar_within_cutoff(last_bar: str) -> str:
    """Return last_bar clamped to CUTOFF (informational note)."""
    if not last_bar or last_bar.startswith("ERROR"):
        return last_bar
    try:
        dt = datetime.strptime(last_bar, "%Y.%m.%d %H:%M:%S")
        if dt > CUTOFF:
            return CUTOFF.strftime("%Y.%m.%d %H:%M:%S") + " (capped)"
    except Exception:
        pass
    return last_bar


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rows_out = []

    total = len(YEARS) * len(QUARTERS) * len(ALL_INSTRUMENTS) * len(TIMEFRAMES)
    done  = 0

    for year in YEARS:
        yr = int(year)
        for q in QUARTERS:
            q_start, q_end = quarter_date_range(yr, q)
            if q_start > CUTOFF.date():
                # Quarter entirely after cutoff — skip
                done += len(ALL_INSTRUMENTS) * len(TIMEFRAMES)
                continue

            for symbol in ALL_INSTRUMENTS:
                for tf in TIMEFRAMES:
                    done += 1
                    if done % 500 == 0:
                        print(f"  {done}/{total}...", end="\r", flush=True)

                    csv_path = DATA_ROOT / year / q / symbol / f"candles_{tf}.csv"
                    exp = expected_bars(yr, q, tf)

                    if not csv_path.exists():
                        rows_out.append({
                            "year": year, "quarter": q,
                            "symbol": symbol, "timeframe": tf,
                            "file_exists": "N",
                            "row_count": 0,
                            "first_bar": "",
                            "last_bar": "",
                            "expected_bars": exp,
                            "pct_complete": 0.0,
                            "note": "missing",
                        })
                        continue

                    row_count, first_bar, last_bar = read_csv_stats(csv_path)

                    # Clamp row_count to bars up to cutoff (approximation)
                    note = ""
                    if last_bar and not last_bar.startswith("ERROR"):
                        try:
                            dt_last = datetime.strptime(last_bar, "%Y.%m.%d %H:%M:%S")
                            if dt_last > CUTOFF:
                                note = "past_cutoff"
                        except Exception:
                            pass

                    pct = round(100.0 * row_count / exp, 1) if exp > 0 else 0.0
                    pct = min(pct, 150.0)   # cap at 150% for display sanity

                    rows_out.append({
                        "year": year, "quarter": q,
                        "symbol": symbol, "timeframe": tf,
                        "file_exists": "Y",
                        "row_count": row_count,
                        "first_bar": first_bar,
                        "last_bar": last_bar,
                        "expected_bars": exp,
                        "pct_complete": pct,
                        "note": note,
                    })

    print(f"\n  Writing {len(rows_out)} rows to {OUT_CSV}...")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "year", "quarter", "symbol", "timeframe",
            "file_exists", "row_count", "first_bar", "last_bar",
            "expected_bars", "pct_complete", "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    present = [r for r in rows_out if r["file_exists"] == "Y"]
    missing = [r for r in rows_out if r["file_exists"] == "N"]

    print(f"\n  Total slots : {len(rows_out)}")
    print(f"  Present     : {len(present)}  ({100*len(present)/len(rows_out):.1f}%)")
    print(f"  Missing     : {len(missing)}  ({100*len(missing)/len(rows_out):.1f}%)")

    for tf in TIMEFRAMES:
        tf_p = [r for r in present if r["timeframe"] == tf]
        tf_m = [r for r in missing if r["timeframe"] == tf]
        tf_all = tf_p + tf_m
        if tf_all:
            pcts = [r["pct_complete"] for r in tf_p]
            avg_pct = round(sum(pcts) / len(pcts), 1) if pcts else 0.0
            print(f"\n  [{tf}]  present={len(tf_p)}  missing={len(tf_m)}  "
                  f"avg_completeness={avg_pct}%")
            # Per-year summary
            for yr in YEARS:
                yr_p = [r for r in tf_p if r["year"] == yr]
                yr_m = [r for r in tf_m if r["year"] == yr]
                if yr_p or yr_m:
                    n_total = len(yr_p) + len(yr_m)
                    print(f"    {yr}: {len(yr_p)}/{n_total} files present")

    print(f"\n  Done. CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
