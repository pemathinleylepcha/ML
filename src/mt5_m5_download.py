"""
mt5_m5_download.py — Download M5 history from MT5 into DataExtractor quarterly CSV format

Saves to: <out_dir>/{year}/{Q1-Q4}/{instrument}/candles_M5.csv
Compatible with train_catboost_v2.py load_quarterly_csv_dir()

Usage:
    python mt5_m5_download.py --out C:/Algo-C2/data/quarterly --years 8
    python mt5_m5_download.py --out C:/Algo-C2/data/quarterly --start 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed. pip install MetaTrader5")
    sys.exit(1)

# All 43 instruments matching universe.py
INSTRUMENTS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
    "BTCUSD",
    "US30", "USDMXN", "USDZAR", "XAGUSD", "XAUUSD", "XBRUSD",
    # Signal-only (no spread/tk needed but useful for graph)
    "EURCAD", "EURNZD", "GBPNZD", "NZDCAD", "NZDCHF", "AUDCHF", "CADCHF",
    "CHFJPY", "GBPAUD", "GBPCAD",
]
# Deduplicate while preserving order
seen = set()
INSTRUMENTS = [x for x in INSTRUMENTS if not (x in seen or seen.add(x))]

QUARTERS = {1: "Q1", 2: "Q1", 3: "Q1",
             4: "Q2", 5: "Q2", 6: "Q2",
             7: "Q3", 8: "Q3", 9: "Q3",
             10: "Q4", 11: "Q4", 12: "Q4"}

CSV_HEADER = "bar_time,open,high,low,close,spread,tick_volume"


def _quarter(month: int) -> str:
    return QUARTERS[month]


def download_instrument(sym: str, start_dt: datetime, out_dir: Path,
                        max_bars: int = 99_999) -> int:
    """Download M5 history for one instrument and save to quarterly CSVs."""
    if not mt5.symbol_select(sym, True):
        print(f"  [{sym}] symbol_select failed: {mt5.last_error()}")
        return 0

    # Pull max available bars from position 0 (current bar) backward
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, max_bars)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  [{sym}] No data: {err}")
        return 0

    import numpy as np
    combined = rates

    # Filter by start date if specified
    if start_dt is not None:
        start_ts = int(start_dt.timestamp())
        mask = combined["time"] >= start_ts
        combined = combined[mask]
        if len(combined) == 0:
            print(f"  [{sym}] All data before {start_dt.date()}, skipping")
            return 0
    df = pd.DataFrame(combined)
    df["dt"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Build DataFrame with required columns
    df["bar_time"] = df["dt"].dt.strftime("%Y.%m.%d %H:%M:%S")
    if "spread" not in df.columns:
        df["spread"] = 0

    # Extract year/quarter before dropping dt
    df["year"]    = df["dt"].dt.year
    df["month"]   = df["dt"].dt.month
    df["quarter"] = df["month"].map(_quarter)

    total_saved = 0
    for (year, q), group in df.groupby(["year", "quarter"]):
        csv_dir = out_dir / str(year) / q / sym
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "candles_M5.csv"

        g = group[["bar_time", "open", "high", "low", "close", "spread", "tick_volume"]]

        if csv_path.exists():
            existing = pd.read_csv(csv_path, dtype=str)
            combined_df = pd.concat([existing, g.astype(str)], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["bar_time"]).sort_values("bar_time")
            combined_df.to_csv(csv_path, index=False)
        else:
            g.astype(str).to_csv(csv_path, index=False)

        total_saved += len(g)

    print(f"  [{sym}] {len(df):,} bars saved ({df['bar_time'].iloc[0]} to {df['bar_time'].iloc[-1]})")
    return total_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="C:/Algo-C2/data/quarterly",
                        help="Output directory for quarterly CSVs")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 8 years ago)")
    parser.add_argument("--years", type=int, default=8,
                        help="Years of history to download (if --start not given)")
    parser.add_argument("--instruments", default=None,
                        help="Comma-separated list (default: all 43)")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"MT5 version: {mt5.version()}")

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        from datetime import timedelta
        start_dt = datetime.now() - pd.Timedelta(days=args.years * 365)
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    instruments = args.instruments.split(",") if args.instruments else INSTRUMENTS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading M5 history from {start_dt.date()} to now")
    print(f"  Output: {out_dir}")
    print(f"  Instruments: {len(instruments)}")

    total = 0
    for i, sym in enumerate(instruments):
        print(f"[{i+1}/{len(instruments)}] {sym}...")
        n = download_instrument(sym, start_dt, out_dir)
        total += n
        time.sleep(0.1)  # gentle on MT5

    mt5.shutdown()
    print(f"\nDone. Total bars saved: {total:,}")
    print(f"Now run: python src/train_catboost_v2.py --data-dir {out_dir} --quarterly")


if __name__ == "__main__":
    main()
