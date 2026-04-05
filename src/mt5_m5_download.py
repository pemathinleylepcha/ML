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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from dataextractor_contract import canonical_symbols

QUARTERS = {1: "Q1", 2: "Q1", 3: "Q1",
             4: "Q2", 5: "Q2", 6: "Q2",
             7: "Q3", 8: "Q3", 9: "Q3",
             10: "Q4", 11: "Q4", 12: "Q4"}

CSV_COLUMNS = ["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def _quarter(month: int) -> str:
    return QUARTERS[month]


def ensure_mt5_available() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed. pip install MetaTrader5")


def _chunk_ranges(start_dt: datetime, end_dt: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    ranges: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    step = timedelta(days=max(1, chunk_days))
    while cursor < end_dt:
        chunk_end = min(cursor + step, end_dt)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end
    return ranges


def _fetch_m5_range(sym: str, start_dt: datetime, end_dt: datetime, chunk_days: int) -> pd.DataFrame:
    ensure_mt5_available()
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_ranges(start_dt, end_dt, chunk_days):
        rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, chunk_start, chunk_end)
        if rates is None:
            print(f"    [{sym}] chunk {chunk_start} -> {chunk_end} failed: {mt5.last_error()}")
            continue
        if len(rates) == 0:
            continue
        frames.append(pd.DataFrame(rates))
        time.sleep(0.05)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["time"], unit="s")
    mask = (df["dt"] >= pd.Timestamp(start_dt)) & (df["dt"] <= pd.Timestamp(end_dt))
    return df.loc[mask].reset_index(drop=True)


def download_instrument(
    sym: str,
    start_dt: datetime,
    end_dt: datetime,
    out_dir: Path,
    chunk_days: int = 31,
) -> int:
    """Download M5 history for one instrument and save to quarterly CSVs."""
    ensure_mt5_available()
    if not mt5.symbol_select(sym, True):
        print(f"  [{sym}] symbol_select failed: {mt5.last_error()}")
        return 0

    df = _fetch_m5_range(sym, start_dt, end_dt, chunk_days=chunk_days)
    if df.empty:
        err = mt5.last_error()
        print(f"  [{sym}] No data: {err}")
        return 0

    df["bar_time"] = df["dt"].dt.strftime("%Y.%m.%d %H:%M:%S")
    if "spread" not in df.columns:
        df["spread"] = 0
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    if "real_volume" not in df.columns:
        df["real_volume"] = 0

    df["year"]    = df["dt"].dt.year
    df["month"]   = df["dt"].dt.month
    df["quarter"] = df["month"].map(_quarter)

    total_saved = 0
    for (year, q), group in df.groupby(["year", "quarter"]):
        csv_dir = out_dir / str(year) / q / sym
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "candles_M5.csv"

        g = group[CSV_COLUMNS]

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            combined_df = pd.concat([existing, g], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["bar_time"]).sort_values("bar_time")
            combined_df.to_csv(csv_path, index=False)
        else:
            g.to_csv(csv_path, index=False)

        total_saved += len(g)

    print(f"  [{sym}] {len(df):,} bars saved ({df['bar_time'].iloc[0]} to {df['bar_time'].iloc[-1]})")
    return total_saved


def download_instruments(
    instruments: list[str],
    out_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int = 31,
) -> int:
    ensure_mt5_available()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        total = 0
        print(f"Downloading M5 history from {start_dt.date()} to {end_dt.date()}")
        print(f"  Output: {out_dir}")
        print(f"  Instruments: {len(instruments)}")
        print(f"  Chunk days: {chunk_days}")
        for idx, sym in enumerate(instruments, start=1):
            print(f"[{idx}/{len(instruments)}] {sym}...")
            total += download_instrument(sym, start_dt, end_dt, out_dir, chunk_days=chunk_days)
            time.sleep(0.1)
        print(f"\nDone. Total bars saved: {total:,}")
        return total
    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="C:/Algo-C2/data/quarterly",
                        help="Output directory for quarterly CSVs")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 8 years ago)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--years", type=int, default=8,
                        help="Years of history to download (if --start not given)")
    parser.add_argument("--instruments", default=None,
                        help="Comma-separated list (default: all 43)")
    parser.add_argument("--chunk-days", type=int, default=31,
                        help="Chunk size for MT5 range requests")
    args = parser.parse_args()

    ensure_mt5_available()
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"MT5 version: {mt5.version()}")

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = datetime.now() - timedelta(days=args.years * 365)
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        end_dt = datetime.now()

    if end_dt <= start_dt:
        raise SystemExit("--end must be after --start")

    instruments = args.instruments.split(",") if args.instruments else list(canonical_symbols())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading M5 history from {start_dt.date()} to {end_dt.date()}")
    print(f"  Output: {out_dir}")
    print(f"  Instruments: {len(instruments)}")
    print(f"  Chunk days: {args.chunk_days}")

    total = 0
    for i, sym in enumerate(instruments):
        print(f"[{i+1}/{len(instruments)}] {sym}...")
        n = download_instrument(sym, start_dt, end_dt, out_dir, chunk_days=args.chunk_days)
        total += n
        time.sleep(0.1)  # gentle on MT5

    mt5.shutdown()
    print(f"\nDone. Total bars saved: {total:,}")
    print(f"Now run: python src/train_catboost_v2.py --data-dir {out_dir} --quarterly")


if __name__ == "__main__":
    main()
