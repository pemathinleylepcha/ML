"""
MT5 Tick Downloader — Algo C2 v2
Downloads tick data for missing/short pairs via MetaTrader5 Python API.
Chunks by week to avoid MT5 memory limits. Resumes from checkpoint.

Usage:
    python mt5_tick_downloader.py [--out D:\\dataset-ml\\2025-March-2026-March]
                                  [--start 2025-03-01] [--end 2026-03-26]
                                  [--symbols AUDCAD,AUDCHF,...]
                                  [--chunk-days 7]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)

# ── Symbol → output folder mapping ───────────────────────────────────────────

CATEGORY_MAP = {
    "BTCUSD": "Crypto-tick", "ETHUSD": "Crypto-tick",
    "XTIUSD": "Energy-tick", "XBRUSD": "Energy-tick", "XNGUSD": "Energy-tick",
    "XAUUSD": "Metals-tick", "XAGUSD": "Metals-tick",
    "AUS200": "Indices-tick", "GER40":  "Indices-tick", "UK100":   "Indices-tick",
    "NAS100": "Indices-tick", "EUSTX50":"Indices-tick", "EuSTX50": "Indices-tick",
    "JPN225": "Indices-tick", "JN225":  "Indices-tick", "SPX500":  "Indices-tick",
    "US30":   "Indices-tick",
}

def _category(sym: str) -> str:
    return CATEGORY_MAP.get(sym, "Forex-ticks")

# ── Target symbols ────────────────────────────────────────────────────────────

# Missing entirely from 2025-March-2026-March
MISSING = [
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD",
    "CADCHF","CADJPY","CHFJPY",
    "NZDCAD","NZDCHF","NZDJPY",
    "USDMXN","USDZAR",
]

# Short history — extend from 2025-03-01
SHORT = [
    "GBPUSD","USDJPY",                         # FX: < 90 days
    "GBPJPY","GBPCAD","EURGBP","EURAUD",        # FX: < 270 days
    "US30","SPX500",                            # Indices: < 160 days
    "XAUUSD","XAGUSD","XBRUSD",                 # Metals/energy
    "BTCUSD","NAS100",                          # Crypto/indices
]

ALL_TARGETS = MISSING + SHORT

# ── CSV header ────────────────────────────────────────────────────────────────

TICK_HEADER = "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"


def _dt_to_str(ts_ms: int) -> tuple[str, str]:
    """Convert millisecond timestamp to (DATE, TIME) strings."""
    dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
    date_str = dt.strftime("%Y.%m.%d")
    time_str = dt.strftime("%H:%M:%S.") + f"{ts_ms % 1000:03d}"
    return date_str, time_str


def _resolve_symbol(sym: str) -> str | None:
    """Return MT5-resolved symbol name or None if not available."""
    info = mt5.symbol_info(sym)
    if info is not None:
        mt5.symbol_select(sym, True)
        return sym

    # Try common name variants
    for variant in [sym.upper(), sym + ".pro", sym + "m"]:
        info = mt5.symbol_info(variant)
        if info is not None:
            mt5.symbol_select(variant, True)
            print(f"  [{sym}] resolved as '{variant}'")
            return variant

    return None


def download_chunk(sym_mt5: str, from_dt: datetime, to_dt: datetime,
                   retries: int = 3) -> np.ndarray | None:
    """Download one week of ticks. Returns numpy array or None on failure."""
    for attempt in range(retries):
        ticks = mt5.copy_ticks_range(
            sym_mt5,
            from_dt, to_dt,
            mt5.COPY_TICKS_ALL
        )
        if ticks is not None and len(ticks) > 0:
            return ticks
        err = mt5.last_error()
        if err[0] == 0:   # No error — just no data in range
            return None
        print(f"    Attempt {attempt+1}/{retries}: error {err} — retrying...")
        time.sleep(1.5 * (attempt + 1))
    return None


def write_ticks(ticks: np.ndarray, fpath: Path, append: bool) -> int:
    """Write tick array to CSV. Returns rows written."""
    mode = "a" if append else "w"
    rows_written = 0
    with open(fpath, mode, encoding="utf-8", newline="") as fh:
        if not append:
            fh.write(TICK_HEADER + "\n")
        for tick in ticks:
            ts_ms  = int(tick["time_msc"])
            bid    = float(tick["bid"])
            ask    = float(tick["ask"])
            last   = float(tick["last"]) if "last" in tick.dtype.names else 0.0
            vol    = int(tick["volume"]) if "volume" in tick.dtype.names else 0
            flags  = int(tick["flags"])  if "flags"  in tick.dtype.names else 0
            date_s, time_s = _dt_to_str(ts_ms)
            bid_s  = f"{bid:.6f}"  if bid  != 0 else ""
            ask_s  = f"{ask:.6f}"  if ask  != 0 else ""
            last_s = f"{last:.6f}" if last != 0 else ""
            vol_s  = str(vol)      if vol  != 0 else ""
            fh.write(f"{date_s}\t{time_s}\t{bid_s}\t{ask_s}\t{last_s}\t{vol_s}\t{flags}\n")
            rows_written += 1
    return rows_written


def load_checkpoint(cp_path: Path) -> dict[str, str]:
    """Load checkpoint: sym → last completed date string (YYYY-MM-DD)."""
    cp: dict[str, str] = {}
    if cp_path.exists():
        with open(cp_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cp[row["symbol"]] = row["last_completed_date"]
    return cp


def save_checkpoint(cp_path: Path, cp: dict[str, str]) -> None:
    with open(cp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["symbol","last_completed_date","output_file"])
        writer.writeheader()
        for sym, last_date in cp.items():
            writer.writerow({"symbol": sym, "last_completed_date": last_date, "output_file": ""})


def run(out_root: Path, start_dt: datetime, end_dt: datetime,
        symbols: list[str], chunk_days: int) -> None:

    cp_path = out_root / "download_checkpoint.csv"
    cp = load_checkpoint(cp_path)

    print(f"\nMT5 Tick Downloader")
    print(f"Output root : {out_root}")
    print(f"Range       : {start_dt.date()} → {end_dt.date()}")
    print(f"Symbols     : {symbols}")
    print(f"Chunk       : {chunk_days} days\n")

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"MT5 connected — build {mt5.version()}\n")

    for sym in symbols:
        print(f"{'='*60}")
        print(f"Symbol: {sym}")

        sym_mt5 = _resolve_symbol(sym)
        if sym_mt5 is None:
            print(f"  SKIP — symbol not available in MT5 terminal")
            cp[sym] = "unavailable"
            save_checkpoint(cp_path, cp)
            continue

        # Output file path
        cat_dir = out_root / _category(sym)
        cat_dir.mkdir(parents=True, exist_ok=True)

        start_s = start_dt.strftime("%Y%m%d%H%M")
        end_s   = end_dt.strftime("%Y%m%d%H%M")
        out_file = cat_dir / f"{sym}_{start_s}_{end_s}.csv"

        # Resume from checkpoint
        effective_start = start_dt
        append_mode = False
        if sym in cp and cp[sym] not in ("unavailable", ""):
            try:
                last_done = datetime.strptime(cp[sym], "%Y-%m-%d")
                if last_done >= end_dt:
                    print(f"  Already complete (checkpoint: {cp[sym]})")
                    continue
                effective_start = last_done + timedelta(days=1)
                append_mode = out_file.exists()
                print(f"  Resuming from {effective_start.date()} (checkpoint)")
            except ValueError:
                pass

        print(f"  Output: {out_file.name}")
        print(f"  Writing from {effective_start.date()}...\n")

        total_rows = 0
        chunk_start = effective_start

        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)

            ticks = download_chunk(sym_mt5, chunk_start, chunk_end)

            if ticks is not None and len(ticks) > 0:
                rows = write_ticks(ticks, out_file, append=append_mode)
                total_rows += rows
                append_mode = True
                print(f"  {chunk_start.date()} → {chunk_end.date()} : {rows:>8,} ticks  "
                      f"(total {total_rows:,})")
            else:
                print(f"  {chunk_start.date()} → {chunk_end.date()} : no data")

            # Checkpoint after each week
            cp[sym] = chunk_end.strftime("%Y-%m-%d")
            save_checkpoint(cp_path, cp)

            chunk_start = chunk_end

        print(f"\n  [{sym}] DONE — {total_rows:,} total ticks → {out_file.name}\n")

    mt5.shutdown()
    print("MT5 shutdown. All done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default=r"D:\dataset-ml\2025-March-2026-March")
    parser.add_argument("--start",      default="2025-03-01")
    parser.add_argument("--end",        default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--symbols",    default=",".join(ALL_TARGETS))
    parser.add_argument("--chunk-days", type=int, default=7)
    args = parser.parse_args()

    syms  = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")

    run(
        out_root   = Path(args.out),
        start_dt   = start,
        end_dt     = end,
        symbols    = syms,
        chunk_days = args.chunk_days,
    )
