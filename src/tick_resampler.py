"""
Algo C2 v2 — 1000ms Bar Resampler
Resamples MT5 tick CSVs to 1-second (1000ms) bars with microstructure features.
Intended for Phase 3 (2024-2026) high-resolution tick data.

Output: one Parquet (or CSV fallback) file per instrument:
    <output_dir>/<INSTRUMENT>_1000ms.parquet

Columns in output:
    dt              datetime64[ns]  bar open timestamp (UTC)
    o               float64         open price (mid)
    h               float64         high price (mid)
    l               float64         low price (mid)
    c               float64         close price (mid)
    sp              float64         mean spread in price units (NOT pips)
    tk              int64           tick count in bar
    tick_velocity   float64         ticks per second (= tk / 1.0)
    spread_z        float64         spread z-score vs 60-bar rolling window
    bid_ask_imbalance float64       (ask_count - bid_count) / (total + 1e-10)
    price_velocity  float64         (c - c.shift(1)) / (c.shift(1) + 1e-10)

Usage:
    python tick_resampler.py --input_dir ./data/csvs --output_dir ./data/1000ms
    python tick_resampler.py --input_dir ./data/csvs --output_dir ./data/1000ms \\
        --instruments EURUSD,BTCUSD,XAUUSD
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from universe import ALL_INSTRUMENTS, PIP_SIZES

# ── Parquet backend detection ─────────────────────────────────────────────────

try:
    import pyarrow  # noqa: F401
    _PARQUET_ENGINE = "pyarrow"
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
        _HAS_PARQUET = True
    except ImportError:
        _HAS_PARQUET = False
        _PARQUET_ENGINE = None


# ── CSV parsing (tick format) ─────────────────────────────────────────────────

_REQUIRED_TICK_COLUMNS = ("BID", "ASK")


def _canonical_tick_column(name: object) -> str:
    text = str(name).strip().strip("\ufeff").strip().strip('"').strip("'")
    return text.strip("<>").upper()


def _detect_tick_csv_encoding(filepath: Path) -> str:
    head = filepath.read_bytes()[:4096]
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _detect_tick_csv_separator(filepath: Path, encoding: str) -> str:
    with open(filepath, "r", encoding=encoding, errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            tab_count = text.count("\t")
            comma_count = text.count(",")
            if tab_count == 0 and comma_count == 0:
                continue
            return "\t" if tab_count > comma_count else ","
    return ","


def _read_mt5_tick_csv(filepath: Path, chunksize: int | None = None):
    encoding = _detect_tick_csv_encoding(filepath)
    separator = _detect_tick_csv_separator(filepath, encoding)
    kwargs = {
        "sep": separator,
        "header": 0,
        "dtype": str,
        "engine": "python",
        "encoding": encoding,
        "on_bad_lines": "skip",
        "skipinitialspace": True,
        "keep_default_na": True,
        "na_values": ["", " "],
    }
    if chunksize is not None:
        kwargs["chunksize"] = chunksize
    return pd.read_csv(filepath, **kwargs)


def _normalize_tick_chunk(frame: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    renamed = frame.rename(columns={col: _canonical_tick_column(col) for col in frame.columns})
    missing = [col for col in _REQUIRED_TICK_COLUMNS if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required tick columns {missing} in {filepath}")
    if "FLAGS" not in renamed.columns:
        renamed["FLAGS"] = "6"
    has_timestamp = (
        ("TIME_MSC" in renamed.columns)
        or ("TIME_SEC" in renamed.columns)
        or ("DATE" in renamed.columns and "TIME" in renamed.columns)
    )
    if not has_timestamp:
        raise ValueError(f"Missing required tick timestamp columns in {filepath}")
    keep = [col for col in ("DATE", "TIME", "TIME_SEC", "TIME_MSC", "BID", "ASK", "FLAGS") if col in renamed.columns]
    return renamed[keep].copy()


def _parse_tick_datetime(frame: pd.DataFrame) -> pd.Series:
    if "TIME_MSC" in frame.columns:
        dt = pd.to_datetime(pd.to_numeric(frame["TIME_MSC"], errors="coerce"), unit="ms", errors="coerce")
        if dt.notna().any():
            return dt
    if "TIME_SEC" in frame.columns:
        dt = pd.to_datetime(frame["TIME_SEC"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
        if dt.notna().any():
            return dt
    dt = pd.to_datetime(frame["DATE"] + " " + frame["TIME"], format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")
    if dt.notna().any():
        return dt
    return pd.to_datetime(frame["DATE"] + " " + frame["TIME"], errors="coerce")

def parse_tick_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse an MT5 broker tick CSV into a DataFrame.

    Supported input quirks:
        - tab- or comma-separated MT5 exports
        - UTF-8, UTF-8 BOM, or UTF-16 encodings
        - angle-bracket MT5 headers (<DATE>, <TIME>, ...)
        - empty LAST / VOLUME fields on quote-only ticks

    FLAGS encoding:
        2 = bid tick only  (ask unchanged, ask is forward-filled)
        4 = ask tick only  (bid unchanged, bid is forward-filled)
        6 = both bid and ask updated

    Returns a DataFrame with columns:
        datetime, BID, ASK, mid, spread, FLAGS
    """
    df = _normalize_tick_chunk(_read_mt5_tick_csv(filepath), filepath)

    df["datetime"] = _parse_tick_datetime(df)
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["FLAGS"] = pd.to_numeric(df["FLAGS"], errors="coerce").fillna(6).astype(int)

    # Forward-fill partial ticks
    df["BID"] = df["BID"].ffill()
    df["ASK"] = df["ASK"].ffill()

    # Drop rows before the first fully-quoted tick
    df = df.dropna(subset=["BID", "ASK"]).copy()

    df["mid"] = (df["BID"] + df["ASK"]) / 2.0
    df["spread"] = df["ASK"] - df["BID"]

    return df[["datetime", "BID", "ASK", "mid", "spread", "FLAGS"]].copy()


def stream_resample_tick_csv(
    filepath: Path,
    instrument: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Stream-resample a raw MT5 tick CSV into 1-second bars without loading the full
    tick file into memory.

    This is intended for very large raw files where a full DataFrame parse may
    cause excessive RAM usage or native parser crashes on constrained machines.
    """
    chunk_bars: list[pd.DataFrame] = []
    last_bid: float | None = None
    last_ask: float | None = None

    reader = _read_mt5_tick_csv(filepath, chunksize=chunksize)

    for chunk in reader:
        if chunk.empty:
            continue
        chunk = _normalize_tick_chunk(chunk, filepath)

        bid = pd.to_numeric(chunk["BID"], errors="coerce")
        ask = pd.to_numeric(chunk["ASK"], errors="coerce")
        flags = pd.to_numeric(chunk["FLAGS"], errors="coerce").fillna(6).astype(np.int16)

        bid = bid.ffill()
        ask = ask.ffill()
        if last_bid is not None:
            bid = bid.fillna(last_bid)
        if last_ask is not None:
            ask = ask.fillna(last_ask)

        valid = bid.notna() & ask.notna()
        if not valid.any():
            continue

        chunk = chunk.loc[valid, ["DATE", "TIME"]].copy()
        bid = bid.loc[valid]
        ask = ask.loc[valid]
        flags = flags.loc[valid]

        last_bid = float(bid.iloc[-1])
        last_ask = float(ask.iloc[-1])

        datetimes = _parse_tick_datetime(chunk)
        valid_dt = datetimes.notna()
        if not valid_dt.any():
            continue

        datetimes = datetimes.loc[valid_dt]
        bid = bid.loc[valid_dt]
        ask = ask.loc[valid_dt]
        flags = flags.loc[valid_dt]

        if start is not None:
            mask = datetimes >= start
            datetimes = datetimes.loc[mask]
            bid = bid.loc[mask]
            ask = ask.loc[mask]
            flags = flags.loc[mask]
        if end is not None:
            mask = datetimes <= end
            datetimes = datetimes.loc[mask]
            bid = bid.loc[mask]
            ask = ask.loc[mask]
            flags = flags.loc[mask]
        if datetimes.empty:
            continue

        mids = ((bid + ask) / 2.0).astype(np.float64)
        spreads = (ask - bid).astype(np.float64)
        sec = datetimes.dt.floor("s")
        bid_tick = ((flags & 2) != 0).astype(np.int32)
        ask_tick = ((flags & 4) != 0).astype(np.int32)

        chunk_df = pd.DataFrame(
            {
                "dt": sec.to_numpy(),
                "mid": mids.to_numpy(),
                "spread": spreads.to_numpy(),
                "bid_tick": bid_tick.to_numpy(),
                "ask_tick": ask_tick.to_numpy(),
            }
        )

        bars = (
            chunk_df.groupby("dt", sort=True)
            .agg(
                o=("mid", "first"),
                h=("mid", "max"),
                l=("mid", "min"),
                c=("mid", "last"),
                spread_sum=("spread", "sum"),
                tk=("mid", "count"),
                _bid_cnt=("bid_tick", "sum"),
                _ask_cnt=("ask_tick", "sum"),
            )
            .reset_index()
        )
        chunk_bars.append(bars)

    if not chunk_bars:
        return pd.DataFrame(columns=["dt", "o", "h", "l", "c", "sp", "tk", "tick_velocity", "spread_z", "bid_ask_imbalance", "price_velocity"])

    merged = pd.concat(chunk_bars, ignore_index=True)
    merged = (
        merged.groupby("dt", sort=True)
        .agg(
            o=("o", "first"),
            h=("h", "max"),
            l=("l", "min"),
            c=("c", "last"),
            spread_sum=("spread_sum", "sum"),
            tk=("tk", "sum"),
            _bid_cnt=("_bid_cnt", "sum"),
            _ask_cnt=("_ask_cnt", "sum"),
        )
        .reset_index()
    )

    merged["sp"] = merged["spread_sum"] / (merged["tk"].replace(0, np.nan))
    merged["tick_velocity"] = merged["tk"].astype(float)

    rolling_mean = merged["sp"].rolling(window=60, min_periods=1).mean()
    rolling_std = merged["sp"].rolling(window=60, min_periods=1).std().fillna(0.0)
    merged["spread_z"] = (merged["sp"] - rolling_mean) / (rolling_std + 1e-10)

    total = merged["_bid_cnt"] + merged["_ask_cnt"]
    merged["bid_ask_imbalance"] = (merged["_ask_cnt"] - merged["_bid_cnt"]) / (total + 1e-10)
    merged["price_velocity"] = (merged["c"] - merged["c"].shift(1)) / (merged["c"].shift(1) + 1e-10)
    merged["price_velocity"] = merged["price_velocity"].fillna(0.0)

    merged["sp"] = merged["sp"].round(7)
    merged["spread_z"] = merged["spread_z"].round(4)
    merged["bid_ask_imbalance"] = merged["bid_ask_imbalance"].round(4)
    merged["price_velocity"] = merged["price_velocity"].round(8)
    merged["tk"] = merged["tk"].astype("int64")
    merged["tick_velocity"] = merged["tick_velocity"].astype("float64")

    merged = merged.drop(columns=["spread_sum", "_bid_cnt", "_ask_cnt"])
    return merged[["dt", "o", "h", "l", "c", "sp", "tk", "tick_velocity", "spread_z", "bid_ask_imbalance", "price_velocity"]]


# ── 1000ms resampling ─────────────────────────────────────────────────────────

def resample_to_1000ms(ticks: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Resample tick data to 1-second (1000ms) OHLC bars.

    Microstructure features computed per bar:
        tick_velocity     ticks per second (= tk, since bar width is exactly 1s)
        spread_z          spread z-score vs 60-bar rolling mean/std
        bid_ask_imbalance (ask_count - bid_count) / (ask_count + bid_count + 1e-10)
                          FLAGS=2 → bid tick, FLAGS=4 → ask tick, FLAGS=6 → both
        price_velocity    (c - c.shift(1)) / (c.shift(1) + 1e-10)

    Returns a DataFrame with columns:
        dt, o, h, l, c, sp, tk,
        tick_velocity, spread_z, bid_ask_imbalance, price_velocity
    """
    ticks = ticks.set_index("datetime")

    # Core OHLC on mid price
    bars = ticks["mid"].resample("1s").ohlc()
    bars.columns = ["o", "h", "l", "c"]

    # Mean spread in price units (not pips — microstructure level)
    bars["sp"] = ticks["spread"].resample("1s").mean()

    # Tick count
    bars["tk"] = ticks["mid"].resample("1s").count()

    # Bid/ask directional counts per bar
    # FLAGS=2: bid-side tick, FLAGS=4: ask-side tick, FLAGS=6: both sides
    flags = pd.to_numeric(ticks["FLAGS"], errors="coerce").fillna(6).astype(int)
    bid_counts = ((flags & 2) != 0).astype(np.int32).resample("1s").sum()
    ask_counts = ((flags & 4) != 0).astype(np.int32).resample("1s").sum()

    bars["_bid_cnt"] = bid_counts
    bars["_ask_cnt"] = ask_counts

    # Drop bars with no ticks
    bars = bars[bars["tk"] > 0].copy()

    # ── Microstructure features ───────────────────────────────────────────────

    # tick_velocity: ticks per second; bar width is exactly 1 second
    bars["tick_velocity"] = bars["tk"].astype(float)

    # spread_z: z-score of spread within 60-bar rolling window
    rolling_mean = bars["sp"].rolling(window=60, min_periods=1).mean()
    rolling_std  = bars["sp"].rolling(window=60, min_periods=1).std().fillna(0.0)
    bars["spread_z"] = (bars["sp"] - rolling_mean) / (rolling_std + 1e-10)

    # bid_ask_imbalance: signed order flow proxy
    total = bars["_bid_cnt"] + bars["_ask_cnt"]
    bars["bid_ask_imbalance"] = (
        (bars["_ask_cnt"] - bars["_bid_cnt"]) / (total + 1e-10)
    )

    # price_velocity: fractional return vs previous bar close
    bars["price_velocity"] = (bars["c"] - bars["c"].shift(1)) / (bars["c"].shift(1) + 1e-10)
    bars["price_velocity"] = bars["price_velocity"].fillna(0.0)

    # Clean up temporary columns
    bars = bars.drop(columns=["_bid_cnt", "_ask_cnt"])

    # Round stored values for storage efficiency
    bars["sp"]              = bars["sp"].round(7)
    bars["spread_z"]        = bars["spread_z"].round(4)
    bars["bid_ask_imbalance"] = bars["bid_ask_imbalance"].round(4)
    bars["price_velocity"]  = bars["price_velocity"].round(8)

    # Promote index to column
    bars.index.name = "dt"
    bars = bars.reset_index()

    # Enforce column types
    bars["tk"] = bars["tk"].astype("int64")
    bars["tick_velocity"] = bars["tick_velocity"].astype("float64")

    return bars


# ── File discovery ────────────────────────────────────────────────────────────

def find_csv_for_instrument(input_dir: Path, instrument: str) -> Path | None:
    """
    Find a CSV file whose stem contains the instrument name (case-insensitive).
    Returns the first match by sorted filename, or None.
    """
    for f in sorted(input_dir.glob("*.csv")):
        if instrument.upper() in f.stem.upper():
            return f
    return None


# ── Output writing ────────────────────────────────────────────────────────────

def write_output(bars: pd.DataFrame, instrument: str, output_dir: Path) -> Path:
    """
    Write bars DataFrame to Parquet (preferred) or CSV (fallback).

    Returns the path of the written file.
    """
    if _HAS_PARQUET:
        out_path = output_dir / f"{instrument}_1000ms.parquet"
        bars.to_parquet(out_path, engine=_PARQUET_ENGINE, index=False)
    else:
        out_path = output_dir / f"{instrument}_1000ms.csv"
        bars.to_csv(out_path, index=False)

    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algo C2 v2: MT5 tick CSV → 1000ms bars with microstructure features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing MT5 tick CSV files",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write per-instrument output files",
    )
    parser.add_argument(
        "--instruments", default=None,
        help="Comma-separated subset of instruments to process (default: all 43). "
             "Example: --instruments EURUSD,BTCUSD",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start datetime filter, inclusive (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End datetime filter, inclusive (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: input_dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not _HAS_PARQUET:
        print(
            "Warning: pyarrow and fastparquet not found. "
            "Falling back to CSV output. Install pyarrow for Parquet: pip install pyarrow",
            file=sys.stderr,
        )

    # Resolve instrument list
    if args.instruments:
        requested = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
        unknown = [p for p in requested if p not in ALL_INSTRUMENTS]
        if unknown:
            print(
                f"Warning: unknown instruments ignored: {', '.join(unknown)}",
                file=sys.stderr,
            )
        instruments = [p for p in requested if p in ALL_INSTRUMENTS]
        if not instruments:
            print("Error: no valid instruments specified", file=sys.stderr)
            sys.exit(1)
    else:
        instruments = list(ALL_INSTRUMENTS)

    # Date filters as Timestamps
    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts   = pd.Timestamp(args.end + " 23:59:59.999") if args.end else None

    print(f"Output format : {'Parquet (' + _PARQUET_ENGINE + ')' if _HAS_PARQUET else 'CSV (fallback)'}")
    print(f"Instruments   : {len(instruments)}")
    print(f"Input dir     : {input_dir}")
    print(f"Output dir    : {output_dir}")
    if start_ts or end_ts:
        print(f"Date filter   : {start_ts or 'any'} → {end_ts or 'any'}")
    print()

    processed = 0
    skipped = 0
    failed = 0

    for instrument in instruments:
        csv_path = find_csv_for_instrument(input_dir, instrument)
        if csv_path is None:
            print(f"  {instrument}: CSV not found — skipping")
            skipped += 1
            continue

        try:
            ticks = parse_tick_csv(csv_path)

            if start_ts is not None:
                ticks = ticks[ticks["datetime"] >= start_ts]
            if end_ts is not None:
                ticks = ticks[ticks["datetime"] <= end_ts]

            if len(ticks) == 0:
                print(f"  {instrument}: 0 ticks in date range — skipping")
                skipped += 1
                continue

            bars = resample_to_1000ms(ticks, instrument)
            out_path = write_output(bars, instrument, output_dir)

            print(
                f"Processing {instrument}: {len(ticks):,} ticks → {len(bars):,} bars"
                f"  [{out_path.name}]"
            )
            processed += 1

        except Exception as exc:
            print(f"  {instrument}: ERROR — {exc}", file=sys.stderr)
            failed += 1

    print()
    print(f"Done. Processed: {processed} | Skipped: {skipped} | Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
