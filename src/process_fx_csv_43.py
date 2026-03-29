"""
Algo C2 v2 — Phase 1: Tick/Candle CSV → 1-min OHLC JSON (43 instruments)
Extends process_fx_csv_35.py to cover all 43 nodes defined in universe.py.

Supports two input modes:
  - tick   (default): MT5 tab-separated tick export (same format as v1)
  - candle: OHLCV CSVs with columns datetime,open,high,low,close,volume (comma-separated, UTC)

Usage:
    # Tick mode (default)
    python process_fx_csv_43.py --input_dir ./data/csvs --output ./data/algo_c2_43.json

    # Candle mode (2019-2023 OHLCV CSVs)
    python process_fx_csv_43.py --input_dir ./data/ohlcv --output ./data/algo_c2_43.json --mode candle

    # Subset of instruments
    python process_fx_csv_43.py --input_dir ./data/csvs --output ./data/fx_only.json \\
        --instruments EURUSD,GBPUSD,USDJPY
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from universe import ALL_INSTRUMENTS, SIGNAL_ONLY, PIP_SIZES


# ── CSV discovery ─────────────────────────────────────────────────────────────

def find_csv_for_instrument(input_dir: Path, instrument: str) -> Path | None:
    """
    Find a CSV file matching an instrument name in the input directory.
    Case-insensitive match against file stem.
    """
    for f in sorted(input_dir.glob("*.csv")):
        if instrument.upper() in f.stem.upper():
            return f
    return None


# ── Tick mode parsing ─────────────────────────────────────────────────────────

def parse_tick_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse an MT5 broker tick CSV (tab-separated) into a DataFrame.

    Expected columns (tab-separated, with header):
        DATE  TIME  BID  ASK  LAST  VOLUME  FLAGS

    FLAGS encoding:
        2 = bid tick only (ask unchanged)
        4 = ask tick only (bid unchanged)
        6 = both bid and ask updated

    Returns a DataFrame with columns:
        datetime, BID, ASK, mid, spread
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=0,
        names=["DATE", "TIME", "BID", "ASK", "LAST", "VOLUME", "FLAGS"],
        dtype={"DATE": str, "TIME": str},
        na_values=["", " "],
    )

    df["datetime"] = pd.to_datetime(
        df["DATE"] + " " + df["TIME"],
        format="%Y.%m.%d %H:%M:%S.%f",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["FLAGS"] = pd.to_numeric(df["FLAGS"], errors="coerce").fillna(6).astype(int)

    # Forward-fill partial ticks: FLAGS=2 means only bid changed, ffill ask;
    # FLAGS=4 means only ask changed, ffill bid.
    df["BID"] = df["BID"].ffill()
    df["ASK"] = df["ASK"].ffill()

    # Drop rows before the first fully-quoted tick
    df = df.dropna(subset=["BID", "ASK"]).copy()

    df["mid"] = (df["BID"] + df["ASK"]) / 2.0
    df["spread"] = df["ASK"] - df["BID"]

    return df[["datetime", "BID", "ASK", "mid", "spread", "FLAGS"]].copy()


# ── Candle mode parsing ───────────────────────────────────────────────────────

def parse_candle_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse a historical OHLCV candle CSV (comma-separated, UTC timestamps).

    Expected columns:
        datetime,open,high,low,close,volume

    The datetime column must be parseable by pandas (ISO 8601 recommended).
    Returns a DataFrame indexed by datetime with columns: open, high, low, close, volume.
    """
    df = pd.read_csv(
        filepath,
        sep=",",
        header=0,
        dtype={"datetime": str},
        na_values=["", " "],
    )

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" not in df.columns:
        # Try common alternatives: date, time, timestamp
        for alt in ("date", "timestamp", "time"):
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"})
                break
        else:
            raise ValueError(
                f"Cannot find datetime column in {filepath.name}. "
                f"Expected one of: datetime, date, timestamp, time. "
                f"Found: {list(df.columns)}"
            )

    required = {"open", "high", "low", "close"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {filepath.name}: {sorted(missing_cols)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, infer_datetime_format=True)
    df["datetime"] = df["datetime"].dt.tz_localize(None)  # strip tz, treat as UTC-naive

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def candle_to_1min_bars(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Convert a candle DataFrame (any timeframe) to 1-minute bar format.

    If the source data is already 1-minute, this is a pass-through with column
    renaming. If coarser (e.g. H1), each candle is emitted as a single bar
    anchored to its open time. Spread is set to 0 (unknown for historical data)
    and tick count is set to 1.

    Returns a DataFrame with columns: dt, o, h, l, c, sp, tk
    """
    df = df.set_index("datetime").sort_index()

    # Detect source resolution: median gap between bars
    if len(df) > 1:
        gaps = df.index.to_series().diff().dropna()
        median_gap_min = gaps.median().total_seconds() / 60.0
    else:
        median_gap_min = 1.0

    if abs(median_gap_min - 1.0) < 0.5:
        # Source is already 1-minute — rename only
        out = pd.DataFrame({
            "o":  df["open"],
            "h":  df["high"],
            "l":  df["low"],
            "c":  df["close"],
            "sp": 0.0,
            "tk": df["volume"].clip(lower=1).astype(int),
        })
        out.index = df.index.strftime("%Y-%m-%d %H:%M")
        out.index.name = "dt"
        return out.reset_index()

    # Coarser resolution: expand each candle to its open minute, keep OHLCV intact
    out = pd.DataFrame({
        "o":  df["open"].values,
        "h":  df["high"].values,
        "l":  df["low"].values,
        "c":  df["close"].values,
        "sp": 0.0,
        "tk": df["volume"].clip(lower=0).fillna(0).astype(int).values,
    }, index=df.index.strftime("%Y-%m-%d %H:%M"))
    out.index.name = "dt"
    return out.reset_index()


# ── Intra-bar tick path (tick mode only) ──────────────────────────────────────

def _compute_bar_tick_path(
    group_mid: pd.Series,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    pip: float,
) -> list:
    """
    Build a compact intra-bar tick path for TP/SL resolution.

    Records every price that sets a new intra-bar extreme so that backtesting
    can determine whether TP or SL was hit first, in chronological order.
    Always begins with open and ends with close.
    """
    if len(group_mid) <= 2:
        return [bar_open, bar_high, bar_low, bar_close]

    prices = group_mid.values
    path = [round(float(prices[0]), 6)]
    running_high = prices[0]
    running_low = prices[0]

    for p in prices[1:-1]:
        if p > running_high:
            running_high = p
            path.append(round(float(p), 6))
        elif p < running_low:
            running_low = p
            path.append(round(float(p), 6))

    path.append(round(float(prices[-1]), 6))
    return path


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_to_1min(ticks: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Resample tick data to 1-minute OHLC bars.

    Produces columns: dt, o, h, l, c, sp (spread in pips), tk (tick count),
    tp (intra-bar tick path for TP/SL resolution).
    """
    pip = PIP_SIZES[instrument]
    ticks = ticks.set_index("datetime")

    bars = ticks["mid"].resample("1min").ohlc()
    bars.columns = ["o", "h", "l", "c"]

    bars["sp"] = ticks["spread"].resample("1min").mean() / pip
    bars["tk"] = ticks["mid"].resample("1min").count()

    bars = bars[bars["tk"] > 0].copy()

    tick_paths = []
    for dt_idx in bars.index:
        end = dt_idx + pd.Timedelta(minutes=1)
        mask = (ticks.index >= dt_idx) & (ticks.index < end)
        group = ticks.loc[mask, "mid"]

        row = bars.loc[dt_idx]
        path = _compute_bar_tick_path(
            group, row["o"], row["h"], row["l"], row["c"], pip
        )
        tick_paths.append(path)

    bars["tp"] = tick_paths
    bars["sp"] = bars["sp"].round(2)

    bars.index = bars.index.strftime("%Y-%m-%d %H:%M")
    bars.index.name = "dt"

    return bars.reset_index()


# ── Timeline alignment ────────────────────────────────────────────────────────

def align_to_master(pair_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Align all instruments to a single master timeline.

    Builds the union of all instrument timestamps, reindexes each instrument
    to the full master index, then forward-fills gaps. Rows before an
    instrument's first valid bar are dropped (no back-fill).
    """
    all_timestamps: set = set()
    for df in pair_dfs.values():
        all_timestamps.update(df["dt"].tolist())
    master_index = sorted(all_timestamps)

    aligned: dict[str, pd.DataFrame] = {}
    for instrument, df in pair_dfs.items():
        df = df.set_index("dt")
        df = df.reindex(master_index)
        df = df.ffill()
        df = df.dropna(subset=["o"])
        df.index.name = "dt"
        aligned[instrument] = df.reset_index()

    return aligned


# ── JSON output ───────────────────────────────────────────────────────────────

def write_json(pair_data: dict[str, pd.DataFrame], output_path: str) -> None:
    """
    Write aligned bar data to compact JSON.

    Format: {instrument: [{dt, o, h, l, c, sp, tk, tp?}, ...]}
    The tick-path field (tp) is omitted when absent (candle mode).
    """
    result: dict = {}
    for instrument in sorted(pair_data.keys()):
        df = pair_data[instrument]
        records = []
        for _, row in df.iterrows():
            rec = {
                "dt": row["dt"],
                "o":  round(float(row["o"]), 6),
                "h":  round(float(row["h"]), 6),
                "l":  round(float(row["l"]), 6),
                "c":  round(float(row["c"]), 6),
                "sp": round(float(row["sp"]), 2),
                "tk": int(row["tk"]),
            }
            if "tp" in row and isinstance(row["tp"], list) and len(row["tp"]) > 0:
                rec["tp"] = row["tp"]
            records.append(rec)
        result[instrument] = records

    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB, {len(result)} instruments)")


# ── Multi-timeframe resampling ────────────────────────────────────────────────

def resample_multi_timeframe(
    json_path: str,
    pairs: list[str] | None = None,
) -> dict:
    """
    Load a 1-min OHLC JSON and resample to 10 standard timeframes.

    Args:
        json_path: Path to 1-min OHLC JSON produced by write_json().
        pairs:     Optional instrument subset; defaults to all instruments
                   present in the JSON file. Accepts any of the 43 nodes
                   defined in universe.ALL_INSTRUMENTS.

    Returns:
        {tf_name: {instrument: DataFrame(dt, o, h, l, c, sp, tk)}}
        for timeframes: M1, M5, M15, M30, H1, H4, H12, D1, W1, MN1.
    """
    TIMEFRAME_NAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]
    TIMEFRAME_FREQS = ["1min", "5min", "15min", "30min", "1h", "4h", "12h", "1D", "1W", "1ME"]

    with open(json_path, "r") as f:
        data: dict = json.load(f)

    if pairs is None:
        # Preserve canonical node ordering for instruments present in the file
        present = set(data.keys())
        pairs = [p for p in ALL_INSTRUMENTS if p in present]

    result: dict = {}
    for tf_name, freq in zip(TIMEFRAME_NAMES, TIMEFRAME_FREQS):
        result[tf_name] = {}

        for instrument in pairs:
            if instrument not in data:
                continue

            bars = data[instrument]
            df = pd.DataFrame(bars)
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.set_index("dt").sort_index()

            if tf_name == "M1":
                resampled = df[["o", "h", "l", "c", "sp", "tk"]].copy()
            else:
                resampled = df.resample(freq).agg({
                    "o":  "first",
                    "h":  "max",
                    "l":  "min",
                    "c":  "last",
                    "sp": "mean",
                    "tk": "sum",
                }).dropna(subset=["c"])

            for col in ["o", "h", "l", "c", "sp"]:
                if resampled[col].isna().any():
                    resampled[col] = resampled[col].interpolate(
                        method="linear", limit_direction="both"
                    )

            resampled["tk"] = resampled["tk"].fillna(0)
            result[tf_name][instrument] = resampled.reset_index()

        n_pairs = len(result[tf_name])
        min_bars = (
            min(len(v) for v in result[tf_name].values())
            if result[tf_name]
            else 0
        )
        print(f"  {tf_name}: {n_pairs} instruments, {min_bars} bars min")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algo C2 v2: CSV → 1-min OHLC JSON (43 instruments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output", default="algo_c2_43_data.json",
        help="Output JSON path (default: algo_c2_43_data.json)",
    )
    parser.add_argument(
        "--mode", choices=["tick", "candle"], default="tick",
        help="Input format: 'tick' = MT5 tab-separated tick export (default), "
             "'candle' = OHLCV comma-separated with datetime,open,high,low,close,volume",
    )
    parser.add_argument(
        "--instruments", default=None,
        help="Comma-separated subset of instruments to process (default: all 43). "
             "Example: --instruments EURUSD,GBPUSD,BTCUSD",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date filter, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date filter, inclusive (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

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

    print(f"Mode: {args.mode} | Instruments: {len(instruments)}")

    pair_dfs: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for instrument in instruments:
        csv_path = find_csv_for_instrument(input_dir, instrument)
        if csv_path is None:
            missing.append(instrument)
            continue

        print(f"Processing {instrument} from {csv_path.name}...")
        try:
            if args.mode == "tick":
                raw = parse_tick_csv(csv_path)

                # Date filter on tick data
                if args.start:
                    raw = raw[raw["datetime"] >= pd.Timestamp(args.start)]
                if args.end:
                    raw = raw[raw["datetime"] <= pd.Timestamp(args.end + " 23:59:59.999")]

                if len(raw) == 0:
                    print(f"  Warning: no ticks for {instrument} in date range")
                    missing.append(instrument)
                    continue

                bars = resample_to_1min(raw, instrument)
                print(f"  {len(raw):,} ticks → {len(bars):,} bars, "
                      f"spread mean={bars['sp'].mean():.2f} pips")

            else:  # candle mode
                raw_candles = parse_candle_csv(csv_path)

                # Date filter on candle data
                if args.start:
                    raw_candles = raw_candles[
                        raw_candles["datetime"] >= pd.Timestamp(args.start)
                    ]
                if args.end:
                    raw_candles = raw_candles[
                        raw_candles["datetime"] <= pd.Timestamp(args.end + " 23:59:59.999")
                    ]

                if len(raw_candles) == 0:
                    print(f"  Warning: no candles for {instrument} in date range")
                    missing.append(instrument)
                    continue

                bars = candle_to_1min_bars(raw_candles, instrument)
                print(f"  {len(raw_candles):,} candles → {len(bars):,} bars")

        except Exception as exc:
            print(f"  Error processing {instrument}: {exc}", file=sys.stderr)
            missing.append(instrument)
            continue

        pair_dfs[instrument] = bars

    if missing:
        role_map = {
            p: ("signal-only" if p in SIGNAL_ONLY else "tradeable")
            for p in missing
        }
        tradeable_missing = [p for p in missing if role_map[p] == "tradeable"]
        signal_missing = [p for p in missing if role_map[p] == "signal-only"]
        print(f"\nMissing {len(missing)} instrument(s):")
        if tradeable_missing:
            print(f"  Tradeable ({len(tradeable_missing)}): {', '.join(tradeable_missing)}")
        if signal_missing:
            print(f"  Signal-only ({len(signal_missing)}): {', '.join(signal_missing)}")
        print("  Expected CSV filenames containing the instrument name, e.g.:")
        print("    EURUSD_202603020005_202603202259.csv  (tick mode)")
        print("    EURUSD_2019_2023.csv                 (candle mode)")

    if not pair_dfs:
        print("Error: no instruments processed successfully", file=sys.stderr)
        sys.exit(1)

    # Align to master timeline
    print(f"\nAligning {len(pair_dfs)} instrument(s) to master timeline...")
    aligned = align_to_master(pair_dfs)

    master_len = max(len(df) for df in aligned.values())
    start_dt = min(df["dt"].iloc[0] for df in aligned.values())
    end_dt = max(df["dt"].iloc[-1] for df in aligned.values())
    print(f"Master timeline: {master_len:,} bars  ({start_dt} → {end_dt})")

    write_json(aligned, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
