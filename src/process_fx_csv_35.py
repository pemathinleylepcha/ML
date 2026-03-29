"""
Algo C2 — Phase 1: Tick CSV → 1-min OHLC JSON
Parses broker tick data for 35 instruments, resamples to 1-minute bars,
aligns to master timeline, and outputs compact JSON.

Usage:
    python process_fx_csv_35.py --input_dir ./data/csvs --output ./data/algo_c2_5day_data.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 35 instruments ──────────────────────────────────────────────────────────

PAIRS_FX = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]

PAIRS_NON_FX = ["BTCUSD", "US30", "USDMXN", "USDZAR", "XAGUSD", "XAUUSD", "XBRUSD"]

PAIRS_ALL = sorted(PAIRS_FX + PAIRS_NON_FX)

# ── Pip sizes per instrument ────────────────────────────────────────────────

PIP_SIZES = {}
_JPY_CROSSES = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
for p in PAIRS_FX:
    PIP_SIZES[p] = 0.01 if p in _JPY_CROSSES else 0.0001
PIP_SIZES.update({
    "BTCUSD": 1.0,
    "US30":   1.0,
    "XAUUSD": 0.1,
    "XAGUSD": 0.01,
    "XBRUSD": 0.01,
    "USDMXN": 0.0001,
    "USDZAR": 0.0001,
})


def find_csv_for_pair(input_dir: Path, pair: str) -> Path | None:
    """Find CSV file matching a pair name in the input directory."""
    for f in input_dir.glob("*.csv"):
        if pair in f.stem.upper():
            return f
    return None


def parse_tick_csv(filepath: Path) -> pd.DataFrame:
    """Parse a broker tick CSV (tab-separated) into a DataFrame with bid/ask/mid."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=0,
        names=["DATE", "TIME", "BID", "ASK", "LAST", "VOLUME", "FLAGS"],
        dtype={"DATE": str, "TIME": str},
        na_values=["", " "],
    )
    # Combine date + time into datetime
    df["datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="%Y.%m.%d %H:%M:%S.%f")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Convert to numeric
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["FLAGS"] = pd.to_numeric(df["FLAGS"], errors="coerce").fillna(6).astype(int)

    # Forward-fill partial ticks based on FLAGS
    # FLAGS=6: both present, FLAGS=2: bid only (ffill ask), FLAGS=4: ask only (ffill bid)
    df["BID"] = df["BID"].ffill()
    df["ASK"] = df["ASK"].ffill()

    # Drop rows where we still have no bid or ask (start-of-file edge)
    df = df.dropna(subset=["BID", "ASK"]).copy()

    # Mid price
    df["mid"] = (df["BID"] + df["ASK"]) / 2.0
    df["spread"] = df["ASK"] - df["BID"]

    return df[["datetime", "BID", "ASK", "mid", "spread"]].copy()


def _compute_bar_tick_path(group_mid: pd.Series, bar_open: float,
                           bar_high: float, bar_low: float,
                           bar_close: float, pip: float) -> list:
    """
    Build a compact intra-bar tick path for TP/SL resolution.

    Returns a list of price levels in chronological order that captures
    every local extreme within the bar. This is the minimum information
    needed to determine whether TP or SL was hit first.

    The path always starts with open and ends with close. Between them,
    we record every tick that sets a new high or new low for the bar.
    """
    if len(group_mid) <= 2:
        # Not enough ticks — return O→H→L→C or O→L→H→C based on
        # whether high or low was hit first
        # With <=2 ticks we can't distinguish, use midpoint heuristic
        return [bar_open, bar_high, bar_low, bar_close]

    prices = group_mid.values
    path = [round(float(prices[0]), 6)]  # open
    running_high = prices[0]
    running_low = prices[0]

    for p in prices[1:-1]:
        if p > running_high:
            running_high = p
            path.append(round(float(p), 6))
        elif p < running_low:
            running_low = p
            path.append(round(float(p), 6))
        # Skip ticks that don't set new extremes

    path.append(round(float(prices[-1]), 6))  # close
    return path


def resample_to_1min(ticks: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Resample tick data to 1-minute OHLC bars with spread, tick count,
    and intra-bar tick path for accurate TP/SL resolution.
    """
    pip = PIP_SIZES[pair]
    ticks = ticks.set_index("datetime")

    bars = ticks["mid"].resample("1min").ohlc()
    bars.columns = ["o", "h", "l", "c"]

    # Spread in pips (mean per bar)
    bars["sp"] = ticks["spread"].resample("1min").mean() / pip

    # Tick count per bar
    bars["tk"] = ticks["mid"].resample("1min").count()

    # Drop bars with zero ticks (no data in that minute)
    bars = bars[bars["tk"] > 0].copy()

    # Build intra-bar tick paths for TP/SL resolution
    # Group ticks by minute and compute path for each bar
    tick_paths = []
    for dt_idx in bars.index:
        # Get ticks in this 1-minute window
        end = dt_idx + pd.Timedelta(minutes=1)
        mask = (ticks.index >= dt_idx) & (ticks.index < end)
        group = ticks.loc[mask, "mid"]

        row = bars.loc[dt_idx]
        path = _compute_bar_tick_path(
            group, row["o"], row["h"], row["l"], row["c"], pip
        )
        tick_paths.append(path)

    bars["tp"] = tick_paths  # tick path

    # Round spread to 2 decimals
    bars["sp"] = bars["sp"].round(2)

    # Format datetime index to string
    bars.index = bars.index.strftime("%Y-%m-%d %H:%M")
    bars.index.name = "dt"

    return bars.reset_index()


def align_to_master(pair_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align all pairs to a single master timeline, forward-filling gaps."""
    # Build master timestamp index (union of all pair timestamps)
    all_timestamps = set()
    for df in pair_dfs.values():
        all_timestamps.update(df["dt"].tolist())
    master_index = sorted(all_timestamps)

    aligned = {}
    for pair, df in pair_dfs.items():
        df = df.set_index("dt")
        df = df.reindex(master_index)
        # Forward-fill missing bars
        df = df.ffill()
        # Drop leading NaN rows (before pair's first valid data)
        df = df.dropna(subset=["o"])
        df.index.name = "dt"
        aligned[pair] = df.reset_index()

    return aligned


def write_json(pair_data: dict[str, pd.DataFrame], output_path: str):
    """Write aligned bar data to compact JSON, including tick paths."""
    result = {}
    for pair in sorted(pair_data.keys()):
        df = pair_data[pair]
        records = []
        for _, row in df.iterrows():
            rec = {
                "dt": row["dt"],
                "o": round(float(row["o"]), 6),
                "h": round(float(row["h"]), 6),
                "l": round(float(row["l"]), 6),
                "c": round(float(row["c"]), 6),
                "sp": round(float(row["sp"]), 2),
                "tk": int(row["tk"]),
            }
            # Include tick path if available (for TP/SL resolution)
            if "tp" in row and isinstance(row["tp"], list) and len(row["tp"]) > 0:
                rec["tp"] = row["tp"]
            records.append(rec)
        result[pair] = records

    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


def resample_multi_timeframe(json_path: str, pairs: list[str] = None) -> dict:
    """
    Load 1-min JSON and resample to 10 timeframes for STGNN consumption.

    Args:
        json_path: Path to 1-min OHLC JSON (output of write_json).
        pairs: Optional pair list; defaults to all 35 instruments.

    Returns:
        {tf_name: {pair: DataFrame(dt, o, h, l, c, sp, tk)}} for
        M1, M5, M15, M30, H1, H4, H12, D1, W1, MN1.
    """
    TIMEFRAME_NAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]
    TIMEFRAME_FREQS = ["1min", "5min", "15min", "30min", "1h", "4h", "12h", "1D", "1W", "1ME"]

    with open(json_path, "r") as f:
        data = json.load(f)

    if pairs is None:
        pairs = sorted(data.keys())

    result = {}
    for tf_name, freq in zip(TIMEFRAME_NAMES, TIMEFRAME_FREQS):
        result[tf_name] = {}

        for pair in pairs:
            if pair not in data:
                continue

            bars = data[pair]
            df = pd.DataFrame(bars)
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.set_index("dt").sort_index()

            if tf_name == "M1":
                resampled = df[["o", "h", "l", "c", "sp", "tk"]].copy()
            else:
                resampled = df.resample(freq).agg({
                    "o": "first", "h": "max", "l": "min", "c": "last",
                    "sp": "mean", "tk": "sum",
                }).dropna(subset=["c"])

            # Interpolate missing price values
            for col in ["o", "h", "l", "c", "sp"]:
                if resampled[col].isna().any():
                    resampled[col] = resampled[col].interpolate(
                        method="linear", limit_direction="both"
                    )

            resampled["tk"] = resampled["tk"].fillna(0)
            result[tf_name][pair] = resampled.reset_index()

        n_pairs = len(result[tf_name])
        min_bars = min(len(v) for v in result[tf_name].values()) if result[tf_name] else 0
        print(f"  {tf_name}: {n_pairs} pairs, {min_bars} bars min")

    return result


def main():
    parser = argparse.ArgumentParser(description="Algo C2: Tick CSV → 1-min OHLC JSON")
    parser.add_argument("--input_dir", required=True, help="Directory containing tick CSVs")
    parser.add_argument("--output", default="algo_c2_5day_data.json", help="Output JSON path")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Process each pair
    pair_dfs = {}
    missing = []
    for pair in PAIRS_ALL:
        csv_path = find_csv_for_pair(input_dir, pair)
        if csv_path is None:
            missing.append(pair)
            continue

        print(f"Processing {pair} from {csv_path.name}...")
        ticks = parse_tick_csv(csv_path)

        # Date filter
        if args.start:
            ticks = ticks[ticks["datetime"] >= pd.Timestamp(args.start)]
        if args.end:
            ticks = ticks[ticks["datetime"] <= pd.Timestamp(args.end + " 23:59:59.999")]

        if len(ticks) == 0:
            print(f"  Warning: no ticks for {pair} in date range")
            missing.append(pair)
            continue

        bars = resample_to_1min(ticks, pair)
        pair_dfs[pair] = bars
        print(f"  {len(bars)} bars, spread mean={bars['sp'].mean():.2f} pips")

    if missing:
        print(f"\nMissing pairs ({len(missing)}): {', '.join(missing)}")
        print("Expected CSV filenames like: EURUSD_202603020005_202603202259.csv")

    if not pair_dfs:
        print("Error: no pairs processed")
        sys.exit(1)

    # Align to master timeline
    print(f"\nAligning {len(pair_dfs)} pairs to master timeline...")
    aligned = align_to_master(pair_dfs)

    master_len = max(len(df) for df in aligned.values())
    print(f"Master timeline: {master_len} bars")

    # Write output
    write_json(aligned, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
