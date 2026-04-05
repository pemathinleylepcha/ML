from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDT"
INTERVAL = "5m"
LIMIT = 1000


@dataclass(slots=True)
class SpreadProfile:
    global_median: float
    by_hour_of_week: dict[int, float]

    def spread_for(self, dt: pd.Timestamp) -> float:
        hour_of_week = int(dt.weekday()) * 24 + int(dt.hour)
        return float(self.by_hour_of_week.get(hour_of_week, self.global_median))


def _unix_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _http_get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "Algo-C2-Codex/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_binance_klines(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor_ms = _unix_ms(start_dt)
    end_ms = _unix_ms(end_dt)

    while cursor_ms < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": BINANCE_SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": LIMIT,
            }
        )
        rows = _http_get_json(f"{BINANCE_BASE_URL}?{params}")
        if not rows:
            break

        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_base_volume",
                "taker_quote_volume",
                "ignore",
            ],
        )
        frames.append(frame)
        last_open_ms = int(frame["open_time"].iloc[-1])
        next_cursor = last_open_ms + 5 * 60 * 1000
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    dt = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    df["dt"] = dt
    mask = (df["dt"] >= pd.Timestamp(start_dt)) & (df["dt"] <= pd.Timestamp(end_dt))
    df = df.loc[mask].reset_index(drop=True)
    return df


def load_reference_spread_profile(root: Path, year: str, exclude_quarter: str = "Q1") -> SpreadProfile:
    spreads: list[pd.DataFrame] = []
    year_dir = root / year
    for quarter_dir in sorted([q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q")]):
        if quarter_dir.name == exclude_quarter:
            continue
        csv_path = quarter_dir / "BTCUSD" / "candles_M5.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, usecols=["bar_time", "spread"])
        frame["dt"] = pd.to_datetime(frame["bar_time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
        frame = frame.dropna(subset=["dt"])
        frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
        frame = frame.dropna(subset=["spread"])
        spreads.append(frame[["dt", "spread"]])

    if not spreads:
        return SpreadProfile(global_median=0.0, by_hour_of_week={})

    ref = pd.concat(spreads, ignore_index=True)
    global_median = float(ref["spread"].median())
    hour_of_week = ref["dt"].dt.weekday.astype(int) * 24 + ref["dt"].dt.hour.astype(int)
    grouped = ref.groupby(hour_of_week)["spread"].median()
    return SpreadProfile(
        global_median=global_median,
        by_hour_of_week={int(idx): float(val) for idx, val in grouped.items()},
    )


def write_quarterly_csvs(df: pd.DataFrame, dest_root: Path, symbol: str = "BTCUSD") -> dict[str, int]:
    if df.empty:
        return {}

    counts: dict[str, int] = {}
    df = df.copy()
    df["year"] = df["dt"].dt.year.astype(str)
    df["quarter"] = "Q" + df["dt"].dt.quarter.astype(str)

    for (year, quarter), group in df.groupby(["year", "quarter"], sort=True):
        out_dir = dest_root / year / quarter / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "candles_M5.csv"
        write_frame = group[
            ["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        ].copy()
        if out_path.exists():
            existing = pd.read_csv(out_path)
            merged = pd.concat([existing, write_frame], ignore_index=True)
            merged = merged.drop_duplicates(subset=["bar_time"]).sort_values("bar_time")
            merged.to_csv(out_path, index=False)
            counts[f"{year}/{quarter}"] = len(write_frame)
        else:
            write_frame.to_csv(out_path, index=False)
            counts[f"{year}/{quarter}"] = len(write_frame)
    return counts


def backfill_btcusd_from_binance(
    dest_root: Path,
    start_dt: datetime,
    end_dt: datetime,
    year: str,
) -> dict:
    raw = fetch_binance_klines(start_dt, end_dt)
    if raw.empty:
        raise RuntimeError("No Binance BTCUSDT 5m data returned for the requested range.")

    spread_profile = load_reference_spread_profile(dest_root, year=year, exclude_quarter="Q1")
    bars = pd.DataFrame(
        {
            "dt": raw["dt"],
            "bar_time": raw["dt"].dt.strftime("%Y.%m.%d %H:%M:%S"),
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "tick_volume": raw["trade_count"].astype(int),
            "spread": raw["dt"].map(spread_profile.spread_for).astype(float),
            "real_volume": raw["quote_volume"].astype(float),
        }
    )

    quarter_counts = write_quarterly_csvs(bars, dest_root=dest_root, symbol="BTCUSD")
    summary = {
        "source": "binance:BTCUSDT",
        "mapped_symbol": "BTCUSD",
        "start": bars["bar_time"].iloc[0],
        "end": bars["bar_time"].iloc[-1],
        "rows": int(len(bars)),
        "quarter_counts": quarter_counts,
        "spread_global_median": spread_profile.global_median,
    }
    summary_path = dest_root / year / "Q1" / "BTCUSD" / "backfill_meta_binance.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BTCUSD M5 using Binance BTCUSDT candles.")
    parser.add_argument("--dest-root", required=True, help="Destination DataExtractor root")
    parser.add_argument("--start", default="2025-01-01", help="Inclusive start date")
    parser.add_argument("--end", default="2025-03-31", help="Inclusive end date")
    parser.add_argument("--year", default="2025", help="Year folder under the destination root")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    summary = backfill_btcusd_from_binance(
        dest_root=Path(args.dest_root),
        start_dt=start_dt,
        end_dt=end_dt,
        year=args.year,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
