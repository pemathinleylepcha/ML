from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dataextractor_contract import resolve_candle_path
from feature_engine import compute_atr
from universe import ALL_INSTRUMENTS, PIP_SIZES, TIMEFRAME_FREQ

BASE_TIMEFRAME = "M5"
DERIVED_TIMEFRAMES = ("M15", "H1", "H4")
RESEARCH_TIMEFRAMES = (BASE_TIMEFRAME,) + DERIVED_TIMEFRAMES
FILL_POLICY_CARRY = "carry"
FILL_POLICY_MASK = "mask"
VALID_FILL_POLICIES = {FILL_POLICY_CARRY, FILL_POLICY_MASK}

_CSV_COLS = ["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
_RENAME = {
    "bar_time": "dt",
    "open": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "tick_volume": "tk",
    "spread": "sp",
}

SESSION_NAMES = ("closed", "tokyo", "london", "newyork", "overlap")
SESSION_CODES = {name: idx for idx, name in enumerate(SESSION_NAMES)}

BUY_CLASS = 1
SELL_CLASS = 0
HOLD_CLASS = 2


@dataclass(slots=True)
class CanonicalResearchDataset:
    base_timeframe: str
    timeframes: tuple[str, ...]
    fill_policy: str
    tf_data: dict[str, dict[str, dict[str, np.ndarray]]]
    base_timestamps: np.ndarray
    session_codes: np.ndarray
    session_names: np.ndarray
    quarter_ids: np.ndarray
    tf_index_for_base: dict[str, np.ndarray]
    outer_holdout_quarters: tuple[str, ...]

    @property
    def n_bars(self) -> int:
        return len(self.base_timestamps)

    @property
    def symbols(self) -> list[str]:
        return list(self.tf_data.get(self.base_timeframe, {}).keys())


def _read_candles_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=_CSV_COLS)
    parsed = pd.to_datetime(df["bar_time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    if parsed.notna().sum() < max(3, len(df) // 4):
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip("<>").lower() for c in df.columns]
        rename_map = {
            "bar_time": "dt",
            "datetime": "dt",
            "time": "dt",
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "tick_volume": "tk",
            "volume": "tk",
            "spread": "sp",
        }
        df = df.rename(columns=rename_map)
    else:
        df = df.rename(columns=_RENAME)
        df["dt"] = parsed

    if "dt" not in df.columns:
        raise ValueError(f"Could not locate datetime column in {csv_path}")

    if not np.issubdtype(df["dt"].dtype, np.datetime64):
        df["dt"] = pd.to_datetime(df["dt"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt"])
    for col in ("o", "h", "l", "c", "sp", "tk"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[["dt", "o", "h", "l", "c", "sp", "tk"]].sort_values("dt").drop_duplicates("dt")
    return df


def _load_base_frames(
    data_root: Path,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    io_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    selected = sorted(set(symbols or ALL_INSTRUMENTS))
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    frames: dict[str, list[pd.DataFrame]] = {}
    tasks: list[tuple[str, Path]] = []

    for year_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for quarter_dir in sorted(q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q")):
            for symbol in selected:
                csv_path, _ = resolve_candle_path(quarter_dir, symbol, BASE_TIMEFRAME)
                if csv_path is None:
                    continue
                tasks.append((symbol, csv_path))

    def _load_task(task: tuple[str, Path]) -> tuple[str, pd.DataFrame] | None:
        symbol, csv_path = task
        try:
            df = _read_candles_csv(csv_path)
        except Exception:
            return None
        if start_ts is not None:
            df = df[df["dt"] >= start_ts]
        if end_ts is not None:
            df = df[df["dt"] <= end_ts]
        if len(df) == 0:
            return None
        return symbol, df

    max_workers = max(1, io_workers or min(8, max(1, (os.cpu_count() or 4) // 2)))
    if max_workers == 1 or len(tasks) <= 1:
        loaded_items = (_load_task(task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            loaded_items = pool.map(_load_task, tasks)

    for item in loaded_items:
        if item is None:
            continue
        symbol, df = item
        frames.setdefault(symbol, []).append(df)

    merged: dict[str, pd.DataFrame] = {}
    for symbol, chunks in frames.items():
        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
        merged[symbol] = df
    return merged


def _session_name(dt: pd.Timestamp) -> str:
    wd = dt.weekday()
    hour = dt.hour
    if wd >= 5 or (wd == 4 and hour >= 22):
        return "closed"
    if 13 <= hour < 17:
        return "overlap"
    if 2 <= hour < 8:
        return "tokyo"
    if 8 <= hour < 13:
        return "london"
    if 17 <= hour < 22:
        return "newyork"
    return "closed"


def encode_session_codes(timestamps: np.ndarray | pd.Index) -> np.ndarray:
    dt_index = pd.DatetimeIndex(timestamps)
    hours = dt_index.hour.to_numpy()
    weekdays = dt_index.weekday.to_numpy()
    session_codes = np.full(len(dt_index), SESSION_CODES["closed"], dtype=np.int8)
    open_day = weekdays < 5
    overlap = open_day & (hours >= 13) & (hours < 17)
    tokyo = open_day & (hours >= 2) & (hours < 8)
    london = open_day & (hours >= 8) & (hours < 13)
    newyork = open_day & (hours >= 17) & (hours < 22)
    session_codes[tokyo] = SESSION_CODES["tokyo"]
    session_codes[london] = SESSION_CODES["london"]
    session_codes[newyork] = SESSION_CODES["newyork"]
    session_codes[overlap] = SESSION_CODES["overlap"]
    return session_codes


def _build_base_index(base_frames: dict[str, pd.DataFrame], fill_policy: str) -> pd.Index:
    if fill_policy == FILL_POLICY_MASK:
        starts = [frame.index.min() for frame in base_frames.values()]
        ends = [frame.index.max() for frame in base_frames.values()]
        return pd.date_range(min(starts), max(ends), freq=TIMEFRAME_FREQ[BASE_TIMEFRAME], name="dt")
    return pd.Index(sorted({ts for frame in base_frames.values() for ts in frame.index}), name="dt")


def _build_resample_index(base_index: pd.Index, tf_name: str) -> pd.DatetimeIndex:
    reference = pd.Series(np.ones(len(base_index), dtype=np.int8), index=pd.DatetimeIndex(base_index))
    return reference.resample(TIMEFRAME_FREQ[tf_name], label="right", closed="right").max().index


def _resample_frame(
    df: pd.DataFrame,
    tf_name: str,
    keep_calendar_gaps: bool = False,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    freq = TIMEFRAME_FREQ[tf_name]
    agg = df.resample(freq, label="right", closed="right").agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "sp": "mean",
        "tk": "sum",
        "real": "max",
    })
    if target_index is not None:
        agg = agg.reindex(target_index)
    if not keep_calendar_gaps:
        agg = agg.dropna(subset=["c"])
    return agg


def _frame_to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "dt": df.index.astype("datetime64[ns]").to_numpy(),
        "o": df["o"].to_numpy(dtype=np.float32),
        "h": df["h"].to_numpy(dtype=np.float32),
        "l": df["l"].to_numpy(dtype=np.float32),
        "c": df["c"].to_numpy(dtype=np.float32),
        "sp": df["sp"].fillna(0.0).to_numpy(dtype=np.float32),
        "tk": df["tk"].fillna(0.0).to_numpy(dtype=np.float32),
        "real": df["real"].fillna(False).to_numpy(dtype=np.bool_),
    }


def _align_base_frame_with_policy(frame: pd.DataFrame, base_index: pd.Index, fill_policy: str) -> pd.DataFrame:
    aligned = frame.reindex(base_index).copy()
    real_mask = aligned["c"].notna().to_numpy(dtype=np.bool_)
    if fill_policy == FILL_POLICY_CARRY:
        price_cols = ["o", "h", "l", "c"]
        aligned[price_cols] = aligned[price_cols].ffill().bfill()
    elif fill_policy != FILL_POLICY_MASK:
        raise ValueError(f"Unsupported fill_policy={fill_policy!r}; expected one of {sorted(VALID_FILL_POLICIES)}")
    aligned["sp"] = aligned["sp"].fillna(0.0)
    aligned["tk"] = aligned["tk"].fillna(0.0)
    aligned["real"] = real_mask
    return aligned


def load_canonical_research_dataset(
    data_root: str,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    io_workers: int | None = None,
    fill_policy: str = FILL_POLICY_CARRY,
) -> CanonicalResearchDataset:
    root = Path(data_root)
    if fill_policy not in VALID_FILL_POLICIES:
        raise ValueError(f"Unsupported fill_policy={fill_policy!r}; expected one of {sorted(VALID_FILL_POLICIES)}")
    base_frames = _load_base_frames(root, symbols=symbols, start=start, end=end, io_workers=io_workers)
    if not base_frames:
        raise ValueError(f"No {BASE_TIMEFRAME} data found under {data_root}")

    base_index = _build_base_index(base_frames, fill_policy=fill_policy)
    tf_data: dict[str, dict[str, dict[str, np.ndarray]]] = {tf: {} for tf in RESEARCH_TIMEFRAMES}
    target_resample_indices = {
        tf_name: _build_resample_index(base_index, tf_name)
        for tf_name in DERIVED_TIMEFRAMES
    }

    for symbol, frame in base_frames.items():
        aligned = _align_base_frame_with_policy(frame, base_index, fill_policy=fill_policy)
        tf_data[BASE_TIMEFRAME][symbol] = _frame_to_arrays(aligned)

        aligned_df = aligned[["o", "h", "l", "c", "sp", "tk", "real"]]
        for tf_name in DERIVED_TIMEFRAMES:
            resampled = _resample_frame(
                aligned_df,
                tf_name,
                keep_calendar_gaps=fill_policy == FILL_POLICY_MASK,
                target_index=target_resample_indices[tf_name] if fill_policy == FILL_POLICY_MASK else None,
            )
            tf_data[tf_name][symbol] = _frame_to_arrays(resampled)

    base_any_symbol = next(iter(tf_data[BASE_TIMEFRAME].values()))
    base_timestamps = base_any_symbol["dt"]
    base_dt_index = pd.Index(base_timestamps)
    dt_index = pd.DatetimeIndex(base_timestamps)
    session_codes = encode_session_codes(base_timestamps)
    session_names = np.array([SESSION_NAMES[idx] for idx in session_codes], dtype=object)
    quarter_ids = dt_index.to_period("Q").astype(str).to_numpy(dtype=object)

    tf_index_for_base: dict[str, np.ndarray] = {}
    for tf_name in RESEARCH_TIMEFRAMES:
        tf_any_symbol = next(iter(tf_data[tf_name].values()))
        tf_dt_index = pd.Index(tf_any_symbol["dt"])
        positions = tf_dt_index.searchsorted(base_dt_index, side="right") - 1
        positions = np.maximum(positions, 0).astype(np.int32)
        tf_index_for_base[tf_name] = positions

    unique_quarters = sorted(pd.unique(quarter_ids).tolist())
    outer_holdout_quarters = tuple(unique_quarters[-2:]) if len(unique_quarters) >= 2 else tuple(unique_quarters)

    return CanonicalResearchDataset(
        base_timeframe=BASE_TIMEFRAME,
        timeframes=RESEARCH_TIMEFRAMES,
        fill_policy=fill_policy,
        tf_data=tf_data,
        base_timestamps=base_timestamps,
        session_codes=session_codes,
        session_names=session_names,
        quarter_ids=quarter_ids,
        tf_index_for_base=tf_index_for_base,
        outer_holdout_quarters=outer_holdout_quarters,
    )


def compute_atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    atr = np.zeros(len(close), dtype=np.float32)
    for idx in range(len(close)):
        eff_period = max(2, min(period, idx)) if idx > 0 else 2
        atr[idx] = compute_atr(high[: idx + 1], low[: idx + 1], close[: idx + 1], period=eff_period)
    return atr


def estimate_spread_cost(pair: str, spread_raw: float, close_price: float) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    if spread_raw <= 0:
        return 0.0
    if spread_raw < pip * 50:
        return float(spread_raw)
    return float(spread_raw) * (pip / 10.0)


def make_triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    spread: np.ndarray,
    pair: str,
    horizon: int = 6,
    atr_period: int = 14,
    binary: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    atr = compute_atr_series(high, low, close, period=atr_period)
    labels = np.full(len(close), HOLD_CLASS if not binary else -1, dtype=np.int32)
    valid = np.zeros(len(close), dtype=np.bool_)

    for idx in range(max(atr_period, 2), len(close) - horizon):
        entry = float(close[idx])
        spread_cost = estimate_spread_cost(pair, float(spread[idx]), entry)
        barrier = max(2.0 * spread_cost, 0.35 * float(atr[idx]))
        upper = entry + barrier
        lower = entry - barrier

        decided = HOLD_CLASS
        for step in range(1, horizon + 1):
            hi = float(high[idx + step])
            lo = float(low[idx + step])
            if hi >= upper and lo <= lower:
                mid_move = float(close[idx + step] - entry)
                decided = BUY_CLASS if mid_move >= 0 else SELL_CLASS
                break
            if hi >= upper:
                decided = BUY_CLASS
                break
            if lo <= lower:
                decided = SELL_CLASS
                break

        if binary:
            if decided in (BUY_CLASS, SELL_CLASS):
                labels[idx] = decided
                valid[idx] = True
        else:
            labels[idx] = decided
            valid[idx] = True

    return labels, valid


def build_split_metadata(
    quarter_ids: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    n_inner_folds: int = 4,
) -> dict:
    inner_quarters = [q for q in pd.unique(quarter_ids).tolist() if q not in set(outer_holdout_quarters)]
    if len(inner_quarters) < 2:
        return {"inner_folds": [], "outer_holdout_quarters": list(outer_holdout_quarters)}

    step = max(1, len(inner_quarters) // (n_inner_folds + 1))
    folds = []
    for fold_idx in range(1, n_inner_folds + 1):
        train_quarters = inner_quarters[: fold_idx * step]
        val_quarters = inner_quarters[fold_idx * step: min(len(inner_quarters), (fold_idx + 1) * step)]
        if not train_quarters or not val_quarters:
            continue
        folds.append({
            "fold": fold_idx - 1,
            "train_quarters": train_quarters,
            "val_quarters": val_quarters,
        })
    return {"inner_folds": folds, "outer_holdout_quarters": list(outer_holdout_quarters)}
