from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object

from dataextractor_contract import resolve_candle_path
from export_research_stgnn_pack import FEATURE_ORDER
from research_dataset import SESSION_CODES, encode_session_codes, estimate_spread_cost, make_triple_barrier_labels
from universe import ALL_INSTRUMENTS, SUBNET_24x5, SUBNET_24x5_TRADEABLE, SUBNET_24x7, TIMEFRAME_FREQ

# Real training uses a shorter upper ladder than the architectural maximum.
# Weekly/monthly branches are too sparse on a 1-year panel and tended to
# dominate exchange dynamics without adding stable edge.
TIMEFRAMES_FROM_M5 = ("M5", "M15", "M30", "H1", "H4", "H12", "D1")
VALIDITY_IDX = FEATURE_ORDER.index("validity")
SESSION_SCALE = float(max(1, len(SESSION_CODES) - 1))
CSV_COLS = ["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
TRADE_SESSION_CODES = (
    SESSION_CODES["london"],
    SESSION_CODES["newyork"],
    SESSION_CODES["overlap"],
)


@dataclass(slots=True)
class CooperativeRealDataset:
    timeframes: tuple[str, ...]
    base_timestamps: np.ndarray
    session_codes: np.ndarray
    quarter_ids: np.ndarray
    outer_holdout_quarters: tuple[str, ...]
    tf_index_for_base: dict[str, np.ndarray]
    tf_session_codes: dict[str, np.ndarray]
    btc_tensors: dict[str, np.ndarray]
    fx_tensors: dict[str, np.ndarray]
    btc_labels: np.ndarray
    btc_entry_labels: np.ndarray
    btc_valid: np.ndarray
    btc_forward_returns: np.ndarray
    fx_labels: np.ndarray
    fx_entry_labels: np.ndarray
    fx_valid: np.ndarray
    fx_forward_returns: np.ndarray
    regime_codes: np.ndarray
    btc_node_names: tuple[str, ...]
    fx_node_names: tuple[str, ...]
    fx_tradable_node_names: tuple[str, ...]


def _read_candle_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS)
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
        df = df.rename(
            columns={
                "bar_time": "dt",
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "tick_volume": "tk",
                "spread": "sp",
            }
        )
        df["dt"] = parsed
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt"])
    for col in ("o", "h", "l", "c", "sp", "tk"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["o", "h", "l", "c"])
    return df[["dt", "o", "h", "l", "c", "sp", "tk"]].sort_values("dt").drop_duplicates("dt")


def _load_timeframe_frames(
    data_root: Path,
    timeframe: str,
    symbols: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    io_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    selected = tuple(symbols)
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    tasks: list[tuple[str, Path]] = []
    for year_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for quarter_dir in sorted(q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q")):
            for symbol in selected:
                csv_path, _ = resolve_candle_path(quarter_dir, symbol, timeframe)
                if csv_path is not None:
                    tasks.append((symbol, csv_path))

    def _load_one(task: tuple[str, Path]) -> tuple[str, pd.DataFrame] | None:
        symbol, csv_path = task
        try:
            df = _read_candle_csv(csv_path)
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
        loaded_items = (_load_one(task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            loaded_items = pool.map(_load_one, tasks)

    by_symbol: dict[str, list[pd.DataFrame]] = {}
    for item in loaded_items:
        if item is None:
            continue
        symbol, df = item
        by_symbol.setdefault(symbol, []).append(df)

    merged: dict[str, pd.DataFrame] = {}
    for symbol, chunks in by_symbol.items():
        frame = pd.concat(chunks, ignore_index=True)
        merged[symbol] = frame.sort_values("dt").drop_duplicates("dt").set_index("dt")
    return merged


def _align_frame(frame: pd.DataFrame | None, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    if frame is None:
        aligned = pd.DataFrame(index=target_index, columns=["o", "h", "l", "c", "sp", "tk"], dtype=np.float32)
    else:
        aligned = frame.reindex(target_index).copy()
    real_mask = aligned["c"].notna().to_numpy(dtype=np.bool_)
    aligned["sp"] = aligned["sp"].fillna(0.0)
    aligned["tk"] = aligned["tk"].fillna(0.0)
    aligned["real"] = real_mask
    return aligned


def _compute_session_transition(session_codes: np.ndarray) -> np.ndarray:
    transition = np.zeros(len(session_codes), dtype=np.float32)
    if len(session_codes) > 1:
        transition[1:] = (session_codes[1:] != session_codes[:-1]).astype(np.float32)
    return transition


def _compute_regime_signal(close: np.ndarray, valid: np.ndarray, lookback: int = 12) -> np.ndarray:
    close_series = pd.Series(close.astype(np.float64, copy=False))
    returns = close_series.pct_change(fill_method=None)
    trend = close_series.pct_change(lookback, fill_method=None)
    realized_vol = returns.abs().rolling(lookback, min_periods=3).mean()
    scaled = trend / (realized_vol * np.sqrt(float(lookback)) + 1e-8)
    regime = scaled.clip(-3.0, 3.0).fillna(0.0).to_numpy(dtype=np.float32) / 3.0
    regime[~valid] = 0.0
    return regime


def _frame_to_feature_matrix(frame: pd.DataFrame, session_codes: np.ndarray) -> np.ndarray:
    valid = frame["real"].to_numpy(dtype=np.bool_)
    matrix = np.zeros((len(frame), len(FEATURE_ORDER)), dtype=np.float32)
    matrix[:, 0] = frame["o"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 1] = frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 2] = frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 3] = frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 4] = frame["sp"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 5] = frame["tk"].fillna(0.0).to_numpy(dtype=np.float32)
    matrix[:, 6] = valid.astype(np.float32)
    matrix[:, 7] = session_codes.astype(np.float32) / SESSION_SCALE
    matrix[:, 8] = _compute_session_transition(session_codes)
    matrix[:, 9] = _compute_regime_signal(matrix[:, 3], valid)
    return matrix


def _compute_global_regime_codes(base_close_by_symbol: dict[str, np.ndarray], base_valid_by_symbol: dict[str, np.ndarray]) -> np.ndarray:
    close_stack = []
    valid_stack = []
    for symbol in SUBNET_24x5_TRADEABLE:
        close = base_close_by_symbol.get(symbol)
        valid = base_valid_by_symbol.get(symbol)
        if close is None or valid is None:
            continue
        close_stack.append(close.astype(np.float64, copy=False))
        valid_stack.append(valid.astype(np.bool_, copy=False))
    if not close_stack:
        return np.ones(len(next(iter(base_close_by_symbol.values()))), dtype=np.float32)
    close_matrix = np.column_stack(close_stack)
    valid_matrix = np.column_stack(valid_stack)
    safe_close = np.maximum(np.nan_to_num(close_matrix, nan=np.nan), 1e-8)
    log_close = np.log(safe_close)
    returns = np.zeros_like(log_close, dtype=np.float64)
    returns[1:] = np.diff(log_close, axis=0)
    valid_ret = valid_matrix.copy()
    valid_ret[1:] &= valid_matrix[:-1]
    returns = np.where(valid_ret, returns, np.nan)
    abs_returns = np.abs(returns)
    stress = np.full(abs_returns.shape[0], np.nan, dtype=np.float64)
    for row_idx in range(abs_returns.shape[0]):
        finite_row = abs_returns[row_idx][np.isfinite(abs_returns[row_idx])]
        if finite_row.size > 0:
            stress[row_idx] = float(np.median(finite_row))
    finite = np.isfinite(stress)
    regime = np.ones(len(stress), dtype=np.float32)
    if finite.any():
        q50, q75, q90 = np.quantile(stress[finite], [0.5, 0.75, 0.9])
        regime[stress >= q50] = 2.0
        regime[stress >= q75] = 3.0
        regime[stress >= q90] = 4.0
    regime[~finite] = 1.0
    return regime


def _compute_session_progress(
    session_codes: np.ndarray,
    trade_session_codes: tuple[int, ...] = TRADE_SESSION_CODES,
) -> np.ndarray:
    codes = np.asarray(session_codes, dtype=np.int32)
    progress = np.zeros(len(codes), dtype=np.int32)
    active_count = 0
    previous = None
    for idx, code in enumerate(codes.tolist()):
        if previous is None or code != previous:
            active_count = 0
        if code in trade_session_codes:
            active_count += 1
            progress[idx] = active_count
        else:
            active_count = 0
        previous = code
    return progress


def _compute_session_open_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    real: np.ndarray,
    session_codes: np.ndarray,
    opening_range_bars: int = 3,
    trade_session_codes: tuple[int, ...] = TRADE_SESSION_CODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    real_mask = np.asarray(real, dtype=np.bool_)
    codes = np.asarray(session_codes, dtype=np.int32)
    progress = _compute_session_progress(codes, trade_session_codes=trade_session_codes)
    open_ready = progress >= int(max(1, opening_range_bars))
    opening_range_width = np.zeros(len(close), dtype=np.float32)
    breakout_signed = np.zeros(len(close), dtype=np.float32)

    start = 0
    while start < len(codes):
        end = start + 1
        while end < len(codes) and codes[end] == codes[start]:
            end += 1
        session_code = int(codes[start])
        if session_code in trade_session_codes:
            seg_real = np.flatnonzero(real_mask[start:end]) + start
            if len(seg_real) >= opening_range_bars:
                opening_positions = seg_real[:opening_range_bars]
                or_high = float(np.max(high[opening_positions]))
                or_low = float(np.min(low[opening_positions]))
                width = max(or_high - or_low, float(max(np.max(close[opening_positions]), 1e-6)) * 1e-6)
                or_mid = 0.5 * (or_high + or_low)
                opening_range_width[start:end] = width / np.maximum(close[start:end], 1e-6)
                breakout_signed[start:end] = (close[start:end] - or_mid) / width
        start = end
    return open_ready.astype(np.bool_), opening_range_width, breakout_signed.astype(np.float32, copy=False)


def _build_scalp_entry_labels(
    symbol: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    spread: np.ndarray,
    tick: np.ndarray,
    real: np.ndarray,
    session_codes: np.ndarray,
    direction_labels: np.ndarray,
    direction_valid: np.ndarray,
    opening_range_bars: int = 3,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    spread = np.asarray(spread, dtype=np.float32)
    tick = np.asarray(tick, dtype=np.float32)
    real_mask = np.asarray(real, dtype=np.bool_)
    label = np.asarray(direction_labels, dtype=np.float32)
    valid = np.asarray(direction_valid, dtype=np.bool_)
    open_ready, _, breakout_signed = _compute_session_open_features(
        close,
        high,
        low,
        real_mask,
        session_codes,
        opening_range_bars=opening_range_bars,
    )
    trade_session = np.isin(np.asarray(session_codes, dtype=np.int32), list(TRADE_SESSION_CODES))
    safe_close = np.maximum(close.astype(np.float64), 1e-6)
    log_close = np.log(safe_close)
    log_ret = np.zeros(len(close), dtype=np.float32)
    if len(close) > 1:
        log_ret[1:] = np.diff(log_close).astype(np.float32)
    momentum_3 = pd.Series(log_ret).rolling(3, min_periods=2).sum().fillna(0.0).to_numpy(dtype=np.float32)
    momentum_6 = pd.Series(log_ret).rolling(6, min_periods=3).sum().fillna(0.0).to_numpy(dtype=np.float32)
    vol_6 = pd.Series(log_ret).rolling(6, min_periods=3).apply(lambda values: float(np.sqrt(np.mean(np.square(values)))), raw=True).fillna(0.0).to_numpy(dtype=np.float32)
    vol_12 = pd.Series(log_ret).rolling(12, min_periods=6).apply(lambda values: float(np.sqrt(np.mean(np.square(values)))), raw=True).fillna(0.0).to_numpy(dtype=np.float32)
    vol_ratio = np.where(vol_12 > 1e-8, vol_6 / np.maximum(vol_12, 1e-8), 0.0).astype(np.float32, copy=False)
    tick_mean = pd.Series(tick).rolling(12, min_periods=3).mean().fillna(0.0).to_numpy(dtype=np.float32)
    tick_std = pd.Series(tick).rolling(12, min_periods=3).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
    tick_shock = np.where(tick_std > 1e-6, (tick - tick_mean) / np.maximum(tick_std, 1e-6), 0.0).astype(np.float32, copy=False)
    spread_cost = np.asarray(
        [estimate_spread_cost(symbol, float(raw_spread), float(price)) for raw_spread, price in zip(spread, close, strict=False)],
        dtype=np.float32,
    )
    vol_scale = np.maximum(vol_6, vol_12).astype(np.float32, copy=False)
    # Keep early-session spreads punitive when volatility has not fully formed yet.
    vol_scale = np.maximum(vol_scale, 5e-5)
    spread_stress = ((spread_cost / np.maximum(close, 1e-6)) / vol_scale).astype(np.float32, copy=False)

    direction_side = np.where(label > 0.5, 1.0, -1.0).astype(np.float32, copy=False)
    momentum_aligned = direction_side * (0.65 * momentum_3 + 0.35 * momentum_6) / np.maximum(vol_12, 1e-6)
    breakout_aligned = direction_side * breakout_signed
    vol_center = np.exp(-np.abs(np.log(np.clip(vol_ratio, 1e-3, 10.0))))
    spread_quality = np.clip(1.0 - (spread_stress / 5.0), 0.0, 1.0)
    tick_quality = np.clip(0.5 + (0.25 * tick_shock), 0.0, 1.0)
    momentum_quality = np.clip((momentum_aligned + 0.05) / 0.55, 0.0, 1.0)
    breakout_quality = np.clip((breakout_aligned + 0.10) / 0.60, 0.0, 1.0)
    entry_score = (
        (0.40 * momentum_quality)
        + (0.25 * breakout_quality)
        + (0.20 * vol_center.astype(np.float32, copy=False))
        + (0.10 * spread_quality)
        + (0.05 * tick_quality)
    ).astype(np.float32, copy=False)
    entry = (
        valid
        & real_mask
        & trade_session
        & open_ready
        & (spread_quality >= 0.20)
        & (entry_score >= 0.48)
    )
    return entry.astype(np.float32, copy=False)


def load_cooperative_real_dataset(
    data_root: str,
    start: str | None = None,
    end: str | None = None,
    io_workers: int | None = None,
    max_base_bars: int | None = None,
    timeframes: tuple[str, ...] = TIMEFRAMES_FROM_M5,
    opening_range_bars: int = 3,
) -> CooperativeRealDataset:
    root = Path(data_root)
    base_frames = _load_timeframe_frames(root, "M5", ALL_INSTRUMENTS, start=start, end=end, io_workers=io_workers)
    if not base_frames:
        raise ValueError(f"No M5 data found under {data_root}")

    base_index = pd.Index(sorted({ts for frame in base_frames.values() for ts in frame.index}), name="dt")
    if max_base_bars is not None and len(base_index) > max_base_bars:
        base_index = pd.DatetimeIndex(base_index[-max_base_bars:])
    base_timestamps = base_index.astype("datetime64[ns]").to_numpy()
    session_codes = encode_session_codes(base_timestamps)
    quarter_ids = pd.DatetimeIndex(base_timestamps).to_period("Q").astype(str).to_numpy(dtype=object)
    unique_quarters = sorted(pd.unique(quarter_ids).tolist())
    outer_holdout_quarters = tuple(unique_quarters[-2:]) if len(unique_quarters) >= 2 else tuple(unique_quarters)

    timeframe_frames: dict[str, dict[str, pd.DataFrame]] = {"M5": base_frames}
    tf_session_codes: dict[str, np.ndarray] = {}
    tf_indices: dict[str, np.ndarray] = {}
    btc_tensors: dict[str, np.ndarray] = {}
    fx_tensors: dict[str, np.ndarray] = {}

    base_close_by_symbol: dict[str, np.ndarray] = {}
    base_valid_by_symbol: dict[str, np.ndarray] = {}

    for timeframe in timeframes:
        raw_frames = base_frames if timeframe == "M5" else _load_timeframe_frames(root, timeframe, ALL_INSTRUMENTS, start=start, end=end, io_workers=io_workers)
        timeframe_frames[timeframe] = raw_frames
        all_timestamps = sorted({ts for frame in raw_frames.values() for ts in frame.index})
        if not all_timestamps:
            raise ValueError(f"No data found for timeframe {timeframe} under {data_root}")
        tf_index = pd.DatetimeIndex(all_timestamps, name="dt")
        if timeframe == "M5":
            tf_index = pd.DatetimeIndex(base_index)
        tf_indices[timeframe] = tf_index.astype("datetime64[ns]").to_numpy()
        tf_session = encode_session_codes(tf_index.astype("datetime64[ns]").to_numpy())
        tf_session_codes[timeframe] = tf_session

        btc_matrix = np.zeros((len(tf_index), len(SUBNET_24x7), len(FEATURE_ORDER)), dtype=np.float32)
        fx_matrix = np.zeros((len(tf_index), len(SUBNET_24x5), len(FEATURE_ORDER)), dtype=np.float32)

        for node_idx, symbol in enumerate(SUBNET_24x7):
            aligned = _align_frame(raw_frames.get(symbol), tf_index)
            btc_matrix[:, node_idx, :] = _frame_to_feature_matrix(aligned, tf_session)
            if timeframe == "M5":
                base_close_by_symbol[symbol] = aligned["c"].fillna(0.0).to_numpy(dtype=np.float32)
                base_valid_by_symbol[symbol] = aligned["real"].to_numpy(dtype=np.bool_)

        for node_idx, symbol in enumerate(SUBNET_24x5):
            aligned = _align_frame(raw_frames.get(symbol), tf_index)
            fx_matrix[:, node_idx, :] = _frame_to_feature_matrix(aligned, tf_session)
            if timeframe == "M5":
                base_close_by_symbol[symbol] = aligned["c"].fillna(0.0).to_numpy(dtype=np.float32)
                base_valid_by_symbol[symbol] = aligned["real"].to_numpy(dtype=np.bool_)

        btc_tensors[timeframe] = btc_matrix
        fx_tensors[timeframe] = fx_matrix

    tf_index_for_base: dict[str, np.ndarray] = {}
    base_dt_index = pd.Index(base_timestamps)
    for timeframe in timeframes:
        tf_dt_index = pd.Index(tf_indices[timeframe])
        positions = tf_dt_index.searchsorted(base_dt_index, side="right") - 1
        positions = np.clip(positions, 0, max(len(tf_dt_index) - 1, 0)).astype(np.int32)
        tf_index_for_base[timeframe] = positions

    n_base = len(base_timestamps)
    btc_labels = np.zeros((n_base, len(SUBNET_24x7)), dtype=np.float32)
    btc_entry_labels = np.zeros((n_base, len(SUBNET_24x7)), dtype=np.float32)
    btc_valid = np.zeros((n_base, len(SUBNET_24x7)), dtype=np.bool_)
    btc_forward_returns = np.zeros((n_base, len(SUBNET_24x7)), dtype=np.float32)
    fx_labels = np.zeros((n_base, len(SUBNET_24x5)), dtype=np.float32)
    fx_entry_labels = np.zeros((n_base, len(SUBNET_24x5)), dtype=np.float32)
    fx_valid = np.zeros((n_base, len(SUBNET_24x5)), dtype=np.bool_)
    fx_forward_returns = np.zeros((n_base, len(SUBNET_24x5)), dtype=np.float32)

    btc_close = btc_tensors["M5"][:, 0, FEATURE_ORDER.index("c")]
    btc_high = btc_tensors["M5"][:, 0, FEATURE_ORDER.index("h")]
    btc_low = btc_tensors["M5"][:, 0, FEATURE_ORDER.index("l")]
    btc_spread = btc_tensors["M5"][:, 0, FEATURE_ORDER.index("sp")]
    btc_real = btc_tensors["M5"][:, 0, VALIDITY_IDX] > 0.5
    labels, valid = make_triple_barrier_labels(btc_close, btc_high, btc_low, btc_spread, "BTCUSD", binary=True)
    btc_labels[:, 0] = labels.astype(np.float32, copy=False)
    btc_valid[:, 0] = valid & btc_real
    btc_entry_labels[:, 0] = btc_labels[:, 0] * btc_valid[:, 0].astype(np.float32, copy=False)
    safe_close = np.maximum(btc_close.astype(np.float64), 1e-10)
    btc_forward_returns[:-1, 0] = np.log(safe_close[1:] / safe_close[:-1]).astype(np.float32)

    for node_idx, symbol in enumerate(SUBNET_24x5):
        close = fx_tensors["M5"][:, node_idx, FEATURE_ORDER.index("c")]
        high = fx_tensors["M5"][:, node_idx, FEATURE_ORDER.index("h")]
        low = fx_tensors["M5"][:, node_idx, FEATURE_ORDER.index("l")]
        spread = fx_tensors["M5"][:, node_idx, FEATURE_ORDER.index("sp")]
        tick = fx_tensors["M5"][:, node_idx, FEATURE_ORDER.index("tk")]
        real = fx_tensors["M5"][:, node_idx, VALIDITY_IDX] > 0.5
        safe_close = np.maximum(close.astype(np.float64), 1e-10)
        fx_forward_returns[:-1, node_idx] = np.log(safe_close[1:] / safe_close[:-1]).astype(np.float32)
        if symbol not in SUBNET_24x5_TRADEABLE:
            continue
        labels, valid = make_triple_barrier_labels(close, high, low, spread, symbol, binary=True)
        fx_labels[:, node_idx] = labels.astype(np.float32, copy=False)
        fx_valid[:, node_idx] = valid & real
        fx_entry_labels[:, node_idx] = _build_scalp_entry_labels(
            symbol,
            close,
            high,
            low,
            spread,
            tick,
            real,
            session_codes,
            fx_labels[:, node_idx],
            fx_valid[:, node_idx],
            opening_range_bars=opening_range_bars,
        )

    regime_codes = _compute_global_regime_codes(base_close_by_symbol, base_valid_by_symbol)

    return CooperativeRealDataset(
        timeframes=timeframes,
        base_timestamps=base_timestamps,
        session_codes=session_codes,
        quarter_ids=quarter_ids,
        outer_holdout_quarters=outer_holdout_quarters,
        tf_index_for_base=tf_index_for_base,
        tf_session_codes=tf_session_codes,
        btc_tensors=btc_tensors,
        fx_tensors=fx_tensors,
        btc_labels=btc_labels,
        btc_entry_labels=btc_entry_labels,
        btc_valid=btc_valid,
        btc_forward_returns=btc_forward_returns,
        fx_labels=fx_labels,
        fx_entry_labels=fx_entry_labels,
        fx_valid=fx_valid,
        fx_forward_returns=fx_forward_returns,
        regime_codes=regime_codes,
        btc_node_names=tuple(SUBNET_24x7),
        fx_node_names=tuple(SUBNET_24x5),
        fx_tradable_node_names=tuple(SUBNET_24x5_TRADEABLE),
    )


def summarize_dataset_coverage(dataset: CooperativeRealDataset) -> dict:
    tradeable_indices = [dataset.fx_node_names.index(symbol) for symbol in dataset.fx_tradable_node_names]
    summary = {
        "base_bars": int(len(dataset.base_timestamps)),
        "btc_label_valid_ratio": float(dataset.btc_valid.mean()),
        "btc_entry_positive_ratio": float(dataset.btc_entry_labels.mean()),
        "fx_label_valid_ratio_all": float(dataset.fx_valid.mean()),
        "fx_label_valid_ratio_tradable": float(dataset.fx_valid[:, tradeable_indices].mean()),
        "fx_entry_positive_ratio_all": float(dataset.fx_entry_labels.mean()),
        "fx_entry_positive_ratio_tradable": float(dataset.fx_entry_labels[:, tradeable_indices].mean()),
        "timeframes": {},
    }
    for timeframe in dataset.timeframes:
        btc_tensor = dataset.btc_tensors[timeframe]
        fx_tensor = dataset.fx_tensors[timeframe]
        btc_valid = btc_tensor[:, :, VALIDITY_IDX] > 0.5
        fx_valid = fx_tensor[:, :, VALIDITY_IDX] > 0.5
        summary["timeframes"][timeframe] = {
            "btc_valid_ratio": float(btc_valid.mean()),
            "fx_valid_ratio_all": float(fx_valid.mean()),
            "fx_valid_ratio_tradable": float(fx_valid[:, tradeable_indices].mean()),
            "btc_feature_nonfinite": int((~np.isfinite(btc_tensor)).sum()),
            "fx_feature_nonfinite": int((~np.isfinite(fx_tensor)).sum()),
        }
    return summary


class CooperativeSequenceDataset(Dataset):
    def __init__(
        self,
        dataset: CooperativeRealDataset,
        base_indices: np.ndarray,
        seq_lens: dict[str, int],
    ):
        self.dataset = dataset
        self.base_indices = np.asarray(base_indices, dtype=np.int32)
        self.seq_lens = {tf: int(seq_lens[tf]) for tf in dataset.timeframes}

    def __len__(self) -> int:
        return len(self.base_indices)

    def _slice_subnet(
        self,
        subnet_name: str,
        tensors: dict[str, np.ndarray],
        labels: np.ndarray,
        entry_labels: np.ndarray,
        valid_labels: np.ndarray,
        base_idx: int,
    ):
        sample = {"subnet_name": subnet_name, "timeframe_batches": {}, "base_index": base_idx}
        for timeframe in self.dataset.timeframes:
            tensor = tensors[timeframe]
            end = int(self.dataset.tf_index_for_base[timeframe][base_idx])
            seq_len = self.seq_lens[timeframe]
            start = max(0, end + 1 - seq_len)
            chunk = tensor[start: end + 1]
            if len(chunk) < seq_len:
                pad = np.zeros((seq_len - len(chunk), tensor.shape[1], tensor.shape[2]), dtype=np.float32)
                chunk = np.concatenate([pad, chunk], axis=0)
            valid_mask = chunk[:, :, VALIDITY_IDX] > 0.5
            session_chunk = self.dataset.tf_session_codes[timeframe][max(0, end + 1 - seq_len): end + 1]
            if len(session_chunk) < seq_len:
                session_pad = np.full(seq_len - len(session_chunk), SESSION_CODES["closed"], dtype=np.int8)
                session_chunk = np.concatenate([session_pad, session_chunk], axis=0)
            overlap = (session_chunk == SESSION_CODES["overlap"])[:, None] & valid_mask
            market_open = (session_chunk != SESSION_CODES["closed"])[:, None] & valid_mask
            sample["timeframe_batches"][timeframe] = {
                "timeframe": timeframe,
                "node_features": chunk,
                "valid_mask": valid_mask,
                "market_open_mask": market_open,
                "overlap_mask": overlap,
                "session_codes": session_chunk,
                "target_direction": labels[base_idx].copy(),
                "target_entry": entry_labels[base_idx].copy(),
                "target_valid": valid_labels[base_idx].copy(),
            }
        return sample

    def __getitem__(self, idx: int):
        base_idx = int(self.base_indices[idx])
        btc = self._slice_subnet("btc", self.dataset.btc_tensors, self.dataset.btc_labels, self.dataset.btc_entry_labels, self.dataset.btc_valid, base_idx)
        fx = self._slice_subnet("fx", self.dataset.fx_tensors, self.dataset.fx_labels, self.dataset.fx_entry_labels, self.dataset.fx_valid, base_idx)
        return {"base_index": base_idx, "btc": btc, "fx": fx}


def collate_cooperative_batches(batch):
    if torch is None:
        raise RuntimeError("PyTorch is required to collate cooperative batches")

    def _build_subnet(subnet_key: str, node_names: tuple[str, ...], tradable_node_names: tuple[str, ...]):
        timeframe_batches = {}
        base_indices = torch.tensor([item["base_index"] for item in batch], dtype=torch.long)
        for timeframe in batch[0][subnet_key]["timeframe_batches"]:
            tf_items = [item[subnet_key]["timeframe_batches"][timeframe] for item in batch]
            n_nodes = tf_items[0]["node_features"].shape[1]
            edge_eye = torch.eye(n_nodes, dtype=torch.float32)
            edge_full = torch.full((n_nodes, n_nodes), 0.2, dtype=torch.float32)
            edge_full.fill_diagonal_(1.0)
            edge_chain = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
            for idx in range(n_nodes):
                edge_chain[idx, idx] = 1.0
                if idx + 1 < n_nodes:
                    edge_chain[idx, idx + 1] = 1.0
                    edge_chain[idx + 1, idx] = 1.0
            timeframe_batches[timeframe] = {
                "timeframe": timeframe,
                "node_names": node_names,
                "node_features": torch.stack([torch.from_numpy(it["node_features"]) for it in tf_items], dim=0),
                "edge_matrices": {
                    "fundamental": edge_eye,
                    "rolling_corr": edge_full,
                    "session": edge_chain,
                },
                "valid_mask": torch.stack([torch.from_numpy(it["valid_mask"]) for it in tf_items], dim=0),
                "market_open_mask": torch.stack([torch.from_numpy(it["market_open_mask"]) for it in tf_items], dim=0),
                "overlap_mask": torch.stack([torch.from_numpy(it["overlap_mask"]) for it in tf_items], dim=0),
                "session_codes": torch.stack([torch.from_numpy(it["session_codes"]) for it in tf_items], dim=0),
                "target_direction": torch.stack([torch.from_numpy(it["target_direction"]) for it in tf_items], dim=0),
                "target_entry": torch.stack([torch.from_numpy(it["target_entry"]) for it in tf_items], dim=0),
                "target_valid": torch.stack([torch.from_numpy(it["target_valid"]) for it in tf_items], dim=0),
            }
        return timeframe_batches, base_indices

    btc_tf, btc_base = _build_subnet("btc", tuple(SUBNET_24x7), tuple(SUBNET_24x7))
    fx_tf, fx_base = _build_subnet("fx", tuple(SUBNET_24x5), tuple(SUBNET_24x5_TRADEABLE))

    from .contracts import SubnetBatch, TimeframeBatch

    btc_batch = SubnetBatch(
        subnet_name="btc",
        timeframe_batches={tf: TimeframeBatch(**payload) for tf, payload in btc_tf.items()},
        node_names=tuple(SUBNET_24x7),
        tradable_node_names=tuple(SUBNET_24x7),
        base_indices=btc_base,
    )
    fx_batch = SubnetBatch(
        subnet_name="fx",
        timeframe_batches={tf: TimeframeBatch(**payload) for tf, payload in fx_tf.items()},
        node_names=tuple(SUBNET_24x5),
        tradable_node_names=tuple(SUBNET_24x5_TRADEABLE),
        base_indices=fx_base,
    )
    return btc_batch, fx_batch
