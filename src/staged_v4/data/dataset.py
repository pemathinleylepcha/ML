from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataextractor_contract import resolve_candle_path
from research_dataset import _read_candles_csv, encode_session_codes
from staged_v4.config import ALL_TIMEFRAMES, BTC_NODE_NAMES, FX_NODE_NAMES, LOWER_TIMEFRAME, TIMEFRAME_FREQ, TPO_SOURCE_TIMEFRAME
from staged_v4.utils.runtime_logging import enforce_memory_guard, guard_worker_budget, log_progress


_CORE_COLS = ["o", "h", "l", "c", "sp", "tk", "real"]
_TICK_FILE_MAP_CACHE: dict[str, dict[str, Path]] = {}
_CHUNK_ROWS = 250_000


@dataclass(slots=True)
class StagedPanels:
    subnet_name: str
    symbols: tuple[str, ...]
    anchor_timeframe: str
    panels: dict[str, dict[str, pd.DataFrame]]
    anchor_timestamps: np.ndarray
    anchor_lookup: dict[str, np.ndarray]
    walkforward_splits: list[dict[str, object]]
    split_frequency: str
    tpo_panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]


def _read_tick_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    rename = {
        "datetime": "dt",
        "bar_time": "dt",
        "time": "dt",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "spread": "sp",
        "tick_volume": "tk",
        "volume": "tk",
    }
    df = df.rename(columns=rename)
    if "dt" not in df.columns:
        raise ValueError(f"Missing dt column in tick frame {path}")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=False)
    df = df.dropna(subset=["dt"])
    for col in ("o", "h", "l", "c"):
        if col not in df.columns:
            raise ValueError(f"Missing price column {col} in tick frame {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("sp", "tk"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[["dt", "o", "h", "l", "c", "sp", "tk"]].sort_values("dt").drop_duplicates("dt")
    df["real"] = True
    return df.set_index("dt")


def _read_bar_chunk_csv(path: Path, chunksize: int = _CHUNK_ROWS):
    usecols = ["dt", "datetime", "bar_time", "time", "o", "h", "l", "c", "sp", "spread", "tk", "tick_volume", "volume"]
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, engine="python", on_bad_lines="skip"):
        rename = {
            "datetime": "dt",
            "bar_time": "dt",
            "time": "dt",
            "spread": "sp",
            "tick_volume": "tk",
            "volume": "tk",
        }
        chunk = chunk.rename(columns=rename)
        if "dt" not in chunk.columns:
            raise ValueError(f"Missing dt column in tick frame {path}")
        chunk["dt"] = pd.to_datetime(chunk["dt"], errors="coerce", utc=False)
        chunk = chunk.dropna(subset=["dt"])
        for col in ("o", "h", "l", "c"):
            if col not in chunk.columns:
                raise ValueError(f"Missing price column {col} in tick frame {path}")
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        for col in ("sp", "tk"):
            if col not in chunk.columns:
                chunk[col] = 0.0
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        chunk = chunk.dropna(subset=["o", "h", "l", "c"])
        if len(chunk) == 0:
            continue
        chunk["real"] = True
        yield chunk[["dt", "o", "h", "l", "c", "sp", "tk", "real"]].copy()


def _get_tick_file_map(tick_root: Path) -> dict[str, Path]:
    root_key = str(tick_root.resolve())
    if root_key not in _TICK_FILE_MAP_CACHE:
        mapping: dict[str, Path] = {}
        for path in tick_root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if not (lower.endswith(".parquet") or lower.endswith(".csv")):
                continue
            stem = path.stem
            if stem.endswith("_1000ms"):
                symbol_name = stem[: -len("_1000ms")]
            else:
                symbol_name = stem.split("_")[0]
            mapping.setdefault(symbol_name.upper(), path)
        _TICK_FILE_MAP_CACHE[root_key] = mapping
    return _TICK_FILE_MAP_CACHE[root_key]


def _find_tick_file(tick_root: Path, symbol: str, tick_file_map: dict[str, Path] | None = None) -> Path | None:
    mapping = tick_file_map if tick_file_map is not None else _get_tick_file_map(tick_root)
    direct = mapping.get(symbol.upper())
    if direct is not None:
        return direct
    patterns = (
        f"{symbol}_1000ms.parquet",
        f"{symbol}_1000ms.csv",
        f"{symbol}.parquet",
        f"{symbol}.csv",
    )
    for pattern in patterns:
        match = next(iter(tick_root.rglob(pattern)), None)
        if match is not None:
            return match
    return None


def _resample_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
    if frame is None or len(frame) == 0:
        return None
    agg = frame.resample(TIMEFRAME_FREQ[timeframe], label="right", closed="right").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "sp": "mean", "tk": "sum", "real": "max"}
    )
    agg = agg.dropna(subset=["o", "h", "l", "c"])
    if len(agg) == 0:
        return None
    agg["sp"] = agg["sp"].fillna(0.0)
    agg["tk"] = agg["tk"].fillna(0.0)
    agg["real"] = agg["real"].fillna(False).astype(bool)
    return agg[_CORE_COLS].copy()


def _derived_timeframe_path(source_path: Path, timeframe: str) -> Path:
    stem = source_path.stem
    if stem.endswith("_1000ms"):
        stem = stem[: -len("_1000ms")]
    return source_path.parent / "_derived" / timeframe / f"{stem}_{timeframe}.csv"


def _aggregate_bar_rows(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=_CORE_COLS, index=pd.Index([], name="dt"))
    bucket_end = frame["dt"].dt.ceil(TIMEFRAME_FREQ[timeframe])
    grouped = frame.groupby(bucket_end, sort=True, observed=False)
    agg = grouped.agg(
        {
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "sp": "mean",
            "tk": "sum",
            "real": "max",
        }
    )
    agg.index.name = "dt"
    agg["sp"] = agg["sp"].fillna(0.0)
    agg["tk"] = agg["tk"].fillna(0.0)
    agg["real"] = agg["real"].fillna(False).astype(bool)
    return agg[_CORE_COLS].sort_index()


def _stream_resample_tick_source(
    source_path: Path,
    timeframe: str,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    logger=None,
) -> pd.DataFrame | None:
    checkpoint_path = _derived_timeframe_path(source_path, timeframe)
    if checkpoint_path.exists():
        if logger is not None:
            logger.info("timeframe=%s source=%s checkpoint=hit path=%s", timeframe, source_path.name, checkpoint_path)
        df = _read_tick_frame(checkpoint_path)
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index <= end_ts]
        return df[_CORE_COLS].copy() if len(df) else None

    if logger is not None:
        logger.info("timeframe=%s source=%s checkpoint=build path=%s", timeframe, source_path.name, checkpoint_path)
    carry: pd.DataFrame | None = None
    parts: list[pd.DataFrame] = []
    for chunk in _read_bar_chunk_csv(source_path):
        if start_ts is not None:
            chunk = chunk[chunk["dt"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["dt"] <= end_ts]
        if len(chunk) == 0:
            continue
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        bucket_end = chunk["dt"].dt.ceil(TIMEFRAME_FREQ[timeframe])
        if len(bucket_end) == 0:
            continue
        last_bucket = bucket_end.iloc[-1]
        complete = chunk[bucket_end != last_bucket]
        carry = chunk[bucket_end == last_bucket].copy()
        if len(complete):
            parts.append(_aggregate_bar_rows(complete, timeframe))
    if carry is not None and len(carry):
        parts.append(_aggregate_bar_rows(carry, timeframe))
    if not parts:
        return None
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(checkpoint_path, index=False)
    return out[_CORE_COLS].copy()


def _load_symbol_timeframe(
    candle_root: Path,
    tick_root: Path | None,
    symbol: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    lower_frame: pd.DataFrame | None = None,
    lower_timeframe: str | None = None,
    relevant_quarters: list[Path] | None = None,
    tick_file_map: dict[str, Path] | None = None,
    logger=None,
) -> pd.DataFrame | None:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    df: pd.DataFrame | None = None
    if timeframe == "tick":
        if tick_root is None:
            return None
        tick_path = _find_tick_file(tick_root, symbol, tick_file_map=tick_file_map)
        if tick_path is None:
            return None
        df = _read_tick_frame(tick_path)
    else:
        chunks: list[pd.DataFrame] = []
        quarter_dirs = relevant_quarters if relevant_quarters is not None else _iter_relevant_quarters(candle_root, start_ts, end_ts)
        for quarter_dir in quarter_dirs:
            csv_path, _ = resolve_candle_path(quarter_dir, symbol, timeframe)
            if csv_path is None:
                continue
            frame = _read_candles_csv(csv_path).set_index("dt")
            frame["real"] = True
            chunks.append(frame)
        if chunks:
            df = pd.concat(chunks).sort_index().drop_duplicates()
        elif lower_timeframe == "tick" and tick_root is not None:
            tick_path = _find_tick_file(tick_root, symbol, tick_file_map=tick_file_map)
            if tick_path is not None:
                df = _stream_resample_tick_source(tick_path, timeframe, start_ts=start_ts, end_ts=end_ts, logger=logger)
                if df is None:
                    return None
            elif lower_frame is not None:
                df = _resample_frame(lower_frame, timeframe)
                if df is None:
                    return None
            else:
                return None
        elif lower_frame is not None:
            df = _resample_frame(lower_frame, timeframe)
            if df is None:
                return None
        else:
            return None
    if df is None:
        return None
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    if len(df) == 0:
        return None
    return df[_CORE_COLS].copy()


def _iter_relevant_quarters(candle_root: Path, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> list[Path]:
    years = sorted(p for p in candle_root.iterdir() if p.is_dir() and p.name.isdigit())
    if start_ts is None and end_ts is None:
        return [quarter for year_dir in years for quarter in sorted(q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q"))]

    start_year = start_ts.year if start_ts is not None else int(years[0].name)
    end_year = end_ts.year if end_ts is not None else int(years[-1].name)
    relevant: list[Path] = []
    for year_dir in years:
        year = int(year_dir.name)
        if year < start_year or year > end_year:
            continue
        for quarter_dir in sorted(q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q")):
            quarter_num = int(quarter_dir.name[1:])
            quarter_start_month = 1 + (quarter_num - 1) * 3
            quarter_start = pd.Timestamp(year=year, month=quarter_start_month, day=1)
            quarter_end = quarter_start + pd.offsets.QuarterEnd()
            if start_ts is not None and quarter_end < start_ts:
                continue
            if end_ts is not None and quarter_start > end_ts:
                continue
            relevant.append(quarter_dir)
    return relevant


def _align_frames(frames: dict[str, pd.DataFrame], expected_symbols: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    if not frames:
        return {}
    union_index = pd.Index(sorted({ts for frame in frames.values() for ts in frame.index}), name="dt")
    aligned: dict[str, pd.DataFrame] = {}
    symbols = expected_symbols or tuple(frames.keys())
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None:
            out = pd.DataFrame(index=union_index, columns=_CORE_COLS, dtype=np.float32)
            out["sp"] = 0.0
            out["tk"] = 0.0
            out["real"] = False
            aligned[symbol] = out
            continue
        out = frame.reindex(union_index)
        real_mask = out["c"].notna()
        out["sp"] = out["sp"].fillna(0.0)
        out["tk"] = out["tk"].fillna(0.0)
        out["real"] = real_mask.fillna(False).astype(bool)
        aligned[symbol] = out
    return aligned


def _presweep_tpo(
    panels: dict[str, dict[str, pd.DataFrame]],
    symbols: tuple[str, ...],
    requested_timeframes: tuple[str, ...],
    logger=None,
    max_workers: int = 0,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    from staged_v4.data.tpo_features import compute_tpo_feature_panel

    source_timeframes: set[str] = set()
    for timeframe in requested_timeframes:
        source_timeframe = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
        if source_timeframe in panels and panels[source_timeframe]:
            source_timeframes.add(source_timeframe)

    tpo_panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    cpu_total = os.cpu_count() or 8
    worker_count = max_workers if max_workers and max_workers > 0 else min(24, max(4, cpu_total - 4))
    for source_timeframe in sorted(source_timeframes):
        symbol_frames = panels[source_timeframe]
        tpo_panels[source_timeframe] = {}
        effective_workers = max(1, min(worker_count, len(symbols)))

        def _one(symbol: str) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
            frame = symbol_frames.get(symbol)
            if frame is None:
                return symbol, None
            high = frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
            low = frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
            close = frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
            return symbol, compute_tpo_feature_panel(high, low, close)

        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix=f"tpo_{source_timeframe}") as executor:
            futures = [executor.submit(_one, symbol) for symbol in symbols]
            for future in as_completed(futures):
                symbol, result = future.result()
                if result is not None:
                    tpo_panels[source_timeframe][symbol] = result
        if logger is not None:
            logger.info(
                "state=tpo_presweep source_tf=%s symbols=%d workers=%d",
                source_timeframe,
                len(tpo_panels[source_timeframe]),
                effective_workers,
            )
    return tpo_panels


def _build_anchor_lookup(anchor_timestamps: np.ndarray, tf_timestamps: pd.Index) -> np.ndarray:
    anchor_index = pd.Index(anchor_timestamps)
    positions = tf_timestamps.searchsorted(anchor_index, side="right") - 1
    positions = np.maximum(positions, 0).astype(np.int32)
    return positions


def _period_labels(anchor_index: pd.DatetimeIndex, split_frequency: str) -> np.ndarray:
    if split_frequency == "month":
        return anchor_index.to_period("M").astype(str)
    if split_frequency == "week":
        return anchor_index.to_period("W-SUN").astype(str)
    raise ValueError(f"Unsupported split_frequency: {split_frequency}")


def build_walkforward_splits(
    anchor_timestamps: np.ndarray,
    split_frequency: str = "week",
    outer_holdout_blocks: int = 1,
    min_train_blocks: int = 2,
    purge_bars: int = 6,
) -> list[dict[str, object]]:
    anchor_index = pd.DatetimeIndex(anchor_timestamps)
    period_labels = _period_labels(anchor_index, split_frequency)
    unique_blocks = pd.Index(period_labels).unique().tolist()
    if len(unique_blocks) <= outer_holdout_blocks:
        return []
    in_sample_blocks = unique_blocks if outer_holdout_blocks == 0 else unique_blocks[:-outer_holdout_blocks]
    splits: list[dict[str, object]] = []
    for split_end in range(min_train_blocks, len(in_sample_blocks)):
        train_blocks = in_sample_blocks[:split_end]
        val_block = in_sample_blocks[split_end]
        train_mask = np.isin(period_labels, train_blocks)
        val_mask = period_labels == val_block
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        if purge_bars > 0:
            train_idx = train_idx[train_idx < max(0, val_idx[0] - purge_bars)]
        if len(train_idx) == 0:
            continue
        splits.append(
            {
                "fold": len(splits),
                "split_frequency": split_frequency,
                "train_blocks": list(train_blocks),
                "val_block": val_block,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )
    return splits


def _load_panels_for_symbols(
    candle_root: Path,
    tick_root: Path | None,
    symbols: tuple[str, ...],
    start: str | None,
    end: str | None,
    strict: bool,
    requested_timeframes: tuple[str, ...] | None = None,
    logger=None,
    status_file: str | None = None,
    subnet_name: str = "subnet",
    relevant_quarters: list[Path] | None = None,
    tick_file_map: dict[str, Path] | None = None,
    max_workers: int = 0,
    memory_guard_min_available_mb: float = 0.0,
    memory_guard_critical_available_mb: float = 0.0,
) -> dict[str, dict[str, pd.DataFrame]]:
    active_timeframes = tuple(requested_timeframes or ALL_TIMEFRAMES)
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    symbol_cache: dict[str, dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}
    cpu_total = os.cpu_count() or 8
    requested_workers = max_workers if max_workers and max_workers > 0 else min(24, max(4, cpu_total - 4))
    for timeframe in active_timeframes:
        worker_count = max(1, min(len(symbols), requested_workers))
        if timeframe == "tick":
            worker_count = 1
        if memory_guard_min_available_mb > 0.0 or memory_guard_critical_available_mb > 0.0:
            guarded_workers, guard = guard_worker_budget(
                worker_count,
                min_available_mb=memory_guard_min_available_mb,
                critical_available_mb=memory_guard_critical_available_mb,
            )
            if logger is not None and guarded_workers != worker_count:
                logger.warning(
                    "subnet=%s timeframe=%s state=memory_guard workers=%d->%d available_mb=%s",
                    subnet_name,
                    timeframe,
                    worker_count,
                    guarded_workers,
                    guard["available_mb"],
                )
            worker_count = guarded_workers
            if logger is not None:
                enforce_memory_guard(
                    logger,
                    status_file,
                    "load_panels",
                    min_available_mb=memory_guard_min_available_mb,
                    critical_available_mb=memory_guard_critical_available_mb,
                    details={"subnet": subnet_name, "timeframe": timeframe, "workers": worker_count},
                )
        if logger is not None:
            logger.info(
                "subnet=%s timeframe=%s state=start symbols=%d workers=%d",
                subnet_name,
                timeframe,
                len(symbols),
                worker_count,
            )
        per_symbol: dict[str, pd.DataFrame] = {}
        lower_tf = LOWER_TIMEFRAME.get(timeframe)
        futures = {}
        if worker_count == 1:
            for idx, symbol in enumerate(symbols, start=1):
                lower_frame = symbol_cache[symbol].get(lower_tf) if lower_tf else None
                if logger is not None:
                    logger.info(
                        "subnet=%s timeframe=%s symbol=%s state=start index=%d/%d",
                        subnet_name,
                        timeframe,
                        symbol,
                        idx,
                        len(symbols),
                    )
                try:
                    frame = _load_symbol_timeframe(
                        candle_root,
                        tick_root,
                        symbol,
                        timeframe,
                        start,
                        end,
                        lower_frame=lower_frame,
                        lower_timeframe=lower_tf,
                        relevant_quarters=relevant_quarters,
                        tick_file_map=tick_file_map,
                        logger=logger,
                    )
                except Exception:
                    if logger is not None:
                        logger.exception(
                            "subnet=%s timeframe=%s symbol=%s state=failed index=%d/%d",
                            subnet_name,
                            timeframe,
                            symbol,
                            idx,
                            len(symbols),
                        )
                    raise
                if frame is None:
                    if logger is not None:
                        logger.warning(
                            "subnet=%s timeframe=%s symbol=%s state=missing index=%d/%d",
                            subnet_name,
                            timeframe,
                            symbol,
                            idx,
                            len(symbols),
                        )
                    if strict:
                        raise FileNotFoundError(f"Missing {timeframe} data for {symbol}")
                    continue
                symbol_cache[symbol][timeframe] = frame
                per_symbol[symbol] = frame
                if logger is not None:
                    logger.info(
                        "subnet=%s timeframe=%s symbol=%s state=done index=%d/%d rows=%d start=%s end=%s",
                        subnet_name,
                        timeframe,
                        symbol,
                        idx,
                        len(symbols),
                        len(frame),
                        frame.index[0] if len(frame) else None,
                        frame.index[-1] if len(frame) else None,
                    )
                    if memory_guard_min_available_mb > 0.0 or memory_guard_critical_available_mb > 0.0:
                        enforce_memory_guard(
                            logger,
                            status_file,
                            "load_panels",
                            min_available_mb=memory_guard_min_available_mb,
                            critical_available_mb=memory_guard_critical_available_mb,
                            details={"subnet": subnet_name, "timeframe": timeframe, "symbol": symbol, "index": idx, "workers": worker_count},
                        )
                if logger is not None:
                    log_progress(logger, status_file, "load_panels", idx, len(symbols), subnet=subnet_name, timeframe=timeframe, valid_symbols=len(per_symbol), workers=worker_count)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{subnet_name}_{timeframe}") as executor:
                for symbol in symbols:
                    if logger is not None:
                        logger.info(
                            "subnet=%s timeframe=%s symbol=%s state=queued workers=%d",
                            subnet_name,
                            timeframe,
                            symbol,
                            worker_count,
                        )
                    lower_frame = symbol_cache[symbol].get(lower_tf) if lower_tf else None
                    future = executor.submit(
                        _load_symbol_timeframe,
                        candle_root,
                        tick_root,
                        symbol,
                        timeframe,
                        start,
                        end,
                        lower_frame,
                        lower_tf,
                        relevant_quarters,
                        tick_file_map,
                        logger,
                    )
                    futures[future] = symbol
                for idx, future in enumerate(as_completed(futures), start=1):
                    symbol = futures[future]
                    try:
                        frame = future.result()
                    except Exception:
                        if logger is not None:
                            logger.exception(
                                "subnet=%s timeframe=%s symbol=%s state=failed index=%d/%d",
                                subnet_name,
                                timeframe,
                                symbol,
                                idx,
                                len(symbols),
                            )
                        raise
                    if frame is None:
                        if logger is not None:
                            logger.warning(
                                "subnet=%s timeframe=%s symbol=%s state=missing index=%d/%d",
                                subnet_name,
                                timeframe,
                                symbol,
                                idx,
                                len(symbols),
                            )
                        if strict:
                            raise FileNotFoundError(f"Missing {timeframe} data for {symbol}")
                        continue
                    symbol_cache[symbol][timeframe] = frame
                    per_symbol[symbol] = frame
                    if logger is not None:
                        logger.info(
                            "subnet=%s timeframe=%s symbol=%s state=done index=%d/%d rows=%d start=%s end=%s",
                            subnet_name,
                            timeframe,
                            symbol,
                            idx,
                            len(symbols),
                            len(frame),
                            frame.index[0] if len(frame) else None,
                            frame.index[-1] if len(frame) else None,
                        )
                        if memory_guard_min_available_mb > 0.0 or memory_guard_critical_available_mb > 0.0:
                            enforce_memory_guard(
                                logger,
                                status_file,
                                "load_panels",
                                min_available_mb=memory_guard_min_available_mb,
                                critical_available_mb=memory_guard_critical_available_mb,
                                details={"subnet": subnet_name, "timeframe": timeframe, "symbol": symbol, "index": idx, "workers": worker_count},
                            )
                    if logger is not None:
                        log_progress(logger, status_file, "load_panels", idx, len(symbols), subnet=subnet_name, timeframe=timeframe, valid_symbols=len(per_symbol), workers=worker_count)
        panels[timeframe] = _align_frames(per_symbol, expected_symbols=symbols)
        if logger is not None:
            aligned_rows = len(next(iter(panels[timeframe].values())).index) if panels[timeframe] else 0
            logger.info("subnet=%s timeframe=%s state=done valid_symbols=%d aligned_rows=%d", subnet_name, timeframe, len(per_symbol), aligned_rows)
    return panels


def load_staged_panels(
    candle_root: str,
    tick_root: str | None = None,
    start: str | None = None,
    end: str | None = None,
    anchor_timeframe: str = "M1",
    strict: bool = True,
    timeframes: tuple[str, ...] | None = None,
    split_frequency: str = "week",
    outer_holdout_blocks: int = 1,
    min_train_blocks: int = 2,
    purge_bars: int = 6,
    logger=None,
    status_file: str | None = None,
    max_workers: int = 0,
    memory_guard_min_available_mb: float = 0.0,
    memory_guard_critical_available_mb: float = 0.0,
) -> tuple[StagedPanels, StagedPanels]:
    candle_path = Path(candle_root)
    tick_path = Path(tick_root) if tick_root else None
    requested_timeframes = tuple(timeframes or ALL_TIMEFRAMES)
    if anchor_timeframe not in requested_timeframes:
        raise ValueError(f"Anchor timeframe {anchor_timeframe} must be included in requested timeframes")
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    relevant_quarters = _iter_relevant_quarters(candle_path, start_ts, end_ts)
    tick_file_map = _get_tick_file_map(tick_path) if tick_path is not None else None
    btc_panels = _load_panels_for_symbols(
        candle_path,
        tick_path,
        BTC_NODE_NAMES,
        start,
        end,
        strict,
        requested_timeframes=requested_timeframes,
        logger=logger,
        status_file=status_file,
        subnet_name="btc",
        relevant_quarters=relevant_quarters,
        tick_file_map=tick_file_map,
        max_workers=max_workers,
        memory_guard_min_available_mb=memory_guard_min_available_mb,
        memory_guard_critical_available_mb=memory_guard_critical_available_mb,
    )
    fx_panels = _load_panels_for_symbols(
        candle_path,
        tick_path,
        FX_NODE_NAMES,
        start,
        end,
        strict,
        requested_timeframes=requested_timeframes,
        logger=logger,
        status_file=status_file,
        subnet_name="fx",
        relevant_quarters=relevant_quarters,
        tick_file_map=tick_file_map,
        max_workers=max_workers,
        memory_guard_min_available_mb=memory_guard_min_available_mb,
        memory_guard_critical_available_mb=memory_guard_critical_available_mb,
    )
    anchor_source = fx_panels[anchor_timeframe] if fx_panels.get(anchor_timeframe) else btc_panels[anchor_timeframe]
    if not anchor_source:
        raise ValueError(f"No anchor timeframe data for {anchor_timeframe}")
    anchor_timestamps = next(iter(anchor_source.values())).index.astype("datetime64[ns]").to_numpy()
    walkforward_splits = build_walkforward_splits(
        anchor_timestamps,
        split_frequency=split_frequency,
        outer_holdout_blocks=outer_holdout_blocks,
        min_train_blocks=min_train_blocks,
        purge_bars=purge_bars,
    )

    def _make(subnet_name: str, symbols: tuple[str, ...], panels: dict[str, dict[str, pd.DataFrame]]) -> StagedPanels:
        lookup = {}
        for timeframe, symbol_map in panels.items():
            if symbol_map:
                tf_index = next(iter(symbol_map.values())).index
                lookup[timeframe] = _build_anchor_lookup(anchor_timestamps, tf_index)
            else:
                lookup[timeframe] = np.zeros(len(anchor_timestamps), dtype=np.int32)
        tpo_panels = _presweep_tpo(panels, symbols, requested_timeframes, logger=logger, max_workers=max_workers)
        return StagedPanels(
            subnet_name=subnet_name,
            symbols=symbols,
            anchor_timeframe=anchor_timeframe,
            panels=panels,
            anchor_timestamps=anchor_timestamps,
            anchor_lookup=lookup,
            walkforward_splits=walkforward_splits,
            split_frequency=split_frequency,
            tpo_panels=tpo_panels,
        )

    return _make("btc", BTC_NODE_NAMES, btc_panels), _make("fx", FX_NODE_NAMES, fx_panels)


def generate_synthetic_panels(
    n_anchor: int = 96,
    anchor_timeframe: str = "M1",
) -> tuple[StagedPanels, StagedPanels]:
    anchor_freq = TIMEFRAME_FREQ[anchor_timeframe]
    anchor_index = pd.date_range("2025-01-01", periods=n_anchor, freq=anchor_freq, name="dt")
    rng = np.random.default_rng(7)

    def _make_symbol_frame(base_price: float) -> pd.DataFrame:
        ret = rng.normal(0.0, 0.0008, size=n_anchor).astype(np.float32)
        close = base_price * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[close[0]], close[:-1]])
        spread = np.full(n_anchor, max(base_price * 1e-4, 1e-5), dtype=np.float32)
        range_size = np.abs(ret) * base_price * 0.8 + spread
        high = np.maximum(open_, close) + range_size
        low = np.minimum(open_, close) - range_size
        tick = rng.integers(20, 200, size=n_anchor).astype(np.float32)
        real = np.ones(n_anchor, dtype=bool)
        return pd.DataFrame(
            {"o": open_, "h": high, "l": low, "c": close, "sp": spread, "tk": tick, "real": real},
            index=anchor_index,
        )

    def _resample(symbol_frames: dict[str, pd.DataFrame], timeframe: str) -> dict[str, pd.DataFrame]:
        if ALL_TIMEFRAMES.index(timeframe) < ALL_TIMEFRAMES.index(anchor_timeframe):
            return {symbol: frame.copy() for symbol, frame in symbol_frames.items()}
        if timeframe == anchor_timeframe:
            return symbol_frames
        freq = TIMEFRAME_FREQ[timeframe]
        out = {}
        for symbol, frame in symbol_frames.items():
            agg = frame.resample(freq, label="right", closed="right").agg(
                {"o": "first", "h": "max", "l": "min", "c": "last", "sp": "mean", "tk": "sum", "real": "max"}
            )
            out[symbol] = agg
        return _align_frames(out)

    btc_anchor = {symbol: _make_symbol_frame(40000.0 + idx * 1000.0) for idx, symbol in enumerate(BTC_NODE_NAMES)}
    fx_anchor = {symbol: _make_symbol_frame(1.0 + idx * 0.05) for idx, symbol in enumerate(FX_NODE_NAMES)}
    btc_panels = {tf: _resample(btc_anchor, tf) for tf in ALL_TIMEFRAMES}
    fx_panels = {tf: _resample(fx_anchor, tf) for tf in ALL_TIMEFRAMES}
    walkforward_splits = build_walkforward_splits(
        anchor_index.astype("datetime64[ns]").to_numpy(),
        split_frequency="week",
        outer_holdout_blocks=1,
        min_train_blocks=1,
        purge_bars=2,
    )

    def _make(subnet_name: str, symbols: tuple[str, ...], panels: dict[str, dict[str, pd.DataFrame]]) -> StagedPanels:
        lookup = {}
        anchor_ts = anchor_index.astype("datetime64[ns]").to_numpy()
        for timeframe, symbol_map in panels.items():
            tf_index = next(iter(symbol_map.values())).index
            lookup[timeframe] = _build_anchor_lookup(anchor_ts, tf_index)
        return StagedPanels(
            subnet_name=subnet_name,
            symbols=symbols,
            anchor_timeframe=anchor_timeframe,
            panels=panels,
            anchor_timestamps=anchor_index.astype("datetime64[ns]").to_numpy(),
            anchor_lookup=lookup,
            walkforward_splits=walkforward_splits,
            split_frequency="week",
            tpo_panels={},
        )

    return _make("btc", BTC_NODE_NAMES, btc_panels), _make("fx", FX_NODE_NAMES, fx_panels)


def build_sequence_dataset(feature_batch, seq_lens: dict[str, int]) -> dict[str, np.ndarray]:
    sequence_indices: dict[str, np.ndarray] = {}
    for timeframe, tf_batch in feature_batch.timeframe_batches.items():
        seq_len = int(seq_lens[timeframe])
        n_steps = tf_batch.timestamps.shape[0]
        if n_steps <= seq_len:
            sequence_indices[timeframe] = np.empty((0, seq_len), dtype=np.int32)
            continue
        windows = []
        for end_idx in range(seq_len - 1, n_steps):
            start_idx = end_idx - seq_len + 1
            windows.append(np.arange(start_idx, end_idx + 1, dtype=np.int32))
        sequence_indices[timeframe] = np.vstack(windows)
    return sequence_indices
