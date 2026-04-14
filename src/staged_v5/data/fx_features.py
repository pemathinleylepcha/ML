from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from math_engine import MathEngine
from research_dataset import SESSION_CODES, BUY_CLASS, SELL_CLASS, encode_session_codes, make_triple_barrier_labels
from staged_v5.config import ALL_TIMEFRAMES, FX_TRADABLE_NAMES, TPO_SOURCE_TIMEFRAME
from staged_v5.contracts import FXFeatureBatch, TimeframeFeatureBatch
from staged_v5.data.dataset import StagedPanels
from staged_v5.data.tpo_features import compute_tpo_feature_panel
from staged_v5.utils.runtime_logging import log_progress, write_status

_MAX_TPO_BARS = 500_000
_DEFAULT_SYMBOL_TIMEOUT_SEC = 300.0
_DEFAULT_BATCH_DEADLINE_SEC = 3600.0
# Lock to serialize compute_tpo_memory_state C extension calls —
# the extension is not thread-safe under concurrent access.
_TPO_LOCK = threading.Lock()


def _lagged_logret_matrix(close: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(close, dtype=np.float32)
    if close.shape[0] <= lag:
        return out
    out[lag:] = (np.log(np.clip(close[lag:], 1e-12, None)) - np.log(np.clip(close[:-lag], 1e-12, None))).astype(np.float32)
    return out


def _atr_norm_matrix(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1, axis=0)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    csum = np.vstack([np.zeros((1, tr.shape[1]), dtype=np.float64), np.cumsum(tr, axis=0, dtype=np.float64)])
    idx = np.arange(close.shape[0], dtype=np.int64)
    start = np.maximum(0, idx - period + 1)
    count = (idx - start + 1).astype(np.float64)[:, None]
    atr = (csum[idx + 1] - csum[start]) / count
    return (atr / np.maximum(np.abs(close), 1e-8)).astype(np.float32)


def _session_transition(session_codes: np.ndarray) -> np.ndarray:
    out = np.zeros(len(session_codes), dtype=np.float32)
    out[1:] = (session_codes[1:] != session_codes[:-1]).astype(np.float32)
    return out


def _regime_signal(ret_3: np.ndarray, atr_norm: np.ndarray) -> np.ndarray:
    return np.clip(ret_3 / np.maximum(np.abs(atr_norm), 1e-6), -5.0, 5.0).astype(np.float32)


def _compute_laplacian_residuals(close: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    engine = MathEngine(n_pairs=close.shape[1], rolling_window=60)
    residuals = np.zeros_like(close, dtype=np.float32)
    log_returns = np.zeros_like(close, dtype=np.float64)
    log_returns[1:] = np.log(np.clip(close[1:], 1e-12, None)) - np.log(np.clip(close[:-1], 1e-12, None))
    for idx in range(close.shape[0]):
        returns = log_returns[idx].copy()
        returns[~valid_mask[idx]] = 0.0
        state = engine.update(returns)
        if state.valid:
            residuals[idx] = state.residuals.astype(np.float32)
    return residuals


def _lookup_to_source_index(source_index: pd.Index, target_index: pd.Index) -> np.ndarray:
    lookup = source_index.searchsorted(target_index, side="right") - 1
    return np.maximum(lookup, 0).astype(np.int32)


def _symbol_shard_path(shard_root: Path, symbol: str) -> Path:
    return shard_root / f"{symbol}.npz"


def _save_symbol_aux_shard(
    shard_path: Path,
    node_tpo: np.ndarray,
    node_vol: np.ndarray,
    direction: np.ndarray,
    entry: np.ndarray,
    label_valid: np.ndarray,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_path,
        node_tpo=node_tpo.astype(np.float32),
        node_vol=node_vol.astype(np.float32),
        direction=direction.astype(np.int32),
        entry=entry.astype(np.int32),
        label_valid=label_valid.astype(np.bool_),
    )


def _load_symbol_aux_shard(shard_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(shard_path, allow_pickle=False)
    return (
        payload["node_tpo"].astype(np.float32),
        payload["node_vol"].astype(np.float32),
        payload["direction"].astype(np.int32),
        payload["entry"].astype(np.int32),
        payload["label_valid"].astype(np.bool_),
    )


def _build_symbol_aux_features(
    symbol: str,
    timeframe: str,
    timestamps: pd.Index,
    high_col: np.ndarray,
    low_col: np.ndarray,
    close_col: np.ndarray,
    spread_col: np.ndarray,
    tradable: bool,
    valid_col: np.ndarray,
    tpo_source_frame: pd.DataFrame | None = None,
    tpo_lookup: np.ndarray | None = None,
    label_source_frame: pd.DataFrame | None = None,
    label_lookup: np.ndarray | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    total_started = time.perf_counter()
    node_tpo = np.zeros((len(close_col), 8), dtype=np.float32)
    node_vol = np.zeros(len(close_col), dtype=np.float32)
    direction = np.full(len(close_col), -1, dtype=np.int32)
    entry = np.zeros(len(close_col), dtype=np.int32)
    label_valid = np.zeros(len(close_col), dtype=np.bool_)
    tpo_duration = 0.0
    label_duration = 0.0
    valid_rows = int(np.count_nonzero(valid_col))
    tpo_skipped_large = False

    if valid_rows > 0:
        tpo_started = time.perf_counter()
        if tpo_source_frame is not None and tpo_lookup is not None:
            source_high = tpo_source_frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
            source_low = tpo_source_frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
            source_close = tpo_source_frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
            with _TPO_LOCK:
                source_tpo, source_vol = compute_tpo_feature_panel(source_high, source_low, source_close)
            node_tpo = source_tpo[tpo_lookup]
            node_vol = source_vol[tpo_lookup]
        elif valid_rows <= _MAX_TPO_BARS and len(close_col) <= _MAX_TPO_BARS:
            with _TPO_LOCK:
                node_tpo, node_vol = compute_tpo_feature_panel(high_col, low_col, close_col)
        else:
            tpo_skipped_large = True
        tpo_duration = time.perf_counter() - tpo_started

    if tradable:
        label_started = time.perf_counter()
        if label_source_frame is not None and label_lookup is not None:
            label_high = label_source_frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
            label_low = label_source_frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
            label_close = label_source_frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
            label_spread = label_source_frame["sp"].fillna(0.0).to_numpy(dtype=np.float32)
            labels, valid = make_triple_barrier_labels(
                label_close,
                label_high,
                label_low,
                label_spread,
                pair=symbol,
                binary=True,
            )
            labels = labels[label_lookup]
            valid = valid[label_lookup]
        else:
            labels, valid = make_triple_barrier_labels(
                close_col,
                high_col,
                low_col,
                spread_col,
                pair=symbol,
                binary=True,
            )
        direction = np.where(labels == BUY_CLASS, 1, np.where(labels == SELL_CLASS, 0, -1)).astype(np.int32)
        entry = (labels == BUY_CLASS).astype(np.int32)
        label_valid = valid
        label_duration = time.perf_counter() - label_started

    diagnostics = {
        "rows": int(len(close_col)),
        "valid_rows": valid_rows,
        "tradable": bool(tradable),
        "used_external_tpo": bool(tpo_source_frame is not None and tpo_lookup is not None),
        "tpo_skipped_large": bool(tpo_skipped_large),
        "tpo_sec": round(tpo_duration, 4),
        "label_sec": round(label_duration, 4),
        "total_sec": round(time.perf_counter() - total_started, 4),
    }
    return symbol, node_tpo, node_vol, direction, entry, label_valid, diagnostics


def build_fx_timeframe_batch(
    panels: StagedPanels,
    timeframe: str,
    include_signal_only_tpo: bool = True,
    max_workers: int = 0,
    logger=None,
    status_file: str | None = None,
    shard_root: str | Path | None = None,
    symbol_timeout_sec: float | None = None,
    batch_deadline_sec: float | None = None,
) -> TimeframeFeatureBatch:
    symbol_frames = panels.panels[timeframe]
    timestamps = next(iter(symbol_frames.values())).index.astype("datetime64[ns]").to_numpy()
    n_steps = len(timestamps)
    n_nodes = len(panels.symbols)
    tradable_mask = np.array([symbol in FX_TRADABLE_NAMES for symbol in panels.symbols], dtype=np.bool_)
    open_ = np.zeros((n_steps, n_nodes), dtype=np.float32)
    high = np.zeros((n_steps, n_nodes), dtype=np.float32)
    low = np.zeros((n_steps, n_nodes), dtype=np.float32)
    close = np.zeros((n_steps, n_nodes), dtype=np.float32)
    spread = np.zeros((n_steps, n_nodes), dtype=np.float32)
    tick = np.zeros((n_steps, n_nodes), dtype=np.float32)
    valid_mask = np.zeros((n_steps, n_nodes), dtype=np.bool_)
    tpo_features = np.zeros((n_steps, n_nodes, 8), dtype=np.float32)
    volatility = np.zeros((n_steps, n_nodes), dtype=np.float32)
    direction_labels = np.full((n_steps, n_nodes), -1, dtype=np.int32)
    entry_labels = np.zeros((n_steps, n_nodes), dtype=np.int32)
    label_valid_mask = np.zeros((n_steps, n_nodes), dtype=np.bool_)

    session_codes = encode_session_codes(timestamps)
    session_transition = _session_transition(session_codes)
    market_open_mask = session_codes != SESSION_CODES["closed"]
    overlap_mask = session_codes == SESSION_CODES["overlap"]
    timestamp_index = pd.Index(timestamps)
    tpo_source_timeframe = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
    tpo_source_frames = panels.panels.get(tpo_source_timeframe, {}) if tpo_source_timeframe != timeframe else {}
    tpo_source_lookup = None
    if tpo_source_frames:
        tpo_source_index = next(iter(tpo_source_frames.values())).index
        tpo_source_lookup = _lookup_to_source_index(tpo_source_index, timestamp_index)

    coarse_tick_frames = panels.panels.get("M1", {}) if timeframe == "tick" else {}
    tick_to_m1_lookup = None
    if timeframe == "tick" and coarse_tick_frames:
        coarse_tick_index = next(iter(coarse_tick_frames.values())).index
        tick_to_m1_lookup = _lookup_to_source_index(coarse_tick_index, timestamp_index)

    total_symbols = len(panels.symbols)
    shard_dir = Path(shard_root) if shard_root is not None else None
    completed = 0
    wait_cycles = 0
    if logger is not None:
        _batch_mb = (n_steps * n_nodes * (4 * 8 + 4 + 4 + 4 * 3 + 1 * 3)) / (1024 * 1024)
        logger.info(
            "stage=build_fx_timeframe_batch state=allocated timeframe=%s n_steps=%d n_nodes=%d batch_mb=%.1f tpo_source=%s",
            timeframe, n_steps, n_nodes, _batch_mb, tpo_source_timeframe,
        )
    for node_idx, symbol in enumerate(panels.symbols, start=1):
        frame = symbol_frames[symbol]
        col_idx = node_idx - 1
        open_[:, col_idx] = frame["o"].fillna(0.0).to_numpy(dtype=np.float32)
        high[:, col_idx] = frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
        low[:, col_idx] = frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
        close[:, col_idx] = frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
        spread[:, col_idx] = frame["sp"].fillna(0.0).to_numpy(dtype=np.float32)
        tick[:, col_idx] = frame["tk"].fillna(0.0).to_numpy(dtype=np.float32)
        valid_mask[:, col_idx] = frame["real"].fillna(False).to_numpy(dtype=np.bool_)

    requested_workers = max_workers if max_workers and max_workers > 0 else min(16, max(4, (os.cpu_count() or 8) // 2))
    worker_count = max(1, min(total_symbols, requested_workers))
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    if symbol_timeout_sec is not None:
        symbol_timeout: float | None = float(symbol_timeout_sec)
    elif timeframe == "tick":
        symbol_timeout = _DEFAULT_SYMBOL_TIMEOUT_SEC
    else:
        symbol_timeout = None
    default_batch_deadline = _DEFAULT_BATCH_DEADLINE_SEC if timeframe == "tick" else max(14400.0, _DEFAULT_BATCH_DEADLINE_SEC)
    batch_deadline = time.perf_counter() + float(
        batch_deadline_sec if batch_deadline_sec is not None else default_batch_deadline
    )
    failed_symbols: list[str] = []
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"fx_{timeframe}")
    try:
        queued_specs = []
        for node_idx, symbol in enumerate(panels.symbols, start=1):
            col_idx = node_idx - 1
            shard_path = _symbol_shard_path(shard_dir, symbol) if shard_dir is not None else None
            if shard_path is not None and shard_path.exists():
                try:
                    node_tpo, node_vol, direction, entry, label_valid = _load_symbol_aux_shard(shard_path)
                    tpo_features[:, col_idx, :] = node_tpo
                    volatility[:, col_idx] = node_vol
                    direction_labels[:, col_idx] = direction
                    entry_labels[:, col_idx] = entry
                    label_valid_mask[:, col_idx] = label_valid
                    completed += 1
                    if logger is not None:
                        logger.info(
                            "stage=build_fx_feature_symbol state=skip timeframe=%s symbol=%s shard=%s",
                            timeframe,
                            symbol,
                            shard_path.name,
                        )
                    if logger is not None and (completed == total_symbols or completed % 5 == 0):
                        log_progress(
                            logger,
                            status_file,
                            "build_fx_feature_symbols",
                            completed,
                            total_symbols,
                            timeframe=timeframe,
                            workers=worker_count,
                        )
                    continue
                except Exception:
                    if logger is not None:
                        logger.warning(
                            "stage=build_fx_feature_symbol state=reload_failed timeframe=%s symbol=%s shard=%s",
                            timeframe,
                            symbol,
                            str(shard_path),
                        )
            should_build_tpo = valid_mask[:, col_idx].any() and (tradable_mask[col_idx] or include_signal_only_tpo)
            tpo_source_frame = tpo_source_frames.get(symbol) if should_build_tpo and tpo_source_lookup is not None else None
            label_source_frame = coarse_tick_frames.get(symbol) if timeframe == "tick" and tradable_mask[col_idx] and tick_to_m1_lookup is not None else None
            queued_specs.append(
                (
                    node_idx,
                    col_idx,
                    symbol,
                    shard_path,
                    tpo_source_frame,
                    label_source_frame,
                )
            )
        pending = {}

        def _submit_spec(spec: tuple[int, int, str, Path | None, pd.DataFrame | None, pd.DataFrame | None]) -> None:
            node_idx, col_idx, symbol, shard_path, tpo_source_frame, label_source_frame = spec
            if logger is not None:
                logger.info("stage=build_fx_feature_symbol state=start timeframe=%s symbol=%s", timeframe, symbol)
            future = executor.submit(
                _build_symbol_aux_features,
                symbol,
                timeframe,
                timestamp_index,
                high[:, col_idx],
                low[:, col_idx],
                close[:, col_idx],
                spread[:, col_idx],
                bool(tradable_mask[col_idx]),
                valid_mask[:, col_idx],
                tpo_source_frame,
                tpo_source_lookup,
                label_source_frame,
                tick_to_m1_lookup,
            )
            pending[future] = (node_idx, col_idx, symbol, shard_path, time.perf_counter())

        while queued_specs and len(pending) < worker_count:
            _submit_spec(queued_specs.pop(0))

        def _mark_failed_symbol(
            node_idx: int,
            col_idx: int,
            symbol: str,
            shard_path: Path | None,
            submitted_at: float,
            reason: str,
        ) -> None:
            nonlocal completed
            tpo_features[:, col_idx, :] = 0.0
            volatility[:, col_idx] = 0.0
            direction_labels[:, col_idx] = -1
            entry_labels[:, col_idx] = 0
            label_valid_mask[:, col_idx] = False
            if shard_path is not None:
                _save_symbol_aux_shard(
                    shard_path,
                    tpo_features[:, col_idx, :],
                    volatility[:, col_idx],
                    direction_labels[:, col_idx],
                    entry_labels[:, col_idx],
                    label_valid_mask[:, col_idx],
                )
            failed_symbols.append(symbol)
            if logger is not None:
                logger.warning(
                    "stage=build_fx_feature_symbol state=failed timeframe=%s symbol=%s running_sec=%.2f reason=%s",
                    timeframe,
                    symbol,
                    time.perf_counter() - submitted_at,
                    reason,
                )
            completed += 1
            if logger is not None and (completed == total_symbols or completed % 5 == 0):
                log_progress(
                    logger,
                    status_file,
                    "build_fx_feature_symbols",
                    completed,
                    total_symbols,
                    timeframe=timeframe,
                    workers=worker_count,
                    failed=len(failed_symbols),
                )

        while pending:
            now = time.perf_counter()
            if now >= batch_deadline:
                for future, spec in list(pending.items()):
                    node_idx, col_idx, symbol, shard_path, submitted_at = spec
                    future.cancel()
                    pending.pop(future, None)
                    _mark_failed_symbol(node_idx, col_idx, symbol, shard_path, submitted_at, "batch_deadline")
                while queued_specs:
                    node_idx, col_idx, symbol, shard_path, _, _ = queued_specs.pop(0)
                    _mark_failed_symbol(node_idx, col_idx, symbol, shard_path, now, "batch_deadline")
                break

            for future, spec in list(pending.items()):
                node_idx, col_idx, symbol, shard_path, submitted_at = spec
                if symbol_timeout is not None and now - submitted_at > symbol_timeout:
                    future.cancel()
                    pending.pop(future, None)
                    _mark_failed_symbol(node_idx, col_idx, symbol, shard_path, submitted_at, f"timeout>{symbol_timeout:.1f}s")

            if not pending:
                break

            done, not_done = wait(tuple(pending.keys()), timeout=30.0, return_when=FIRST_COMPLETED)
            if not done:
                wait_cycles += 1
                if logger is not None:
                    pending_infos = sorted(
                        (
                            {
                                "symbol": pending[future][2],
                                "running_sec": round(now - pending[future][4], 2),
                            }
                            for future in not_done
                        ),
                        key=lambda item: item["running_sec"],
                        reverse=True,
                    )
                    pending_symbols = pending_infos[:8]
                    logger.warning(
                        "stage=build_fx_feature_symbols state=waiting timeframe=%s pending=%d oldest_sec=%.2f waits=%d symbols=%s",
                        timeframe,
                        len(not_done),
                        pending_infos[0]["running_sec"] if pending_infos else 0.0,
                        wait_cycles,
                        ",".join(f"{item['symbol']}:{item['running_sec']:.1f}s" for item in pending_symbols),
                    )
                write_status(
                    status_file,
                    {
                        "state": "running",
                        "stage": "build_fx_feature_symbols_wait",
                        "progress": {
                            "current": int(completed),
                            "total": int(total_symbols),
                            "ratio": float(completed / total_symbols) if total_symbols else 0.0,
                        },
                        "details": {
                            "timeframe": timeframe,
                            "workers": worker_count,
                            "pending_symbols": [pending[future][2] for future in list(not_done)[:8]],
                            "pending_durations_sec": {
                                pending[future][2]: round(now - pending[future][4], 2)
                                for future in list(not_done)[:8]
                            },
                            "pending_count": len(not_done),
                            "oldest_pending_sec": round(max(now - pending[future][4] for future in not_done), 2),
                            "wait_cycles": wait_cycles,
                            "wait_timeout_sec": 30.0,
                            "symbol_timeout_sec": symbol_timeout,
                            "batch_deadline_remaining_sec": round(max(batch_deadline - now, 0.0), 2),
                            "failed_symbols": failed_symbols[:12],
                        },
                    },
                )
                continue
            for future in done:
                node_idx, col_idx, symbol, shard_path, submitted_at = pending.pop(future)
                try:
                    _, node_tpo, node_vol, direction, entry, label_valid, diagnostics = future.result()
                except Exception as exc:
                    _mark_failed_symbol(node_idx, col_idx, symbol, shard_path, submitted_at, f"exception:{type(exc).__name__}")
                    if logger is not None:
                        logger.exception(
                            "stage=build_fx_feature_symbol state=exception timeframe=%s symbol=%s running_sec=%.2f",
                            timeframe,
                            symbol,
                            time.perf_counter() - submitted_at,
                        )
                    continue
                tpo_features[:, col_idx, :] = node_tpo
                volatility[:, col_idx] = node_vol
                direction_labels[:, col_idx] = direction
                entry_labels[:, col_idx] = entry
                label_valid_mask[:, col_idx] = label_valid
                if shard_path is not None:
                    _save_symbol_aux_shard(shard_path, node_tpo, node_vol, direction, entry, label_valid)
                if logger is not None:
                    logger.info(
                        "stage=build_fx_feature_symbol state=done timeframe=%s symbol=%s running_sec=%.2f tpo_sec=%.2f label_sec=%.2f total_sec=%.2f valid_rows=%d",
                        timeframe,
                        symbol,
                        time.perf_counter() - submitted_at,
                        float(diagnostics["tpo_sec"]),
                        float(diagnostics["label_sec"]),
                        float(diagnostics["total_sec"]),
                        int(diagnostics["valid_rows"]),
                    )
                completed += 1
                if logger is not None and (completed == total_symbols or completed % 5 == 0):
                    log_progress(
                        logger,
                        status_file,
                        "build_fx_feature_symbols",
                        completed,
                        total_symbols,
                        timeframe=timeframe,
                        workers=worker_count,
                    )
                if queued_specs:
                    _submit_spec(queued_specs.pop(0))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if logger is not None and failed_symbols:
        logger.warning(
            "stage=build_fx_feature_symbols state=partial_complete timeframe=%s failed=%d symbols=%s",
            timeframe,
            len(failed_symbols),
            ",".join(failed_symbols[:16]),
        )

    if logger is not None:
        logger.info("stage=build_fx_timeframe_batch state=symbols_done timeframe=%s completed=%d failed=%d", timeframe, completed, len(failed_symbols))
    ret_1 = _lagged_logret_matrix(close, 1)
    ret_3 = _lagged_logret_matrix(close, 3)
    atr_norm = _atr_norm_matrix(high, low, close)
    range_norm = ((high - low) / np.maximum(np.abs(close), 1e-8)).astype(np.float32)
    if logger is not None:
        logger.info("stage=build_fx_timeframe_batch state=pre_laplacian timeframe=%s", timeframe)
    if timeframe == "tick" and coarse_tick_frames and tick_to_m1_lookup is not None:
        coarse_close = np.zeros((len(coarse_tick_index), n_nodes), dtype=np.float32)
        coarse_valid = np.zeros((len(coarse_tick_index), n_nodes), dtype=np.bool_)
        for col_idx, symbol in enumerate(panels.symbols):
            coarse_frame = coarse_tick_frames[symbol]
            coarse_close[:, col_idx] = coarse_frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
            coarse_valid[:, col_idx] = coarse_frame["real"].fillna(False).to_numpy(dtype=np.bool_)
        coarse_lap = _compute_laplacian_residuals(coarse_close, coarse_valid)
        lap_residual = coarse_lap[tick_to_m1_lookup]
    else:
        lap_residual = _compute_laplacian_residuals(close, valid_mask)
    regime_signal = _regime_signal(ret_3, atr_norm)
    raw = np.stack(
        [
            open_,
            high,
            low,
            close,
            spread,
            tick,
            ret_1,
            ret_3,
            atr_norm,
            range_norm,
            lap_residual,
            regime_signal,
            np.broadcast_to(session_codes[:, None], close.shape).astype(np.float32),
            np.broadcast_to(session_transition[:, None], close.shape),
        ],
        axis=-1,
    )

    return TimeframeFeatureBatch(
        timeframe=timeframe,
        timestamps=timestamps,
        node_names=panels.symbols,
        tradable_mask=tradable_mask.copy(),
        node_features=raw.astype(np.float32),
        tpo_features=tpo_features.astype(np.float32),
        volatility=volatility.astype(np.float32),
        valid_mask=valid_mask,
        market_open_mask=market_open_mask.astype(np.bool_),
        overlap_mask=overlap_mask.astype(np.bool_),
        session_codes=session_codes,
        direction_labels=direction_labels,
        entry_labels=entry_labels,
        label_valid_mask=label_valid_mask,
    )


def build_fx_feature_batch(
    panels: StagedPanels,
    timeframes: tuple[str, ...] | None = None,
    include_signal_only_tpo: bool = True,
    max_workers: int = 0,
    logger=None,
    status_file: str | None = None,
) -> FXFeatureBatch:
    timeframe_batches: dict[str, TimeframeFeatureBatch] = {}
    selected_timeframes = tuple(timeframes or ALL_TIMEFRAMES)
    for timeframe in selected_timeframes:
        timeframe_batches[timeframe] = build_fx_timeframe_batch(
            panels,
            timeframe,
            include_signal_only_tpo=include_signal_only_tpo,
            max_workers=max_workers,
            logger=logger,
            status_file=status_file,
        )
    return FXFeatureBatch(
        anchor_timeframe=panels.anchor_timeframe,
        anchor_timestamps=panels.anchor_timestamps,
        timeframe_batches=timeframe_batches,
        node_names=panels.symbols,
        tradable_node_names=FX_TRADABLE_NAMES,
        anchor_lookup=panels.anchor_lookup,
    )
