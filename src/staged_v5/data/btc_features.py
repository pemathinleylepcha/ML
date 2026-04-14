from __future__ import annotations

import numpy as np
import pandas as pd

from research_dataset import BUY_CLASS, SELL_CLASS, make_triple_barrier_labels
from staged_v5.config import ALL_TIMEFRAMES, TPO_SOURCE_TIMEFRAME
from staged_v5.contracts import BTCFeatureBatch, TimeframeFeatureBatch
from staged_v5.data.dataset import StagedPanels
from staged_v5.data.tpo_features import compute_tpo_feature_panel
from staged_v5.utils.runtime_logging import log_progress


_BTC_FEATURE_CHUNK_ROWS = 200_000
_BTC_FEATURE_BACK_OVERLAP = 256
_BTC_FEATURE_FORWARD_OVERLAP = 8


def _lagged_logret(close: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.float32)
    if len(close) <= lag:
        return out
    numer = np.log(np.clip(close[lag:], 1e-12, None))
    denom = np.log(np.clip(close[:-lag], 1e-12, None))
    out[lag:] = (numer - denom).astype(np.float32)
    return out


def _atr_norm(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out = np.zeros(len(close), dtype=np.float32)
    for idx in range(len(close)):
        start = max(0, idx - period + 1)
        atr = np.mean(tr[start : idx + 1])
        out[idx] = float(atr / max(abs(close[idx]), 1e-8))
    return out


def _session_transition(session_codes: np.ndarray) -> np.ndarray:
    if len(session_codes) == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(len(session_codes), dtype=np.float32)
    out[1:] = (session_codes[1:] != session_codes[:-1]).astype(np.float32)
    return out


def _regime_signal(ret_3: np.ndarray, atr_norm: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(atr_norm), 1e-6)
    return np.clip(ret_3 / denom, -5.0, 5.0).astype(np.float32)


def _resolve_tpo_source(
    panels: StagedPanels,
    symbol: str,
    timeframe: str,
    timestamps: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    source_timeframe = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
    if source_timeframe == timeframe:
        return None, None
    source_frame = panels.panels.get(source_timeframe, {}).get(symbol)
    if source_frame is None or len(source_frame) == 0:
        return None, None
    source_high = source_frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
    source_low = source_frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
    source_close = source_frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
    source_tpo, source_vol = compute_tpo_feature_panel(source_high, source_low, source_close)
    source_lookup = pd.Index(source_frame.index).searchsorted(pd.Index(timestamps), side="right") - 1
    source_lookup = np.maximum(source_lookup, 0).astype(np.int32)
    return source_tpo[source_lookup].astype(np.float32), source_vol[source_lookup].astype(np.float32)


def build_btc_timeframe_batch(
    panels: StagedPanels,
    timeframe: str,
    logger=None,
    status_file: str | None = None,
    chunk_rows: int = _BTC_FEATURE_CHUNK_ROWS,
) -> TimeframeFeatureBatch:
    symbol = panels.symbols[0]
    frame = panels.panels[timeframe][symbol]
    timestamps = frame.index.astype("datetime64[ns]").to_numpy()
    close = frame["c"].to_numpy(dtype=np.float32)
    high = frame["h"].to_numpy(dtype=np.float32)
    low = frame["l"].to_numpy(dtype=np.float32)
    open_ = frame["o"].to_numpy(dtype=np.float32)
    spread = frame["sp"].fillna(0.0).to_numpy(dtype=np.float32)
    tick = frame["tk"].fillna(0.0).to_numpy(dtype=np.float32)
    valid_mask = frame["real"].fillna(False).to_numpy(dtype=np.bool_)[:, None]

    n_rows = len(close)
    raw = np.zeros((n_rows, 1, 14), dtype=np.float32)
    tpo = np.zeros((n_rows, 8), dtype=np.float32)
    volatility = np.zeros(n_rows, dtype=np.float32)
    direction_labels = np.full(n_rows, -1, dtype=np.int32)
    entry_labels = np.zeros(n_rows, dtype=np.int32)
    label_valid = np.zeros(n_rows, dtype=np.bool_)
    session_codes = np.zeros(n_rows, dtype=np.int8)
    session_transition = _session_transition(session_codes)
    source_tpo, source_vol = _resolve_tpo_source(panels, symbol, timeframe, timestamps)

    use_chunking = n_rows > chunk_rows
    windows: list[tuple[int, int]]
    if use_chunking:
        windows = [(start, min(start + chunk_rows, n_rows)) for start in range(0, n_rows, chunk_rows)]
    else:
        windows = [(0, n_rows)]

    total_windows = len(windows)
    for window_idx, (start, end) in enumerate(windows, start=1):
        ext_start = max(0, start - _BTC_FEATURE_BACK_OVERLAP)
        ext_end = min(n_rows, end + _BTC_FEATURE_FORWARD_OVERLAP)
        offset = start - ext_start
        length = end - start

        ext_close = close[ext_start:ext_end]
        ext_high = high[ext_start:ext_end]
        ext_low = low[ext_start:ext_end]
        ext_open = open_[ext_start:ext_end]
        ext_spread = spread[ext_start:ext_end]
        ext_tick = tick[ext_start:ext_end]

        ext_ret_1 = _lagged_logret(ext_close, 1)
        ext_ret_3 = _lagged_logret(ext_close, 3)
        ext_atr_norm = _atr_norm(ext_high, ext_low, ext_close)
        ext_range_norm = ((ext_high - ext_low) / np.maximum(np.abs(ext_close), 1e-8)).astype(np.float32)
        ext_regime_signal = _regime_signal(ext_ret_3, ext_atr_norm)
        ext_labels, ext_label_valid = make_triple_barrier_labels(ext_close, ext_high, ext_low, ext_spread, pair=symbol, binary=True)

        center = slice(offset, offset + length)
        target = slice(start, end)

        raw[target, 0, 0] = ext_open[center]
        raw[target, 0, 1] = ext_high[center]
        raw[target, 0, 2] = ext_low[center]
        raw[target, 0, 3] = ext_close[center]
        raw[target, 0, 4] = ext_spread[center]
        raw[target, 0, 5] = ext_tick[center]
        raw[target, 0, 6] = ext_ret_1[center]
        raw[target, 0, 7] = ext_ret_3[center]
        raw[target, 0, 8] = ext_atr_norm[center]
        raw[target, 0, 9] = ext_range_norm[center]
        raw[target, 0, 10] = 0.0
        raw[target, 0, 11] = ext_regime_signal[center]
        raw[target, 0, 12] = session_codes[target].astype(np.float32)
        raw[target, 0, 13] = session_transition[target]

        if source_tpo is None or source_vol is None:
            ext_tpo, ext_vol = compute_tpo_feature_panel(ext_high, ext_low, ext_close)
            tpo[target] = ext_tpo[center].astype(np.float32)
            volatility[target] = ext_vol[center].astype(np.float32)
        else:
            tpo[target] = source_tpo[target]
            volatility[target] = source_vol[target]
        center_labels = ext_labels[center]
        center_valid = ext_label_valid[center]
        direction_labels[target] = np.where(center_labels == BUY_CLASS, 1, np.where(center_labels == SELL_CLASS, 0, -1)).astype(np.int32)
        entry_labels[target] = (center_labels == BUY_CLASS).astype(np.int32)
        label_valid[target] = center_valid

        if logger is not None and total_windows > 1:
            log_progress(
                logger,
                status_file,
                "build_btc_feature_window",
                window_idx,
                total_windows,
                timeframe=timeframe,
                rows=n_rows,
                start=int(start),
                end=int(end),
            )

    return TimeframeFeatureBatch(
        timeframe=timeframe,
        timestamps=timestamps,
        node_names=(symbol,),
        tradable_mask=np.array([True], dtype=np.bool_),
        node_features=raw.astype(np.float32),
        tpo_features=tpo[:, None, :].astype(np.float32),
        volatility=volatility[:, None].astype(np.float32),
        valid_mask=valid_mask,
        market_open_mask=np.ones(n_rows, dtype=np.bool_),
        overlap_mask=np.zeros(n_rows, dtype=np.bool_),
        session_codes=session_codes,
        direction_labels=direction_labels[:, None],
        entry_labels=entry_labels[:, None],
        label_valid_mask=label_valid[:, None],
    )


def build_btc_feature_batch(
    panels: StagedPanels,
    timeframes: tuple[str, ...] | None = None,
    logger=None,
    status_file: str | None = None,
) -> BTCFeatureBatch:
    timeframe_batches: dict[str, TimeframeFeatureBatch] = {}
    selected_timeframes = tuple(timeframes or ALL_TIMEFRAMES)
    for timeframe in selected_timeframes:
        timeframe_batches[timeframe] = build_btc_timeframe_batch(panels, timeframe, logger=logger, status_file=status_file)
    return BTCFeatureBatch(
        anchor_timeframe=panels.anchor_timeframe,
        anchor_timestamps=panels.anchor_timestamps,
        timeframe_batches=timeframe_batches,
        node_names=panels.symbols,
        anchor_lookup=panels.anchor_lookup,
    )
