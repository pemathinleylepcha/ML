from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from research_dataset import BUY_CLASS, SELL_CLASS, SESSION_CODES, encode_session_codes, make_triple_barrier_labels
from staged_v4.config import FX_TRADABLE_NAMES, TPO_SOURCE_TIMEFRAME
from staged_v4.contracts import SubnetSequenceBatch, TimeframeSequenceBatch
from staged_v4.data.dataset import StagedPanels
from staged_v4.data.tpo_features import compute_tpo_feature_panel
from staged_v4.data.btc_features import _atr_norm as _btc_atr_norm
from staged_v4.data.btc_features import _lagged_logret as _btc_lagged_logret
from staged_v4.data.btc_features import _regime_signal as _btc_regime_signal
from staged_v4.data.fx_features import _atr_norm_matrix, _compute_laplacian_residuals, _lagged_logret_matrix, _regime_signal as _fx_regime_signal
from staged_v4.data.fx_features import _session_transition as _fx_session_transition


_JIT_BACK_CONTEXT = 192
_JIT_LABEL_BACK_CONTEXT = 32
_JIT_LABEL_FORWARD = 8
_JIT_TPO_LOOKBACK = 192
_JIT_LAPLACIAN_LOOKBACK = 60


def _to_device(arr: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    cpu_tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype)
    if device.type == "cuda":
        return cpu_tensor.pin_memory().to(device, non_blocking=True)
    return cpu_tensor.to(device)


def _pad_window(values: np.ndarray, seq_len: int, fill_value=0) -> np.ndarray:
    out = np.full((seq_len,) + values.shape[1:], fill_value, dtype=values.dtype)
    if len(values):
        out[-len(values) :] = values
    return out


def _frame_series(frame: pd.DataFrame, column: str, start: int, end: int, dtype) -> np.ndarray:
    series = frame.iloc[start:end][column]
    if dtype == np.bool_:
        return series.fillna(False).to_numpy(dtype=np.bool_)
    return series.fillna(0.0).to_numpy(dtype=dtype)


def _aligned_matrix(symbol_frames: dict[str, pd.DataFrame], symbols: tuple[str, ...], start: int, end: int, column: str, dtype) -> np.ndarray:
    cols = [_frame_series(symbol_frames[symbol], column, start, end, dtype) for symbol in symbols]
    if not cols:
        return np.zeros((0, 0), dtype=dtype)
    return np.stack(cols, axis=1)


def _direction_entry_from_label(label_value: int, valid: bool) -> tuple[int, int, bool]:
    if not valid:
        return -1, 0, False
    direction = 1 if label_value == BUY_CLASS else 0 if label_value == SELL_CLASS else -1
    entry = 1 if label_value == BUY_CLASS else 0
    return direction, entry, direction >= 0


def _label_at_position(frame: pd.DataFrame, position: int, symbol: str) -> tuple[int, int, bool]:
    start = max(0, position - _JIT_LABEL_BACK_CONTEXT)
    end = min(len(frame), position + _JIT_LABEL_FORWARD + 1)
    if end - start <= 0:
        return -1, 0, False
    close = _frame_series(frame, "c", start, end, np.float32)
    high = _frame_series(frame, "h", start, end, np.float32)
    low = _frame_series(frame, "l", start, end, np.float32)
    spread = _frame_series(frame, "sp", start, end, np.float32)
    labels, valid = make_triple_barrier_labels(close, high, low, spread, pair=symbol, binary=True)
    local_idx = position - start
    if local_idx < 0 or local_idx >= len(labels):
        return -1, 0, False
    return _direction_entry_from_label(int(labels[local_idx]), bool(valid[local_idx]))


def _preswept_tpo_for_target_window(
    panels: StagedPanels,
    source_timeframe: str,
    symbol: str,
    target_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    symbol_tpo = panels.tpo_panels.get(source_timeframe, {}).get(symbol)
    if symbol_tpo is None or len(target_timestamps) == 0:
        return None
    source_frame = panels.panels.get(source_timeframe, {}).get(symbol)
    if source_frame is None or len(source_frame.index) == 0:
        return None
    all_tpo, all_vol = symbol_tpo
    if len(all_tpo) == 0:
        return (
            np.zeros((len(target_timestamps), 8), dtype=np.float32),
            np.zeros(len(target_timestamps), dtype=np.float32),
        )
    source_lookup = source_frame.index.searchsorted(pd.Index(target_timestamps), side="right") - 1
    source_lookup = np.clip(source_lookup, 0, len(all_tpo) - 1).astype(np.int32)
    return all_tpo[source_lookup], all_vol[source_lookup]


def _tpo_for_target_window(
    source_frames: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
    target_timestamps: np.ndarray,
    include_signal_only_tpo: bool,
    tradable_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = len(target_timestamps)
    n_nodes = len(symbols)
    tpo_out = np.zeros((n_steps, n_nodes, 8), dtype=np.float32)
    vol_out = np.zeros((n_steps, n_nodes), dtype=np.float32)
    if n_steps == 0:
        return tpo_out, vol_out

    source_index = next(iter(source_frames.values())).index
    source_lookup = source_index.searchsorted(pd.Index(target_timestamps), side="right") - 1
    source_lookup = np.clip(source_lookup, 0, len(source_index) - 1).astype(np.int32)
    source_start = max(0, int(source_lookup.min()) - _JIT_TPO_LOOKBACK)
    source_end = int(source_lookup.max()) + 1
    local_lookup = source_lookup - source_start

    for col_idx, symbol in enumerate(symbols):
        if not include_signal_only_tpo and not tradable_mask[col_idx]:
            continue
        frame = source_frames[symbol]
        high = _frame_series(frame, "h", source_start, source_end, np.float32)
        low = _frame_series(frame, "l", source_start, source_end, np.float32)
        close = _frame_series(frame, "c", source_start, source_end, np.float32)
        features, volatility = compute_tpo_feature_panel(high, low, close)
        tpo_out[:, col_idx, :] = features[local_lookup]
        vol_out[:, col_idx] = volatility[local_lookup]
    return tpo_out, vol_out


def build_btc_sequence_batch_from_panels(
    panels: StagedPanels,
    anchor_indices: np.ndarray,
    seq_lens: dict[str, int],
    device: torch.device,
) -> SubnetSequenceBatch:
    symbol = panels.symbols[0]
    timeframe_batches: dict[str, TimeframeSequenceBatch] = {}

    for timeframe, symbol_frames in panels.panels.items():
        frame = symbol_frames[symbol]
        full_index = frame.index
        seq_len = int(seq_lens[timeframe])
        end_indices = panels.anchor_lookup[timeframe][anchor_indices]
        batch_size = len(anchor_indices)

        node_features = np.zeros((batch_size, seq_len, 1, 14), dtype=np.float32)
        tpo_features = np.zeros((batch_size, seq_len, 1, 8), dtype=np.float32)
        volatility = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        valid_mask = np.zeros((batch_size, seq_len, 1), dtype=np.bool_)
        market_open_mask = np.ones((batch_size, seq_len), dtype=np.bool_)
        overlap_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        session_codes = np.zeros((batch_size, seq_len), dtype=np.int64)
        timestamps = np.zeros((batch_size, seq_len), dtype="datetime64[ns]")
        direction_labels = np.full((batch_size, 1), -1, dtype=np.float32)
        entry_labels = np.zeros((batch_size, 1), dtype=np.float32)
        label_valid_mask = np.zeros((batch_size, 1), dtype=np.bool_)

        source_timeframe = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
        source_frames = panels.panels[source_timeframe]

        for batch_idx, end_idx in enumerate(end_indices):
            end_idx = int(end_idx)
            seq_start = max(0, end_idx - seq_len + 1)
            ext_start = max(0, seq_start - _JIT_BACK_CONTEXT)
            ext_end = end_idx + 1
            local_start = seq_start - ext_start
            local_end = ext_end - ext_start

            ext_open = _frame_series(frame, "o", ext_start, ext_end, np.float32)
            ext_high = _frame_series(frame, "h", ext_start, ext_end, np.float32)
            ext_low = _frame_series(frame, "l", ext_start, ext_end, np.float32)
            ext_close = _frame_series(frame, "c", ext_start, ext_end, np.float32)
            ext_spread = _frame_series(frame, "sp", ext_start, ext_end, np.float32)
            ext_tick = _frame_series(frame, "tk", ext_start, ext_end, np.float32)
            ext_valid = _frame_series(frame, "real", ext_start, ext_end, np.bool_)
            ext_ts = full_index[ext_start:ext_end].to_numpy(dtype="datetime64[ns]")

            window_ts = full_index[seq_start : end_idx + 1].to_numpy(dtype="datetime64[ns]")
            window_open = ext_open[local_start:local_end]
            window_high = ext_high[local_start:local_end]
            window_low = ext_low[local_start:local_end]
            window_close = ext_close[local_start:local_end]
            window_spread = ext_spread[local_start:local_end]
            window_tick = ext_tick[local_start:local_end]
            window_valid = ext_valid[local_start:local_end]
            ret_1 = _btc_lagged_logret(ext_close, 1)[local_start:local_end]
            ret_3 = _btc_lagged_logret(ext_close, 3)[local_start:local_end]
            atr_norm = _btc_atr_norm(ext_high, ext_low, ext_close)[local_start:local_end]
            range_norm = ((window_high - window_low) / np.maximum(np.abs(window_close), 1e-8)).astype(np.float32)
            regime_signal = _btc_regime_signal(ret_3, atr_norm)

            preswept = _preswept_tpo_for_target_window(panels, source_timeframe, symbol, window_ts)
            if preswept is not None:
                window_tpo, window_vol = preswept
            elif source_timeframe == timeframe:
                source_tpo, source_vol = compute_tpo_feature_panel(ext_high, ext_low, ext_close)
                window_tpo = source_tpo[local_start:local_end]
                window_vol = source_vol[local_start:local_end]
            else:
                window_tpo, window_vol = _tpo_for_target_window(source_frames, (symbol,), window_ts, True, np.array([True], dtype=np.bool_))
                window_tpo = window_tpo[:, 0, :]
                window_vol = window_vol[:, 0]

            label_direction, label_entry, label_valid = _label_at_position(frame, end_idx, symbol)

            packed = np.stack(
                [
                    window_open,
                    window_high,
                    window_low,
                    window_close,
                    window_spread,
                    window_tick,
                    ret_1,
                    ret_3,
                    atr_norm,
                    range_norm,
                    np.zeros_like(window_close, dtype=np.float32),
                    regime_signal,
                    np.zeros_like(window_close, dtype=np.float32),
                    np.zeros_like(window_close, dtype=np.float32),
                ],
                axis=-1,
            )[:, None, :]

            node_features[batch_idx] = _pad_window(packed, seq_len, 0.0)
            tpo_features[batch_idx] = _pad_window(window_tpo[:, None, :].astype(np.float32), seq_len, 0.0)
            volatility[batch_idx] = _pad_window(window_vol[:, None].astype(np.float32), seq_len, 0.0)
            valid_mask[batch_idx] = _pad_window(window_valid[:, None].astype(np.bool_), seq_len, False)
            timestamps[batch_idx] = _pad_window(window_ts, seq_len, np.datetime64("1970-01-01T00:00:00"))
            direction_labels[batch_idx, 0] = float(label_direction)
            entry_labels[batch_idx, 0] = float(label_entry)
            label_valid_mask[batch_idx, 0] = label_valid

        timeframe_batches[timeframe] = TimeframeSequenceBatch(
            timeframe=timeframe,
            node_names=(symbol,),
            tradable_indices=(0,),
            timestamps=timestamps,
            node_features=_to_device(node_features, torch.float32, device),
            tpo_features=_to_device(tpo_features, torch.float32, device),
            volatility=_to_device(volatility, torch.float32, device),
            valid_mask=_to_device(valid_mask, torch.bool, device),
            market_open_mask=_to_device(market_open_mask, torch.bool, device),
            overlap_mask=_to_device(overlap_mask, torch.bool, device),
            session_codes=_to_device(session_codes, torch.long, device),
            direction_labels=_to_device(direction_labels, torch.float32, device),
            entry_labels=_to_device(entry_labels, torch.float32, device),
            label_valid_mask=_to_device(label_valid_mask, torch.bool, device),
        )

    return SubnetSequenceBatch(
        subnet_name=panels.subnet_name,
        timeframe_batches=timeframe_batches,
        node_names=panels.symbols,
        tradable_node_names=panels.symbols,
        anchor_timestamps=panels.anchor_timestamps[anchor_indices],
    )


def build_fx_sequence_batch_from_panels(
    panels: StagedPanels,
    anchor_indices: np.ndarray,
    seq_lens: dict[str, int],
    device: torch.device,
    include_signal_only_tpo: bool = True,
) -> SubnetSequenceBatch:
    symbols = panels.symbols
    tradable_mask = np.array([symbol in FX_TRADABLE_NAMES for symbol in symbols], dtype=np.bool_)
    tradable_indices = tuple(np.flatnonzero(tradable_mask).tolist())
    timeframe_batches: dict[str, TimeframeSequenceBatch] = {}

    for timeframe, symbol_frames in panels.panels.items():
        full_index = next(iter(symbol_frames.values())).index
        seq_len = int(seq_lens[timeframe])
        end_indices = panels.anchor_lookup[timeframe][anchor_indices]
        batch_size = len(anchor_indices)
        n_nodes = len(symbols)

        node_features = np.zeros((batch_size, seq_len, n_nodes, 14), dtype=np.float32)
        tpo_features = np.zeros((batch_size, seq_len, n_nodes, 8), dtype=np.float32)
        volatility = np.zeros((batch_size, seq_len, n_nodes), dtype=np.float32)
        valid_mask = np.zeros((batch_size, seq_len, n_nodes), dtype=np.bool_)
        market_open_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        overlap_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        session_codes = np.zeros((batch_size, seq_len), dtype=np.int64)
        timestamps = np.zeros((batch_size, seq_len), dtype="datetime64[ns]")
        direction_labels = np.full((batch_size, n_nodes), -1, dtype=np.float32)
        entry_labels = np.zeros((batch_size, n_nodes), dtype=np.float32)
        label_valid_mask = np.zeros((batch_size, n_nodes), dtype=np.bool_)

        source_timeframe = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
        source_frames = panels.panels[source_timeframe]
        coarse_tick_frames = panels.panels.get("M1", {}) if timeframe == "tick" else {}
        coarse_tick_index = next(iter(coarse_tick_frames.values())).index if coarse_tick_frames else None

        for batch_idx, end_idx in enumerate(end_indices):
            end_idx = int(end_idx)
            seq_start = max(0, end_idx - seq_len + 1)
            ext_start = max(0, seq_start - _JIT_BACK_CONTEXT)
            ext_end = end_idx + 1
            local_start = seq_start - ext_start
            local_end = ext_end - ext_start

            ext_ts = full_index[ext_start:ext_end].to_numpy(dtype="datetime64[ns]")
            target_ts = full_index[seq_start : end_idx + 1].to_numpy(dtype="datetime64[ns]")

            ext_open = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "o", np.float32)
            ext_high = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "h", np.float32)
            ext_low = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "l", np.float32)
            ext_close = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "c", np.float32)
            ext_spread = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "sp", np.float32)
            ext_tick = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "tk", np.float32)
            ext_valid = _aligned_matrix(symbol_frames, symbols, ext_start, ext_end, "real", np.bool_)

            window_open = ext_open[local_start:local_end]
            window_high = ext_high[local_start:local_end]
            window_low = ext_low[local_start:local_end]
            window_close = ext_close[local_start:local_end]
            window_spread = ext_spread[local_start:local_end]
            window_tick = ext_tick[local_start:local_end]
            window_valid = ext_valid[local_start:local_end]

            ret_1 = _lagged_logret_matrix(ext_close, 1)[local_start:local_end]
            ret_3 = _lagged_logret_matrix(ext_close, 3)[local_start:local_end]
            atr_norm = _atr_norm_matrix(ext_high, ext_low, ext_close)[local_start:local_end]
            range_norm = ((window_high - window_low) / np.maximum(np.abs(window_close), 1e-8)).astype(np.float32)
            regime_signal = _fx_regime_signal(ret_3, atr_norm)

            ext_session = encode_session_codes(ext_ts)
            ext_session_transition = _fx_session_transition(ext_session)
            window_session = ext_session[local_start:local_end]
            window_session_transition = ext_session_transition[local_start:local_end]
            window_market_open = window_session != SESSION_CODES["closed"]
            window_overlap = window_session == SESSION_CODES["overlap"]

            if timeframe == "tick" and coarse_tick_frames and coarse_tick_index is not None:
                coarse_lookup = coarse_tick_index.searchsorted(pd.Index(target_ts), side="right") - 1
                coarse_lookup = np.clip(coarse_lookup, 0, len(coarse_tick_index) - 1).astype(np.int32)
                coarse_start = max(0, int(coarse_lookup.min()) - _JIT_LAPLACIAN_LOOKBACK)
                coarse_end = int(coarse_lookup.max()) + 1
                coarse_close = _aligned_matrix(coarse_tick_frames, symbols, coarse_start, coarse_end, "c", np.float32)
                coarse_valid = _aligned_matrix(coarse_tick_frames, symbols, coarse_start, coarse_end, "real", np.bool_)
                coarse_lap = _compute_laplacian_residuals(coarse_close, coarse_valid)
                lap_residual = coarse_lap[coarse_lookup - coarse_start]
            else:
                lap_residual = _compute_laplacian_residuals(ext_close, ext_valid)[local_start:local_end]

            window_tpo = np.zeros((len(target_ts), n_nodes, 8), dtype=np.float32)
            window_vol = np.zeros((len(target_ts), n_nodes), dtype=np.float32)
            missing_preswept: list[int] = []
            for col_idx, symbol in enumerate(symbols):
                if not include_signal_only_tpo and not tradable_mask[col_idx]:
                    continue
                preswept = _preswept_tpo_for_target_window(panels, source_timeframe, symbol, target_ts)
                if preswept is None:
                    missing_preswept.append(col_idx)
                    continue
                symbol_tpo, symbol_vol = preswept
                window_tpo[:, col_idx, :] = symbol_tpo
                window_vol[:, col_idx] = symbol_vol
            if len(missing_preswept) == n_nodes:
                if source_timeframe == timeframe:
                    for col_idx, symbol in enumerate(symbols):
                        if not include_signal_only_tpo and not tradable_mask[col_idx]:
                            continue
                        features, vol = compute_tpo_feature_panel(ext_high[:, col_idx], ext_low[:, col_idx], ext_close[:, col_idx])
                        window_tpo[:, col_idx, :] = features[local_start:local_end]
                        window_vol[:, col_idx] = vol[local_start:local_end]
                else:
                    window_tpo, window_vol = _tpo_for_target_window(source_frames, symbols, target_ts, include_signal_only_tpo, tradable_mask)
            elif missing_preswept:
                for col_idx in missing_preswept:
                    if not include_signal_only_tpo and not tradable_mask[col_idx]:
                        continue
                    if source_timeframe == timeframe:
                        features, vol = compute_tpo_feature_panel(ext_high[:, col_idx], ext_low[:, col_idx], ext_close[:, col_idx])
                        window_tpo[:, col_idx, :] = features[local_start:local_end]
                        window_vol[:, col_idx] = vol[local_start:local_end]
                    else:
                        fallback_tpo, fallback_vol = _tpo_for_target_window(
                            source_frames,
                            (symbols[col_idx],),
                            target_ts,
                            True,
                            np.array([True], dtype=np.bool_),
                        )
                        window_tpo[:, col_idx, :] = fallback_tpo[:, 0, :]
                        window_vol[:, col_idx] = fallback_vol[:, 0]

            for col_idx, symbol in enumerate(symbols):
                if not tradable_mask[col_idx]:
                    continue
                if timeframe == "tick" and coarse_tick_frames and coarse_tick_index is not None:
                    label_end = int(coarse_tick_index.searchsorted(target_ts[-1], side="right") - 1)
                    label_end = max(label_end, 0)
                    label_frame = coarse_tick_frames[symbol]
                else:
                    label_end = end_idx
                    label_frame = symbol_frames[symbol]
                label_direction, label_entry, label_valid = _label_at_position(label_frame, label_end, symbol)
                direction_labels[batch_idx, col_idx] = float(label_direction)
                entry_labels[batch_idx, col_idx] = float(label_entry)
                label_valid_mask[batch_idx, col_idx] = label_valid

            packed = np.stack(
                [
                    window_open,
                    window_high,
                    window_low,
                    window_close,
                    window_spread,
                    window_tick,
                    ret_1,
                    ret_3,
                    atr_norm,
                    range_norm,
                    lap_residual.astype(np.float32),
                    regime_signal,
                    np.broadcast_to(window_session[:, None], window_close.shape).astype(np.float32),
                    np.broadcast_to(window_session_transition[:, None], window_close.shape).astype(np.float32),
                ],
                axis=-1,
            ).astype(np.float32)

            node_features[batch_idx] = _pad_window(packed, seq_len, 0.0)
            tpo_features[batch_idx] = _pad_window(window_tpo.astype(np.float32), seq_len, 0.0)
            volatility[batch_idx] = _pad_window(window_vol.astype(np.float32), seq_len, 0.0)
            valid_mask[batch_idx] = _pad_window(window_valid.astype(np.bool_), seq_len, False)
            market_open_mask[batch_idx] = _pad_window(window_market_open.astype(np.bool_), seq_len, False)
            overlap_mask[batch_idx] = _pad_window(window_overlap.astype(np.bool_), seq_len, False)
            session_codes[batch_idx] = _pad_window(window_session.astype(np.int64), seq_len, 0)
            timestamps[batch_idx] = _pad_window(target_ts, seq_len, np.datetime64("1970-01-01T00:00:00"))

        timeframe_batches[timeframe] = TimeframeSequenceBatch(
            timeframe=timeframe,
            node_names=symbols,
            tradable_indices=tradable_indices,
            timestamps=timestamps,
            node_features=_to_device(node_features, torch.float32, device),
            tpo_features=_to_device(tpo_features, torch.float32, device),
            volatility=_to_device(volatility, torch.float32, device),
            valid_mask=_to_device(valid_mask, torch.bool, device),
            market_open_mask=_to_device(market_open_mask, torch.bool, device),
            overlap_mask=_to_device(overlap_mask, torch.bool, device),
            session_codes=_to_device(session_codes, torch.long, device),
            direction_labels=_to_device(direction_labels, torch.float32, device),
            entry_labels=_to_device(entry_labels, torch.float32, device),
            label_valid_mask=_to_device(label_valid_mask, torch.bool, device),
        )

    return SubnetSequenceBatch(
        subnet_name=panels.subnet_name,
        timeframe_batches=timeframe_batches,
        node_names=symbols,
        tradable_node_names=FX_TRADABLE_NAMES,
        anchor_timestamps=panels.anchor_timestamps[anchor_indices],
    )
