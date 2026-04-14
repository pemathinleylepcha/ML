from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research_dataset import SESSION_CODES, encode_session_codes
from staged_v5.config import DEFAULT_SEQ_LENS
from staged_v5.contracts import TimeframeSequenceBatch
from staged_v5.data.memory_budget import compute_tick_chunk_size, get_available_ram_mb, get_available_vram_mb
from staged_v5.data.tick_features import build_tick_raw_features
from staged_v5.data.tpo_features import compute_tpo_feature_panel
from staged_v5.execution_gate.features import TickProxyStore


_LOGGER = logging.getLogger(__name__)
_REQUIRED_COLUMNS = (
    "dt",
    "o",
    "h",
    "l",
    "c",
    "sp",
    "tk",
    "tick_velocity",
    "spread_z",
    "bid_ask_imbalance",
    "price_velocity",
)


def _pad_window(values: np.ndarray, seq_len: int, fill_value=0) -> np.ndarray:
    out = np.full((seq_len,) + values.shape[1:], fill_value, dtype=values.dtype)
    if len(values):
        out[-len(values) :] = values
    return out


def _session_transition(session_codes: np.ndarray) -> np.ndarray:
    out = np.zeros(len(session_codes), dtype=np.int32)
    if len(session_codes) > 1:
        out[1:] = (session_codes[1:] != session_codes[:-1]).astype(np.int32)
    return out


def _read_csv_range(path: Path, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    usecols = lambda c: c in _REQUIRED_COLUMNS or c in {"datetime", "bar_time", "time", "open", "high", "low", "close", "spread", "tick_volume", "volume"}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, engine="python", on_bad_lines="skip"):
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _normalize_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
    frame = frame.rename(columns=rename)
    missing = [col for col in _REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required tick columns {missing}")
    frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce", utc=False)
    frame = frame.dropna(subset=["dt"]).copy()
    for col in _REQUIRED_COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame.sort_values("dt").drop_duplicates("dt")


def _load_time_range(path: Path, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = _read_csv_range(path, start_ts, end_ts)
    frame = _normalize_tick_frame(frame)
    if start_ts is not None:
        frame = frame[frame["dt"] >= start_ts]
    if end_ts is not None:
        frame = frame[frame["dt"] <= end_ts]
    return frame


class JITTickLoader:
    """Chunk-aware tick sequence loader aligned to TimeframeSequenceBatch."""

    def __init__(
        self,
        tick_root: str | Path,
        node_names: tuple[str, ...],
        *,
        chunk_bars: int | None = None,
        tick_seq_len: int | None = None,
        budget_fraction_ram: float = 0.25,
        budget_fraction_vram: float = 0.15,
        min_chunk_bars: int = 1_000,
        max_chunk_bars: int = 2_000_000,
        device_type: str = "cpu",
    ):
        self.tick_root = Path(tick_root)
        self.node_names = tuple(node_names)
        self.tick_seq_len = int(tick_seq_len or DEFAULT_SEQ_LENS["tick"])
        self.tick_store = TickProxyStore(self.tick_root)
        available_ram_mb = get_available_ram_mb()
        ram_chunk = compute_tick_chunk_size(
            available_ram_mb=available_ram_mb,
            n_nodes=len(self.node_names),
            n_features=22,
            budget_fraction=budget_fraction_ram,
            min_chunk_bars=min_chunk_bars,
            max_chunk_bars=max_chunk_bars,
        )
        # On CUDA, also consider VRAM budget and take the larger of the two
        # since tick chunks are loaded to CPU first then transferred to GPU
        if device_type == "cuda":
            vram_mb = get_available_vram_mb(device_type)
            vram_chunk = compute_tick_chunk_size(
                available_ram_mb=vram_mb,
                n_nodes=len(self.node_names),
                n_features=22,
                budget_fraction=budget_fraction_vram,
                min_chunk_bars=min_chunk_bars,
                max_chunk_bars=max_chunk_bars,
            )
            auto_chunk = max(ram_chunk, vram_chunk)
        else:
            auto_chunk = ram_chunk
        self.chunk_bars = int(chunk_bars if chunk_bars is not None else auto_chunk)
        self._reference_symbol = self.node_names[0]
        self._reference_frame = self.tick_store.get_frame(self._reference_symbol)
        self._chunk_load_count = 0
        self._chunk_start_idx: int | None = None
        self._chunk_end_idx: int | None = None
        self._chunk_index: pd.DatetimeIndex | None = None
        self._chunk_timestamps: np.ndarray | None = None
        self._node_feature_cache: dict[str, np.ndarray] = {}
        self._tpo_feature_cache: dict[str, np.ndarray] = {}
        self._volatility_cache: dict[str, np.ndarray] = {}
        self._valid_mask_cache: dict[str, np.ndarray] = {}
        self._market_open_mask: np.ndarray | None = None
        self._overlap_mask: np.ndarray | None = None
        self._session_codes: np.ndarray | None = None
        _LOGGER.info("JITTickLoader initialized chunk_bars=%d nodes=%d", self.chunk_bars, len(self.node_names))

    def _resolve_chunk_bounds(self, anchor_timestamps: np.ndarray) -> tuple[int, int]:
        anchor_index = pd.DatetimeIndex(pd.to_datetime(anchor_timestamps))
        positions = self._reference_frame.index.searchsorted(anchor_index, side="right") - 1
        positions = np.clip(positions, 0, max(len(self._reference_frame.index) - 1, 0))
        min_pos = int(np.min(positions))
        max_pos = int(np.max(positions))
        base_start = (min_pos // self.chunk_bars) * self.chunk_bars
        start_idx = max(0, base_start - self.tick_seq_len + 1)
        end_idx = min(len(self._reference_frame.index), base_start + self.chunk_bars)
        while max_pos >= end_idx and end_idx < len(self._reference_frame.index):
            end_idx = min(len(self._reference_frame.index), end_idx + self.chunk_bars)
        return start_idx, end_idx

    def _align_symbol_frame(self, frame: pd.DataFrame, reference_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, np.ndarray]:
        if len(frame) == 0:
            empty = pd.DataFrame(index=reference_index, columns=_REQUIRED_COLUMNS[1:])
            valid_mask = np.zeros(len(reference_index), dtype=np.bool_)
            return empty, valid_mask
        indexed = frame.set_index("dt")[list(_REQUIRED_COLUMNS[1:])].sort_index()
        aligned = indexed.reindex(reference_index)
        valid_mask = aligned["o"].notna().to_numpy(dtype=np.bool_)
        price_cols = ["o", "h", "l", "c"]
        aligned[price_cols] = aligned[price_cols].ffill().bfill().fillna(0.0)
        fill_zero_cols = ["sp", "tk", "tick_velocity", "spread_z", "bid_ask_imbalance", "price_velocity"]
        aligned[fill_zero_cols] = aligned[fill_zero_cols].fillna(0.0)
        return aligned, valid_mask

    def _load_chunk(self, start_idx: int, end_idx: int) -> None:
        reference_index = self._reference_frame.index[start_idx:end_idx]
        if len(reference_index) == 0:
            raise ValueError("Cannot load empty tick chunk")
        start_ts = pd.Timestamp(reference_index[0])
        end_ts = pd.Timestamp(reference_index[-1])

        self._chunk_index = pd.DatetimeIndex(reference_index)
        self._chunk_timestamps = reference_index.to_numpy(dtype="datetime64[ns]")
        self._node_feature_cache = {}
        self._tpo_feature_cache = {}
        self._volatility_cache = {}
        self._valid_mask_cache = {}

        session_codes = encode_session_codes(reference_index.to_numpy(dtype="datetime64[ns]"))
        self._session_codes = np.asarray(session_codes, dtype=np.int64)
        self._market_open_mask = self._session_codes != SESSION_CODES["closed"]
        self._overlap_mask = self._session_codes == SESSION_CODES["overlap"]
        session_transition = _session_transition(self._session_codes)

        for symbol in self.node_names:
            path = self.tick_store.find_path(symbol)
            if path is None:
                raise FileNotFoundError(f"Could not find 1000ms proxy bars for {symbol} under {self.tick_root}")
            frame = _load_time_range(path, start_ts, end_ts)
            aligned, valid_mask = self._align_symbol_frame(frame, self._chunk_index)
            features, _session_codes = build_tick_raw_features(
                aligned.reset_index().rename(columns={"index": "dt"}),
                session_code=0,
            )
            features[:, 12] = self._session_codes.astype(np.float32)
            features[:, 13] = session_transition.astype(np.float32)
            tpo_features, volatility = compute_tpo_feature_panel(
                aligned["h"].to_numpy(dtype=np.float32),
                aligned["l"].to_numpy(dtype=np.float32),
                aligned["c"].to_numpy(dtype=np.float32),
            )
            self._node_feature_cache[symbol] = features
            self._tpo_feature_cache[symbol] = tpo_features.astype(np.float32)
            self._volatility_cache[symbol] = volatility.astype(np.float32)
            self._valid_mask_cache[symbol] = valid_mask

        self._chunk_start_idx = start_idx
        self._chunk_end_idx = end_idx
        self._chunk_load_count += 1
        _LOGGER.info(
            "JITTickLoader loaded chunk [%s, %s] (%d bars) for %d nodes (load #%d)",
            start_ts,
            end_ts,
            len(reference_index),
            len(self.node_names),
            self._chunk_load_count,
        )

    def _ensure_chunk(self, anchor_timestamps: np.ndarray) -> None:
        start_idx, end_idx = self._resolve_chunk_bounds(anchor_timestamps)
        if (
            self._chunk_start_idx is None
            or self._chunk_end_idx is None
            or start_idx < self._chunk_start_idx
            or end_idx > self._chunk_end_idx
        ):
            self._load_chunk(start_idx, end_idx)

    def get_tick_sequence_batch(
        self,
        anchor_timestamps: np.ndarray,
        device_type: str = "cpu",
    ) -> TimeframeSequenceBatch:
        anchor_timestamps = np.asarray(anchor_timestamps)
        self._ensure_chunk(anchor_timestamps)
        assert self._chunk_index is not None
        assert self._chunk_timestamps is not None
        assert self._market_open_mask is not None
        assert self._overlap_mask is not None
        assert self._session_codes is not None

        batch_size = len(anchor_timestamps)
        seq_len = self.tick_seq_len
        n_nodes = len(self.node_names)
        node_features = np.zeros((batch_size, seq_len, n_nodes, 14), dtype=np.float32)
        tpo_features = np.zeros((batch_size, seq_len, n_nodes, 8), dtype=np.float32)
        volatility = np.zeros((batch_size, seq_len, n_nodes), dtype=np.float32)
        valid_mask = np.zeros((batch_size, seq_len, n_nodes), dtype=np.bool_)
        market_open_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        overlap_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        session_codes = np.zeros((batch_size, seq_len), dtype=np.int64)
        timestamps = np.full((batch_size, seq_len), np.datetime64("1970-01-01T00:00:00"), dtype="datetime64[ns]")

        anchor_index = pd.DatetimeIndex(pd.to_datetime(anchor_timestamps))
        end_positions = self._chunk_index.searchsorted(anchor_index, side="right") - 1

        for batch_idx, end_pos in enumerate(end_positions):
            end_pos = int(end_pos)
            if end_pos < 0:
                continue
            window_timestamps = self._chunk_timestamps[max(0, end_pos - seq_len + 1) : end_pos + 1]
            timestamps[batch_idx] = _pad_window(window_timestamps, seq_len, np.datetime64("1970-01-01T00:00:00"))
            market_open_mask[batch_idx] = _pad_window(
                self._market_open_mask[max(0, end_pos - seq_len + 1) : end_pos + 1, None],
                seq_len,
                False,
            ).squeeze(-1)
            overlap_mask[batch_idx] = _pad_window(
                self._overlap_mask[max(0, end_pos - seq_len + 1) : end_pos + 1, None],
                seq_len,
                False,
            ).squeeze(-1)
            session_codes[batch_idx] = _pad_window(
                self._session_codes[max(0, end_pos - seq_len + 1) : end_pos + 1, None],
                seq_len,
                0,
            ).squeeze(-1)

            for node_idx, symbol in enumerate(self.node_names):
                symbol_features = self._node_feature_cache[symbol]
                symbol_tpo = self._tpo_feature_cache[symbol]
                symbol_vol = self._volatility_cache[symbol]
                symbol_valid = self._valid_mask_cache[symbol]
                start_pos = max(0, end_pos - seq_len + 1)
                node_features[batch_idx, :, node_idx, :] = _pad_window(symbol_features[start_pos : end_pos + 1], seq_len, 0.0)
                tpo_features[batch_idx, :, node_idx, :] = _pad_window(symbol_tpo[start_pos : end_pos + 1], seq_len, 0.0)
                volatility[batch_idx, :, node_idx] = _pad_window(symbol_vol[start_pos : end_pos + 1, None], seq_len, 0.0).squeeze(-1)
                valid_mask[batch_idx, :, node_idx] = _pad_window(symbol_valid[start_pos : end_pos + 1, None], seq_len, False).squeeze(-1)

        return TimeframeSequenceBatch(
            timeframe="tick",
            node_names=self.node_names,
            tradable_indices=tuple(range(len(self.node_names))),
            timestamps=timestamps,
            node_features=node_features,
            tpo_features=tpo_features,
            volatility=volatility,
            valid_mask=valid_mask,
            market_open_mask=market_open_mask,
            overlap_mask=overlap_mask,
            session_codes=session_codes,
        )
