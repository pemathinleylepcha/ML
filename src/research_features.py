from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from math_engine import MathEngine
from research_bridge import BridgeContextEncoder
from research_dataset import CanonicalResearchDataset
from universe import PIP_SIZES

REGIME_MAP = {"LOW_VOL": 0.0, "NORMAL": 1.0, "TRANSITIONAL": 2.0, "HIGH_STRESS": 3.0, "FRAGMENTED": 4.0}
LOCAL_FEATURE_NAMES = [
    "local_ret_1",
    "local_ret_3",
    "local_ret_6",
    "local_atr_norm",
    "local_range_norm",
    "local_body_ratio",
    "local_spread_z",
    "local_tick_z",
    "local_liquidity_stress",
    "local_lap_residual",
    "local_residual_streak",
    "local_regime",
    "local_session_code",
]


@dataclass(slots=True)
class StateFeatureCache:
    residuals: np.ndarray
    regime_codes: np.ndarray


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    vals = values.astype(np.float64, copy=False)
    csum = np.concatenate(([0.0], np.cumsum(vals)))
    end = np.arange(1, len(vals) + 1, dtype=np.int64)
    start = np.maximum(0, end - window)
    counts = end - start
    out = (csum[end] - csum[start]) / np.maximum(counts, 1)
    return out.astype(np.float32)


def _rolling_zscore_from_past(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    vals = values.astype(np.float64, copy=False)
    csum = np.concatenate(([0.0], np.cumsum(vals)))
    csum_sq = np.concatenate(([0.0], np.cumsum(vals * vals)))
    idx = np.arange(len(vals), dtype=np.int64)
    start = np.maximum(0, idx - window)
    count = idx - start

    sums = csum[idx] - csum[start]
    sums_sq = csum_sq[idx] - csum_sq[start]
    means = np.divide(sums, count, out=np.zeros(len(vals), dtype=np.float64), where=count > 0)
    variances = np.divide(sums_sq, count, out=np.zeros(len(vals), dtype=np.float64), where=count > 0) - means * means
    variances = np.maximum(variances, 0.0)

    z = np.zeros(len(vals), dtype=np.float64)
    mask = count > 0
    z[mask] = (vals[mask] - means[mask]) / np.sqrt(variances[mask] + 1e-8)
    return z.astype(np.float32)


def _lagged_logret_array(close: np.ndarray, lag: int) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return out
    safe = np.maximum(close.astype(np.float64, copy=False), 1e-10)
    idx = np.arange(n, dtype=np.int64)
    prev_idx = np.maximum(0, idx - lag)
    valid = idx > 0
    out[valid] = np.log(safe[valid] / safe[prev_idx[valid]]).astype(np.float32)
    return out


def _spread_cost_array(symbol: str, spread_raw: np.ndarray) -> np.ndarray:
    pip = float(PIP_SIZES.get(symbol, 1e-4))
    spread = spread_raw.astype(np.float64, copy=False)
    return np.where(
        spread <= 0.0,
        0.0,
        np.where(spread < pip * 50.0, spread, spread * (pip / 10.0)),
    ).astype(np.float32)


def _residual_streak_array(residuals: np.ndarray, dead_zone: float = 1e-5) -> np.ndarray:
    streak = np.zeros(len(residuals), dtype=np.float32)
    last_sign = 0
    current = 0
    for idx, residual in enumerate(residuals):
        if abs(float(residual)) <= dead_zone:
            last_sign = 0
            current = 0
        else:
            sign = 1 if residual > 0 else -1
            if sign == last_sign:
                current += 1
            else:
                last_sign = sign
                current = 1
        streak[idx] = float(current)
    return streak


class CompactFeatureExtractor:
    def __init__(self, symbols: list[str], bridge_encoder: BridgeContextEncoder):
        self.symbols = list(symbols)
        self.bridge_encoder = bridge_encoder
        self._feature_names = list(LOCAL_FEATURE_NAMES)
        self._state_symbols: list[str] = []
        self._state_symbol_to_idx: dict[str, int] = {}
        self._static_feature_cache: dict[str, dict[str, np.ndarray]] = {}

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def all_feature_names(self, use_bridge: bool) -> list[str]:
        if not use_bridge:
            return list(self._feature_names)
        return list(self._feature_names) + self.bridge_encoder.feature_names

    def build_state_features(self, dataset: CanonicalResearchDataset) -> StateFeatureCache:
        base_tf = dataset.base_timeframe
        self._state_symbols = [symbol for symbol in self.symbols if symbol in dataset.tf_data[base_tf]]
        self._state_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self._state_symbols)}
        engine = MathEngine(n_pairs=len(self._state_symbols))
        n_bars = dataset.n_bars
        returns_matrix = np.zeros((n_bars, len(self._state_symbols)), dtype=np.float64)

        for col, symbol in enumerate(self._state_symbols):
            close = np.maximum(dataset.tf_data[base_tf][symbol]["c"].astype(np.float64), 1e-10)
            limit = min(n_bars, len(close))
            if limit > 1:
                returns_matrix[1:limit, col] = np.diff(np.log(close[:limit]))

        residuals = np.zeros((n_bars, len(self._state_symbols)), dtype=np.float32)
        regime_codes = np.full(n_bars, REGIME_MAP["NORMAL"], dtype=np.float32)
        for bar_idx in range(n_bars):
            state = engine.update(returns_matrix[bar_idx])
            if state.valid:
                residuals[bar_idx] = state.residuals.astype(np.float32, copy=False)
                regime_codes[bar_idx] = REGIME_MAP.get(state.regime, REGIME_MAP["NORMAL"])
        return StateFeatureCache(residuals=residuals, regime_codes=regime_codes)

    def prepare_static_feature_cache(
        self,
        dataset: CanonicalResearchDataset,
        state_features: StateFeatureCache,
    ) -> dict[str, dict[str, np.ndarray]]:
        if self._static_feature_cache:
            return self._static_feature_cache

        base_tf = dataset.base_timeframe
        for symbol in self.symbols:
            if symbol not in dataset.tf_data[base_tf]:
                continue

            frame = dataset.tf_data[base_tf][symbol]
            close = frame["c"].astype(np.float64)
            high = frame["h"].astype(np.float64)
            low = frame["l"].astype(np.float64)
            open_ = frame["o"].astype(np.float64)
            tick = frame["tk"].astype(np.float64)
            safe_close = np.maximum(np.abs(close), 1e-10)
            range_now = (high - low).astype(np.float32)

            tr = np.zeros(len(close), dtype=np.float32)
            if len(close) > 1:
                tr[1:] = np.maximum.reduce([
                    high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ]).astype(np.float32)
            atr_proxy = range_now.astype(np.float32, copy=True)
            if len(close) > 2:
                atr_proxy[2:] = _rolling_mean(tr[1:], window=24)[1:]

            spread_cost = _spread_cost_array(symbol, frame["sp"])
            residual = np.zeros(len(close), dtype=np.float32)
            if symbol in self._state_symbol_to_idx:
                residual = state_features.residuals[:, self._state_symbol_to_idx[symbol]]
            streak = _residual_streak_array(residual)

            self._static_feature_cache[symbol] = {
                "local_ret_1": _lagged_logret_array(close, lag=1),
                "local_ret_3": _lagged_logret_array(close, lag=3),
                "local_ret_6": _lagged_logret_array(close, lag=6),
                "local_atr_norm": (atr_proxy / safe_close).astype(np.float32),
                "local_range_norm": (range_now / safe_close).astype(np.float32),
                "local_body_ratio": (np.abs(close - open_) / np.maximum(range_now, 1e-10)).astype(np.float32),
                "local_spread_z": _rolling_zscore_from_past(spread_cost, window=64),
                "local_tick_z": _rolling_zscore_from_past(tick, window=64),
                "local_liquidity_stress": (spread_cost / safe_close + 1.0 / np.sqrt(tick + 1.0)).astype(np.float32),
                "local_lap_residual": residual.astype(np.float32, copy=False),
                "local_residual_streak": streak,
            }

        return self._static_feature_cache

    def build_pair_matrix(
        self,
        dataset: CanonicalResearchDataset,
        symbol: str,
        bar_indices: np.ndarray,
        state_features: StateFeatureCache,
        bridge_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        cache = self.prepare_static_feature_cache(dataset, state_features)[symbol]
        local_matrix = np.column_stack([
            cache["local_ret_1"][bar_indices],
            cache["local_ret_3"][bar_indices],
            cache["local_ret_6"][bar_indices],
            cache["local_atr_norm"][bar_indices],
            cache["local_range_norm"][bar_indices],
            cache["local_body_ratio"][bar_indices],
            cache["local_spread_z"][bar_indices],
            cache["local_tick_z"][bar_indices],
            cache["local_liquidity_stress"][bar_indices],
            cache["local_lap_residual"][bar_indices],
            cache["local_residual_streak"][bar_indices],
            state_features.regime_codes[bar_indices],
            dataset.session_codes[bar_indices].astype(np.float32),
        ]).astype(np.float32, copy=False)
        if bridge_matrix is None:
            return local_matrix
        return np.hstack([local_matrix, bridge_matrix[bar_indices]]).astype(np.float32, copy=False)
