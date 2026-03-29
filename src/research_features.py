from __future__ import annotations

from collections import deque

import numpy as np

from math_engine import MathEngine, MathState
from research_bridge import BridgeContextEncoder
from research_dataset import CanonicalResearchDataset, estimate_spread_cost


def _safe_logret(close: np.ndarray, lag: int = 1) -> float:
    if len(close) <= lag:
        return 0.0
    return float(np.log(max(close[-1], 1e-10) / max(close[-1 - lag], 1e-10)))


class CompactFeatureExtractor:
    def __init__(self, symbols: list[str], bridge_encoder: BridgeContextEncoder):
        self.symbols = list(symbols)
        self.bridge_encoder = bridge_encoder
        self._spread_windows = {symbol: deque(maxlen=64) for symbol in self.symbols}
        self._tick_windows = {symbol: deque(maxlen=64) for symbol in self.symbols}
        self._residual_streak = {symbol: 0 for symbol in self.symbols}
        self._residual_sign = {symbol: 0 for symbol in self.symbols}
        self._feature_names = None
        self._state_symbols: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Feature names are available after the first extraction call")
        return list(self._feature_names)

    def _zscore(self, value: float, window: deque[float]) -> float:
        arr = np.array(window) if window else np.array([value], dtype=np.float64)
        z = float((value - arr.mean()) / (arr.std() + 1e-8))
        window.append(value)
        return z

    def _update_streak(self, symbol: str, residual: float, dead_zone: float = 1e-5) -> int:
        if abs(residual) <= dead_zone:
            self._residual_streak[symbol] = 0
            self._residual_sign[symbol] = 0
            return 0
        sign = 1 if residual > 0 else -1
        if sign == self._residual_sign[symbol]:
            self._residual_streak[symbol] += 1
        else:
            self._residual_sign[symbol] = sign
            self._residual_streak[symbol] = 1
        return self._residual_streak[symbol]

    def build_state_series(self, dataset: CanonicalResearchDataset) -> list[MathState]:
        base_tf = dataset.base_timeframe
        self._state_symbols = [symbol for symbol in self.symbols if symbol in dataset.tf_data[base_tf]]
        engine = MathEngine(n_pairs=len(self._state_symbols))
        states: list[MathState] = []

        for bar_idx in range(dataset.n_bars):
            rets = np.zeros(len(self._state_symbols), dtype=np.float64)
            for col, symbol in enumerate(self._state_symbols):
                close = dataset.tf_data[base_tf][symbol]["c"]
                if bar_idx > 0 and bar_idx < len(close):
                    rets[col] = np.log(max(float(close[bar_idx]), 1e-10) / max(float(close[bar_idx - 1]), 1e-10))
            states.append(engine.update(rets))
        return states

    def extract_pair_row(
        self,
        dataset: CanonicalResearchDataset,
        symbol: str,
        bar_idx: int,
        math_state: MathState,
        bridge_context,
    ) -> dict[str, float]:
        frame = dataset.tf_data[dataset.base_timeframe][symbol]
        lo = max(0, bar_idx - 24)
        close = frame["c"][lo: bar_idx + 1].astype(np.float64)
        high = frame["h"][lo: bar_idx + 1].astype(np.float64)
        low = frame["l"][lo: bar_idx + 1].astype(np.float64)
        open_ = frame["o"][lo: bar_idx + 1].astype(np.float64)

        spread_cost = estimate_spread_cost(symbol, float(frame["sp"][bar_idx]), float(frame["c"][bar_idx]))
        spread_z = self._zscore(spread_cost, self._spread_windows[symbol])
        tick_z = self._zscore(float(frame["tk"][bar_idx]), self._tick_windows[symbol])

        range_now = float(high[-1] - low[-1]) if len(high) else 0.0
        atr_proxy = float(np.mean(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))) if len(close) > 2 else range_now

        residual = 0.0
        if math_state.valid and symbol in self._state_symbols:
            residual = float(math_state.residuals[self._state_symbols.index(symbol)])
        streak = float(self._update_streak(symbol, residual))

        regime_map = {"LOW_VOL": 0.0, "NORMAL": 1.0, "TRANSITIONAL": 2.0, "HIGH_STRESS": 3.0, "FRAGMENTED": 4.0}
        session_code = float(dataset.session_codes[bar_idx])

        features = {
            "local_ret_1": _safe_logret(close, 1),
            "local_ret_3": _safe_logret(close, min(3, len(close) - 1)),
            "local_ret_6": _safe_logret(close, min(6, len(close) - 1)),
            "local_atr_norm": float(atr_proxy / max(abs(float(close[-1])), 1e-10)),
            "local_range_norm": float(range_now / max(abs(float(close[-1])), 1e-10)),
            "local_body_ratio": float(abs(close[-1] - open_[-1]) / max(range_now, 1e-10)),
            "local_spread_z": spread_z,
            "local_tick_z": tick_z,
            "local_liquidity_stress": float(spread_cost / max(abs(float(close[-1])), 1e-10) + 1.0 / np.sqrt(float(frame["tk"][bar_idx]) + 1.0)),
            "local_lap_residual": residual,
            "local_residual_streak": streak,
            "local_regime": regime_map.get(math_state.regime if math_state.valid else "NORMAL", 1.0),
            "local_session_code": session_code,
        }
        features.update(bridge_context.features)
        if self._feature_names is None:
            self._feature_names = list(features.keys())
        return features
