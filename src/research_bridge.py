from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research_dataset import CanonicalResearchDataset, estimate_spread_cost
from universe import ENERGY, FX_PAIRS, INDICES, METALS

FX_MAJORS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
}
FX_CROSSES = set(FX_PAIRS) - FX_MAJORS
BUCKETS = {
    "BTC": {"BTCUSD"},
    "FX_MAJORS": set(FX_MAJORS),
    "FX_CROSSES": set(FX_CROSSES),
    "INDICES": set(INDICES),
    "METALS": set(METALS),
    "ENERGY": set(ENERGY),
}
STATS = ("trend", "dispersion", "stress")


@dataclass(slots=True)
class BridgeContext:
    features: dict[str, float]

    def to_array(self, feature_names: list[str]) -> np.ndarray:
        return np.array([self.features.get(name, 0.0) for name in feature_names], dtype=np.float32)


class BridgeContextEncoder:
    def __init__(self, lookback_bars: int = 8):
        self.lookback_bars = lookback_bars
        self._feature_names = [
            f"{tf}_{bucket}_{stat}"
            for tf in ("M5", "M15", "H1", "H4")
            for bucket in BUCKETS
            for stat in STATS
        ]
        self._cached_dataset_id: int | None = None
        self._cached_matrix: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _bucket_members(self, dataset: CanonicalResearchDataset, tf_name: str, bucket_name: str) -> list[str]:
        tf_symbols = set(dataset.tf_data[tf_name].keys())
        return sorted(tf_symbols.intersection(BUCKETS[bucket_name]))

    @staticmethod
    def _sample_on_base(series: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if len(series) == 0:
            return np.zeros(len(positions), dtype=np.float32)
        safe_positions = np.clip(positions, 0, len(series) - 1)
        return series[safe_positions].astype(np.float32, copy=False)

    @staticmethod
    def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = mask.sum(axis=1).astype(np.float32)
        safe = np.where(mask, values, 0.0).astype(np.float32, copy=False)
        denom = np.maximum(counts, 1.0)
        mean = safe.sum(axis=1, dtype=np.float64) / denom
        second_moment = (safe * safe).sum(axis=1, dtype=np.float64) / denom
        var = np.maximum(second_moment - mean * mean, 0.0)
        mean = mean.astype(np.float32)
        std = np.sqrt(var, dtype=np.float64).astype(np.float32)
        empty = counts <= 0
        mean[empty] = 0.0
        std[empty] = 0.0
        return mean, std

    def build_feature_matrix(self, dataset: CanonicalResearchDataset) -> np.ndarray:
        dataset_id = id(dataset)
        if self._cached_dataset_id == dataset_id and self._cached_matrix is not None:
            return self._cached_matrix

        n_bars = dataset.n_bars
        matrix = np.zeros((n_bars, len(self._feature_names)), dtype=np.float32)
        col = 0

        for tf_name in dataset.timeframes:
            tf_positions = dataset.tf_index_for_base[tf_name]
            symbols = dataset.tf_data[tf_name]
            returns_by_symbol: dict[str, np.ndarray] = {}
            stress_by_symbol: dict[str, np.ndarray] = {}
            real_by_symbol: dict[str, np.ndarray] = {}

            for symbol, frame in symbols.items():
                close = np.maximum(frame["c"].astype(np.float64), 1e-10)
                ret = np.zeros(len(close), dtype=np.float32)
                if len(close) > 1:
                    ret[1:] = np.diff(np.log(close)).astype(np.float32)

                spread_cost = np.array(
                    [estimate_spread_cost(symbol, float(sp), float(cp)) for sp, cp in zip(frame["sp"], frame["c"])],
                    dtype=np.float32,
                )
                tick = frame["tk"].astype(np.float64)
                stress = (spread_cost / close + 1.0 / np.sqrt(tick + 1.0)).astype(np.float32)

                returns_by_symbol[symbol] = ret
                stress_by_symbol[symbol] = stress
                real_by_symbol[symbol] = frame["real"].astype(np.bool_, copy=False)

            for bucket_name in BUCKETS:
                members = self._bucket_members(dataset, tf_name, bucket_name)
                if members:
                    ret_stack = np.column_stack([
                        self._sample_on_base(returns_by_symbol[symbol], tf_positions) for symbol in members
                    ])
                    stress_stack = np.column_stack([
                        self._sample_on_base(stress_by_symbol[symbol], tf_positions) for symbol in members
                    ])
                    real_stack = np.column_stack([
                        self._sample_on_base(real_by_symbol[symbol].astype(np.float32), tf_positions) > 0.5
                        for symbol in members
                    ])
                    trend, dispersion = self._masked_mean_std(ret_stack, real_stack)
                    stress_mean, _ = self._masked_mean_std(stress_stack, real_stack)
                    matrix[:, col] = trend
                    matrix[:, col + 1] = dispersion
                    matrix[:, col + 2] = stress_mean
                else:
                    matrix[:, col: col + 3] = 0.0
                col += 3

        self._cached_dataset_id = dataset_id
        self._cached_matrix = matrix
        return matrix

    def encode(self, dataset: CanonicalResearchDataset, base_bar_idx: int) -> BridgeContext:
        row = self.build_feature_matrix(dataset)[base_bar_idx]
        features = {name: float(row[idx]) for idx, name in enumerate(self._feature_names)}
        return BridgeContext(features=features)
