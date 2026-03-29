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

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _bucket_members(self, dataset: CanonicalResearchDataset, tf_name: str, bucket_name: str) -> list[str]:
        tf_symbols = set(dataset.tf_data[tf_name].keys())
        return sorted(tf_symbols.intersection(BUCKETS[bucket_name]))

    def encode(self, dataset: CanonicalResearchDataset, base_bar_idx: int) -> BridgeContext:
        features: dict[str, float] = {}
        for tf_name in dataset.timeframes:
            tf_bar_idx = int(dataset.tf_index_for_base[tf_name][base_bar_idx])
            for bucket_name in BUCKETS:
                members = self._bucket_members(dataset, tf_name, bucket_name)
                returns = []
                stresses = []
                for symbol in members:
                    frame = dataset.tf_data[tf_name][symbol]
                    if tf_bar_idx <= 0 or tf_bar_idx >= len(frame["c"]):
                        continue
                    lo = max(1, tf_bar_idx - self.lookback_bars + 1)
                    close = frame["c"][lo - 1: tf_bar_idx + 1].astype(np.float64)
                    if len(close) < 2:
                        continue
                    log_rets = np.diff(np.log(np.maximum(close, 1e-10)))
                    returns.append(float(log_rets[-1]))
                    spread_cost = estimate_spread_cost(symbol, float(frame["sp"][tf_bar_idx]), float(frame["c"][tf_bar_idx]))
                    tick = float(frame["tk"][tf_bar_idx])
                    stress = spread_cost / max(float(frame["c"][tf_bar_idx]), 1e-10) + 1.0 / np.sqrt(tick + 1.0)
                    stresses.append(float(stress))
                key_prefix = f"{tf_name}_{bucket_name}"
                if returns:
                    ret_arr = np.array(returns, dtype=np.float64)
                    features[f"{key_prefix}_trend"] = float(ret_arr.mean())
                    features[f"{key_prefix}_dispersion"] = float(ret_arr.std(ddof=0))
                    features[f"{key_prefix}_stress"] = float(np.mean(stresses)) if stresses else 0.0
                else:
                    features[f"{key_prefix}_trend"] = 0.0
                    features[f"{key_prefix}_dispersion"] = 0.0
                    features[f"{key_prefix}_stress"] = 0.0
        return BridgeContext(features=features)
