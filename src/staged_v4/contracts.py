from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TimeframeFeatureBatch:
    timeframe: str
    timestamps: np.ndarray
    node_names: tuple[str, ...]
    tradable_mask: np.ndarray
    node_features: np.ndarray
    tpo_features: np.ndarray
    volatility: np.ndarray
    valid_mask: np.ndarray
    market_open_mask: np.ndarray
    overlap_mask: np.ndarray
    session_codes: np.ndarray
    direction_labels: np.ndarray | None = None
    entry_labels: np.ndarray | None = None
    label_valid_mask: np.ndarray | None = None


@dataclass(slots=True)
class BTCFeatureBatch:
    anchor_timeframe: str
    anchor_timestamps: np.ndarray
    timeframe_batches: dict[str, TimeframeFeatureBatch]
    node_names: tuple[str, ...]
    anchor_lookup: dict[str, np.ndarray]


@dataclass(slots=True)
class FXFeatureBatch:
    anchor_timeframe: str
    anchor_timestamps: np.ndarray
    timeframe_batches: dict[str, TimeframeFeatureBatch]
    node_names: tuple[str, ...]
    tradable_node_names: tuple[str, ...]
    anchor_lookup: dict[str, np.ndarray]

    @property
    def tradable_indices(self) -> tuple[int, ...]:
        lookup = {name: idx for idx, name in enumerate(self.node_names)}
        return tuple(lookup[name] for name in self.tradable_node_names if name in lookup)


@dataclass(slots=True)
class BridgeBatch:
    timeframe: str
    fx_timestamps: np.ndarray
    btc_index_for_fx: np.ndarray
    overlap_mask: np.ndarray


@dataclass(slots=True)
class TimeframeSequenceBatch:
    timeframe: str
    node_names: tuple[str, ...]
    tradable_indices: tuple[int, ...]
    timestamps: np.ndarray
    node_features: Any
    tpo_features: Any
    volatility: Any
    valid_mask: Any
    market_open_mask: Any
    overlap_mask: Any
    session_codes: Any
    direction_labels: Any | None = None
    entry_labels: Any | None = None
    label_valid_mask: Any | None = None
    edge_matrices: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubnetSequenceBatch:
    subnet_name: str
    timeframe_batches: dict[str, TimeframeSequenceBatch]
    node_names: tuple[str, ...]
    tradable_node_names: tuple[str, ...]
    anchor_timestamps: Any

    @property
    def tradable_indices(self) -> tuple[int, ...]:
        lookup = {name: idx for idx, name in enumerate(self.node_names)}
        return tuple(lookup[name] for name in self.tradable_node_names if name in lookup)


@dataclass(slots=True)
class TimeframeState:
    timeframe: str
    node_embeddings: Any
    pooled_context: Any
    directional_logits: Any
    entry_logits: Any | None = None
    edge_type_attention: Any | None = None
    tpo_gate: Any | None = None


@dataclass(slots=True)
class SubnetState:
    subnet_name: str
    timeframe_states: dict[str, TimeframeState]
    next_exchange_contexts: dict[str, Any] = field(default_factory=dict)
    active_timeframe: str | None = None


@dataclass(slots=True)
class CalibrationArtifact:
    coef: float
    intercept: float

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        calibrated = self.coef * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(calibrated, -30.0, 30.0)))
