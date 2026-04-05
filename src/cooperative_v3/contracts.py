from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TimeframeBatch:
    timeframe: str
    node_names: Sequence[str]
    node_features: Any
    edge_matrices: dict[str, Any]
    valid_mask: Any
    market_open_mask: Any
    overlap_mask: Any
    session_codes: Any
    target_direction: Any | None = None
    target_entry: Any | None = None
    target_valid: Any | None = None
    is_bar_real: Any | None = None
    is_stale_fill: Any | None = None
    is_low_liquidity: Any | None = None


@dataclass
class SubnetBatch:
    subnet_name: str
    timeframe_batches: dict[str, TimeframeBatch]
    node_names: Sequence[str]
    tradable_node_names: Sequence[str]
    base_indices: Any | None = None

    @property
    def tradable_indices(self) -> tuple[int, ...]:
        lookup = {name: idx for idx, name in enumerate(self.node_names)}
        return tuple(lookup[name] for name in self.tradable_node_names if name in lookup)


@dataclass
class TimeframeState:
    timeframe: str
    node_embeddings: Any
    pooled_context: Any
    directional_logits: Any
    entry_logits: Any | None = None
    edge_type_attention: Any | None = None


@dataclass
class SubnetState:
    subnet_name: str
    timeframe_states: dict[str, TimeframeState]
    next_exchange_contexts: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemOutput:
    btc_state: SubnetState
    fx_state: SubnetState
    fx_bridge_contexts: dict[str, Any]
    meta_features: Any | None = None
