from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from universe import SUBNET_24x5, SUBNET_24x5_TRADEABLE, SUBNET_24x7, TIMEFRAMES


COOPERATIVE_TIMEFRAMES = ("tick",) + tuple(TIMEFRAMES)
EDGE_TYPES = ("fundamental", "rolling_corr", "session")


@dataclass(frozen=True)
class HeteroGraphConfig:
    input_dim: int = 10
    hidden_dim: int = 48
    output_dim: int = 48
    edge_types: Sequence[str] = EDGE_TYPES
    dropout: float = 0.10


@dataclass(frozen=True)
class TemporalAttentionConfig:
    hidden_dim: int = 48
    n_heads: int = 4
    ff_multiplier: int = 2
    dropout: float = 0.10


@dataclass(frozen=True)
class CooperativeExchangeConfig:
    exchange_every_k_batches: int = 8
    dropout: float = 0.05


@dataclass(frozen=True)
class SubnetConfig:
    name: str
    node_names: Sequence[str]
    tradable_node_names: Sequence[str]
    timeframe_order: Sequence[str] = COOPERATIVE_TIMEFRAMES
    enable_entry_head: bool = False
    hidden_dim: int = 48
    input_dim: int = 10

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def tradable_indices(self) -> tuple[int, ...]:
        lookup = {name: idx for idx, name in enumerate(self.node_names)}
        return tuple(lookup[name] for name in self.tradable_node_names if name in lookup)


@dataclass(frozen=True)
class DualSubnetSystemConfig:
    btc_subnet: SubnetConfig = field(default_factory=lambda: default_btc_subnet_config())
    fx_subnet: SubnetConfig = field(default_factory=lambda: default_fx_subnet_config())
    graph: HeteroGraphConfig = field(default_factory=HeteroGraphConfig)
    temporal: TemporalAttentionConfig = field(default_factory=TemporalAttentionConfig)
    exchange: CooperativeExchangeConfig = field(default_factory=CooperativeExchangeConfig)


def default_btc_subnet_config() -> SubnetConfig:
    return SubnetConfig(
        name="btc",
        node_names=tuple(SUBNET_24x7),
        tradable_node_names=tuple(SUBNET_24x7),
        timeframe_order=COOPERATIVE_TIMEFRAMES,
        enable_entry_head=True,
    )


def default_fx_subnet_config() -> SubnetConfig:
    return SubnetConfig(
        name="fx",
        node_names=tuple(SUBNET_24x5),
        tradable_node_names=tuple(SUBNET_24x5_TRADEABLE),
        timeframe_order=COOPERATIVE_TIMEFRAMES,
        enable_entry_head=True,
    )
