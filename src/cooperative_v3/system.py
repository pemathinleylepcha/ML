from __future__ import annotations

import torch
from torch import nn

from .bridge import OverlapGatedContextBridge
from .config import DualSubnetSystemConfig
from .contracts import SubnetBatch, SubnetState, SystemOutput
from .meta import MetaFeatureBuilder
from .subnet import CooperativeTimeframeSubnet


class DualSubnetCooperativeSystem(nn.Module):
    def __init__(self, config: DualSubnetSystemConfig | None = None):
        super().__init__()
        self.config = config or DualSubnetSystemConfig()
        self.btc_subnet = CooperativeTimeframeSubnet(
            subnet_cfg=self.config.btc_subnet,
            graph_cfg=self.config.graph,
            temporal_cfg=self.config.temporal,
            exchange_cfg=self.config.exchange,
        )
        self.fx_subnet = CooperativeTimeframeSubnet(
            subnet_cfg=self.config.fx_subnet,
            graph_cfg=self.config.graph,
            temporal_cfg=self.config.temporal,
            exchange_cfg=self.config.exchange,
        )
        self.bridge = OverlapGatedContextBridge(hidden_dim=self.config.graph.output_dim)
        self.meta_builder = MetaFeatureBuilder()

    def reset_cooperative_state(self) -> None:
        self.btc_subnet.reset_exchange_memory()
        self.fx_subnet.reset_exchange_memory()

    def forward_btc_only(self, btc_batch: SubnetBatch, batch_index: int = 0) -> SubnetState:
        return self.btc_subnet(btc_batch, batch_index=batch_index)

    def forward_fx_only(
        self,
        fx_batch: SubnetBatch,
        batch_index: int = 0,
        incoming_contexts: dict[str, torch.Tensor] | None = None,
    ) -> SubnetState:
        return self.fx_subnet(fx_batch, batch_index=batch_index, incoming_contexts=incoming_contexts)

    def forward(
        self,
        btc_batch: SubnetBatch,
        fx_batch: SubnetBatch,
        batch_index: int = 0,
        bridge_enabled: bool = True,
    ) -> SystemOutput:
        btc_state = self.forward_btc_only(btc_batch, batch_index=batch_index)
        bridge_contexts = self.bridge.build_fx_contexts(btc_state, fx_batch) if bridge_enabled else {}
        fx_state = self.forward_fx_only(fx_batch, batch_index=batch_index, incoming_contexts=bridge_contexts)
        output = SystemOutput(
            btc_state=btc_state,
            fx_state=fx_state,
            fx_bridge_contexts=bridge_contexts,
            meta_features=None,
        )
        output.meta_features = self.meta_builder.build(output, fx_batch)
        return output
