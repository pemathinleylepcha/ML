from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import CooperativeExchangeConfig, HeteroGraphConfig, SubnetConfig, TemporalAttentionConfig
from .contracts import SubnetBatch, SubnetState, TimeframeBatch, TimeframeState
from .layers import (
    CausalTemporalSelfAttention,
    ExchangeInjectionGate,
    HeteroTypeAttentionConv,
    TimeframeNodeEncoder,
    masked_mean,
)

LOWER_TIMEFRAME_NEIGHBOR_SPAN = {
    "M5": 2,
    "M15": 2,
    "M30": 2,
    "H1": 1,
    "H4": 1,
    "H12": 1,
    "D1": 0,
    "W1": 0,
    "MN1": 0,
}


def _take_last_valid(sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    batch_size, n_steps, n_nodes, hidden_dim = sequence.shape
    step_ids = torch.arange(n_steps, device=sequence.device).view(1, n_steps, 1)
    weighted = valid_mask.long() * (step_ids + 1)
    last_index = weighted.argmax(dim=1)
    gather_index = last_index.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, n_nodes, hidden_dim)
    gathered = sequence.gather(1, gather_index).squeeze(1)
    any_valid = valid_mask.any(dim=1).unsqueeze(-1)
    return gathered * any_valid.to(gathered.dtype)


class PerTimeframeSTGNN(nn.Module):
    def __init__(
        self,
        timeframe: str,
        graph_cfg: HeteroGraphConfig,
        temporal_cfg: TemporalAttentionConfig,
        enable_entry_head: bool,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.encoder = TimeframeNodeEncoder(graph_cfg.input_dim, graph_cfg.hidden_dim, dropout=graph_cfg.dropout)
        self.exchange_gate = ExchangeInjectionGate(graph_cfg.hidden_dim)
        self.graph = HeteroTypeAttentionConv(
            graph_cfg.hidden_dim,
            graph_cfg.output_dim,
            graph_cfg.edge_types,
            dropout=graph_cfg.dropout,
        )
        self.temporal = CausalTemporalSelfAttention(
            temporal_cfg.hidden_dim,
            temporal_cfg.n_heads,
            ff_multiplier=temporal_cfg.ff_multiplier,
            dropout=temporal_cfg.dropout,
        )
        self.direction_head = nn.Linear(graph_cfg.output_dim, 1)
        self.entry_head = nn.Linear(graph_cfg.output_dim, 1) if enable_entry_head else None

    def forward(
        self,
        batch: TimeframeBatch,
        incoming_context: torch.Tensor | None = None,
    ) -> TimeframeState:
        encoded = self.encoder(batch.node_features, batch.session_codes)
        encoded = self.exchange_gate(encoded, incoming_context)
        graph_steps = []
        edge_weights = []
        for step_idx in range(encoded.shape[1]):
            step_mask = batch.valid_mask[:, step_idx]
            graph_step, step_weights = self.graph(encoded[:, step_idx], batch.edge_matrices, step_mask)
            graph_steps.append(graph_step)
            edge_weights.append(step_weights)
        graph_sequence = torch.stack(graph_steps, dim=1)
        temporal_sequence = self.temporal(
            graph_sequence,
            batch.valid_mask,
            session_codes=batch.session_codes,
            regime_signal=batch.node_features[:, :, :, 9],
            session_transition=batch.node_features[:, :, :, 8],
        )
        last_embeddings = _take_last_valid(temporal_sequence, batch.valid_mask)
        final_mask = batch.valid_mask[:, -1]
        pooled_context = masked_mean(last_embeddings, final_mask, dim=1)
        directional_logits = self.direction_head(last_embeddings).squeeze(-1)
        entry_logits = None
        if self.entry_head is not None:
            entry_logits = self.entry_head(last_embeddings).squeeze(-1)
        mean_edge_weights = torch.stack(edge_weights, dim=0).mean(dim=0)
        return TimeframeState(
            timeframe=batch.timeframe,
            node_embeddings=last_embeddings,
            pooled_context=pooled_context,
            directional_logits=directional_logits,
            entry_logits=entry_logits,
            edge_type_attention=mean_edge_weights,
        )


@dataclass(frozen=True)
class AdjacentTimeframeExchangeController:
    timeframe_order: tuple[str, ...]
    exchange_every_k_batches: int

    def should_exchange(self, batch_index: int) -> bool:
        return batch_index > 0 and (batch_index % self.exchange_every_k_batches) == 0

    def build_next_contexts(self, states: dict[str, TimeframeState]) -> dict[str, torch.Tensor]:
        contexts: dict[str, torch.Tensor] = {}
        for idx, timeframe in enumerate(self.timeframe_order):
            # Timeframe order is fast -> slow. Lower frames benefit from a slightly
            # wider local neighborhood, while higher frames stay lightly coupled.
            span = LOWER_TIMEFRAME_NEIGHBOR_SPAN.get(timeframe, 1)
            neighbor_contexts = []
            neighbor_weights = []
            for offset in range(1, span + 1):
                slower_idx = idx + offset
                if slower_idx >= len(self.timeframe_order):
                    break
                slower_timeframe = self.timeframe_order[slower_idx]
                if slower_timeframe not in states:
                    continue
                weight = 0.7 if offset == 1 else 0.3
                neighbor_contexts.append(states[slower_timeframe].pooled_context)
                neighbor_weights.append(weight)
            if neighbor_contexts:
                total_weight = sum(neighbor_weights)
                merged = sum(ctx * (weight / total_weight) for ctx, weight in zip(neighbor_contexts, neighbor_weights, strict=False))
                contexts[timeframe] = merged
        return contexts


class CooperativeTimeframeSubnet(nn.Module):
    def __init__(
        self,
        subnet_cfg: SubnetConfig,
        graph_cfg: HeteroGraphConfig,
        temporal_cfg: TemporalAttentionConfig,
        exchange_cfg: CooperativeExchangeConfig,
    ):
        super().__init__()
        self.subnet_cfg = subnet_cfg
        self.timeframe_order = tuple(subnet_cfg.timeframe_order)
        self.controllers = AdjacentTimeframeExchangeController(
            timeframe_order=self.timeframe_order,
            exchange_every_k_batches=exchange_cfg.exchange_every_k_batches,
        )
        self.models = nn.ModuleDict(
            {
                timeframe: PerTimeframeSTGNN(
                    timeframe=timeframe,
                    graph_cfg=graph_cfg,
                    temporal_cfg=temporal_cfg,
                    enable_entry_head=subnet_cfg.enable_entry_head,
                )
                for timeframe in self.timeframe_order
            }
        )
        self.exchange_memory: dict[str, torch.Tensor] = {}

    def reset_exchange_memory(self) -> None:
        self.exchange_memory = {}

    def _expand_context(self, context: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if context is None:
            return None
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.dim() == 2 and context.shape[0] == batch_size:
            return context
        if context.dim() == 2 and context.shape[0] == 1:
            return context.expand(batch_size, -1)
        if context.dim() == 2:
            return context.mean(dim=0, keepdim=True).expand(batch_size, -1)
        return context

    def _merge_context(
        self,
        timeframe: str,
        incoming_context: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        external = self._expand_context(incoming_context, batch_size)
        memory = self._expand_context(self.exchange_memory.get(timeframe), batch_size)
        if external is None:
            return memory
        if memory is None:
            return external
        return 0.5 * (external + memory)

    def _combine_contexts(self, *contexts: torch.Tensor | None) -> torch.Tensor | None:
        active = [ctx for ctx in contexts if ctx is not None]
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        return torch.stack(active, dim=0).mean(dim=0)

    def forward(
        self,
        subnet_batch: SubnetBatch,
        batch_index: int = 0,
        incoming_contexts: dict[str, torch.Tensor] | None = None,
    ) -> SubnetState:
        states: dict[str, TimeframeState] = {}
        incoming_contexts = incoming_contexts or {}
        for timeframe in self.timeframe_order:
            batch = subnet_batch.timeframe_batches[timeframe]
            merged_context = self._merge_context(timeframe, incoming_contexts.get(timeframe), batch.node_features.shape[0])
            states[timeframe] = self.models[timeframe](batch, merged_context)
        next_exchange_contexts = {}
        if self.controllers.should_exchange(batch_index):
            local_contexts = self.controllers.build_next_contexts(states)
            refined_states: dict[str, TimeframeState] = {}
            for timeframe in self.timeframe_order:
                batch = subnet_batch.timeframe_batches[timeframe]
                bridge_or_external = incoming_contexts.get(timeframe)
                local = local_contexts.get(timeframe)
                batch_size = batch.node_features.shape[0]
                refined_context = self._combine_contexts(
                    self._merge_context(timeframe, bridge_or_external, batch_size),
                    self._expand_context(local, batch_size),
                )
                refined_states[timeframe] = self.models[timeframe](batch, refined_context)
            states = refined_states
            next_exchange_contexts = self.controllers.build_next_contexts(states)
            self.exchange_memory = {
                timeframe: context.detach().mean(dim=0, keepdim=True)
                for timeframe, context in next_exchange_contexts.items()
            }
        return SubnetState(
            subnet_name=subnet_batch.subnet_name,
            timeframe_states=states,
            next_exchange_contexts=next_exchange_contexts,
        )
