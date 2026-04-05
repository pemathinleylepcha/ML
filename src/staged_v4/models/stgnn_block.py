from __future__ import annotations

import torch
from torch import nn

from cooperative_v3.layers import (
    CausalTemporalSelfAttention,
    ExchangeInjectionGate,
    HeteroTypeAttentionConv,
    TimeframeNodeEncoder,
    masked_mean,
)
from staged_v4.config import EDGE_TYPES, STGNNBlockConfig
from staged_v4.contracts import TimeframeState


def _take_last_valid(sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    batch_size, n_steps, n_nodes, hidden_dim = sequence.shape
    step_ids = torch.arange(n_steps, device=sequence.device).view(1, n_steps, 1)
    weighted = valid_mask.long() * (step_ids + 1)
    last_index = weighted.argmax(dim=1)
    gather_index = last_index.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, n_nodes, hidden_dim)
    gathered = sequence.gather(1, gather_index).squeeze(1)
    any_valid = valid_mask.any(dim=1).unsqueeze(-1)
    return gathered * any_valid.to(gathered.dtype)


class STGNNBlock(nn.Module):
    def __init__(
        self,
        timeframe: str,
        raw_input_dim: int,
        tpo_input_dim: int,
        cfg: STGNNBlockConfig,
        enable_entry_head: bool = True,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.cfg = cfg
        self.raw_encoder = TimeframeNodeEncoder(raw_input_dim, cfg.hidden_dim, dropout=cfg.dropout)
        self.tpo_encoder = nn.Linear(tpo_input_dim, cfg.hidden_dim)
        self.tpo_alpha = nn.Parameter(torch.tensor(float(cfg.tpo_gate_alpha))) if cfg.trainable_tpo_gate else None
        self.tpo_threshold = nn.Parameter(torch.tensor(float(cfg.tpo_gate_threshold))) if cfg.trainable_tpo_gate else None
        self.fixed_tpo_alpha = float(cfg.tpo_gate_alpha)
        self.fixed_tpo_threshold = float(cfg.tpo_gate_threshold)
        self.exchange_gate = ExchangeInjectionGate(cfg.hidden_dim)
        self.bridge_gate = ExchangeInjectionGate(cfg.hidden_dim)
        self.graph = HeteroTypeAttentionConv(cfg.hidden_dim, cfg.output_dim, EDGE_TYPES, dropout=cfg.dropout)
        self.temporal = CausalTemporalSelfAttention(
            cfg.output_dim,
            cfg.n_heads,
            ff_multiplier=cfg.ff_multiplier,
            dropout=cfg.dropout,
        )
        self.direction_head = nn.Linear(cfg.output_dim, 1)
        self.entry_head = nn.Linear(cfg.output_dim, 1) if enable_entry_head else None

    def _tpo_gate(self, volatility: torch.Tensor) -> torch.Tensor:
        alpha = self.tpo_alpha if self.tpo_alpha is not None else torch.tensor(self.fixed_tpo_alpha, device=volatility.device)
        threshold = self.tpo_threshold if self.tpo_threshold is not None else torch.tensor(self.fixed_tpo_threshold, device=volatility.device)
        return torch.sigmoid(alpha * (threshold - volatility))

    def _expand_context(self, context: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if context is None:
            return None
        if context.dim() >= 2 and context.shape[0] == batch_size:
            return context
        if context.dim() >= 2 and context.shape[0] == 1:
            expand_shape = (batch_size,) + tuple(context.shape[1:])
            return context.expand(*expand_shape)
        return context

    def forward(
        self,
        node_features: torch.Tensor,
        tpo_features: torch.Tensor,
        volatility: torch.Tensor,
        valid_mask: torch.Tensor,
        session_codes: torch.Tensor,
        edge_matrices: dict[str, torch.Tensor],
        transfer_context: torch.Tensor | None = None,
        bridge_context: torch.Tensor | None = None,
    ) -> TimeframeState:
        raw_state = self.raw_encoder(node_features, session_codes)
        tpo_state = self.tpo_encoder(tpo_features)
        tpo_gate = self._tpo_gate(volatility).unsqueeze(-1)
        state = raw_state + tpo_state * tpo_gate
        transfer_context = self._expand_context(transfer_context, state.shape[0])
        bridge_context = self._expand_context(bridge_context, state.shape[0])
        state = self.exchange_gate(state, transfer_context)
        state = self.bridge_gate(state, bridge_context)

        graph_steps = []
        edge_weights = []
        for step_idx in range(state.shape[1]):
            step_mask = valid_mask[:, step_idx]
            graph_step, step_weight = self.graph(state[:, step_idx], edge_matrices, step_mask)
            graph_steps.append(graph_step)
            edge_weights.append(step_weight)
        graph_sequence = torch.stack(graph_steps, dim=1)
        temporal_sequence = self.temporal(
            graph_sequence,
            valid_mask=valid_mask,
            session_codes=session_codes,
            regime_signal=node_features[:, :, :, 11],
            session_transition=node_features[:, :, :, 13],
        )
        last_embeddings = _take_last_valid(temporal_sequence, valid_mask)
        pooled_context = masked_mean(last_embeddings, valid_mask[:, -1], dim=1)
        directional_logits = self.direction_head(last_embeddings).squeeze(-1)
        entry_logits = self.entry_head(last_embeddings).squeeze(-1) if self.entry_head is not None else None
        return TimeframeState(
            timeframe=self.timeframe,
            node_embeddings=last_embeddings,
            pooled_context=pooled_context,
            directional_logits=directional_logits,
            entry_logits=entry_logits,
            edge_type_attention=torch.stack(edge_weights, dim=0).mean(dim=0),
            tpo_gate=tpo_gate.detach().mean(dim=(1, 2)),
        )
