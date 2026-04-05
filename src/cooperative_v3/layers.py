from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def _broadcast_adjacency(adj: torch.Tensor, batch_size: int) -> torch.Tensor:
    if adj.dim() == 2:
        return adj.unsqueeze(0).expand(batch_size, -1, -1)
    return adj


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / degree


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    expanded = weights
    while expanded.dim() < values.dim():
        expanded = expanded.unsqueeze(-1)
    numer = (values * expanded).sum(dim=dim)
    denom = expanded.sum(dim=dim).clamp_min(1e-6)
    return numer / denom


class TimeframeNodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_sessions: int = 8, dropout: float = 0.10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.session_embedding = nn.Embedding(n_sessions, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, session_codes: torch.Tensor) -> torch.Tensor:
        encoded = self.input_proj(node_features)
        session_codes = session_codes.clamp_min(0).clamp_max(self.session_embedding.num_embeddings - 1)
        session_bias = self.session_embedding(session_codes.long()).unsqueeze(2)
        return self.dropout(self.norm(encoded + session_bias))


class HeteroTypeAttentionConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_types: Sequence[str], dropout: float = 0.10):
        super().__init__()
        self.edge_types = tuple(edge_types)
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.edge_projs = nn.ModuleDict({edge_type: nn.Linear(in_dim, out_dim) for edge_type in self.edge_types})
        self.score = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_matrices: dict[str, torch.Tensor],
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = node_states.shape[0]
        base = self.self_proj(node_states)
        valid_float = None
        if valid_mask is not None:
            valid_float = valid_mask.unsqueeze(-1).to(base.dtype)
            base = base * valid_float
        messages = []
        scores = []
        for edge_type in self.edge_types:
            adj = _normalize_adjacency(_broadcast_adjacency(edge_matrices[edge_type], batch_size))
            msg = torch.matmul(adj, self.edge_projs[edge_type](node_states))
            if valid_float is not None:
                msg = msg * valid_float
            messages.append(msg)
            scores.append(self.score(torch.tanh(msg)))
        stacked_messages = torch.stack(messages, dim=2)
        stacked_scores = torch.stack(scores, dim=2).squeeze(-1)
        weights = torch.softmax(stacked_scores, dim=2)
        mixed = (stacked_messages * weights.unsqueeze(-1)).sum(dim=2)
        out = self.norm(base + self.dropout(mixed))
        if valid_float is not None:
            out = out * valid_float
        return out, weights.mean(dim=(0, 1))


class ExchangeInjectionGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_states: torch.Tensor, incoming_context: torch.Tensor | None) -> torch.Tensor:
        if incoming_context is None:
            return node_states
        context = incoming_context
        if node_states.dim() == 4:
            if context.dim() == 2:
                context = context.unsqueeze(1).unsqueeze(2).expand(
                    -1,
                    node_states.shape[1],
                    node_states.shape[2],
                    -1,
                )
            elif context.dim() == 3:
                context = context.unsqueeze(1).expand(-1, node_states.shape[1], -1, -1)
        elif node_states.dim() == 3 and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, node_states.shape[1], -1)
        context = self.context_proj(context)
        pair = torch.cat([node_states, context], dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        update = torch.tanh(self.update(pair))
        return node_states + gate * update


class CausalTemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ff_multiplier: int = 2, dropout: float = 0.10):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.session_embedding = nn.Embedding(8, hidden_dim)
        self.regime_proj = nn.Linear(1, hidden_dim)
        self.transition_proj = nn.Linear(1, hidden_dim)
        self.time_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_k_proj = nn.Linear(hidden_dim, hidden_dim)
        ff_dim = hidden_dim * ff_multiplier
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps, _ = tensor.shape
        return tensor.view(batch_size, n_steps, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        session_codes: torch.Tensor | None = None,
        regime_signal: torch.Tensor | None = None,
        session_transition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_steps, n_nodes, hidden_dim = sequence.shape
        flat = sequence.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, n_steps, hidden_dim)

        if session_codes is None:
            session_flat = torch.zeros((batch_size * n_nodes, n_steps), dtype=torch.long, device=sequence.device)
        else:
            session = session_codes.long().clamp_min(0).clamp_max(self.session_embedding.num_embeddings - 1)
            session_flat = session.unsqueeze(1).expand(-1, n_nodes, -1).reshape(batch_size * n_nodes, n_steps)

        time_state = self.session_embedding(session_flat)
        if regime_signal is not None:
            if regime_signal.dim() == 2:
                regime_flat = regime_signal.unsqueeze(1).expand(-1, n_nodes, -1).reshape(batch_size * n_nodes, n_steps)
            else:
                regime_flat = regime_signal.permute(0, 2, 1).reshape(batch_size * n_nodes, n_steps)
            time_state = time_state + self.regime_proj(regime_flat.unsqueeze(-1).to(flat.dtype))
        if session_transition is not None:
            if session_transition.dim() == 2:
                transition_flat = session_transition.unsqueeze(1).expand(-1, n_nodes, -1).reshape(batch_size * n_nodes, n_steps)
            else:
                transition_flat = session_transition.permute(0, 2, 1).reshape(batch_size * n_nodes, n_steps)
            time_state = time_state + self.transition_proj(transition_flat.unsqueeze(-1).to(flat.dtype))

        q = self._reshape_heads(self.q_proj(flat + time_state))
        k = self._reshape_heads(self.k_proj(flat + time_state))
        v = self._reshape_heads(self.v_proj(flat))
        time_q = self._reshape_heads(self.time_q_proj(time_state))
        time_k = self._reshape_heads(self.time_k_proj(time_state))

        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + (0.25 * torch.matmul(time_q, time_k.transpose(-2, -1)) * scale)

        causal_mask = torch.triu(
            torch.ones(n_steps, n_steps, device=sequence.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.view(1, 1, n_steps, n_steps), -1e4)

        valid = None
        all_invalid = None
        if valid_mask is not None:
            valid = valid_mask.permute(0, 2, 1).reshape(batch_size * n_nodes, n_steps)
            all_invalid = ~valid.bool().any(dim=1)
            key_valid = valid.bool().view(batch_size * n_nodes, 1, 1, n_steps)
            scores = scores.masked_fill(~key_valid, -1e4)
            if all_invalid.any():
                scores = scores.clone()
                scores[all_invalid, :, :, 0] = 0.0

        attn = torch.softmax(scores, dim=-1)
        if valid is not None:
            key_valid = valid.bool().view(batch_size * n_nodes, 1, 1, n_steps)
            attn = attn * key_valid.to(attn.dtype)
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom
            if all_invalid is not None and all_invalid.any():
                attn = attn.clone()
                attn[all_invalid] = 0.0

        attn_out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch_size * n_nodes, n_steps, hidden_dim)
        attn_out = self.out_proj(attn_out)
        flat = self.norm1(flat + self.dropout(attn_out))
        ff_out = self.ff(flat)
        flat = self.norm2(flat + self.dropout(ff_out))
        if valid is not None:
            flat = flat * valid.to(flat.dtype).unsqueeze(-1)
        return flat.reshape(batch_size, n_nodes, n_steps, hidden_dim).permute(0, 2, 1, 3)
