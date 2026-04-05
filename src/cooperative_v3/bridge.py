from __future__ import annotations

import torch
from torch import nn

from .contracts import SubnetBatch, SubnetState


class OverlapGatedContextBridge(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def build_fx_contexts(self, btc_state: SubnetState, fx_batch: SubnetBatch) -> dict[str, torch.Tensor]:
        contexts: dict[str, torch.Tensor] = {}
        for timeframe, fx_tf_batch in fx_batch.timeframe_batches.items():
            btc_tf_state = btc_state.timeframe_states.get(timeframe)
            if btc_tf_state is None:
                continue
            overlap_open = fx_tf_batch.overlap_mask.any(dim=(1, 2)) & fx_tf_batch.market_open_mask.any(dim=(1, 2))
            gate = overlap_open.to(btc_tf_state.pooled_context.dtype).unsqueeze(-1)
            contexts[timeframe] = self.proj(btc_tf_state.pooled_context) * gate
        return contexts
