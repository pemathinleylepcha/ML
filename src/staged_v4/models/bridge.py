from __future__ import annotations

import torch
from torch import nn


class ConditionalBridge(nn.Module):
    def __init__(self, btc_dim: int, fx_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(btc_dim, fx_dim),
            nn.GELU(),
            nn.Linear(fx_dim, fx_dim),
        )

    def forward(self, btc_context: torch.Tensor, overlap_mask: torch.Tensor, n_nodes: int) -> torch.Tensor:
        projected = self.projector(btc_context)
        if projected.dim() == 2:
            projected = projected.unsqueeze(1).expand(-1, n_nodes, -1)
        mask = overlap_mask.to(projected.dtype).view(-1, 1, 1)
        return projected * mask
