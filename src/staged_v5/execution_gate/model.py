from __future__ import annotations

import torch
from torch import nn

from staged_v5.execution_gate.contracts import NeuralGateAction


class MicrostructureGate(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = len(NeuralGateAction)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
