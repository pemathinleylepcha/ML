from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from staged_v5.config import RAW_FEATURE_NAMES, STGNNBlockConfig, SubnetConfig, TPO_FEATURE_NAMES
from staged_v5.contracts import SubnetState, TimeframeSequenceBatch
from staged_v5.models.stgnn_block import STGNNBlock


@dataclass(slots=True)
class CooperativeStepOutput:
    state: SubnetState
    next_active_idx: int


class BTCSubnet(nn.Module):
    def __init__(self, cfg: SubnetConfig, block_cfg: STGNNBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.timeframe_order = tuple(cfg.timeframe_order)
        self.blocks = nn.ModuleDict(
            {
                timeframe: STGNNBlock(
                    timeframe=timeframe,
                    raw_input_dim=len(RAW_FEATURE_NAMES),
                    tpo_input_dim=len(TPO_FEATURE_NAMES),
                    cfg=block_cfg,
                    enable_entry_head=cfg.enable_entry_head,
                )
                for timeframe in self.timeframe_order
            }
        )
        self.exchange_memory: dict[str, torch.Tensor] = {}

    def reset_exchange_memory(self) -> None:
        self.exchange_memory = {}

    def _adjacent_transfer(self, states: dict[str, torch.Tensor], timeframe: str) -> torch.Tensor | None:
        idx = self.timeframe_order.index(timeframe)
        contexts = []
        if idx > 0 and self.timeframe_order[idx - 1] in states:
            contexts.append(states[self.timeframe_order[idx - 1]].mean(dim=0, keepdim=True))
        if idx + 1 < len(self.timeframe_order) and self.timeframe_order[idx + 1] in states:
            contexts.append(states[self.timeframe_order[idx + 1]].mean(dim=0, keepdim=True))
        if timeframe in self.exchange_memory:
            contexts.append(self.exchange_memory[timeframe])
        if not contexts:
            return None
        return torch.stack(contexts, dim=0).mean(dim=0)

    def cooperative_step(
        self,
        timeframe_batches: dict[str, TimeframeSequenceBatch],
        edge_matrices: dict[str, dict[str, torch.Tensor]],
        active_idx: int,
        batch_index: int,
    ) -> CooperativeStepOutput:
        states = {}
        pooled = {}
        active_timeframe = self.timeframe_order[active_idx % len(self.timeframe_order)]
        for timeframe in self.timeframe_order:
            if timeframe not in timeframe_batches:
                continue
            tf_batch = timeframe_batches[timeframe]
            transfer = self._adjacent_transfer(pooled, timeframe) if pooled else self.exchange_memory.get(timeframe)
            state = self.blocks[timeframe](
                node_features=tf_batch.node_features,
                tpo_features=tf_batch.tpo_features,
                volatility=tf_batch.volatility,
                valid_mask=tf_batch.valid_mask,
                session_codes=tf_batch.session_codes,
                edge_matrices=edge_matrices[timeframe],
                transfer_context=transfer,
            )
            states[timeframe] = state
            pooled[timeframe] = state.pooled_context.detach()
        if batch_index > 0 and batch_index % self.cfg.exchange_every_k_batches == 0:
            self.exchange_memory = {
                timeframe: state.pooled_context.detach().mean(dim=0, keepdim=True)
                for timeframe, state in states.items()
            }
            active_idx = (active_idx + 1) % len(self.timeframe_order)
        return CooperativeStepOutput(
            state=SubnetState(subnet_name="btc", timeframe_states=states, next_exchange_contexts=self.exchange_memory, active_timeframe=active_timeframe),
            next_active_idx=active_idx,
        )
