from __future__ import annotations

import torch
from torch import nn

from staged_v4.config import RAW_FEATURE_NAMES, STGNNBlockConfig, SubnetConfig, TPO_FEATURE_NAMES
from staged_v4.contracts import SubnetState, TimeframeSequenceBatch
from staged_v4.models.stgnn_block import STGNNBlock


class FXSubnet(nn.Module):
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

    def cooperative_step(
        self,
        timeframe_batches: dict[str, TimeframeSequenceBatch],
        edge_matrices: dict[str, dict[str, torch.Tensor]],
        bridge_contexts: dict[str, torch.Tensor],
        active_idx: int,
        batch_index: int,
    ) -> tuple[SubnetState, int]:
        states = {}
        pooled = {}
        active_timeframe = self.timeframe_order[active_idx % len(self.timeframe_order)]
        for timeframe in self.timeframe_order:
            tf_batch = timeframe_batches[timeframe]
            transfer = None
            if timeframe in self.exchange_memory:
                transfer = self.exchange_memory[timeframe]
            elif pooled:
                contexts = [ctx.mean(dim=0, keepdim=True) for ctx in list(pooled.values())[-2:]]
                transfer = torch.stack(contexts, dim=0).mean(dim=0)
            state = self.blocks[timeframe](
                node_features=tf_batch.node_features,
                tpo_features=tf_batch.tpo_features,
                volatility=tf_batch.volatility,
                valid_mask=tf_batch.valid_mask,
                session_codes=tf_batch.session_codes,
                edge_matrices=edge_matrices[timeframe],
                transfer_context=transfer,
                bridge_context=bridge_contexts.get(timeframe),
            )
            states[timeframe] = state
            pooled[timeframe] = state.pooled_context.detach()
        if batch_index > 0 and batch_index % self.cfg.exchange_every_k_batches == 0:
            self.exchange_memory = {
                timeframe: state.pooled_context.detach().mean(dim=0, keepdim=True)
                for timeframe, state in states.items()
            }
            active_idx = (active_idx + 1) % len(self.timeframe_order)
        return SubnetState(
            subnet_name="fx",
            timeframe_states=states,
            next_exchange_contexts=self.exchange_memory,
            active_timeframe=active_timeframe,
        ), active_idx
