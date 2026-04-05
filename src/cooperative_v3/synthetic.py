from __future__ import annotations

from typing import Iterable

import torch

from .config import COOPERATIVE_TIMEFRAMES, DualSubnetSystemConfig
from .contracts import SubnetBatch, TimeframeBatch


def _make_edge_matrices(n_nodes: int) -> dict[str, torch.Tensor]:
    eye = torch.eye(n_nodes, dtype=torch.float32)
    full = torch.ones(n_nodes, n_nodes, dtype=torch.float32)
    chain = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    for idx in range(n_nodes):
        chain[idx, idx] = 1.0
        if idx + 1 < n_nodes:
            chain[idx, idx + 1] = 1.0
            chain[idx + 1, idx] = 1.0
    return {
        "fundamental": eye,
        "rolling_corr": 0.35 * full + 0.65 * eye,
        "session": chain,
    }


def build_synthetic_subnet_batch(
    subnet_name: str,
    node_names: Iterable[str],
    tradable_node_names: Iterable[str],
    timeframe_order: Iterable[str] = COOPERATIVE_TIMEFRAMES,
    batch_size: int = 2,
    n_steps: int = 6,
    input_dim: int = 10,
) -> SubnetBatch:
    node_names = tuple(node_names)
    tradable_node_names = tuple(tradable_node_names)
    n_nodes = len(node_names)
    timeframe_batches: dict[str, TimeframeBatch] = {}
    edges = _make_edge_matrices(n_nodes)
    base_sessions = torch.arange(n_steps, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) % 4
    for timeframe in timeframe_order:
        features = torch.randn(batch_size, n_steps, n_nodes, input_dim, dtype=torch.float32)
        valid_mask = torch.ones(batch_size, n_steps, n_nodes, dtype=torch.bool)
        market_open_mask = torch.ones(batch_size, n_steps, n_nodes, dtype=torch.bool)
        overlap_mask = torch.ones(batch_size, n_steps, n_nodes, dtype=torch.bool)
        target_direction = torch.randint(0, 2, (batch_size, n_nodes), dtype=torch.float32)
        target_entry = torch.randint(0, 2, (batch_size, n_nodes), dtype=torch.float32)
        timeframe_batches[timeframe] = TimeframeBatch(
            timeframe=timeframe,
            node_names=node_names,
            node_features=features,
            edge_matrices=edges,
            valid_mask=valid_mask,
            market_open_mask=market_open_mask,
            overlap_mask=overlap_mask,
            session_codes=base_sessions.clone(),
            target_direction=target_direction,
            target_entry=target_entry,
            is_bar_real=valid_mask.clone(),
            is_stale_fill=torch.zeros_like(valid_mask),
            is_low_liquidity=torch.zeros_like(valid_mask),
        )
    return SubnetBatch(
        subnet_name=subnet_name,
        timeframe_batches=timeframe_batches,
        node_names=node_names,
        tradable_node_names=tradable_node_names,
    )


def build_synthetic_dual_subnet_batches(
    config: DualSubnetSystemConfig | None = None,
    batch_size: int = 2,
    n_steps: int = 6,
) -> tuple[SubnetBatch, SubnetBatch]:
    cfg = config or DualSubnetSystemConfig()
    btc_batch = build_synthetic_subnet_batch(
        subnet_name=cfg.btc_subnet.name,
        node_names=cfg.btc_subnet.node_names,
        tradable_node_names=cfg.btc_subnet.tradable_node_names,
        timeframe_order=cfg.btc_subnet.timeframe_order,
        batch_size=batch_size,
        n_steps=n_steps,
        input_dim=cfg.btc_subnet.input_dim,
    )
    fx_batch = build_synthetic_subnet_batch(
        subnet_name=cfg.fx_subnet.name,
        node_names=cfg.fx_subnet.node_names,
        tradable_node_names=cfg.fx_subnet.tradable_node_names,
        timeframe_order=cfg.fx_subnet.timeframe_order,
        batch_size=batch_size,
        n_steps=n_steps,
        input_dim=cfg.fx_subnet.input_dim,
    )
    return btc_batch, fx_batch
