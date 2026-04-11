from __future__ import annotations
import gc
import numpy as np
import torch

from staged_v4.config import DEFAULT_SEQ_LENS, SubnetConfig, STGNNBlockConfig, TrainingConfig
from staged_v4.data.cache import load_feature_batches
from staged_v4.training.train_staged import _resolve_cached_splits, _iter_batches, _run_btc_forward, _run_fx_forward, _compute_subnet_loss
from staged_v4.models import BTCSubnet, FXSubnet, ConditionalBridge
from staged_v4.utils.runtime_logging import configure_logging

logger = configure_logging('stage2_boundary_probe')
cache_root = r'D:\work\Algo-C2-Codex\data\staged_v4_remote_cache_20260301_20260331_weekly'
logger.info('probe_state=start')

btc_batch, fx_batch, cached_splits, split_meta = load_feature_batches(cache_root)
loaded_timeframes = tuple(fx_batch.timeframe_batches.keys())
training_cfg = TrainingConfig(anchor_timeframe=fx_batch.anchor_timeframe, outer_holdout_blocks=0, min_train_blocks=2)
splits, _ = _resolve_cached_splits(fx_batch.anchor_timestamps, cached_splits, split_meta, training_cfg, logger, None)
split = splits[0]
train_idx = np.asarray(split['train_idx'], dtype=np.int32)
calib_cut = max(4, int(len(train_idx) * 0.2))
train_main_idx = train_idx[:-calib_cut] if len(train_idx) > calib_cut else train_idx
stage1_batches = _iter_batches(train_main_idx, training_cfg.batch_size)
logger.info('probe_state=split_ready stage1_batches=%d', len(stage1_batches))

device = torch.device('cpu')
subnet_cfg = SubnetConfig(timeframe_order=loaded_timeframes)
block_cfg = STGNNBlockConfig()
btc_subnet = BTCSubnet(subnet_cfg, block_cfg).to(device)
fx_subnet = FXSubnet(subnet_cfg, block_cfg).to(device)
bridge = ConditionalBridge(block_cfg.output_dim, block_cfg.hidden_dim).to(device)
btc_optimizer = torch.optim.Adam(btc_subnet.parameters(), lr=training_cfg.learning_rate)
fx_params = list(fx_subnet.parameters()) + list(bridge.parameters())
fx_optimizer = torch.optim.Adam(fx_params, lr=training_cfg.learning_rate)

btc_active = 0
fx_active = 0
batch_counter = 0
logger.info('probe_state=stage1_start')
btc_subnet.train()
btc_subnet.reset_exchange_memory()
for batch_no, batch_indices in enumerate(stage1_batches, start=1):
    if batch_no == 1 or batch_no % 100 == 0 or batch_no == len(stage1_batches):
        logger.info('probe_state=stage1_progress batch=%d/%d', batch_no, len(stage1_batches))
    btc_optimizer.zero_grad()
    btc_sequence, btc_state, btc_active = _run_btc_forward(btc_subnet, btc_batch, batch_indices, DEFAULT_SEQ_LENS, device, btc_active, batch_counter)
    loss = _compute_subnet_loss(btc_state, btc_sequence, btc_state.active_timeframe or fx_batch.anchor_timeframe, subnet_cfg.active_loss_boost)
    if loss.requires_grad:
        loss.backward()
        btc_optimizer.step()
    batch_counter += 1
logger.info('probe_state=stage1_done btc_active=%d batch_counter=%d', btc_active, batch_counter)

for param in btc_subnet.parameters():
    param.requires_grad = False
logger.info('probe_state=btc_frozen')
gc.collect()
logger.info('probe_state=gc_done')

fx_subnet.train()
bridge.train()
fx_subnet.reset_exchange_memory()
first_stage2_batch = stage1_batches[0]
logger.info('probe_state=stage2_first_batch_start batch_size=%d first_idx=%d last_idx=%d', len(first_stage2_batch), int(first_stage2_batch[0]), int(first_stage2_batch[-1]))
fx_optimizer.zero_grad()
with torch.no_grad():
    _, btc_state, btc_active = _run_btc_forward(btc_subnet, btc_batch, first_stage2_batch, DEFAULT_SEQ_LENS, device, btc_active, batch_counter)
logger.info('probe_state=after_btc_replay btc_active=%d', btc_active)
fx_sequence, fx_state, fx_active = _run_fx_forward(fx_subnet, bridge, fx_batch, btc_state, first_stage2_batch, DEFAULT_SEQ_LENS, device, fx_active, batch_counter)
logger.info('probe_state=after_fx_forward fx_active=%d', fx_active)
loss2 = _compute_subnet_loss(fx_state, fx_sequence, fx_state.active_timeframe or fx_batch.anchor_timeframe, subnet_cfg.active_loss_boost)
logger.info('probe_state=after_loss requires_grad=%s loss_value=%s', loss2.requires_grad, float(loss2.detach().cpu()))
if loss2.requires_grad:
    loss2.backward()
    logger.info('probe_state=after_backward')
    fx_optimizer.step()
    logger.info('probe_state=after_optimizer_step')
logger.info('probe_state=done')
