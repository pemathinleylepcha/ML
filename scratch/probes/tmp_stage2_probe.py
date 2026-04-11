from __future__ import annotations
import numpy as np
import torch

from staged_v4.config import DEFAULT_SEQ_LENS, SubnetConfig, STGNNBlockConfig, TrainingConfig
from staged_v4.data.cache import load_feature_batches
from staged_v4.training.train_staged import _resolve_cached_splits, _iter_batches, _run_btc_forward, _run_fx_forward
from staged_v4.models import BTCSubnet, FXSubnet, ConditionalBridge
from staged_v4.utils.runtime_logging import configure_logging

logger = configure_logging('stage2_probe')
cache_root = r'D:\work\Algo-C2-Codex\data\staged_v4_remote_cache_20260301_20260331_weekly'
logger.info('probe_state=start')

btc_batch, fx_batch, cached_splits, split_meta = load_feature_batches(cache_root)
loaded_timeframes = tuple(fx_batch.timeframe_batches.keys())
logger.info('probe_state=cache_loaded cached_splits=%d loaded_timeframes=%s', len(cached_splits), loaded_timeframes)
training_cfg = TrainingConfig(anchor_timeframe=fx_batch.anchor_timeframe, outer_holdout_blocks=0, min_train_blocks=2)
splits, split_frequency = _resolve_cached_splits(fx_batch.anchor_timestamps, cached_splits, split_meta, training_cfg, logger, None)
logger.info('probe_state=splits_resolved n_splits=%d split_frequency=%s', len(splits), split_frequency)
split = splits[0]
train_idx = np.asarray(split['train_idx'], dtype=np.int32)
calib_cut = max(4, int(len(train_idx) * 0.2))
train_main_idx = train_idx[:-calib_cut] if len(train_idx) > calib_cut else train_idx
batch_indices = _iter_batches(train_main_idx, training_cfg.batch_size)[0]
logger.info('probe_state=batch_selected batch_size=%d first_idx=%d last_idx=%d', len(batch_indices), int(batch_indices[0]), int(batch_indices[-1]))

device = torch.device('cpu')
logger.info('probe_state=device_selected device=%s cuda_available=%s', device.type, torch.cuda.is_available())
subnet_cfg = SubnetConfig(timeframe_order=loaded_timeframes)
block_cfg = STGNNBlockConfig()
btc_subnet = BTCSubnet(subnet_cfg, block_cfg).to(device)
fx_subnet = FXSubnet(subnet_cfg, block_cfg).to(device)
bridge = ConditionalBridge(block_cfg.output_dim, block_cfg.hidden_dim).to(device)
logger.info('probe_state=models_built')

btc_optimizer = torch.optim.Adam(btc_subnet.parameters(), lr=training_cfg.learning_rate)
fx_params = list(fx_subnet.parameters()) + list(bridge.parameters())
fx_optimizer = torch.optim.Adam(fx_params, lr=training_cfg.learning_rate)
logger.info('probe_state=optimizers_built')

btc_active = 0
fx_active = 0
batch_counter = 0

logger.info('probe_state=stage1_single_batch_start')
btc_optimizer.zero_grad()
btc_sequence, btc_state, btc_active = _run_btc_forward(btc_subnet, btc_batch, batch_indices, DEFAULT_SEQ_LENS, device, btc_active, batch_counter)
logger.info('probe_state=stage1_forward_done active=%s', btc_state.active_timeframe)
loss = sum((state.directional_logits.mean() * 0.0) for state in btc_state.timeframe_states.values())
loss.backward()
btc_optimizer.step()
logger.info('probe_state=stage1_single_batch_done')

for param in btc_subnet.parameters():
    param.requires_grad = False
logger.info('probe_state=btc_frozen')

logger.info('probe_state=stage2_single_batch_start')
fx_optimizer.zero_grad()
logger.info('probe_state=before_btc_replay')
with torch.no_grad():
    _, btc_state, btc_active = _run_btc_forward(btc_subnet, btc_batch, batch_indices, DEFAULT_SEQ_LENS, device, btc_active, batch_counter)
logger.info('probe_state=after_btc_replay active=%s', btc_state.active_timeframe)
logger.info('probe_state=before_fx_forward')
fx_sequence, fx_state, fx_active = _run_fx_forward(fx_subnet, bridge, fx_batch, btc_state, batch_indices, DEFAULT_SEQ_LENS, device, fx_active, batch_counter)
logger.info('probe_state=after_fx_forward active=%s', fx_state.active_timeframe)
loss2 = sum((state.directional_logits.mean() * 0.0) for state in fx_state.timeframe_states.values())
loss2.backward()
fx_optimizer.step()
logger.info('probe_state=stage2_single_batch_done')
