from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from staged_v4.config import BacktestConfig, DEFAULT_SEQ_LENS, FX_TRADABLE_NAMES, GAConfig, STGNNBlockConfig, SubnetConfig, TrainingConfig
from staged_v4.contracts import FXFeatureBatch, SubnetSequenceBatch, TimeframeSequenceBatch
from staged_v4.data import (
    StagedPanels,
    build_bridge_batches,
    build_btc_feature_batch,
    build_btc_sequence_batch_from_panels,
    build_walkforward_splits,
    build_fx_feature_batch,
    build_fx_sequence_batch_from_panels,
    generate_synthetic_panels,
    load_feature_batches,
    load_staged_panels,
)
from staged_v4.evaluation.backtest import adjusted_backtest_config, backtest_probabilities
from staged_v4.evaluation.metrics import (
    binary_accuracy,
    binary_auc,
    binary_brier,
    binary_ece,
    binary_log_loss,
    summarize_fold_metrics,
)
from staged_v4.models import BTCSubnet, ConditionalBridge, FXSubnet
from staged_v4.utils.calibration_helpers import apply_platt_scaler, fit_platt_scaler
from staged_v4.utils.ga_search import run_binary_ga
from staged_v4.utils.graph_helpers import build_edge_matrices
from staged_v4.utils.runtime_logging import (
    configure_logging,
    enforce_memory_guard,
    guard_worker_budget,
    log_exception,
    log_progress,
    log_throughput,
    stage_context,
    write_status,
)


def _default_split(n_items: int, purge_bars: int) -> list[dict[str, object]]:
    split = max(int(n_items * 0.7), 1)
    train_idx = np.arange(max(0, split - purge_bars), dtype=np.int32)
    val_idx = np.arange(split, n_items, dtype=np.int32)
    if len(val_idx) == 0:
        val_idx = np.arange(max(1, n_items // 2), n_items, dtype=np.int32)
    return [{"fold": 0, "train_idx": train_idx, "val_idx": val_idx, "split_frequency": "synthetic", "train_blocks": [], "val_block": "synthetic"}]


def _run_memory_guard(
    logger: logging.Logger,
    status_file: str | None,
    training_cfg: TrainingConfig,
    stage: str,
    *,
    device: torch.device,
    gc_collect: bool = True,
    details: dict[str, object] | None = None,
) -> None:
    guard = enforce_memory_guard(
        logger,
        status_file,
        stage,
        min_available_mb=training_cfg.memory_guard_min_available_mb,
        critical_available_mb=training_cfg.memory_guard_critical_available_mb,
        details=details,
        raise_on_critical=False,
    )
    if guard["state"] in {"low", "critical"} and gc_collect:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        gpu_guard = _gpu_memory_state(
            device,
            min_available_mb=training_cfg.gpu_memory_guard_min_mb,
            critical_available_mb=training_cfg.gpu_memory_guard_critical_mb,
        )
        gpu_payload = {
            "state": "running",
            "stage": "gpu_memory_guard",
            "details": {
                "target_stage": stage,
                **(details or {}),
                "gpu_memory_state": gpu_guard["state"],
                "vram_free_mb": gpu_guard["vram_free_mb"],
                "vram_total_mb": gpu_guard["vram_total_mb"],
                "vram_allocated_mb": gpu_guard["vram_allocated_mb"],
                "vram_reserved_mb": gpu_guard["vram_reserved_mb"],
            },
        }
        if gpu_guard["state"] == "low":
            logger.warning(
                "stage=%s gpu_memory_guard=low vram_free_mb=%.2f vram_total_mb=%.2f",
                stage,
                gpu_guard["vram_free_mb"] or 0.0,
                gpu_guard["vram_total_mb"] or 0.0,
            )
            write_status(status_file, gpu_payload)
        elif gpu_guard["state"] == "critical":
            logger.error(
                "stage=%s gpu_memory_guard=critical vram_free_mb=%.2f vram_total_mb=%.2f",
                stage,
                gpu_guard["vram_free_mb"] or 0.0,
                gpu_guard["vram_total_mb"] or 0.0,
            )
            write_status(status_file, gpu_payload)
            if gc_collect:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gpu_guard = _gpu_memory_state(
                    device,
                    min_available_mb=training_cfg.gpu_memory_guard_min_mb,
                    critical_available_mb=training_cfg.gpu_memory_guard_critical_mb,
                )
            if gpu_guard["state"] == "critical":
                raise MemoryError(
                    f"GPU memory guard tripped at stage={stage}: "
                    f"vram_free_mb={gpu_guard['vram_free_mb']} < "
                    f"critical_available_mb={gpu_guard['critical_available_mb']}"
                )
    if guard["state"] == "critical":
        enforce_memory_guard(
            logger,
            status_file,
            stage,
            min_available_mb=training_cfg.memory_guard_min_available_mb,
            critical_available_mb=training_cfg.memory_guard_critical_available_mb,
            details=details,
            raise_on_critical=True,
        )


def _gpu_memory_state(
    device: torch.device,
    *,
    min_available_mb: float,
    critical_available_mb: float,
) -> dict[str, float | str | None]:
    if device.type != "cuda":
        return {
            "state": "ok",
            "vram_free_mb": None,
            "vram_total_mb": None,
            "vram_allocated_mb": None,
            "vram_reserved_mb": None,
            "min_available_mb": float(min_available_mb),
            "critical_available_mb": float(critical_available_mb),
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mb = free_bytes / (1024 ** 2)
    state = "ok"
    if free_mb < critical_available_mb:
        state = "critical"
    elif free_mb < min_available_mb:
        state = "low"
    return {
        "state": state,
        "vram_free_mb": float(free_mb),
        "vram_total_mb": float(total_bytes / (1024 ** 2)),
        "vram_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
        "vram_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        "min_available_mb": float(min_available_mb),
        "critical_available_mb": float(critical_available_mb),
    }


def _log_gpu_readiness(
    logger: logging.Logger,
    status_file: str | None,
    device: torch.device,
    training_cfg: TrainingConfig,
    *,
    fold: int,
    fold_total: int,
    amp_dtype: torch.dtype | None,
    use_grad_scaler: bool,
) -> None:
    if device.type != "cuda":
        return
    gpu_guard = _gpu_memory_state(
        device,
        min_available_mb=training_cfg.gpu_memory_guard_min_mb,
        critical_available_mb=training_cfg.gpu_memory_guard_critical_mb,
    )
    logger.info(
        "fold=%d/%d state=gpu_ready amp_dtype=%s use_grad_scaler=%s vram_free_mb=%.2f vram_total_mb=%.2f",
        fold + 1,
        fold_total,
        str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
        use_grad_scaler,
        gpu_guard["vram_free_mb"] or 0.0,
        gpu_guard["vram_total_mb"] or 0.0,
    )
    write_status(
        status_file,
        {
            "state": "running",
            "stage": "gpu_ready",
            "fold": fold,
            "fold_total": fold_total,
            "details": {
                "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
                "use_grad_scaler": use_grad_scaler,
                "use_torch_compile": training_cfg.use_torch_compile,
                "gpu_memory_state": gpu_guard["state"],
                "vram_free_mb": gpu_guard["vram_free_mb"],
                "vram_total_mb": gpu_guard["vram_total_mb"],
                "vram_allocated_mb": gpu_guard["vram_allocated_mb"],
                "vram_reserved_mb": gpu_guard["vram_reserved_mb"],
            },
        },
    )


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()


def _maybe_compile_cuda_forward(module: nn.Module, logger: logging.Logger, label: str, enabled: bool) -> None:
    if not enabled:
        return
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.warning("state=torch_compile_unavailable target=%s reason=missing_api", label)
        return
    try:
        module.forward = compile_fn(module.forward, mode="reduce-overhead")
        logger.info("state=torch_compile_enabled target=%s mode=reduce-overhead", label)
    except Exception as exc:
        logger.warning("state=torch_compile_disabled target=%s reason=%s", label, exc)


def _slice_window(arr: np.ndarray, end_idx: int, seq_len: int) -> np.ndarray:
    out = np.zeros((seq_len,) + arr.shape[1:], dtype=arr.dtype)
    start = max(0, end_idx - seq_len + 1)
    window = arr[start : end_idx + 1]
    out[-len(window) :] = window
    return out


def _build_subnet_sequence_batch(feature_batch, anchor_indices: np.ndarray, seq_lens: dict[str, int], device: torch.device) -> SubnetSequenceBatch:
    timeframe_batches: dict[str, TimeframeSequenceBatch] = {}
    node_names = feature_batch.node_names
    tradable_names = getattr(feature_batch, "tradable_node_names", feature_batch.node_names)
    tradable_indices = tuple(getattr(feature_batch, "tradable_indices", tuple(range(len(node_names)))))
    for timeframe, tf_batch in feature_batch.timeframe_batches.items():
        seq_len = int(seq_lens[timeframe])
        lookup = feature_batch.anchor_lookup[timeframe]
        end_indices = lookup[anchor_indices]
        node_features = np.stack([_slice_window(tf_batch.node_features, int(end_idx), seq_len) for end_idx in end_indices])
        tpo_features = np.stack([_slice_window(tf_batch.tpo_features, int(end_idx), seq_len) for end_idx in end_indices])
        volatility = np.stack([_slice_window(tf_batch.volatility, int(end_idx), seq_len) for end_idx in end_indices])
        valid_mask = np.stack([_slice_window(tf_batch.valid_mask, int(end_idx), seq_len) for end_idx in end_indices])
        market_open_mask = np.stack([_slice_window(tf_batch.market_open_mask[:, None], int(end_idx), seq_len).squeeze(-1) for end_idx in end_indices])
        overlap_mask = np.stack([_slice_window(tf_batch.overlap_mask[:, None], int(end_idx), seq_len).squeeze(-1) for end_idx in end_indices])
        session_codes = np.stack([_slice_window(tf_batch.session_codes[:, None], int(end_idx), seq_len).squeeze(-1) for end_idx in end_indices])
        timestamps = np.stack([_slice_window(tf_batch.timestamps[:, None], int(end_idx), seq_len).squeeze(-1) for end_idx in end_indices])

        direction_labels = None
        entry_labels = None
        label_valid_mask = None
        if tf_batch.direction_labels is not None:
            direction_labels = tf_batch.direction_labels[end_indices]
            entry_labels = tf_batch.entry_labels[end_indices]
            label_valid_mask = tf_batch.label_valid_mask[end_indices]
        timeframe_batches[timeframe] = TimeframeSequenceBatch(
            timeframe=timeframe,
            node_names=node_names,
            tradable_indices=tradable_indices,
            timestamps=timestamps,
            node_features=torch.tensor(node_features, dtype=torch.float32, device=device),
            tpo_features=torch.tensor(tpo_features, dtype=torch.float32, device=device),
            volatility=torch.tensor(volatility, dtype=torch.float32, device=device),
            valid_mask=torch.tensor(valid_mask, dtype=torch.bool, device=device),
            market_open_mask=torch.tensor(market_open_mask, dtype=torch.bool, device=device),
            overlap_mask=torch.tensor(overlap_mask, dtype=torch.bool, device=device),
            session_codes=torch.tensor(session_codes, dtype=torch.long, device=device),
            direction_labels=torch.tensor(direction_labels, dtype=torch.float32, device=device) if direction_labels is not None else None,
            entry_labels=torch.tensor(entry_labels, dtype=torch.float32, device=device) if entry_labels is not None else None,
            label_valid_mask=torch.tensor(label_valid_mask, dtype=torch.bool, device=device) if label_valid_mask is not None else None,
        )
    return SubnetSequenceBatch(
        subnet_name=getattr(feature_batch, "subnet_name", "subnet"),
        timeframe_batches=timeframe_batches,
        node_names=node_names,
        tradable_node_names=tradable_names,
        anchor_timestamps=feature_batch.anchor_timestamps[anchor_indices],
    )


def _build_edge_tensors(sequence_batch: SubnetSequenceBatch, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for timeframe, tf_batch in sequence_batch.timeframe_batches.items():
        edge_accumulator: dict[str, list[np.ndarray]] = {"rolling_corr": [], "fundamental": [], "session": []}
        for batch_idx in range(tf_batch.node_features.shape[0]):
            close_window = tf_batch.node_features[batch_idx, :, :, 3].detach().cpu().numpy()
            market_open = np.full(len(tf_batch.node_names), bool(tf_batch.market_open_mask[batch_idx, -1].item()))
            edge_map = build_edge_matrices(close_window, tuple(tf_batch.node_names), market_open)
            for key, value in edge_map.items():
                edge_accumulator[key].append(value)
        out[timeframe] = {
            key: torch.tensor(np.stack(values), dtype=torch.float32, device=device)
            for key, values in edge_accumulator.items()
        }
    return out


def _compute_subnet_loss(
    subnet_state,
    sequence_batch: SubnetSequenceBatch,
    active_timeframe: str,
    active_boost: float,
    label_smooth: float = 0.0,
) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for timeframe, state in subnet_state.timeframe_states.items():
        tf_batch = sequence_batch.timeframe_batches[timeframe]
        if tf_batch.direction_labels is None or tf_batch.label_valid_mask is None:
            continue
        tradable_mask = torch.zeros(state.directional_logits.shape[1], dtype=torch.bool, device=state.directional_logits.device)
        tradable_mask[list(tf_batch.tradable_indices)] = True
        valid = tf_batch.label_valid_mask & tradable_mask.unsqueeze(0)
        if not valid.any():
            continue
        dir_targets = tf_batch.direction_labels[valid]
        if label_smooth > 0.0:
            dir_targets = dir_targets * (1.0 - label_smooth) + 0.5 * label_smooth
        dir_loss = loss_fn(state.directional_logits[valid], dir_targets)
        total = dir_loss
        if state.entry_logits is not None and tf_batch.entry_labels is not None:
            entry_targets = tf_batch.entry_labels[valid]
            if label_smooth > 0.0:
                entry_targets = entry_targets * (1.0 - label_smooth) + 0.5 * label_smooth
            entry_loss = loss_fn(state.entry_logits[valid], entry_targets)
            total = total + 0.5 * entry_loss
        if timeframe == active_timeframe:
            total = total * active_boost
        losses.append(total)
    if not losses:
        return torch.tensor(0.0, device=next(iter(subnet_state.timeframe_states.values())).directional_logits.device)
    return torch.stack(losses).mean()


def _iter_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _run_btc_forward(btc_subnet: BTCSubnet, btc_source, anchor_indices: np.ndarray, seq_lens: dict[str, int], device: torch.device, active_idx: int, batch_index: int):
    if isinstance(btc_source, StagedPanels):
        sequence_batch = build_btc_sequence_batch_from_panels(btc_source, anchor_indices, seq_lens, device)
    else:
        sequence_batch = _build_subnet_sequence_batch(btc_source, anchor_indices, seq_lens, device)
    edges = _build_edge_tensors(sequence_batch, device)
    step_output = btc_subnet.cooperative_step(sequence_batch.timeframe_batches, edges, active_idx, batch_index)
    return sequence_batch, step_output.state, step_output.next_active_idx


def _run_fx_forward(
    fx_subnet: FXSubnet,
    bridge: ConditionalBridge,
    fx_source,
    btc_state,
    anchor_indices: np.ndarray,
    seq_lens: dict[str, int],
    device: torch.device,
    active_idx: int,
    batch_index: int,
    include_signal_only_tpo: bool = True,
):
    if isinstance(fx_source, StagedPanels):
        sequence_batch = build_fx_sequence_batch_from_panels(
            fx_source,
            anchor_indices,
            seq_lens,
            device,
            include_signal_only_tpo=include_signal_only_tpo,
        )
    else:
        sequence_batch = _build_subnet_sequence_batch(fx_source, anchor_indices, seq_lens, device)
    edges = _build_edge_tensors(sequence_batch, device)
    bridge_contexts = {}
    for timeframe, tf_batch in sequence_batch.timeframe_batches.items():
        btc_context = btc_state.timeframe_states[timeframe].pooled_context
        overlap_mask = tf_batch.overlap_mask[:, -1]
        bridge_contexts[timeframe] = bridge(btc_context, overlap_mask, n_nodes=len(tf_batch.node_names))
    fx_state, next_active = fx_subnet.cooperative_step(sequence_batch.timeframe_batches, edges, bridge_contexts, active_idx, batch_index)
    return sequence_batch, fx_state, next_active


def _collect_anchor_outputs(
    btc_subnet: BTCSubnet,
    fx_subnet: FXSubnet,
    bridge: ConditionalBridge,
    btc_source,
    fx_source,
    anchor_indices: np.ndarray,
    seq_lens: dict[str, int],
    device: torch.device,
    include_signal_only_tpo: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    btc_subnet.eval()
    fx_subnet.eval()
    btc_subnet.reset_exchange_memory()
    fx_subnet.reset_exchange_memory()
    direction_logits_list = []
    entry_logits_list = []
    direction_labels_list = []
    entry_labels_list = []
    valid_list = []
    close_list = []
    high_list = []
    low_list = []
    volatility_list = []
    session_codes_list = []
    active_btc = 0
    active_fx = 0
    with torch.no_grad():
        for batch_index, batch_indices in enumerate(_iter_batches(anchor_indices, 32)):
            _, btc_state, active_btc = _run_btc_forward(btc_subnet, btc_source, batch_indices, seq_lens, device, active_btc, batch_index)
            fx_sequence, fx_state, active_fx = _run_fx_forward(
                fx_subnet,
                bridge,
                fx_source,
                btc_state,
                batch_indices,
                seq_lens,
                device,
                active_fx,
                batch_index,
                include_signal_only_tpo=include_signal_only_tpo,
            )
            anchor_tf = fx_source.anchor_timeframe
            anchor_state = fx_state.timeframe_states[anchor_tf]
            anchor_batch = fx_sequence.timeframe_batches[anchor_tf]
            tradable = list(anchor_batch.tradable_indices)
            direction_logits = anchor_state.directional_logits[:, tradable].detach().cpu().numpy()
            direction_labels = anchor_batch.direction_labels[:, tradable].detach().cpu().numpy()
            if anchor_state.entry_logits is not None:
                entry_logits = anchor_state.entry_logits[:, tradable].detach().cpu().numpy()
            else:
                entry_logits = np.zeros_like(direction_logits)
            if anchor_batch.entry_labels is not None:
                entry_labels = anchor_batch.entry_labels[:, tradable].detach().cpu().numpy()
            else:
                entry_labels = np.zeros_like(direction_labels)
            direction_logits_list.append(direction_logits)
            entry_logits_list.append(entry_logits)
            direction_labels_list.append(direction_labels)
            entry_labels_list.append(entry_labels)
            valid_list.append(anchor_batch.label_valid_mask[:, tradable].detach().cpu().numpy())
            close_list.append(anchor_batch.node_features[:, -1, tradable, 3].detach().cpu().numpy())
            high_list.append(anchor_batch.node_features[:, -1, tradable, 1].detach().cpu().numpy())
            low_list.append(anchor_batch.node_features[:, -1, tradable, 2].detach().cpu().numpy())
            volatility_list.append(anchor_batch.volatility[:, -1, tradable].detach().cpu().numpy())
            session_codes_list.append(anchor_batch.session_codes[:, -1].detach().cpu().numpy())
    return (
        np.vstack(direction_logits_list),
        np.vstack(entry_logits_list),
        np.vstack(direction_labels_list),
        np.vstack(entry_labels_list),
        np.vstack(valid_list),
        np.vstack(close_list),
        np.vstack(high_list),
        np.vstack(low_list),
        np.vstack(volatility_list),
        np.concatenate(session_codes_list),
    )


def _ga_optimize_backtest(
    prob_buy: np.ndarray,
    prob_entry: np.ndarray | None,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volatility: np.ndarray,
    session_codes: np.ndarray,
    pair_names: tuple[str, ...],
    base_cfg: BacktestConfig,
    ga_cfg: GAConfig,
) -> BacktestConfig:
    def _decode(genome: np.ndarray) -> BacktestConfig:
        cfg = deepcopy(base_cfg)
        cfg.base_entry_threshold = 0.55 if genome[0] else 0.60
        cfg.threshold_volatility_coeff = 8.0 if genome[1] else 12.0
        cfg.probability_spread_threshold = 0.08 if genome[2] else 0.12
        cfg.max_group_exposure = 1 if genome[3] else 2
        return cfg

    def _objective(genome: np.ndarray) -> float:
        cfg = _decode(genome)
        result = backtest_probabilities(prob_buy, prob_entry, close, high, low, volatility, session_codes, pair_names, cfg)
        return float(result["strategy_sharpe"])

    result = run_binary_ga(4, _objective, population_size=ga_cfg.population_size, generations=ga_cfg.generations, mutation_rate=ga_cfg.mutation_rate, crossover_rate=ga_cfg.crossover_rate)
    return _decode(result.best_genome)


def _load_references() -> dict[str, object]:
    refs = {}
    benchmark = Path("data/remote_clean_2025_runs/clean_2025_no_bridge_optk10_report.json")
    tpo_branch = Path("data/remote_runs/tpo_normal_backtest_memory_2025.json")
    for name, path in (("compact_benchmark", benchmark), ("tpo_branch", tpo_branch)):
        if path.exists():
            refs[name] = json.loads(path.read_text(encoding="utf-8"))
    return refs


def _resolve_cached_splits(
    anchor_timestamps: np.ndarray,
    cached_splits: list[dict[str, object]],
    split_meta: dict[str, object],
    training_cfg: TrainingConfig,
    logger: logging.Logger,
    status_file: str | None,
) -> tuple[list[dict[str, object]], str]:
    cached_frequency = str(split_meta.get("split_frequency", training_cfg.split_frequency))
    cached_outer_holdout = split_meta.get("outer_holdout_blocks")
    needs_regen = (
        cached_frequency != training_cfg.split_frequency
        or cached_outer_holdout is None
        or int(cached_outer_holdout) != int(training_cfg.outer_holdout_blocks)
    )
    if not needs_regen:
        return cached_splits, cached_frequency

    with stage_context(
        logger,
        status_file,
        "regenerate_cached_splits",
        cached_split_frequency=cached_frequency,
        requested_split_frequency=training_cfg.split_frequency,
        cached_outer_holdout_blocks=cached_outer_holdout,
        requested_outer_holdout_blocks=training_cfg.outer_holdout_blocks,
    ):
        regenerated_splits = build_walkforward_splits(
            anchor_timestamps,
            split_frequency=training_cfg.split_frequency,
            outer_holdout_blocks=training_cfg.outer_holdout_blocks,
            min_train_blocks=training_cfg.min_train_blocks,
            purge_bars=training_cfg.purge_bars,
        )
    if regenerated_splits:
        return regenerated_splits, training_cfg.split_frequency
    logger.warning(
        "state=regenerate_cached_splits_fallback reason=no_regenerated_splits cached_frequency=%s requested_frequency=%s cached_outer_holdout=%s requested_outer_holdout=%s",
        cached_frequency,
        training_cfg.split_frequency,
        cached_outer_holdout,
        training_cfg.outer_holdout_blocks,
    )
    return cached_splits, cached_frequency


def run_staged_experiment(
    mode: str,
    output: str,
    cache_root: str | None = None,
    candle_root: str | None = None,
    tick_root: str | None = None,
    start: str | None = None,
    end: str | None = None,
    anchor_timeframe: str = "M1",
    max_folds: int | None = None,
    strict: bool = False,
    training_cfg: TrainingConfig | None = None,
    subnet_cfg: SubnetConfig | None = None,
    block_cfg: STGNNBlockConfig | None = None,
    backtest_cfg: BacktestConfig | None = None,
    ga_cfg: GAConfig | None = None,
    synthetic_n_anchor: int = 96,
    include_signal_only_tpo: bool = True,
    max_workers: int = 0,
    device_override: str = "auto",
    logger: logging.Logger | None = None,
    status_file: str | None = None,
) -> dict:
    logger = logger or logging.getLogger("train_staged")
    training_cfg = training_cfg or TrainingConfig(anchor_timeframe=anchor_timeframe)
    subnet_cfg = subnet_cfg or SubnetConfig()
    block_cfg = block_cfg or STGNNBlockConfig()
    backtest_cfg = backtest_cfg or BacktestConfig()
    ga_cfg = ga_cfg or GAConfig(population_size=6, generations=0)
    resolved_device = device_override.lower()
    if resolved_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device_override={device_override!r}; expected auto, cpu, or cuda")
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this host")
    device = torch.device(resolved_device)
    logger.info("state=device_selected device=%s cuda_available=%s", device.type, torch.cuda.is_available())

    feature_mode = "prebuilt"
    if cache_root is not None:
        with stage_context(logger, status_file, "load_feature_batches", cache_root=cache_root):
            btc_batch, fx_batch, cached_splits, split_meta = load_feature_batches(cache_root)
            anchor_timeframe = fx_batch.anchor_timeframe
            splits, split_frequency = _resolve_cached_splits(
                fx_batch.anchor_timestamps,
                cached_splits,
                split_meta,
                training_cfg,
                logger,
                status_file,
            )
            split_source = "regenerated" if splits is not cached_splits else "cached"
            logger.info(
                "state=splits_resolved n_splits=%d split_frequency=%s source=%s",
                len(splits),
                split_frequency,
                split_source,
            )
            loaded_timeframes = tuple(fx_batch.timeframe_batches.keys())
            if tuple(subnet_cfg.timeframe_order) != loaded_timeframes:
                subnet_cfg = SubnetConfig(
                    timeframe_order=loaded_timeframes,
                    exchange_every_k_batches=subnet_cfg.exchange_every_k_batches,
                    active_loss_boost=subnet_cfg.active_loss_boost,
                    enable_entry_head=subnet_cfg.enable_entry_head,
                )
        btc_source = btc_batch
        fx_source = fx_batch
        feature_mode = "cached"
    elif mode == "synthetic":
        with stage_context(logger, status_file, "generate_synthetic_panels", n_anchor=synthetic_n_anchor, anchor_timeframe=anchor_timeframe):
            btc_panels, fx_panels = generate_synthetic_panels(n_anchor=synthetic_n_anchor, anchor_timeframe=anchor_timeframe)
        btc_batch = build_btc_feature_batch(btc_panels, timeframes=subnet_cfg.timeframe_order)
        fx_batch = build_fx_feature_batch(
            fx_panels,
            timeframes=subnet_cfg.timeframe_order,
            include_signal_only_tpo=include_signal_only_tpo,
            max_workers=max_workers,
        )
        splits = fx_panels.walkforward_splits
        split_frequency = fx_panels.split_frequency
        btc_source = btc_batch
        fx_source = fx_batch
    else:
        if candle_root is None:
            raise ValueError("candle_root is required for real mode")
        effective_max_workers = max_workers
        if training_cfg.memory_guard_min_available_mb > 0.0 or training_cfg.memory_guard_critical_available_mb > 0.0:
            auto_workers = 4 if device.type == "cuda" else 16
            requested_workers = max_workers if max_workers and max_workers > 0 else auto_workers
            guarded_workers, guard = guard_worker_budget(
                requested_workers,
                min_available_mb=training_cfg.memory_guard_min_available_mb,
                critical_available_mb=training_cfg.memory_guard_critical_available_mb,
            )
            if max_workers and max_workers > 0:
                effective_max_workers = guarded_workers
            elif guard["state"] in {"low", "critical"}:
                effective_max_workers = guarded_workers
            if effective_max_workers != max_workers:
                logger.warning(
                    "stage=load_staged_panels state=memory_guard max_workers=%d->%d available_mb=%s",
                    max_workers,
                    effective_max_workers,
                    guard["available_mb"],
                )
            _run_memory_guard(
                logger,
                status_file,
                training_cfg,
                "load_staged_panels",
                device=device,
                gc_collect=True,
                details={"requested_max_workers": max_workers, "effective_max_workers": effective_max_workers},
            )
        with stage_context(
            logger,
            status_file,
            "load_staged_panels",
            candle_root=candle_root,
            tick_root=tick_root,
            start=start,
            end=end,
            max_workers=effective_max_workers,
        ):
            btc_panels, fx_panels = load_staged_panels(
                candle_root=candle_root,
                tick_root=tick_root,
                start=start,
                end=end,
                anchor_timeframe=anchor_timeframe,
                strict=strict,
                timeframes=subnet_cfg.timeframe_order,
                split_frequency=training_cfg.split_frequency,
                outer_holdout_blocks=training_cfg.outer_holdout_blocks,
                min_train_blocks=training_cfg.min_train_blocks,
                purge_bars=training_cfg.purge_bars,
                logger=logger,
                status_file=status_file,
                max_workers=effective_max_workers,
                memory_guard_min_available_mb=training_cfg.memory_guard_min_available_mb,
                memory_guard_critical_available_mb=training_cfg.memory_guard_critical_available_mb,
            )
        splits = fx_panels.walkforward_splits
        split_frequency = fx_panels.split_frequency
        btc_source = btc_panels
        fx_source = fx_panels
        feature_mode = "jit_panels"

    if not isinstance(fx_source, StagedPanels):
        _ = build_bridge_batches(btc_source, fx_source)
    logger.info("state=feature_mode mode=%s", feature_mode)

    splits = splits or _default_split(len(fx_source.anchor_timestamps), training_cfg.purge_bars)
    if max_folds is not None:
        splits = splits[:max_folds]
    logger.info("state=splits_final n_splits=%d max_folds=%s", len(splits), max_folds)

    fold_results = []
    model_dir = Path(output).with_suffix("")
    model_dir.mkdir(parents=True, exist_ok=True)
    jit_cleanup = isinstance(fx_source if "fx_source" in locals() else None, StagedPanels)

    for fold_index, split in enumerate(splits):
        write_status(status_file, {"state": "running", "stage": "fold", "fold": fold_index, "fold_total": len(splits), "split": {k: v for k, v in split.items() if k not in {"train_idx", "val_idx"}}})
        logger.info("fold=%d/%d state=start split=%s", fold_index + 1, len(splits), json.dumps({k: v for k, v in split.items() if k not in {"train_idx", "val_idx"}}, default=str))
        train_idx = np.asarray(split["train_idx"], dtype=np.int32)
        val_idx = np.asarray(split["val_idx"], dtype=np.int32)
        if len(train_idx) < 8 or len(val_idx) < 4:
            continue
        calib_cut = max(4, int(len(train_idx) * 0.2))
        calib_idx = train_idx[-calib_cut:]
        train_main_idx = train_idx[:-calib_cut] if len(train_idx) > calib_cut else train_idx

        btc_subnet = BTCSubnet(subnet_cfg, block_cfg).to(device)
        fx_subnet = FXSubnet(subnet_cfg, block_cfg).to(device)
        bridge = ConditionalBridge(block_cfg.output_dim, block_cfg.hidden_dim).to(device)
        amp_dtype: torch.dtype | None = None
        grad_scaler: torch.cuda.amp.GradScaler | None = None
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            use_grad_scaler = amp_dtype == torch.float16
            grad_scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
            for timeframe in btc_subnet.timeframe_order:
                _maybe_compile_cuda_forward(
                    btc_subnet.blocks[timeframe],
                    logger,
                    f"btc_block_{timeframe}",
                    training_cfg.use_torch_compile,
                )
            for timeframe in fx_subnet.timeframe_order:
                _maybe_compile_cuda_forward(
                    fx_subnet.blocks[timeframe],
                    logger,
                    f"fx_block_{timeframe}",
                    training_cfg.use_torch_compile,
                )
            _maybe_compile_cuda_forward(bridge, logger, "bridge", training_cfg.use_torch_compile)
            _log_gpu_readiness(
                logger,
                status_file,
                device,
                training_cfg,
                fold=fold_index,
                fold_total=len(splits),
                amp_dtype=amp_dtype,
                use_grad_scaler=use_grad_scaler,
            )

        btc_optimizer = torch.optim.Adam(btc_subnet.parameters(), lr=training_cfg.learning_rate)
        fx_params = list(fx_subnet.parameters()) + list(bridge.parameters())
        fx_optimizer = torch.optim.Adam(fx_params, lr=training_cfg.learning_rate)
        fine_tune_params = list(bridge.parameters()) + list(fx_subnet.blocks[anchor_timeframe].direction_head.parameters())
        fine_tune_optimizer = torch.optim.Adam(fine_tune_params, lr=training_cfg.fine_tune_learning_rate)
        stage1_batches = _iter_batches(train_main_idx, training_cfg.batch_size)
        stage2_batches = _iter_batches(train_main_idx, training_cfg.batch_size)
        stage3_batches = _iter_batches(train_main_idx, training_cfg.batch_size)
        stage1_total_steps = max(len(stage1_batches) * max(training_cfg.epochs_stage1, 1), 1)
        stage2_total_steps = max(len(stage2_batches) * max(training_cfg.epochs_stage2, 1), 1)
        stage3_total_steps = max(len(stage3_batches) * max(training_cfg.epochs_stage3, 1), 1)
        btc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(btc_optimizer, T_max=stage1_total_steps)
        fx_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fx_optimizer, T_max=stage2_total_steps)
        fine_tune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fine_tune_optimizer, T_max=stage3_total_steps)

        btc_active = 0
        fx_active = 0
        batch_counter = 0
        guard_interval = max(1, training_cfg.memory_guard_check_interval)

        for _epoch in range(training_cfg.epochs_stage1):
            btc_subnet.train()
            btc_subnet.reset_exchange_memory()
            epoch_started = time.perf_counter()
            for batch_no, batch_indices in enumerate(stage1_batches, start=1):
                if batch_no == 1 or batch_no % guard_interval == 0 or batch_no == len(stage1_batches):
                    log_progress(logger, status_file, "stage1_btc", batch_no, len(stage1_batches), fold=fold_index, epoch=_epoch)
                    _run_memory_guard(
                        logger,
                        status_file,
                        training_cfg,
                        "stage1_btc",
                        device=device,
                        gc_collect=jit_cleanup,
                        details={"fold": fold_index, "epoch": _epoch, "batch": batch_no, "total_batches": len(stage1_batches)},
                    )
                btc_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_dtype):
                    btc_sequence, btc_state, btc_active = _run_btc_forward(
                        btc_subnet,
                        btc_source,
                        batch_indices,
                        DEFAULT_SEQ_LENS,
                        device,
                        btc_active,
                        batch_counter,
                    )
                    loss = _compute_subnet_loss(
                        btc_state,
                        btc_sequence,
                        btc_state.active_timeframe or anchor_timeframe,
                        subnet_cfg.active_loss_boost,
                        label_smooth=training_cfg.label_smooth,
                    )
                if not loss.requires_grad:
                    batch_counter += 1
                    del btc_sequence, btc_state, loss
                    if jit_cleanup and batch_no % 10 == 0:
                        gc.collect()
                    continue
                _optimizer_step(loss, btc_optimizer, btc_scheduler, grad_scaler)
                batch_counter += 1
                del btc_sequence, btc_state, loss
                if jit_cleanup and batch_no % 10 == 0:
                    gc.collect()
            log_throughput(
                logger,
                "stage1_btc",
                len(stage1_batches),
                time.perf_counter() - epoch_started,
                status_file=status_file,
                fold=fold_index,
                epoch=_epoch,
                batch_size=training_cfg.batch_size,
            )

        for param in btc_subnet.parameters():
            param.requires_grad = False

        for _epoch in range(training_cfg.epochs_stage2):
            fx_subnet.train()
            bridge.train()
            fx_subnet.reset_exchange_memory()
            epoch_started = time.perf_counter()
            for batch_no, batch_indices in enumerate(stage2_batches, start=1):
                if batch_no == 1 or batch_no % guard_interval == 0 or batch_no == len(stage2_batches):
                    log_progress(logger, status_file, "stage2_fx", batch_no, len(stage2_batches), fold=fold_index, epoch=_epoch)
                    _run_memory_guard(
                        logger,
                        status_file,
                        training_cfg,
                        "stage2_fx",
                        device=device,
                        gc_collect=jit_cleanup,
                        details={"fold": fold_index, "epoch": _epoch, "batch": batch_no, "total_batches": len(stage2_batches)},
                    )
                fx_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with _autocast_context(device, amp_dtype):
                        _, btc_state, btc_active = _run_btc_forward(
                            btc_subnet,
                            btc_source,
                            batch_indices,
                            DEFAULT_SEQ_LENS,
                            device,
                            btc_active,
                            batch_counter,
                        )
                with _autocast_context(device, amp_dtype):
                    fx_sequence, fx_state, fx_active = _run_fx_forward(
                        fx_subnet,
                        bridge,
                        fx_source,
                        btc_state,
                        batch_indices,
                        DEFAULT_SEQ_LENS,
                        device,
                        fx_active,
                        batch_counter,
                        include_signal_only_tpo=include_signal_only_tpo,
                    )
                    loss = _compute_subnet_loss(
                        fx_state,
                        fx_sequence,
                        fx_state.active_timeframe or anchor_timeframe,
                        subnet_cfg.active_loss_boost,
                        label_smooth=training_cfg.label_smooth,
                    )
                if not loss.requires_grad:
                    batch_counter += 1
                    del btc_state, fx_sequence, fx_state, loss
                    if jit_cleanup and batch_no % 10 == 0:
                        gc.collect()
                    continue
                _optimizer_step(loss, fx_optimizer, fx_scheduler, grad_scaler)
                batch_counter += 1
                del btc_state, fx_sequence, fx_state, loss
                if jit_cleanup and batch_no % 10 == 0:
                    gc.collect()
            log_throughput(
                logger,
                "stage2_fx",
                len(stage2_batches),
                time.perf_counter() - epoch_started,
                status_file=status_file,
                fold=fold_index,
                epoch=_epoch,
                batch_size=training_cfg.batch_size,
            )

        if training_cfg.epochs_stage3 > 0:
            for param in fx_subnet.blocks[anchor_timeframe].direction_head.parameters():
                param.requires_grad = True
            for _epoch in range(training_cfg.epochs_stage3):
                fx_subnet.train()
                bridge.train()
                epoch_started = time.perf_counter()
                for batch_no, batch_indices in enumerate(stage3_batches, start=1):
                    if batch_no == 1 or batch_no % guard_interval == 0 or batch_no == len(stage3_batches):
                        log_progress(logger, status_file, "stage3_finetune", batch_no, len(stage3_batches), fold=fold_index, epoch=_epoch)
                        _run_memory_guard(
                            logger,
                            status_file,
                            training_cfg,
                            "stage3_finetune",
                            device=device,
                            gc_collect=jit_cleanup,
                            details={"fold": fold_index, "epoch": _epoch, "batch": batch_no, "total_batches": len(stage3_batches)},
                        )
                    fine_tune_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        with _autocast_context(device, amp_dtype):
                            _, btc_state, btc_active = _run_btc_forward(
                                btc_subnet,
                                btc_source,
                                batch_indices,
                                DEFAULT_SEQ_LENS,
                                device,
                                btc_active,
                                batch_counter,
                            )
                    with _autocast_context(device, amp_dtype):
                        fx_sequence, fx_state, fx_active = _run_fx_forward(
                            fx_subnet,
                            bridge,
                            fx_source,
                            btc_state,
                            batch_indices,
                            DEFAULT_SEQ_LENS,
                            device,
                            fx_active,
                            batch_counter,
                            include_signal_only_tpo=include_signal_only_tpo,
                        )
                        loss = _compute_subnet_loss(
                            fx_state,
                            fx_sequence,
                            anchor_timeframe,
                            subnet_cfg.active_loss_boost,
                            label_smooth=training_cfg.label_smooth,
                        )
                    if not loss.requires_grad:
                        batch_counter += 1
                        del btc_state, fx_sequence, fx_state, loss
                        if jit_cleanup and batch_no % 10 == 0:
                            gc.collect()
                        continue
                    _optimizer_step(loss, fine_tune_optimizer, fine_tune_scheduler, grad_scaler)
                    batch_counter += 1
                    del btc_state, fx_sequence, fx_state, loss
                    if jit_cleanup and batch_no % 10 == 0:
                        gc.collect()
                log_throughput(
                    logger,
                    "stage3_finetune",
                    len(stage3_batches),
                    time.perf_counter() - epoch_started,
                    status_file=status_file,
                    fold=fold_index,
                    epoch=_epoch,
                    batch_size=training_cfg.batch_size,
                )

        calib_dir_logits, calib_entry_logits, calib_dir_labels, calib_entry_labels, calib_valid, _, _, _, _, _ = _collect_anchor_outputs(
            btc_subnet,
            fx_subnet,
            bridge,
            btc_source,
            fx_source,
            calib_idx,
            DEFAULT_SEQ_LENS,
            device,
            include_signal_only_tpo=include_signal_only_tpo,
        )
        val_dir_logits, val_entry_logits, val_dir_labels, val_entry_labels, val_valid, close, high, low, volatility, session_codes = _collect_anchor_outputs(
            btc_subnet,
            fx_subnet,
            bridge,
            btc_source,
            fx_source,
            val_idx,
            DEFAULT_SEQ_LENS,
            device,
            include_signal_only_tpo=include_signal_only_tpo,
        )
        calib_dir_logits_flat = calib_dir_logits.reshape(-1)
        calib_entry_logits_flat = calib_entry_logits.reshape(-1)
        calib_dir_labels_flat = calib_dir_labels.reshape(-1)
        calib_entry_labels_flat = calib_entry_labels.reshape(-1)
        calib_valid_flat = calib_valid.reshape(-1)
        calib_dir_valid_flat = calib_valid_flat & (calib_dir_labels_flat >= 0)
        calib_entry_valid_flat = calib_valid_flat & (calib_entry_labels_flat >= 0)
        dir_scaler = fit_platt_scaler(calib_dir_logits_flat, calib_dir_labels_flat, calib_dir_valid_flat)
        entry_scaler = fit_platt_scaler(calib_entry_logits_flat, calib_entry_labels_flat, calib_entry_valid_flat)

        val_dir_logits_flat = val_dir_logits.reshape(-1)
        val_entry_logits_flat = val_entry_logits.reshape(-1)
        val_dir_labels_flat = val_dir_labels.reshape(-1)
        val_valid_flat = val_valid.reshape(-1) & (val_dir_labels_flat >= 0)
        val_prob_flat = apply_platt_scaler(dir_scaler, val_dir_logits_flat)
        val_entry_prob_flat = apply_platt_scaler(entry_scaler, val_entry_logits_flat)
        val_prob = val_prob_flat.reshape(val_dir_logits.shape)
        val_entry_prob = val_entry_prob_flat.reshape(val_entry_logits.shape)

        pair_names = tuple(getattr(fx_source, "tradable_node_names", FX_TRADABLE_NAMES))

        val_ece = binary_ece(val_prob_flat, val_dir_labels_flat, val_valid_flat)
        fold_backtest_cfg = adjusted_backtest_config(backtest_cfg, val_ece)
        if ga_cfg.generations > 0:
            fold_backtest_cfg = _ga_optimize_backtest(
                val_prob,
                val_entry_prob,
                close,
                high,
                low,
                volatility,
                session_codes,
                pair_names,
                backtest_cfg,
                ga_cfg,
            )
            fold_backtest_cfg = adjusted_backtest_config(fold_backtest_cfg, val_ece)
        backtest = backtest_probabilities(val_prob, val_entry_prob, close, high, low, volatility, session_codes, pair_names, fold_backtest_cfg)

        fold_result = {
            "fold": int(split["fold"]),
            "split_frequency": split.get("split_frequency", split_frequency),
            "train_blocks": split.get("train_blocks", []),
            "val_block": split.get("val_block", ""),
            "auc": binary_auc(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "log_loss": binary_log_loss(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "brier": binary_brier(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "ece": val_ece,
            "directional_accuracy": binary_accuracy(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "stage1_val_loss": binary_log_loss(apply_platt_scaler(dir_scaler, calib_dir_logits_flat), calib_dir_labels_flat, calib_dir_valid_flat),
            "n_bars": int(len(val_idx)),
            **{k: v for k, v in backtest.items() if k != "bar_returns"},
            "backtest_cfg": asdict(fold_backtest_cfg),
        }
        fold_results.append(fold_result)
        logger.info("fold=%d state=done auc=%.4f sharpe=%.4f trades=%d", fold_index, fold_result["auc"], fold_result["strategy_sharpe"], fold_result["trade_count"])
        write_status(status_file, {"state": "running", "stage": "fold_complete", "fold": fold_index, "fold_result": fold_result})

        torch.save(
            {
                "btc_subnet": btc_subnet.state_dict(),
                "fx_subnet": fx_subnet.state_dict(),
                "bridge": bridge.state_dict(),
                "direction_scaler": asdict(dir_scaler),
                "entry_scaler": asdict(entry_scaler),
                "fold_result": fold_result,
            },
            model_dir / f"fold_{fold_index}.pt",
        )
        del btc_subnet, fx_subnet, bridge
        del btc_optimizer, fx_optimizer, fine_tune_optimizer
        del btc_scheduler, fx_scheduler, fine_tune_scheduler
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    summary = summarize_fold_metrics(fold_results)
    report = {
        "mode": mode,
        "cache_root": cache_root,
        "anchor_timeframe": anchor_timeframe,
        "split_frequency": split_frequency,
        "timeframes": list(subnet_cfg.timeframe_order),
        "seq_lens": {tf: DEFAULT_SEQ_LENS[tf] for tf in subnet_cfg.timeframe_order},
        "training_config": asdict(training_cfg),
        "subnet_config": asdict(subnet_cfg),
        "block_config": asdict(block_cfg),
        "backtest_config": asdict(backtest_cfg),
        "ga_config": asdict(ga_cfg),
        "feature_mode": feature_mode,
        "include_signal_only_tpo": include_signal_only_tpo,
        "fold_results": fold_results,
        "summary": summary,
        "references": _load_references(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_status(status_file, {"state": "completed", "stage": "train_staged", "summary": summary, "output": str(output_path)})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the staged_v4 BTC/FX cooperative STGNN pipeline.")
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--candle-root", default=None)
    parser.add_argument("--tick-root", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--anchor-timeframe", default="M1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-stage1", type=int, default=1)
    parser.add_argument("--epochs-stage2", type=int, default=1)
    parser.add_argument("--epochs-stage3", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--ga-population", type=int, default=6)
    parser.add_argument("--ga-generations", type=int, default=0)
    parser.add_argument("--synthetic-n-anchor", type=int, default=96)
    parser.add_argument("--tradeable-tpo-only", action="store_true")
    parser.add_argument("--split-frequency", choices=("week", "month"), default="week")
    parser.add_argument("--outer-holdout-blocks", type=int, default=1)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--purge-bars", type=int, default=6)
    parser.add_argument("--label-smooth", type=float, default=0.10)
    parser.add_argument("--max-workers", type=int, default=0, help="Parallel symbol-load workers for real-mode panel builds; 0 = auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--memory-guard-min-mb", type=float, default=4096.0)
    parser.add_argument("--memory-guard-critical-mb", type=float, default=2048.0)
    parser.add_argument("--memory-guard-check-interval", type=int, default=25)
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--gpu-memory-guard-min-mb", type=float, default=4096.0)
    parser.add_argument("--gpu-memory-guard-critical-mb", type=float, default=2048.0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)

    logger = configure_logging("train_staged", args.log_file)
    training_cfg = TrainingConfig(
        anchor_timeframe=args.anchor_timeframe,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        epochs_stage3=args.epochs_stage3,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        purge_bars=args.purge_bars,
        split_frequency=args.split_frequency,
        outer_holdout_blocks=args.outer_holdout_blocks,
        min_train_blocks=args.min_train_blocks,
        label_smooth=args.label_smooth,
        memory_guard_min_available_mb=args.memory_guard_min_mb,
        memory_guard_critical_available_mb=args.memory_guard_critical_mb,
        memory_guard_check_interval=args.memory_guard_check_interval,
        use_torch_compile=not args.disable_torch_compile,
        gpu_memory_guard_min_mb=args.gpu_memory_guard_min_mb,
        gpu_memory_guard_critical_mb=args.gpu_memory_guard_critical_mb,
    )
    ga_cfg = GAConfig(population_size=args.ga_population, generations=args.ga_generations)
    try:
        with stage_context(logger, args.status_file, "train_staged_main", mode=args.mode, output=args.output, cache_root=args.cache_root):
            run_staged_experiment(
                mode=args.mode,
                output=args.output,
                cache_root=args.cache_root,
                candle_root=args.candle_root,
                tick_root=args.tick_root,
                start=args.start,
                end=args.end,
                anchor_timeframe=args.anchor_timeframe,
                max_folds=args.max_folds,
                strict=args.strict,
                training_cfg=training_cfg,
                ga_cfg=ga_cfg,
                synthetic_n_anchor=args.synthetic_n_anchor,
                include_signal_only_tpo=not args.tradeable_tpo_only,
                max_workers=args.max_workers,
                device_override=args.device,
                logger=logger,
                status_file=args.status_file,
            )
        return 0
    except Exception as exc:
        log_exception(logger, args.status_file, "train_staged_main", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
