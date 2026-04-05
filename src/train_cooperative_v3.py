from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from calibration import expected_calibration_error
from cooperative_v3.benchmark import load_best_compact_benchmark
from cooperative_v3.config import (
    CooperativeExchangeConfig,
    DualSubnetSystemConfig,
    HeteroGraphConfig,
    SubnetConfig,
    TemporalAttentionConfig,
)
from cooperative_v3.execution import KellyConfig, run_fractional_kelly_backtest
from cooperative_v3.meta import CatBoostMetaClassifier, MetaFeatureBuilder, MetaFeatureMatrix
from cooperative_v3.real_data import (
    TIMEFRAMES_FROM_M5,
    CooperativeSequenceDataset,
    collate_cooperative_batches,
    load_cooperative_real_dataset,
    summarize_dataset_coverage,
)
from cooperative_v3.synthetic import build_synthetic_dual_subnet_batches
from cooperative_v3.system import DualSubnetCooperativeSystem
from cooperative_v3.contracts import SubnetBatch, TimeframeBatch
from pbo_analysis import compute_pbo
from universe import SUBNET_24x5, SUBNET_24x5_TRADEABLE, SUBNET_24x7

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = object


DEFAULT_SEQ_LENS = {
    "M5": 64,
    "M15": 40,
    "M30": 28,
    "H1": 20,
    "H4": 12,
    "H12": 8,
    "D1": 6,
}

COOPERATION_PAIR_WEIGHT = {
    "M5": 1.35,
    "M15": 1.15,
    "M30": 0.95,
    "H1": 0.75,
    "H4": 0.55,
    "H12": 0.40,
    "D1": 0.25,
    "W1": 0.10,
    "MN1": 0.05,
}


@dataclass(slots=True)
class CooperativeFoldResult:
    fold: int
    auc: float
    log_loss: float
    ece: float
    kelly_sharpe: float
    trade_count: int
    win_rate: float
    confidence_hit_rate: float
    avg_fraction: float
    stage1_val_loss: float | None
    n_bars: int
    n_rows: int


def _safe_float(value) -> float | None:
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _matrix_diagnostics(matrix: MetaFeatureMatrix) -> dict:
    x = np.asarray(matrix.X, dtype=np.float32)
    y = None if matrix.y is None else np.asarray(matrix.y, dtype=np.float32)
    if x.size == 0:
        return {
            "rows": 0,
            "cols": 0,
            "nonfinite": 0,
            "constant_feature_count": 0,
            "nonconstant_feature_count": 0,
            "label_mean": None if y is None or len(y) == 0 else float(y.mean()),
        }
    variances = np.nanvar(x, axis=0)
    nonconstant = variances > 1e-12
    return {
        "rows": int(x.shape[0]),
        "cols": int(x.shape[1]),
        "nonfinite": int((~np.isfinite(x)).sum()),
        "constant_feature_count": int((~nonconstant).sum()),
        "nonconstant_feature_count": int(nonconstant.sum()),
        "label_mean": None if y is None or len(y) == 0 else float(y.mean()),
    }


def _feature_column(matrix: MetaFeatureMatrix, feature_name: str) -> np.ndarray | None:
    if matrix.X.size == 0:
        return None
    try:
        idx = matrix.feature_names.index(feature_name)
    except ValueError:
        return None
    if idx >= matrix.X.shape[1]:
        return None
    return matrix.X[:, idx].astype(np.float32, copy=False)


def _split_csv(raw: str, expected_keys: tuple[str, ...]) -> dict[str, int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} comma-separated values, got {raw!r}")
    return {key: int(value) for key, value in zip(expected_keys, values)}


def _move_subnet_batch(batch: SubnetBatch, device: torch.device) -> SubnetBatch:
    for tf_batch in batch.timeframe_batches.values():
        for attr in ("node_features", "valid_mask", "market_open_mask", "overlap_mask", "session_codes", "target_direction", "target_entry", "target_valid"):
            value = getattr(tf_batch, attr)
            if torch.is_tensor(value):
                setattr(tf_batch, attr, value.to(device))
        tf_batch.edge_matrices = {key: value.to(device) for key, value in tf_batch.edge_matrices.items()}
    if torch.is_tensor(batch.base_indices):
        batch.base_indices = batch.base_indices.to(device)
    return batch


def _build_overlap_day_splits(
    base_timestamps: np.ndarray,
    quarter_ids: np.ndarray,
    session_codes: np.ndarray,
    valid_panel: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    overlap_fold_days: int,
    min_train_blocks: int,
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    base_ts = base_timestamps.astype("datetime64[ns]")
    base_dates = base_ts.astype("datetime64[D]")
    eligible_base = valid_panel.any(axis=1)
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) & eligible_base
    inner_mask = (~np.isin(quarter_ids, list(outer_holdout_quarters))) & eligible_base

    overlap_days = np.unique(base_dates[inner_mask & (session_codes == 4)])
    if len(overlap_days) == 0:
        overlap_days = np.unique(base_dates[inner_mask])

    blocks = []
    for start in range(0, len(overlap_days), overlap_fold_days):
        chunk = overlap_days[start: start + overlap_fold_days]
        if len(chunk) > 0:
            blocks.append(chunk)

    if len(blocks) <= min_train_blocks:
        return [], holdout_mask

    unique_ts = np.unique(base_ts[inner_mask])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block_idx in range(min_train_blocks, len(blocks)):
        train_days = np.concatenate(blocks[:block_idx])
        val_days = blocks[block_idx]
        train_mask = inner_mask & np.isin(base_dates, train_days)
        val_mask = inner_mask & np.isin(base_dates, val_days)
        if purge_bars > 0 and val_mask.any():
            val_start = base_ts[val_mask].min()
            val_pos = int(np.searchsorted(unique_ts, val_start, side="left"))
            purge_ts = unique_ts[max(0, val_pos - purge_bars):val_pos]
            if len(purge_ts) > 0:
                train_mask &= ~np.isin(base_ts, purge_ts)
        if train_mask.any() and val_mask.any():
            splits.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))
    return splits, holdout_mask


def _build_fallback_split(valid_panel: np.ndarray, purge_bars: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    eligible = np.flatnonzero(valid_panel.any(axis=1))
    if len(eligible) < 256:
        raise ValueError("Not enough eligible bars for fallback cooperative split")
    cut = max(128, int(len(eligible) * 0.8))
    train_idx = eligible[:cut]
    val_idx = eligible[cut:]
    if purge_bars > 0 and len(train_idx) > purge_bars:
        train_idx = train_idx[:-purge_bars]
    holdout_mask = np.zeros(valid_panel.shape[0], dtype=bool)
    holdout_mask[val_idx] = True
    return [(train_idx, val_idx)], holdout_mask


def _subnet_loss(
    subnet_state,
    subnet_batch: SubnetBatch,
    tradable_only: bool,
    entry_timeframe: str,
    entry_weight: float,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    indices = list(subnet_batch.tradable_indices if tradable_only else range(len(subnet_batch.node_names)))
    total_loss = None
    total_weight = None
    for timeframe, state in subnet_state.timeframe_states.items():
        tf_batch = subnet_batch.timeframe_batches[timeframe]
        target = tf_batch.target_direction[:, indices].float()
        valid = tf_batch.target_valid[:, indices].float()
        logits = torch.nan_to_num(state.directional_logits[:, indices], nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(logits, target)
        weighted = loss * valid
        total_loss = weighted.sum() if total_loss is None else total_loss + weighted.sum()
        total_weight = valid.sum() if total_weight is None else total_weight + valid.sum()
        if state.entry_logits is not None and timeframe == entry_timeframe:
            entry_logits = torch.nan_to_num(state.entry_logits[:, indices], nan=0.0, posinf=0.0, neginf=0.0)
            entry_target = tf_batch.target_entry[:, indices].float()
            entry_loss = criterion(entry_logits, entry_target) * valid
            total_loss = total_loss + (entry_loss.sum() * entry_weight)
    return total_loss / total_weight.clamp_min(1.0)


def _cooperative_consistency_loss(
    subnet_state,
    subnet_batch: SubnetBatch,
    tradable_only: bool,
    prob_weight: float,
    context_weight: float,
    temperature: float,
    confidence_floor: float,
    regime_floor: float,
) -> torch.Tensor:
    state_values = tuple(subnet_state.timeframe_states.values())
    device = state_values[0].pooled_context.device
    dtype = state_values[0].pooled_context.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if prob_weight <= 0.0 and context_weight <= 0.0:
        return zero

    timeframe_order = tuple(subnet_state.timeframe_states.keys())
    if len(timeframe_order) < 2:
        return zero

    indices = list(subnet_batch.tradable_indices if tradable_only else range(len(subnet_batch.node_names)))
    prob_terms: list[torch.Tensor] = []
    context_terms: list[torch.Tensor] = []

    for student_tf, teacher_tf in zip(timeframe_order[:-1], timeframe_order[1:], strict=False):
        student_state = subnet_state.timeframe_states[student_tf]
        teacher_state = subnet_state.timeframe_states[teacher_tf]
        student_batch = subnet_batch.timeframe_batches[student_tf]
        teacher_batch = subnet_batch.timeframe_batches[teacher_tf]

        student_valid = student_batch.target_valid[:, indices].float()
        teacher_valid = teacher_batch.target_valid[:, indices].float()
        joint_valid = student_valid * teacher_valid

        teacher_logits_raw = torch.nan_to_num(teacher_state.directional_logits[:, indices], nan=0.0, posinf=0.0, neginf=0.0)
        teacher_prob_raw = torch.sigmoid(teacher_logits_raw)
        teacher_confidence = (teacher_prob_raw - 0.5).abs() * 2.0

        teacher_valid_steps = teacher_batch.valid_mask[:, :, indices].float()
        teacher_regime_seq = teacher_batch.node_features[:, :, indices, 9].abs().float()
        teacher_regime = (teacher_regime_seq * teacher_valid_steps).sum(dim=1) / teacher_valid_steps.sum(dim=1).clamp_min(1.0)

        confidence_gate = ((teacher_confidence - confidence_floor) / max(1e-6, 1.0 - confidence_floor)).clamp(0.0, 1.0)
        regime_gate = ((teacher_regime - regime_floor) / max(1e-6, 1.0 - regime_floor)).clamp(0.0, 1.0)
        pair_weight = COOPERATION_PAIR_WEIGHT.get(student_tf, 1.0)
        teacher_gate = joint_valid * confidence_gate * regime_gate * pair_weight

        if prob_weight > 0.0 and float(teacher_gate.sum().item()) > 0.0:
            temp = max(float(temperature), 1e-3)
            student_logit = torch.nan_to_num(student_state.directional_logits[:, indices], nan=0.0, posinf=0.0, neginf=0.0) / temp
            teacher_logit = teacher_logits_raw / temp
            student_prob = torch.sigmoid(student_logit).clamp(1e-5, 1.0 - 1e-5)
            teacher_prob = torch.sigmoid(teacher_logit).clamp(1e-5, 1.0 - 1e-5).detach()
            student_log_prob = torch.stack([torch.log1p(-student_prob), torch.log(student_prob)], dim=-1)
            teacher_prob_2 = torch.stack([1.0 - teacher_prob, teacher_prob], dim=-1)
            teacher_to_student = nn.functional.kl_div(student_log_prob, teacher_prob_2, reduction="none").sum(dim=-1)
            prob_terms.append((teacher_to_student * teacher_gate).sum() / teacher_gate.sum().clamp_min(1.0))

        if context_weight > 0.0:
            active = (teacher_gate.sum(dim=1) > 0).float()
            if float(active.sum().item()) > 0.0:
                student_ctx = torch.nan_to_num(student_state.pooled_context, nan=0.0, posinf=0.0, neginf=0.0)
                teacher_ctx = torch.nan_to_num(teacher_state.pooled_context, nan=0.0, posinf=0.0, neginf=0.0).detach()
                cosine = nn.functional.cosine_similarity(student_ctx, teacher_ctx, dim=-1)
                context_terms.append(((1.0 - cosine) * active).sum() / active.sum().clamp_min(1.0))

    total = zero
    if prob_terms:
        total = total + (prob_weight * torch.stack(prob_terms).mean())
    if context_terms:
        total = total + (context_weight * torch.stack(context_terms).mean())
    return total


def _batch_input_diagnostics(subnet_batch: SubnetBatch) -> dict:
    payload = {}
    for timeframe, tf_batch in subnet_batch.timeframe_batches.items():
        node_features = tf_batch.node_features.detach()
        valid_mask = tf_batch.valid_mask.detach()
        target_valid = tf_batch.target_valid.detach() if tf_batch.target_valid is not None else None
        payload[timeframe] = {
            "feature_nonfinite": int((~torch.isfinite(node_features)).sum().item()),
            "valid_ratio": float(valid_mask.float().mean().item()),
            "all_invalid_sequences": int((~valid_mask.any(dim=1)).sum().item()),
            "target_valid_ratio": None if target_valid is None else float(target_valid.float().mean().item()),
        }
    return payload


def _state_output_diagnostics(subnet_state) -> dict:
    payload = {}
    for timeframe, state in subnet_state.timeframe_states.items():
        payload[timeframe] = {
            "direction_logit_nonfinite": int((~torch.isfinite(state.directional_logits)).sum().item()),
            "entry_logit_nonfinite": 0 if state.entry_logits is None else int((~torch.isfinite(state.entry_logits)).sum().item()),
            "embedding_nonfinite": int((~torch.isfinite(state.node_embeddings)).sum().item()),
            "context_nonfinite": int((~torch.isfinite(state.pooled_context)).sum().item()),
        }
    return payload


def _inspect_first_batch(system, loader, device, bridge_enabled: bool = True) -> dict:
    try:
        system.reset_cooperative_state()
        batch_index, (btc_batch, fx_batch) = next(enumerate(loader, start=1))
    except StopIteration:
        return {"empty_loader": True}

    with torch.no_grad():
        btc_batch = _move_subnet_batch(btc_batch, device)
        fx_batch = _move_subnet_batch(fx_batch, device)
        output = system(btc_batch, fx_batch, batch_index=batch_index, bridge_enabled=bridge_enabled)
        meta = output.meta_features
        return {
            "btc_input": _batch_input_diagnostics(btc_batch),
            "fx_input": _batch_input_diagnostics(fx_batch),
            "btc_output": _state_output_diagnostics(output.btc_state),
            "fx_output": _state_output_diagnostics(output.fx_state),
            "meta": {
                "rows": int(meta.X.shape[0]),
                "cols": int(meta.X.shape[1]) if meta.X.ndim == 2 else 0,
                "feature_nonfinite": int((~np.isfinite(meta.X)).sum()) if meta.X.size > 0 else 0,
            },
        }


def _train_subnet_stage(system, loader, optimizer, device, stage: str, args) -> float:
    system.train()
    system.reset_cooperative_state()
    total_loss = 0.0
    n_batches = 0
    for batch_index, (btc_batch, fx_batch) in enumerate(loader, start=1):
        btc_batch = _move_subnet_batch(btc_batch, device)
        fx_batch = _move_subnet_batch(fx_batch, device)

        if stage == "btc":
            state = system.forward_btc_only(btc_batch, batch_index=batch_index)
            loss = _subnet_loss(state, btc_batch, tradable_only=False, entry_timeframe=args.entry_timeframe, entry_weight=args.entry_loss_weight)
            loss = loss + _cooperative_consistency_loss(
                state,
                btc_batch,
                tradable_only=False,
                prob_weight=args.cooperation_prob_weight,
                context_weight=args.cooperation_context_weight,
                temperature=args.cooperation_temperature,
                confidence_floor=args.cooperation_confidence_floor,
                regime_floor=args.cooperation_regime_floor,
            )
        elif stage == "fx":
            state = system.forward_fx_only(fx_batch, batch_index=batch_index, incoming_contexts=None)
            loss = _subnet_loss(state, fx_batch, tradable_only=True, entry_timeframe=args.entry_timeframe, entry_weight=args.entry_loss_weight)
            loss = loss + _cooperative_consistency_loss(
                state,
                fx_batch,
                tradable_only=True,
                prob_weight=args.cooperation_prob_weight,
                context_weight=args.cooperation_context_weight,
                temperature=args.cooperation_temperature,
                confidence_floor=args.cooperation_confidence_floor,
                regime_floor=args.cooperation_regime_floor,
            )
        elif stage == "bridge":
            output = system(btc_batch, fx_batch, batch_index=batch_index, bridge_enabled=True)
            loss = _subnet_loss(output.fx_state, fx_batch, tradable_only=True, entry_timeframe=args.entry_timeframe, entry_weight=args.entry_loss_weight)
            loss = loss + _cooperative_consistency_loss(
                output.fx_state,
                fx_batch,
                tradable_only=True,
                prob_weight=args.cooperation_prob_weight,
                context_weight=args.cooperation_context_weight,
                temperature=args.cooperation_temperature,
                confidence_floor=args.cooperation_confidence_floor,
                regime_floor=args.cooperation_regime_floor,
            )
        else:
            raise ValueError(f"Unsupported stage {stage!r}")

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected in stage={stage} batch={batch_index}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _collect_meta_matrix(system, loader, device, bridge_enabled: bool, args) -> tuple[MetaFeatureMatrix, float]:
    system.eval()
    system.reset_cooperative_state()
    rows = []
    refs = []
    y_parts = []
    feature_names: list[str] | None = None
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_index, (btc_batch, fx_batch) in enumerate(loader, start=1):
            btc_batch = _move_subnet_batch(btc_batch, device)
            fx_batch = _move_subnet_batch(fx_batch, device)
            output = system(btc_batch, fx_batch, batch_index=batch_index, bridge_enabled=bridge_enabled)
            loss = _subnet_loss(output.fx_state, fx_batch, tradable_only=True, entry_timeframe=system.meta_builder.micro_timeframe, entry_weight=0.0)
            loss = loss + _cooperative_consistency_loss(
                output.fx_state,
                fx_batch,
                tradable_only=True,
                prob_weight=args.cooperation_prob_weight if hasattr(args, "cooperation_prob_weight") else 0.0,
                context_weight=args.cooperation_context_weight if hasattr(args, "cooperation_context_weight") else 0.0,
                temperature=args.cooperation_temperature if hasattr(args, "cooperation_temperature") else 1.0,
                confidence_floor=args.cooperation_confidence_floor if hasattr(args, "cooperation_confidence_floor") else 0.0,
                regime_floor=args.cooperation_regime_floor if hasattr(args, "cooperation_regime_floor") else 0.0,
            )
            total_loss += float(loss.item())
            n_batches += 1
            matrix = output.meta_features
            if matrix.X.size == 0:
                continue
            if feature_names is None:
                feature_names = list(matrix.feature_names)
            rows.append(matrix.X)
            refs.extend(matrix.references)
            if matrix.y is not None:
                y_parts.append(matrix.y)
    if rows:
        X = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        y = None if not y_parts else np.concatenate(y_parts).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, 0), dtype=np.float32)
        y = np.zeros(0, dtype=np.float32)
    return MetaFeatureMatrix(X=X, feature_names=feature_names or [], references=refs, y=y), total_loss / max(n_batches, 1)


def _compute_metrics(probabilities: np.ndarray, matrix: MetaFeatureMatrix, dataset, args) -> dict:
    if matrix.y is None or len(matrix.y) == 0:
        return {
            "auc": 0.5,
            "log_loss": 0.6931471805599453,
            "ece": 0.0,
            "kelly_sharpe": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "confidence_hit_rate": 0.0,
            "avg_fraction": 0.0,
            "n_rows": 0,
        }
    y_true = matrix.y.astype(np.int32, copy=False)
    prob = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    proba_2col = np.column_stack([1.0 - prob, prob])
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.5
    ll = log_loss(y_true, proba_2col, labels=[0, 1])
    ece = expected_calibration_error(np.column_stack([1.0 - prob, np.zeros_like(prob), prob]), y_true * 2, n_bins=10)
    refs = matrix.references
    base_idx = np.asarray([int(ref["base_index"]) for ref in refs], dtype=np.int32)
    node_idx = np.asarray([int(ref["node_index"]) for ref in refs], dtype=np.int32)
    forward_returns = dataset.fx_forward_returns[base_idx, node_idx]
    trade_session_mask = None
    open_ready = None
    momentum_score = None
    volatility_ratio = None
    breakout_signed = None
    if not getattr(args, "disable_scalp_gate", False):
        trade_session_mask = _feature_column(matrix, "trade_session")
        open_ready = _feature_column(matrix, "open_ready")
        momentum_score = _feature_column(matrix, "momentum_score")
        volatility_ratio = _feature_column(matrix, "volatility_ratio")
        breakout_signed = _feature_column(matrix, "breakout_signed")
    lap_neighbor_momentum = None if getattr(args, "disable_laplacian_gate", False) else _feature_column(matrix, "lap_neighbor_momentum")
    lap_laggard_score = None if getattr(args, "disable_laplacian_gate", False) else _feature_column(matrix, "lap_laggard_score")
    kelly = run_fractional_kelly_backtest(
        probabilities=prob,
        labels=y_true,
        forward_returns=forward_returns,
        threshold=args.kelly_threshold,
        payoff_ratio=args.kelly_payoff_ratio,
        config=KellyConfig(
            fractional_scale=args.kelly_fractional_scale,
            max_fraction_per_trade=args.kelly_max_fraction,
            portfolio_cap=args.kelly_portfolio_cap,
        ),
        trade_session_mask=None if trade_session_mask is None else trade_session_mask >= 0.5,
        open_ready=None if open_ready is None else open_ready >= 0.5,
        momentum_score=momentum_score,
        momentum_floor=args.scalp_momentum_floor,
        volatility_ratio=volatility_ratio,
        volatility_ratio_floor=args.scalp_volatility_ratio_floor,
        volatility_ratio_cap=args.scalp_volatility_ratio_cap,
        breakout_signed=breakout_signed,
        breakout_floor=args.scalp_breakout_floor,
        scalp_hard_gate=args.scalp_hard_gate,
        lap_neighbor_momentum=lap_neighbor_momentum,
        lap_laggard_score=lap_laggard_score,
        lap_laggard_floor=args.lap_laggard_floor,
        lap_allocation_scale=args.lap_allocation_scale,
        lap_alignment_scale=args.lap_alignment_scale,
        lap_hard_gate=args.lap_hard_gate,
    )
    return {
        "auc": float(auc),
        "log_loss": float(ll),
        "ece": float(ece),
        "kelly_sharpe": float(kelly.sharpe),
        "trade_count": int(kelly.trade_count),
        "win_rate": float(kelly.win_rate),
        "confidence_hit_rate": float(kelly.confidence_hit_rate),
        "avg_fraction": float(kelly.avg_fraction),
        "n_rows": int(len(prob)),
    }


def _build_system(args, timeframes: tuple[str, ...]) -> DualSubnetCooperativeSystem:
    config = DualSubnetSystemConfig(
        btc_subnet=SubnetConfig(
            name="btc",
            node_names=tuple(SUBNET_24x7),
            tradable_node_names=tuple(SUBNET_24x7),
            timeframe_order=timeframes,
            enable_entry_head=True,
            hidden_dim=args.hidden_dim,
            input_dim=10,
        ),
        fx_subnet=SubnetConfig(
            name="fx",
            node_names=tuple(SUBNET_24x5),
            tradable_node_names=tuple(SUBNET_24x5_TRADEABLE),
            timeframe_order=timeframes,
            enable_entry_head=True,
            hidden_dim=args.hidden_dim,
            input_dim=10,
        ),
        graph=HeteroGraphConfig(input_dim=10, hidden_dim=args.hidden_dim, output_dim=args.hidden_dim, dropout=args.dropout),
        temporal=TemporalAttentionConfig(hidden_dim=args.hidden_dim, n_heads=args.attn_heads, ff_multiplier=2, dropout=args.dropout),
        exchange=CooperativeExchangeConfig(exchange_every_k_batches=args.exchange_every_k_batches, dropout=args.dropout),
    )
    system = DualSubnetCooperativeSystem(config)
    system.meta_builder = MetaFeatureBuilder(
        macro_timeframe=args.meta_macro_timeframe,
        micro_timeframe=args.meta_micro_timeframe,
        opening_range_bars=args.opening_range_bars,
    )
    return system


def _fit_one_fold(dataset, seq_lens: dict[str, int], train_base_idx: np.ndarray, val_base_idx: np.ndarray, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CooperativeSequenceDataset(dataset, train_base_idx, seq_lens)
    val_ds = CooperativeSequenceDataset(dataset, val_base_idx, seq_lens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_cooperative_batches)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_cooperative_batches)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_cooperative_batches)

    system = _build_system(args, dataset.timeframes).to(device)

    btc_optimizer = torch.optim.AdamW(system.btc_subnet.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    fx_optimizer = torch.optim.AdamW(system.fx_subnet.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bridge_params = list(system.fx_subnet.parameters()) + list(system.bridge.parameters())
    bridge_optimizer = torch.optim.AdamW(bridge_params, lr=args.learning_rate, weight_decay=1e-4)

    for _ in range(args.btc_epochs):
        _train_subnet_stage(system, train_loader, btc_optimizer, device, "btc", args)
    for _ in range(args.fx_epochs):
        _train_subnet_stage(system, train_loader, fx_optimizer, device, "fx", args)
    for param in system.btc_subnet.parameters():
        param.requires_grad_(False)
    for _ in range(args.bridge_epochs):
        _train_subnet_stage(system, train_loader, bridge_optimizer, device, "bridge", args)
    for param in system.btc_subnet.parameters():
        param.requires_grad_(True)

    train_matrix, _ = _collect_meta_matrix(system, train_eval_loader, device, bridge_enabled=True, args=args)
    val_matrix, val_loss = _collect_meta_matrix(system, val_loader, device, bridge_enabled=True, args=args)
    diagnostics = {
        "train_first_batch": _inspect_first_batch(system, train_eval_loader, device, bridge_enabled=True),
        "val_first_batch": _inspect_first_batch(system, val_loader, device, bridge_enabled=True),
        "meta_train": _matrix_diagnostics(train_matrix),
        "meta_val": _matrix_diagnostics(val_matrix),
    }

    meta_model = CatBoostMetaClassifier(
        iterations=args.meta_iterations,
        depth=args.meta_depth,
        learning_rate=args.meta_learning_rate,
        l2_leaf_reg=args.meta_l2_leaf_reg,
        min_data_in_leaf=args.meta_min_data_in_leaf,
        random_strength=args.meta_random_strength,
        subsample=args.meta_subsample,
        rsm=args.meta_rsm,
        max_features=None if args.meta_max_features <= 0 else args.meta_max_features,
    )
    meta_model.fit(train_matrix)
    diagnostics["meta_fit_mode"] = meta_model.fit_mode
    diagnostics["meta_fallback_probability"] = meta_model.fallback_probability
    diagnostics["meta_selected_feature_count"] = 0 if meta_model.selected_feature_indices is None else int(len(meta_model.selected_feature_indices))
    diagnostics["meta_selected_feature_names"] = [] if meta_model.selected_feature_indices is None else [
        train_matrix.feature_names[int(idx)]
        for idx in meta_model.selected_feature_indices.tolist()
        if int(idx) < len(train_matrix.feature_names)
    ]
    val_prob = meta_model.predict_proba(val_matrix)[:, 1]
    metrics = _compute_metrics(val_prob, val_matrix, dataset, args)
    fold_result = CooperativeFoldResult(
        fold=0,
        auc=metrics["auc"],
        log_loss=metrics["log_loss"],
        ece=metrics["ece"],
        kelly_sharpe=metrics["kelly_sharpe"],
        trade_count=metrics["trade_count"],
        win_rate=metrics["win_rate"],
        confidence_hit_rate=metrics["confidence_hit_rate"],
        avg_fraction=metrics["avg_fraction"],
        stage1_val_loss=_safe_float(val_loss),
        n_bars=int(len(val_base_idx)),
        n_rows=metrics["n_rows"],
    )
    return fold_result, system, meta_model, diagnostics


def run_real_training(args: argparse.Namespace) -> dict:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for real cooperative training")
    dataset = load_cooperative_real_dataset(
        data_root=args.data_dir,
        start=args.start,
        end=args.end,
        io_workers=args.io_workers,
        max_base_bars=args.max_base_bars,
        timeframes=TIMEFRAMES_FROM_M5,
        opening_range_bars=args.opening_range_bars,
    )
    dataset_diag = summarize_dataset_coverage(dataset)
    seq_lens = DEFAULT_SEQ_LENS.copy()
    seq_lens.update(_split_csv(args.seq_lens, TIMEFRAMES_FROM_M5))
    valid_panel = dataset.fx_valid[:, [dataset.fx_node_names.index(sym) for sym in dataset.fx_tradable_node_names]]
    fold_splits, holdout_mask = _build_overlap_day_splits(
        dataset.base_timestamps,
        dataset.quarter_ids,
        dataset.session_codes,
        valid_panel,
        dataset.outer_holdout_quarters,
        overlap_fold_days=args.overlap_fold_days,
        min_train_blocks=args.min_train_blocks,
        purge_bars=args.purge_bars,
    )
    split_strategy = "overlap"
    if not fold_splits:
        fold_splits, holdout_mask = _build_fallback_split(valid_panel, purge_bars=args.purge_bars)
        split_strategy = "fallback_chrono"
    if args.max_folds is not None:
        fold_splits = fold_splits[: args.max_folds]

    fold_results: list[CooperativeFoldResult] = []
    fold_diagnostics: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        result, _, _, diagnostics = _fit_one_fold(dataset, seq_lens, train_idx, val_idx, args)
        result.fold = fold_idx
        fold_results.append(result)
        fold_diagnostics.append({"fold": fold_idx, **diagnostics})

    final_train_idx = np.flatnonzero((~holdout_mask) & valid_panel.any(axis=1))
    final_val_idx = np.flatnonzero(holdout_mask)
    outer_result, final_system, final_meta, outer_diag = _fit_one_fold(dataset, seq_lens, final_train_idx, final_val_idx, args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "cooperative_v3_stage1.pt"
    meta_path = model_dir / "cooperative_v3_meta.cbm"
    meta_json = model_dir / "cooperative_v3_meta.json"
    torch.save(
        {
            "state_dict": final_system.state_dict(),
            "timeframes": list(dataset.timeframes),
            "seq_lens": seq_lens,
            "exchange_every_k_batches": args.exchange_every_k_batches,
            "cooperation_prob_weight": args.cooperation_prob_weight,
            "cooperation_context_weight": args.cooperation_context_weight,
            "cooperation_temperature": args.cooperation_temperature,
            "cooperation_confidence_floor": args.cooperation_confidence_floor,
            "cooperation_regime_floor": args.cooperation_regime_floor,
            "meta_macro_timeframe": args.meta_macro_timeframe,
            "meta_micro_timeframe": args.meta_micro_timeframe,
            "opening_range_bars": args.opening_range_bars,
            "meta_iterations": args.meta_iterations,
            "meta_depth": args.meta_depth,
            "meta_learning_rate": args.meta_learning_rate,
            "meta_l2_leaf_reg": args.meta_l2_leaf_reg,
            "meta_min_data_in_leaf": args.meta_min_data_in_leaf,
            "meta_random_strength": args.meta_random_strength,
            "meta_subsample": args.meta_subsample,
            "meta_rsm": args.meta_rsm,
            "meta_max_features": args.meta_max_features,
        },
        weights_path,
    )
    final_meta.save(meta_path)
    meta_json.write_text(
        json.dumps(
            {
                "timeframes": list(dataset.timeframes),
                "seq_lens": seq_lens,
                "exchange_every_k_batches": args.exchange_every_k_batches,
                "cooperation_prob_weight": args.cooperation_prob_weight,
                "cooperation_context_weight": args.cooperation_context_weight,
                "cooperation_temperature": args.cooperation_temperature,
                "cooperation_confidence_floor": args.cooperation_confidence_floor,
                "cooperation_regime_floor": args.cooperation_regime_floor,
                "meta_macro_timeframe": args.meta_macro_timeframe,
                "meta_micro_timeframe": args.meta_micro_timeframe,
                "opening_range_bars": args.opening_range_bars,
                "meta_iterations": args.meta_iterations,
                "meta_depth": args.meta_depth,
                "meta_learning_rate": args.meta_learning_rate,
                "meta_l2_leaf_reg": args.meta_l2_leaf_reg,
                "meta_min_data_in_leaf": args.meta_min_data_in_leaf,
                "meta_random_strength": args.meta_random_strength,
                "meta_subsample": args.meta_subsample,
                "meta_rsm": args.meta_rsm,
                "meta_max_features": args.meta_max_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pbo_report = compute_pbo([asdict(result) for result in fold_results]) if len(fold_results) >= 2 else {
        "pbo": None,
        "interpretation": "Insufficient folds for PBO (need >= 2)",
    }
    benchmark = load_best_compact_benchmark()
    payload = {
        "mode": "real",
        "benchmark": asdict(benchmark),
        "config": {
            "timeframes": list(dataset.timeframes),
            "seq_lens": seq_lens,
            "start": args.start,
            "end": args.end,
            "max_base_bars": args.max_base_bars,
            "split_strategy": split_strategy,
            "overlap_fold_days": args.overlap_fold_days,
            "min_train_blocks": args.min_train_blocks,
            "purge_bars": args.purge_bars,
            "btc_epochs": args.btc_epochs,
            "fx_epochs": args.fx_epochs,
            "bridge_epochs": args.bridge_epochs,
            "exchange_every_k_batches": args.exchange_every_k_batches,
            "cooperation_prob_weight": args.cooperation_prob_weight,
            "cooperation_context_weight": args.cooperation_context_weight,
            "cooperation_temperature": args.cooperation_temperature,
            "cooperation_confidence_floor": args.cooperation_confidence_floor,
            "cooperation_regime_floor": args.cooperation_regime_floor,
            "meta_macro_timeframe": args.meta_macro_timeframe,
            "meta_micro_timeframe": args.meta_micro_timeframe,
            "opening_range_bars": args.opening_range_bars,
            "meta_iterations": args.meta_iterations,
            "meta_depth": args.meta_depth,
            "meta_learning_rate": args.meta_learning_rate,
            "meta_l2_leaf_reg": args.meta_l2_leaf_reg,
            "meta_min_data_in_leaf": args.meta_min_data_in_leaf,
            "meta_random_strength": args.meta_random_strength,
            "meta_subsample": args.meta_subsample,
            "meta_rsm": args.meta_rsm,
            "meta_max_features": args.meta_max_features,
            "scalp_gate_enabled": not args.disable_scalp_gate,
            "scalp_hard_gate": args.scalp_hard_gate,
            "scalp_momentum_floor": args.scalp_momentum_floor,
            "scalp_volatility_ratio_floor": args.scalp_volatility_ratio_floor,
            "scalp_volatility_ratio_cap": args.scalp_volatility_ratio_cap,
            "scalp_breakout_floor": args.scalp_breakout_floor,
            "laplacian_gate_enabled": not args.disable_laplacian_gate,
            "lap_laggard_floor": args.lap_laggard_floor,
            "lap_allocation_scale": args.lap_allocation_scale,
            "lap_alignment_scale": args.lap_alignment_scale,
            "lap_hard_gate": args.lap_hard_gate,
        },
        "dataset": {
            "n_bars": int(len(dataset.base_timestamps)),
            "outer_holdout_quarters": list(dataset.outer_holdout_quarters),
            "n_overlap_folds": int(len(fold_splits)),
        },
        "diagnostics": {
            "dataset_coverage": dataset_diag,
            "folds": fold_diagnostics,
            "outer_holdout": outer_diag,
        },
        "artifacts": {
            "stage1_weights": str(weights_path),
            "meta_model": str(meta_path),
            "meta_config": str(meta_json),
        },
        "folds": [asdict(result) for result in fold_results],
        "outer_holdout": {
            "quarters": list(dataset.outer_holdout_quarters),
            "n_bars": int(len(final_val_idx)),
            "auc": float(outer_result.auc),
            "log_loss": float(outer_result.log_loss),
            "ece": float(outer_result.ece),
            "kelly_sharpe": float(outer_result.kelly_sharpe),
            "trade_count": int(outer_result.trade_count),
            "win_rate": float(outer_result.win_rate),
            "confidence_hit_rate": float(outer_result.confidence_hit_rate),
            "avg_fraction": float(outer_result.avg_fraction),
            "n_rows": int(outer_result.n_rows),
        },
        "pbo": pbo_report,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_synthetic_smoke(args: argparse.Namespace) -> dict:
    config = _build_system(args, TIMEFRAMES_FROM_M5).config
    system = DualSubnetCooperativeSystem(config)
    system.meta_builder = MetaFeatureBuilder(
        macro_timeframe=args.meta_macro_timeframe,
        micro_timeframe=args.meta_micro_timeframe,
        opening_range_bars=args.opening_range_bars,
    )
    btc_batch, fx_batch = build_synthetic_dual_subnet_batches(config=config)
    with torch.no_grad():
        output = system(
            btc_batch=btc_batch,
            fx_batch=fx_batch,
            batch_index=args.exchange_every_k_batches,
            bridge_enabled=not args.disable_bridge,
        )
    benchmark = load_best_compact_benchmark()
    summary = {
        "benchmark": asdict(benchmark),
        "active_timeframes": list(config.fx_subnet.timeframe_order),
        "bridge_enabled": not args.disable_bridge,
        "bridge_context_timeframes": sorted(output.fx_bridge_contexts.keys()),
        "meta_feature_count": int(output.meta_features.X.shape[1]),
        "meta_row_count": int(output.meta_features.X.shape[0]),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cooperative v3 trainer")
    parser.add_argument("--mode", choices=("real", "smoke"), default="real")
    parser.add_argument("--data-dir", help="DataExtractor root for real training")
    parser.add_argument("--output", default="data/cooperative_v3_report.json")
    parser.add_argument("--model-dir", default="models/cooperative_v3")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-base-bars", type=int, default=4096)
    parser.add_argument("--io-workers", type=int)
    parser.add_argument("--seq-lens", default="64,40,28,20,12,8,6")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--exchange-every-k-batches", type=int, default=8)
    parser.add_argument("--cooperation-prob-weight", type=float, default=0.0125)
    parser.add_argument("--cooperation-context-weight", type=float, default=0.004)
    parser.add_argument("--cooperation-temperature", type=float, default=1.75)
    parser.add_argument("--cooperation-confidence-floor", type=float, default=0.20)
    parser.add_argument("--cooperation-regime-floor", type=float, default=0.15)
    parser.add_argument("--btc-epochs", type=int, default=1)
    parser.add_argument("--fx-epochs", type=int, default=1)
    parser.add_argument("--bridge-epochs", type=int, default=1)
    parser.add_argument("--entry-timeframe", default="M5")
    parser.add_argument("--entry-loss-weight", type=float, default=0.25)
    parser.add_argument("--meta-macro-timeframe", default="H1")
    parser.add_argument("--meta-micro-timeframe", default="M5")
    parser.add_argument("--opening-range-bars", type=int, default=3)
    parser.add_argument("--meta-iterations", type=int, default=200)
    parser.add_argument("--meta-depth", type=int, default=6)
    parser.add_argument("--meta-learning-rate", type=float, default=0.05)
    parser.add_argument("--meta-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--meta-min-data-in-leaf", type=int, default=1)
    parser.add_argument("--meta-random-strength", type=float, default=1.0)
    parser.add_argument("--meta-subsample", type=float, default=1.0)
    parser.add_argument("--meta-rsm", type=float, default=1.0)
    parser.add_argument("--meta-max-features", type=int, default=0)
    parser.add_argument("--overlap-fold-days", type=int, default=10)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--purge-bars", type=int, default=6)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--kelly-threshold", type=float, default=0.06)
    parser.add_argument("--kelly-payoff-ratio", type=float, default=1.0)
    parser.add_argument("--kelly-fractional-scale", type=float, default=0.25)
    parser.add_argument("--kelly-max-fraction", type=float, default=0.05)
    parser.add_argument("--kelly-portfolio-cap", type=float, default=0.25)
    parser.add_argument("--disable-scalp-gate", action="store_true")
    parser.add_argument("--scalp-hard-gate", dest="scalp_hard_gate", action="store_true")
    parser.add_argument("--scalp-soft-gate", dest="scalp_hard_gate", action="store_false")
    parser.set_defaults(scalp_hard_gate=True)
    parser.add_argument("--scalp-momentum-floor", type=float, default=0.15)
    parser.add_argument("--scalp-volatility-ratio-floor", type=float, default=0.50)
    parser.add_argument("--scalp-volatility-ratio-cap", type=float, default=2.50)
    parser.add_argument("--scalp-breakout-floor", type=float, default=0.01)
    parser.add_argument("--disable-laplacian-gate", action="store_true")
    parser.add_argument("--lap-laggard-floor", type=float, default=0.02)
    parser.add_argument("--lap-allocation-scale", type=float, default=0.25)
    parser.add_argument("--lap-alignment-scale", type=float, default=0.35)
    parser.add_argument("--lap-hard-gate", action="store_true")
    parser.add_argument("--disable-bridge", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "real":
        if not args.data_dir:
            parser.error("--data-dir is required for --mode real")
        result = run_real_training(args)
    else:
        result = run_synthetic_smoke(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
