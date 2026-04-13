from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from replay_staged_v5 import _write_json_atomic
from staged_v5.config import (
    FX_TRADABLE_NAMES,
    NeuralGateConfig,
    STGNNBlockConfig,
    SubnetConfig,
    TrainingConfig,
)
from staged_v5.contracts import CalibrationArtifact
from staged_v5.data import load_feature_batches
from staged_v5.execution_gate import (
    NeuralGateAction,
    GATE_STATE_FEATURE_NAMES,
    MicrostructureGate,
    TickProxyStore,
    build_gate_events,
    build_gate_state_vector,
    extract_tradable_anchor_views,
    grpo_update_step,
    simulate_action,
)
from staged_v5.evaluation.backtest import _compute_atr_matrix
from staged_v5.models import BTCSubnet, ConditionalBridge, FXSubnet
from staged_v5.training.train_staged import DEFAULT_SEQ_LENS, _collect_anchor_outputs, _resolve_cached_splits
from staged_v5.utils.calibration_helpers import apply_platt_scaler
from staged_v5.utils.runtime_logging import configure_logging, log_exception, write_status


ACTION_NAMES = tuple(action.name.lower() for action in NeuralGateAction)


def _merge_dataclass_config(cls, payload: dict | None):
    merged = {field.name: getattr(cls(), field.name) for field in fields(cls)}
    if payload:
        for key, value in payload.items():
            if key in merged:
                merged[key] = value
    return cls(**merged)


def _load_run_metadata(run_path: Path, cache_root: str | None) -> tuple[TrainingConfig, SubnetConfig, STGNNBlockConfig, bool]:
    report_path = run_path / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        include_signal_only_tpo = bool(report.get("include_signal_only_tpo", True))
        return (
            _merge_dataclass_config(TrainingConfig, report.get("training_config")),
            _merge_dataclass_config(SubnetConfig, report.get("subnet_config")),
            _merge_dataclass_config(STGNNBlockConfig, report.get("block_config")),
            include_signal_only_tpo,
        )
    include_signal_only_tpo = True
    if cache_root is not None:
        manifest_path = Path(cache_root) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            include_signal_only_tpo = bool(manifest.get("include_signal_only_tpo", True))
    return TrainingConfig(), SubnetConfig(), STGNNBlockConfig(), include_signal_only_tpo


def _load_checkpoint_paths(run_path: Path) -> list[Path]:
    checkpoint_dir = run_path / "report"
    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")
    return checkpoint_paths


def _sample_group_actions(
    logits: torch.Tensor,
    group_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_logits = logits.squeeze(0)
    if base_logits.ndim != 1:
        raise ValueError(f"Expected logits with shape [1, num_actions] or [num_actions], got {tuple(logits.shape)}")
    dist = torch.distributions.Categorical(logits=base_logits / max(float(temperature), 1e-6))
    actions = dist.sample((group_size,))
    old_log_probs = dist.log_prob(actions).detach()
    return actions, old_log_probs


def _empty_action_counts() -> dict[str, int]:
    return {name: 0 for name in ACTION_NAMES}


def _empty_action_reward_sums() -> dict[str, float]:
    return {name: 0.0 for name in ACTION_NAMES}


def _record_action_result(
    action_counts: dict[str, int],
    fill_counts: dict[str, int],
    no_fill_counts: dict[str, int],
    near_miss_counts: dict[str, int],
    reward_sums: dict[str, float],
    result,
) -> None:
    action_counts[result.action] += 1
    reward_sums[result.action] += float(result.reward)
    if result.filled:
        fill_counts[result.action] += 1
    if result.no_fill:
        no_fill_counts[result.action] += 1
    if result.near_miss:
        near_miss_counts[result.action] += 1


def _rate_by_action(action_counts: dict[str, int], metric_counts: dict[str, int]) -> dict[str, float]:
    return {
        name: float(metric_counts[name] / action_counts[name]) if action_counts[name] > 0 else 0.0
        for name in ACTION_NAMES
    }


def _mean_reward_by_action(action_counts: dict[str, int], reward_sums: dict[str, float]) -> dict[str, float]:
    return {
        name: float(reward_sums[name] / action_counts[name]) if action_counts[name] > 0 else 0.0
        for name in ACTION_NAMES
    }


def _evaluate_gate_model(
    model: MicrostructureGate,
    events,
    tick_store: TickProxyStore,
    gate_cfg: NeuralGateConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    action_counts = _empty_action_counts()
    fill_counts = _empty_action_counts()
    no_fill_counts = _empty_action_counts()
    near_miss_counts = _empty_action_counts()
    reward_sums = _empty_action_reward_sums()
    filled = 0
    rewards: list[float] = []
    with torch.no_grad():
        for event in events:
            state = build_gate_state_vector(
                prob_buy=event.prob_buy,
                prob_entry=event.prob_entry,
                atr=event.atr,
                volatility=event.volatility,
                spread=event.spread,
                tick_count=event.tick_count,
                session_code=event.session_code,
                tpo_features=event.tpo_features,
                pair_name=event.pair_name,
                anchor_timestamp=event.timestamp,
                tick_store=tick_store,
            )
            logits = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            action_idx = int(torch.argmax(logits, dim=-1).item())
            result = simulate_action(
                event,
                action_idx,
                tick_store,
                reference_price_mode=gate_cfg.reference_price_mode,
                reject_wait_penalty=gate_cfg.reject_wait_penalty,
                market_order_slippage_ticks=gate_cfg.market_order_slippage_ticks,
                limit_at_poc_near_miss_reward=gate_cfg.limit_at_poc_near_miss_reward,
                limit_at_poc_no_fill_penalty=gate_cfg.limit_at_poc_no_fill_penalty,
                passive_limit_no_fill_penalty=gate_cfg.passive_limit_no_fill_penalty,
                horizon_bars=max(gate_cfg.mfe_horizon_bars, gate_cfg.mae_horizon_bars),
            )
            _record_action_result(action_counts, fill_counts, no_fill_counts, near_miss_counts, reward_sums, result)
            if result.filled:
                filled += 1
            rewards.append(float(result.reward))
    total = max(len(events), 1)
    return {
        "n_events": len(events),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "fill_rate": float(filled / total),
        "reject_rate": float(action_counts["reject_wait"] / total),
        "action_counts": action_counts,
        "fill_counts": fill_counts,
        "no_fill_counts": no_fill_counts,
        "near_miss_counts": near_miss_counts,
        "mean_reward_by_action": _mean_reward_by_action(action_counts, reward_sums),
        "fill_rate_by_action": _rate_by_action(action_counts, fill_counts),
        "no_fill_rate_by_action": _rate_by_action(action_counts, no_fill_counts),
        "near_miss_rate_by_action": _rate_by_action(action_counts, near_miss_counts),
    }


def train_execution_gate(
    *,
    run_dir: str,
    cache_root: str,
    tick_root: str,
    output_dir: str,
    oracle_device: str = "cpu",
    gate_device: str = "cpu",
    epochs: int = 1,
    learning_rate: float = 1e-3,
    max_folds: int | None = None,
    max_events_per_fold: int = 0,
    neural_gate_overrides: dict | None = None,
    log_file: str | None = None,
    status_file: str | None = None,
) -> dict[str, object]:
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("train_execution_gate_v5", log_file=log_file)
    checkpoint_paths = _load_checkpoint_paths(run_path)
    training_cfg, subnet_cfg, block_cfg, include_signal_only_tpo = _load_run_metadata(run_path, cache_root)
    oracle_device_obj = torch.device(oracle_device)
    gate_device_obj = torch.device(gate_device)
    btc_panels, fx_panels, cached_splits, split_meta = load_feature_batches(cache_root)
    splits, split_frequency = _resolve_cached_splits(
        fx_panels.anchor_timestamps,
        cached_splits,
        split_meta,
        training_cfg,
        logger,
        status_file,
    )
    n_folds = min(len(splits), len(checkpoint_paths))
    if max_folds is not None:
        n_folds = min(n_folds, max_folds)
    splits = splits[:n_folds]
    checkpoint_paths = checkpoint_paths[:n_folds]
    tick_store = TickProxyStore(tick_root)
    missing_pairs = tick_store.missing_pairs(FX_TRADABLE_NAMES)
    if missing_pairs:
        raise FileNotFoundError(
            f"Missing 1000ms proxy bars for required pairs under {tick_root}: {', '.join(missing_pairs)}"
        )
    fold_reports: list[dict[str, object]] = []
    write_status(status_file, {"state": "running", "stage": "gate_train_start", "fold_total": n_folds})

    for fold_index, (split, checkpoint_path) in enumerate(zip(splits, checkpoint_paths)):
        write_status(status_file, {"state": "running", "stage": "gate_train_fold", "fold": fold_index, "fold_total": n_folds})
        checkpoint = torch.load(checkpoint_path, map_location=oracle_device_obj)
        btc_subnet = BTCSubnet(subnet_cfg, block_cfg).to(oracle_device_obj)
        fx_subnet = FXSubnet(subnet_cfg, block_cfg).to(oracle_device_obj)
        bridge = ConditionalBridge(block_cfg.output_dim, block_cfg.hidden_dim).to(oracle_device_obj)
        btc_subnet.load_state_dict(checkpoint["btc_subnet"])
        fx_subnet.load_state_dict(checkpoint["fx_subnet"])
        bridge.load_state_dict(checkpoint["bridge"])
        dir_scaler = CalibrationArtifact(**checkpoint["direction_scaler"])
        entry_scaler = CalibrationArtifact(**checkpoint["entry_scaler"])

        train_idx = np.asarray(split["train_idx"], dtype=np.int32)
        val_idx = np.asarray(split["val_idx"], dtype=np.int32)

        (
            train_dir_logits,
            train_entry_logits,
            _train_dir_labels,
            _train_entry_labels,
            train_valid,
            train_close,
            train_high,
            train_low,
            train_volatility,
            train_session_codes,
        ) = _collect_anchor_outputs(
            btc_subnet,
            fx_subnet,
            bridge,
            btc_panels,
            fx_panels,
            train_idx,
            DEFAULT_SEQ_LENS,
            oracle_device_obj,
            include_signal_only_tpo=include_signal_only_tpo,
        )
        (
            val_dir_logits,
            val_entry_logits,
            _val_dir_labels,
            _val_entry_labels,
            val_valid,
            val_close,
            val_high,
            val_low,
            val_volatility,
            val_session_codes,
        ) = _collect_anchor_outputs(
            btc_subnet,
            fx_subnet,
            bridge,
            btc_panels,
            fx_panels,
            val_idx,
            DEFAULT_SEQ_LENS,
            oracle_device_obj,
            include_signal_only_tpo=include_signal_only_tpo,
        )

        train_prob_buy = apply_platt_scaler(dir_scaler, train_dir_logits.reshape(-1)).reshape(train_dir_logits.shape)
        train_prob_entry = apply_platt_scaler(entry_scaler, train_entry_logits.reshape(-1)).reshape(train_entry_logits.shape)
        val_prob_buy = apply_platt_scaler(dir_scaler, val_dir_logits.reshape(-1)).reshape(val_dir_logits.shape)
        val_prob_entry = apply_platt_scaler(entry_scaler, val_entry_logits.reshape(-1)).reshape(val_entry_logits.shape)
        train_atr = _compute_atr_matrix(train_high, train_low, train_close)
        val_atr = _compute_atr_matrix(val_high, val_low, val_close)
        train_timestamps, train_anchor_node_features, train_anchor_tpo_features = extract_tradable_anchor_views(
            fx_panels,
            train_idx,
            FX_TRADABLE_NAMES,
        )
        val_timestamps, val_anchor_node_features, val_anchor_tpo_features = extract_tradable_anchor_views(
            fx_panels,
            val_idx,
            FX_TRADABLE_NAMES,
        )
        train_events = build_gate_events(
            timestamps=train_timestamps,
            prob_buy=train_prob_buy,
            prob_entry=train_prob_entry,
            close=train_close,
            atr=train_atr,
            volatility=train_volatility,
            session_codes=train_session_codes,
            pair_names=FX_TRADABLE_NAMES,
            anchor_node_features=train_anchor_node_features,
            anchor_tpo_features=train_anchor_tpo_features,
            valid_mask=train_valid,
        )
        val_events = build_gate_events(
            timestamps=val_timestamps,
            prob_buy=val_prob_buy,
            prob_entry=val_prob_entry,
            close=val_close,
            atr=val_atr,
            volatility=val_volatility,
            session_codes=val_session_codes,
            pair_names=FX_TRADABLE_NAMES,
            anchor_node_features=val_anchor_node_features,
            anchor_tpo_features=val_anchor_tpo_features,
            valid_mask=val_valid,
        )
        if max_events_per_fold > 0:
            train_events = train_events[:max_events_per_fold]
            val_events = val_events[:max_events_per_fold]

        gate_cfg = NeuralGateConfig(enabled=True, **(neural_gate_overrides or {}))
        gate_model = MicrostructureGate(input_dim=len(GATE_STATE_FEATURE_NAMES)).to(gate_device_obj)
        reference_model = deepcopy(gate_model).to(gate_device_obj)
        reference_model.eval()
        optimizer = torch.optim.AdamW(gate_model.parameters(), lr=learning_rate)
        epoch_reports: list[dict[str, object]] = []

        for epoch in range(epochs):
            gate_model.train()
            zero_variance_groups = 0
            action_counts = _empty_action_counts()
            fill_counts = _empty_action_counts()
            no_fill_counts = _empty_action_counts()
            near_miss_counts = _empty_action_counts()
            reward_sums = _empty_action_reward_sums()
            filled_actions = 0
            processed_groups = 0
            rng = np.random.default_rng(epoch + fold_index)
            shuffled_events = list(train_events)
            rng.shuffle(shuffled_events)
            for event in shuffled_events:
                state = build_gate_state_vector(
                    prob_buy=event.prob_buy,
                    prob_entry=event.prob_entry,
                    atr=event.atr,
                    volatility=event.volatility,
                    spread=event.spread,
                    tick_count=event.tick_count,
                    session_code=event.session_code,
                    tpo_features=event.tpo_features,
                    pair_name=event.pair_name,
                    anchor_timestamp=event.timestamp,
                    tick_store=tick_store,
                )
                state_tensor = torch.tensor(state, dtype=torch.float32, device=gate_device_obj).unsqueeze(0)
                logits = gate_model(state_tensor)
                actions, old_log_probs = _sample_group_actions(
                    logits,
                    gate_cfg.grpo_group_size,
                    gate_cfg.action_temperature,
                )
                repeated_states = state_tensor.repeat(gate_cfg.grpo_group_size, 1)
                reference_logits = reference_model(repeated_states).detach()
                rewards = []
                filled_mask = []
                for sampled_action in actions.detach().cpu().tolist():
                    result = simulate_action(
                        event,
                        sampled_action,
                        tick_store,
                        reference_price_mode=gate_cfg.reference_price_mode,
                        reject_wait_penalty=gate_cfg.reject_wait_penalty,
                        market_order_slippage_ticks=gate_cfg.market_order_slippage_ticks,
                        limit_at_poc_near_miss_reward=gate_cfg.limit_at_poc_near_miss_reward,
                        limit_at_poc_no_fill_penalty=gate_cfg.limit_at_poc_no_fill_penalty,
                        passive_limit_no_fill_penalty=gate_cfg.passive_limit_no_fill_penalty,
                        horizon_bars=max(gate_cfg.mfe_horizon_bars, gate_cfg.mae_horizon_bars),
                    )
                    _record_action_result(action_counts, fill_counts, no_fill_counts, near_miss_counts, reward_sums, result)
                    rewards.append(result.reward)
                    filled_mask.append(result.filled)
                    if result.filled:
                        filled_actions += 1
                update = grpo_update_step(
                    gate_model,
                    optimizer,
                    repeated_states,
                    actions,
                    torch.tensor(rewards, dtype=torch.float32, device=gate_device_obj),
                    old_log_probs,
                    reference_logits=reference_logits,
                    filled_mask=torch.tensor(filled_mask, dtype=torch.bool, device=gate_device_obj),
                    clip_epsilon=gate_cfg.grpo_clip_epsilon,
                    kl_beta=gate_cfg.grpo_kl_beta,
                    positive_fill_reward_boost=gate_cfg.positive_fill_reward_boost,
                    zero_variance_skip_epsilon=gate_cfg.zero_variance_skip_epsilon,
                )
                processed_groups += 1
                if bool(update["zero_variance"]):
                    zero_variance_groups += 1
            epoch_report = {
                "epoch": epoch,
                "reject_share": float(action_counts["reject_wait"] / max(sum(action_counts.values()), 1)),
                "fill_share": float(filled_actions / max(sum(action_counts.values()), 1)),
                "zero_variance_group_share": float(zero_variance_groups / max(processed_groups, 1)),
                "action_counts": action_counts,
                "fill_counts": fill_counts,
                "no_fill_counts": no_fill_counts,
                "near_miss_counts": near_miss_counts,
                "mean_reward_by_action": _mean_reward_by_action(action_counts, reward_sums),
                "fill_rate_by_action": _rate_by_action(action_counts, fill_counts),
                "no_fill_rate_by_action": _rate_by_action(action_counts, no_fill_counts),
                "near_miss_rate_by_action": _rate_by_action(action_counts, near_miss_counts),
            }
            logger.info("fold=%d epoch=%d state=gate_epoch_metrics %s", fold_index, epoch, json.dumps(epoch_report, sort_keys=True))
            epoch_reports.append(epoch_report)

        val_metrics = _evaluate_gate_model(gate_model, val_events, tick_store, gate_cfg, gate_device_obj)
        checkpoint_out = output_path / f"fold_{fold_index}_gate.pt"
        torch.save(
            {
                "state_dict": gate_model.state_dict(),
                "fold": fold_index,
                "neural_gate_config": asdict(gate_cfg),
                "validation_metrics": val_metrics,
                "source_checkpoint": str(checkpoint_path),
            },
            checkpoint_out,
        )
        fold_report = {
            "fold": fold_index,
            "source_checkpoint": str(checkpoint_path),
            "train_events": len(train_events),
            "val_events": len(val_events),
            "epochs": epoch_reports,
            "validation": val_metrics,
            "gate_checkpoint": str(checkpoint_out),
        }
        fold_reports.append(fold_report)
        _write_json_atomic(
            output_path / "report_partial.json",
            {
                "mode": "train_execution_gate_v5_partial",
                "run_dir": str(run_path),
                "cache_root": cache_root,
                "tick_root": tick_root,
                "fold_results": fold_reports,
                "split_frequency": split_frequency,
            },
        )

    summary = {
        "folds": len(fold_reports),
        "mean_val_reward": float(np.mean([fr["validation"]["mean_reward"] for fr in fold_reports])) if fold_reports else 0.0,
        "mean_val_fill_rate": float(np.mean([fr["validation"]["fill_rate"] for fr in fold_reports])) if fold_reports else 0.0,
        "mean_val_reject_rate": float(np.mean([fr["validation"]["reject_rate"] for fr in fold_reports])) if fold_reports else 0.0,
    }
    report = {
        "mode": "train_execution_gate_v5",
        "run_dir": str(run_path),
        "cache_root": cache_root,
        "tick_root": tick_root,
        "fold_results": fold_reports,
        "summary": summary,
    }
    _write_json_atomic(output_path / "report.json", report)
    write_status(status_file, {"state": "completed", "stage": "train_execution_gate_v5", "summary": summary})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the staged_v5.1 neural execution gate from frozen staged_v5 checkpoints.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tick-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gate-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-events-per-fold", type=int, default=0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)
    try:
        train_execution_gate(
            run_dir=args.run_dir,
            cache_root=args.cache_root,
            tick_root=args.tick_root,
            output_dir=args.output_dir,
            oracle_device=args.oracle_device,
            gate_device=args.gate_device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_folds=args.max_folds,
            max_events_per_fold=args.max_events_per_fold,
            neural_gate_overrides=None,
            log_file=args.log_file,
            status_file=args.status_file,
        )
    except Exception as exc:
        logger = configure_logging("train_execution_gate_v5", log_file=args.log_file)
        log_exception(logger, args.status_file, "train_execution_gate_v5", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
