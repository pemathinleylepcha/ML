from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from staged_v5.config import BacktestConfig, FX_TRADABLE_NAMES, GA_PARAM_SPACE, GAConfig, STGNNBlockConfig, SubnetConfig, TrainingConfig, decode_ga_genome
from staged_v5.contracts import CalibrationArtifact
from staged_v5.data import load_feature_batches, load_staged_panels
from staged_v5.execution_gate import GATE_STATE_FEATURE_NAMES, MicrostructureGate, NeuralGateRuntime, TickProxyStore, extract_tradable_anchor_views
from staged_v5.evaluation.backtest import adjusted_backtest_config, backtest_probabilities
from staged_v5.evaluation.metrics import (
    binary_accuracy,
    binary_auc,
    binary_brier,
    binary_ece,
    binary_log_loss,
    summarize_fold_metrics,
)
from staged_v5.models import BTCSubnet, ConditionalBridge, FXSubnet
from staged_v5.training.train_staged import DEFAULT_SEQ_LENS, _collect_anchor_outputs, _resolve_cached_splits
from staged_v5.utils.calibration_helpers import apply_platt_scaler
from staged_v5.utils.runtime_logging import configure_logging, log_exception, stage_context, write_status


def _merge_dataclass_config(cls, payload: dict | None):
    merged = {field.name: getattr(cls(), field.name) for field in fields(cls)}
    if payload:
        for key, value in payload.items():
            if key in merged:
                merged[key] = value
    return cls(**merged)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _pbo_scalar(summary: dict) -> float | None:
    value = summary.get("pbo")
    if isinstance(value, dict):
        value = value.get("pbo")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_gate_checkpoint(gate_checkpoint: str | None, fold_index: int) -> Path | None:
    if gate_checkpoint is None:
        return None
    gate_path = Path(gate_checkpoint)
    if gate_path.is_file():
        return gate_path
    if not gate_path.exists():
        raise FileNotFoundError(f"Gate checkpoint path does not exist: {gate_path}")
    candidates = sorted(gate_path.glob("fold_*_gate.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not candidates:
        candidates = sorted(gate_path.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if fold_index >= len(candidates):
        raise FileNotFoundError(f"No gate checkpoint for fold {fold_index} under {gate_path}")
    return candidates[fold_index]


def _load_gate_runtime(
    gate_checkpoint: str | None,
    tick_root: str | None,
    fold_index: int,
    gate_device: str,
    backtest_cfg: BacktestConfig,
) -> NeuralGateRuntime | None:
    if backtest_cfg.entry.entry_type != "neural_gate":
        return None
    if tick_root is None:
        raise ValueError("entry_type='neural_gate' requires --tick-root")
    gate_checkpoint_path = _resolve_gate_checkpoint(gate_checkpoint, fold_index)
    if gate_checkpoint_path is None:
        raise ValueError("entry_type='neural_gate' requires --gate-checkpoint")
    resolved_device = gate_device.lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("gate device cuda requested but CUDA is not available")
    gate_device_obj = torch.device(resolved_device)
    payload = torch.load(gate_checkpoint_path, map_location=gate_device_obj)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = MicrostructureGate(input_dim=len(GATE_STATE_FEATURE_NAMES)).to(gate_device_obj)
    model.load_state_dict(state_dict)
    model.eval()
    return NeuralGateRuntime(
        model=model,
        config=backtest_cfg.neural_gate,
        tick_store=TickProxyStore(tick_root),
        device=gate_device_obj,
    )


def replay_backtest(
    run_dir: str,
    candle_root: str | None,
    tick_root: str | None,
    start: str | None,
    end: str | None,
    output: str | None = None,
    *,
    cache_root: str | None = None,
    strict: bool = False,
    max_workers: int = 0,
    device_override: str = "cpu",
    backtest_overrides: dict | None = None,
    ga_sweep: bool = False,
    ga_cfg: GAConfig | None = None,
    gate_checkpoint: str | None = None,
    gate_device: str = "cpu",
    log_file: str | None = None,
    status_file: str | None = None,
) -> dict:
    run_path = Path(run_dir)
    base_report_path = run_path / "report.json"
    checkpoint_dir = run_path / "report"
    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")

    logger = configure_logging("replay_staged_v5", log_file=log_file)
    fallback_cache_manifest: dict[str, object] = {}
    if cache_root is not None:
        fallback_cache_manifest_path = Path(cache_root) / "manifest.json"
        if fallback_cache_manifest_path.exists():
            fallback_cache_manifest = json.loads(fallback_cache_manifest_path.read_text(encoding="utf-8"))
    if base_report_path.exists():
        base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
        source_report_path: str | None = str(base_report_path)
    else:
        checkpoint_preview = torch.load(checkpoint_paths[0], map_location="cpu")
        checkpoint_fold_result = dict(checkpoint_preview.get("fold_result", {}))
        fallback_training_cfg = asdict(TrainingConfig())
        if fallback_cache_manifest:
            for key in ("anchor_timeframe", "split_frequency", "outer_holdout_blocks", "min_train_blocks", "purge_bars"):
                if key in fallback_cache_manifest:
                    fallback_training_cfg[key] = fallback_cache_manifest[key]
        fallback_timeframes = fallback_cache_manifest.get("fx_timeframes") or list(SubnetConfig().timeframe_order)
        base_report = {
            "training_config": fallback_training_cfg,
            "subnet_config": asdict(SubnetConfig()),
            "block_config": asdict(STGNNBlockConfig()),
            "backtest_config": checkpoint_fold_result.get("backtest_cfg", {}),
            "summary": {},
            "timeframes": fallback_timeframes,
            "include_signal_only_tpo": bool(fallback_cache_manifest.get("include_signal_only_tpo", True)),
        }
        source_report_path = None
        logger.warning(
            "state=base_report_missing_fallback run_dir=%s checkpoint=%s",
            run_path,
            checkpoint_paths[0].name,
        )
    training_cfg = _merge_dataclass_config(TrainingConfig, base_report.get("training_config"))
    subnet_cfg = _merge_dataclass_config(SubnetConfig, base_report.get("subnet_config"))
    block_cfg = _merge_dataclass_config(STGNNBlockConfig, base_report.get("block_config"))
    # Replay should use the current execution defaults, not the stale saved ones,
    # so execution-only bundles can be validated against frozen model checkpoints.
    source_backtest_cfg = base_report.get("backtest_config", {})
    if backtest_overrides:
        backtest_cfg = BacktestConfig.from_flat(backtest_overrides)
    else:
        backtest_cfg = BacktestConfig()
    timeframes = tuple(base_report.get("timeframes", list(subnet_cfg.timeframe_order)))
    if tuple(subnet_cfg.timeframe_order) != timeframes:
        subnet_cfg = SubnetConfig(
            timeframe_order=timeframes,
            exchange_every_k_batches=subnet_cfg.exchange_every_k_batches,
            active_loss_boost=subnet_cfg.active_loss_boost,
            enable_entry_head=subnet_cfg.enable_entry_head,
        )
    include_signal_only_tpo = bool(base_report.get("include_signal_only_tpo", True))

    resolved_device = device_override.lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this host")
    device = torch.device(resolved_device)
    logger.info("state=device_selected device=%s cuda_available=%s", device.type, torch.cuda.is_available())
    if backtest_cfg.entry.entry_type == "neural_gate" and cache_root is None:
        raise ValueError("entry_type='neural_gate' currently requires --cache-root")

    feature_mode = "prebuilt"
    cache_manifest: dict[str, object] = {}
    if cache_root is not None:
        cache_root_path = Path(cache_root)
        cache_manifest_path = cache_root_path / "manifest.json"
        if cache_manifest_path.exists():
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        with stage_context(
            logger,
            status_file,
            "load_feature_batches",
            cache_root=cache_root,
            run_dir=str(run_path),
        ):
            btc_panels, fx_panels, cached_splits, split_meta = load_feature_batches(cache_root)
            splits, split_frequency = _resolve_cached_splits(
                fx_panels.anchor_timestamps,
                cached_splits,
                split_meta,
                training_cfg,
                logger,
                status_file,
            )
            split_source = "regenerated" if splits is not cached_splits else "cached"
            logger.info(
                "state=replay_splits_resolved n_splits=%d split_frequency=%s source=%s",
                len(splits),
                split_frequency,
                split_source,
            )
            loaded_timeframes = tuple(fx_panels.timeframe_batches.keys())
            if tuple(subnet_cfg.timeframe_order) != loaded_timeframes:
                subnet_cfg = SubnetConfig(
                    timeframe_order=loaded_timeframes,
                    exchange_every_k_batches=subnet_cfg.exchange_every_k_batches,
                    active_loss_boost=subnet_cfg.active_loss_boost,
                    enable_entry_head=subnet_cfg.enable_entry_head,
                )
            timeframes = loaded_timeframes
        if start is None:
            start = cache_manifest.get("start") if cache_manifest else None
        if end is None:
            end = cache_manifest.get("end") if cache_manifest else None
        if start is None and len(fx_panels.anchor_timestamps):
            start = str(fx_panels.anchor_timestamps[0])
        if end is None and len(fx_panels.anchor_timestamps):
            end = str(fx_panels.anchor_timestamps[-1])
        feature_mode = "cached"
        logger.info("state=feature_mode mode=%s cache_root=%s", feature_mode, cache_root)
    else:
        if candle_root is None:
            raise ValueError("candle_root is required when cache_root is not provided")
        if start is None or end is None:
            raise ValueError("start and end are required when replaying from raw panels")
        with stage_context(
            logger,
            status_file,
            "load_staged_panels",
            candle_root=candle_root,
            tick_root=tick_root,
            start=start,
            end=end,
            max_workers=max_workers,
            run_dir=str(run_path),
        ):
            btc_panels, fx_panels = load_staged_panels(
                candle_root=candle_root,
                tick_root=tick_root,
                start=start,
                end=end,
                anchor_timeframe=training_cfg.anchor_timeframe,
                strict=strict,
                timeframes=timeframes,
                split_frequency=training_cfg.split_frequency,
                outer_holdout_blocks=training_cfg.outer_holdout_blocks,
                min_train_blocks=training_cfg.min_train_blocks,
                purge_bars=training_cfg.purge_bars,
                logger=logger,
                status_file=status_file,
                max_workers=max_workers,
                memory_guard_min_available_mb=training_cfg.memory_guard_min_available_mb,
                memory_guard_critical_available_mb=training_cfg.memory_guard_critical_available_mb,
            )
        splits = list(fx_panels.walkforward_splits)
        split_frequency = training_cfg.split_frequency
        logger.info("state=feature_mode mode=%s", feature_mode)

    n_folds = min(len(checkpoint_paths), len(splits))
    splits = splits[:n_folds]
    checkpoint_paths = checkpoint_paths[:n_folds]
    logger.info("state=replay_splits n_splits=%d checkpoints=%d device=%s", len(splits), len(checkpoint_paths), device.type)
    write_status(status_file, {"state": "running", "stage": "replay_start", "fold_total": len(splits), "device": device.type})

    output_path = Path(output) if output else (run_path / "report_bundle_c_replay.json")
    partial_output_path = output_path.with_name(output_path.stem + "_partial.json")

    fold_results: list[dict] = []
    for fold_index, (split, checkpoint_path) in enumerate(zip(splits, checkpoint_paths)):
        write_status(status_file, {"state": "running", "stage": "replay_fold", "fold": fold_index, "fold_total": len(splits)})
        logger.info("fold=%d/%d state=start checkpoint=%s", fold_index + 1, len(splits), checkpoint_path.name)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        btc_subnet = BTCSubnet(subnet_cfg, block_cfg).to(device)
        fx_subnet = FXSubnet(subnet_cfg, block_cfg).to(device)
        bridge = ConditionalBridge(block_cfg.output_dim, block_cfg.hidden_dim).to(device)
        btc_subnet.load_state_dict(checkpoint["btc_subnet"])
        fx_subnet.load_state_dict(checkpoint["fx_subnet"])
        bridge.load_state_dict(checkpoint["bridge"])
        dir_scaler = CalibrationArtifact(**checkpoint["direction_scaler"])
        entry_scaler = CalibrationArtifact(**checkpoint["entry_scaler"])

        val_idx = np.asarray(split["val_idx"], dtype=np.int32)
        (
            val_dir_logits,
            val_entry_logits,
            val_dir_labels,
            _val_entry_labels,
            val_valid,
            close,
            high,
            low,
            volatility,
            session_codes,
        ) = _collect_anchor_outputs(
            btc_subnet,
            fx_subnet,
            bridge,
            btc_panels,
            fx_panels,
            val_idx,
            DEFAULT_SEQ_LENS,
            device,
            include_signal_only_tpo=include_signal_only_tpo,
        )
        val_dir_logits_flat = val_dir_logits.reshape(-1)
        val_entry_logits_flat = val_entry_logits.reshape(-1)
        val_dir_labels_flat = val_dir_labels.reshape(-1)
        val_valid_flat = val_valid.reshape(-1) & (val_dir_labels_flat >= 0)
        val_prob_flat = apply_platt_scaler(dir_scaler, val_dir_logits_flat)
        val_entry_prob_flat = apply_platt_scaler(entry_scaler, val_entry_logits_flat)
        val_prob = val_prob_flat.reshape(val_dir_logits.shape)
        val_entry_prob = val_entry_prob_flat.reshape(val_entry_logits.shape)
        val_ece = binary_ece(val_prob_flat, val_dir_labels_flat, val_valid_flat)
        fold_backtest_cfg = adjusted_backtest_config(backtest_cfg, val_ece)
        val_timestamps, val_anchor_node_features, val_anchor_tpo_features = extract_tradable_anchor_views(
            fx_panels,
            val_idx,
            FX_TRADABLE_NAMES,
        )
        gate_runtime = _load_gate_runtime(gate_checkpoint, tick_root, fold_index, gate_device, fold_backtest_cfg)
        backtest = backtest_probabilities(
            val_prob,
            val_entry_prob,
            close,
            high,
            low,
            volatility,
            session_codes,
            FX_TRADABLE_NAMES,
            fold_backtest_cfg,
            timestamps=val_timestamps,
            anchor_node_features=val_anchor_node_features,
            anchor_tpo_features=val_anchor_tpo_features,
            neural_gate_runtime=gate_runtime,
        )
        ga_result = None
        if ga_sweep and ga_cfg and ga_cfg.generations > 0:
            from staged_v5.utils.ga_search import run_continuous_ga

            n_val_bars = val_prob.shape[0]

            def _ga_objective(genome: list[float]) -> float:
                opt_cfg = decode_ga_genome(genome, fold_backtest_cfg)
                opt_gate_runtime = _load_gate_runtime(gate_checkpoint, tick_root, fold_index, gate_device, opt_cfg)
                result = backtest_probabilities(
                    val_prob,
                    val_entry_prob,
                    close,
                    high,
                    low,
                    volatility,
                    session_codes,
                    FX_TRADABLE_NAMES,
                    opt_cfg,
                    timestamps=val_timestamps,
                    anchor_node_features=val_anchor_node_features,
                    anchor_tpo_features=val_anchor_tpo_features,
                    neural_gate_runtime=opt_gate_runtime,
                )
                score = float(result["net_return"]) if ga_cfg.ga_objective == "net_return" else float(result["strategy_sharpe"])
                trades = result["trade_count"]
                if trades < 10:
                    return -999.0
                if trades > n_val_bars * 0.5:
                    score *= 0.5
                return score

            ga_search_result = run_continuous_ga(
                n_genes=len(GA_PARAM_SPACE),
                scorer=_ga_objective,
                population_size=ga_cfg.population_size,
                generations=ga_cfg.generations,
                mutation_rate=ga_cfg.mutation_rate,
                crossover_rate=ga_cfg.crossover_rate,
            )
            ga_optimized_cfg = decode_ga_genome(ga_search_result.best_genome, fold_backtest_cfg)
            ga_gate_runtime = _load_gate_runtime(gate_checkpoint, tick_root, fold_index, gate_device, ga_optimized_cfg)
            ga_backtest = backtest_probabilities(
                val_prob,
                val_entry_prob,
                close,
                high,
                low,
                volatility,
                session_codes,
                FX_TRADABLE_NAMES,
                ga_optimized_cfg,
                timestamps=val_timestamps,
                anchor_node_features=val_anchor_node_features,
                anchor_tpo_features=val_anchor_tpo_features,
                neural_gate_runtime=ga_gate_runtime,
            )
            ga_result = {
                "ga_sharpe": ga_backtest["strategy_sharpe"],
                "ga_trades": ga_backtest["trade_count"],
                "ga_net_return": ga_backtest["net_return"],
                "ga_win_rate": ga_backtest["win_rate"],
                "ga_config": ga_optimized_cfg.to_flat(),
                "ga_best_score": ga_search_result.best_score,
            }
            logger.info(
                "fold=%d state=ga_done base_sharpe=%.4f ga_sharpe=%.4f ga_trades=%d",
                fold_index,
                backtest["strategy_sharpe"],
                ga_result["ga_sharpe"],
                ga_result["ga_trades"],
            )
        fold_result = {
            "fold": int(split["fold"]),
            "split_frequency": split.get("split_frequency", training_cfg.split_frequency),
            "train_blocks": split.get("train_blocks", []),
            "val_block": split.get("val_block", ""),
            "auc": binary_auc(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "log_loss": binary_log_loss(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "brier": binary_brier(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "ece": val_ece,
            "directional_accuracy": binary_accuracy(val_prob_flat, val_dir_labels_flat, val_valid_flat),
            "n_bars": int(len(val_idx)),
            **{k: v for k, v in backtest.items() if k != "bar_returns"},
            "backtest_cfg": asdict(fold_backtest_cfg),
            "source_checkpoint": str(checkpoint_path),
        }
        if ga_result:
            fold_result["ga_optimization"] = ga_result
        fold_results.append(fold_result)
        logger.info(
            "fold=%d state=done auc=%.4f sharpe=%.4f net=%.4f trades=%d",
            fold_index,
            fold_result["auc"],
            fold_result["strategy_sharpe"],
            fold_result["net_return"],
            fold_result["trade_count"],
        )
        logger.info(
            "fold=%d state=entry_rejections %s",
            fold_index,
            json.dumps(fold_result.get("entry_rejection_counters", {}), sort_keys=True),
        )
        partial_report = {
            "mode": "replay_backtest_partial",
            "source_run_dir": str(run_path),
            "source_report": source_report_path,
            "cache_root": cache_root,
            "candle_root": candle_root,
            "tick_root": tick_root,
            "start": start,
            "end": end,
            "timeframes": list(timeframes),
            "training_config": asdict(training_cfg),
            "subnet_config": asdict(subnet_cfg),
            "block_config": asdict(block_cfg),
            "backtest_config": asdict(backtest_cfg),
            "backtest_overrides": backtest_overrides or {},
            "ga_sweep": bool(ga_sweep),
            "ga_config": asdict(ga_cfg) if ga_cfg is not None else None,
            "gate_checkpoint": gate_checkpoint,
            "gate_device": gate_device,
            "source_backtest_config": source_backtest_cfg,
            "include_signal_only_tpo": include_signal_only_tpo,
            "feature_mode": feature_mode,
            "cache_manifest": cache_manifest,
            "completed_folds": len(fold_results),
            "fold_results": fold_results,
            "summary": summarize_fold_metrics(fold_results) if fold_results else {},
            "baseline_summary": base_report.get("summary", {}),
        }
        _write_json_atomic(partial_output_path, partial_report)
        write_status(status_file, {"state": "running", "stage": "replay_fold_complete", "fold": fold_index, "fold_result": fold_result})

    summary = summarize_fold_metrics(fold_results)
    baseline_mean_trade_count = float(base_report.get("summary", {}).get("mean_trade_count", 0.0))
    replay_mean_trade_count = float(summary.get("mean_trade_count", 0.0))
    trade_count_reduction = 0.0
    if baseline_mean_trade_count > 0.0:
        trade_count_reduction = 1.0 - (replay_mean_trade_count / baseline_mean_trade_count)
    pbo_value = _pbo_scalar(summary)
    acceptance = {
        "mean_trade_count_drop_gte_40pct": bool(trade_count_reduction >= 0.40),
        "mean_sharpe_gte_6": bool(summary.get("mean_sharpe", -999.0) >= 6.0),
        "worst_fold_sharpe_gt_neg2": bool(summary.get("worst_fold_sharpe", -999.0) > -2.0),
        "all_fold_net_return_gt_neg1": bool(all(fr.get("net_return", -999.0) > -1.0 for fr in fold_results)),
        "pbo_lt_0_55": bool((pbo_value if pbo_value is not None else 1.0) < 0.55),
        "trade_count_reduction": float(trade_count_reduction),
        "baseline_mean_trade_count": baseline_mean_trade_count,
    }
    replay_report = {
        "mode": "replay_backtest",
        "source_run_dir": str(run_path),
        "source_report": source_report_path,
        "cache_root": cache_root,
        "candle_root": candle_root,
        "tick_root": tick_root,
        "start": start,
        "end": end,
        "timeframes": list(timeframes),
        "training_config": asdict(training_cfg),
        "subnet_config": asdict(subnet_cfg),
        "block_config": asdict(block_cfg),
        "backtest_config": asdict(backtest_cfg),
        "backtest_overrides": backtest_overrides or {},
        "ga_sweep": bool(ga_sweep),
        "ga_config": asdict(ga_cfg) if ga_cfg is not None else None,
        "gate_checkpoint": gate_checkpoint,
        "gate_device": gate_device,
        "source_backtest_config": source_backtest_cfg,
        "include_signal_only_tpo": include_signal_only_tpo,
        "feature_mode": feature_mode,
        "cache_manifest": cache_manifest,
        "split_frequency": split_frequency,
        "fold_results": fold_results,
        "summary": summary,
        "acceptance": acceptance,
        "baseline_summary": base_report.get("summary", {}),
    }
    _write_json_atomic(output_path, replay_report)
    write_status(status_file, {"state": "completed", "stage": "replay_backtest", "summary": summary, "output": str(output_path), "acceptance": acceptance})
    return replay_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay staged_v5 backtests from saved fold checkpoints.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--candle-root", default=None)
    parser.add_argument("--tick-root", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--entry-type", choices=("limit", "market", "neural_gate"), default=None)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--gate-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--base-entry-threshold", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument("--max-loss-pct-per-trade", type=float, default=None)
    parser.add_argument("--stop-loss-atr", type=float, default=None)
    parser.add_argument("--trailing-activate-atr", type=float, default=None)
    parser.add_argument("--ga-sweep", action="store_true", help="Run GA optimization per fold after base replay")
    parser.add_argument("--ga-population", type=int, default=30)
    parser.add_argument("--ga-generations", type=int, default=10)
    parser.add_argument("--ga-objective", choices=("sharpe", "net_return"), default="sharpe")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)

    if args.cache_root and args.candle_root:
        parser.error("--cache-root cannot be combined with --candle-root")
    if not args.cache_root and not args.candle_root:
        parser.error("either --cache-root or --candle-root is required")

    try:
        backtest_overrides = {}
        if args.base_entry_threshold is not None:
            backtest_overrides["base_entry_threshold"] = args.base_entry_threshold
        if args.entry_type is not None:
            backtest_overrides["entry_type"] = args.entry_type
        if args.max_hold_bars is not None:
            backtest_overrides["max_hold_bars"] = args.max_hold_bars
        if args.max_loss_pct_per_trade is not None:
            backtest_overrides["max_loss_pct_per_trade"] = args.max_loss_pct_per_trade
        if args.stop_loss_atr is not None:
            backtest_overrides["stop_loss_atr"] = args.stop_loss_atr
        if args.trailing_activate_atr is not None:
            backtest_overrides["trailing_activate_atr"] = args.trailing_activate_atr
        ga_cfg_obj = GAConfig(
            population_size=args.ga_population,
            generations=args.ga_generations,
            ga_objective=args.ga_objective,
        ) if args.ga_sweep else None
        replay_backtest(
            run_dir=args.run_dir,
            cache_root=args.cache_root,
            candle_root=args.candle_root,
            tick_root=args.tick_root,
            start=args.start,
            end=args.end,
            output=args.output,
            strict=args.strict,
            max_workers=args.max_workers,
            device_override=args.device,
            backtest_overrides=backtest_overrides,
            ga_sweep=args.ga_sweep,
            ga_cfg=ga_cfg_obj,
            gate_checkpoint=args.gate_checkpoint,
            gate_device=args.gate_device,
            log_file=args.log_file,
            status_file=args.status_file,
        )
    except Exception as exc:
        logger = configure_logging("replay_staged_v5", log_file=args.log_file)
        log_exception(logger, args.status_file, "replay_staged_v5", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
