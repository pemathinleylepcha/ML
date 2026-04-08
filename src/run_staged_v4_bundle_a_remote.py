from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the remote staged_v4 Bundle A March weekly validation.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candle-root", required=True)
    parser.add_argument("--tick-root", required=True)
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-folds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs-stage1", type=int, default=1)
    parser.add_argument("--epochs-stage2", type=int, default=1)
    parser.add_argument("--epochs-stage3", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--timeframes", default="M1,M5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output = output_root / "report.json"
    train_log = output_root / "train.log"
    status_file = output_root / "status.json"
    timeframe_order = tuple(part.strip() for part in args.timeframes.split(",") if part.strip())

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))

    import torch
    from staged_v4.config import BacktestConfig, GAConfig, SubnetConfig, TrainingConfig
    from staged_v4.training.train_staged import run_staged_experiment
    from staged_v4.utils.runtime_logging import configure_logging

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(max(1, args.torch_interop_threads))
    except RuntimeError:
        pass

    logger = configure_logging("train_staged", str(train_log))
    logger.info(
        "state=torch_runtime torch_threads=%d torch_interop_threads=%d omp_threads=%s mkl_threads=%s",
        torch.get_num_threads(),
        getattr(torch, "get_num_interop_threads", lambda: args.torch_interop_threads)(),
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
    )

    training_cfg = TrainingConfig(
        anchor_timeframe="M1",
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        epochs_stage3=args.epochs_stage3,
        learning_rate=1e-3,
        fine_tune_learning_rate=3e-4,
        purge_bars=6,
        split_frequency="week",
        outer_holdout_blocks=0,
        min_train_blocks=2,
    )
    subnet_cfg = SubnetConfig(
        timeframe_order=timeframe_order,
        exchange_every_k_batches=2,
        active_loss_boost=2.0,
        enable_entry_head=True,
    )
    backtest_cfg = BacktestConfig(
        base_entry_threshold=0.60,
        threshold_volatility_coeff=12.0,
        exit_threshold=0.52,
        probability_spread_threshold=0.10,
        latency_bars=1,
        cooldown_bars=3,
        max_positions=6,
        max_hold_bars=6,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.70,
        max_group_exposure=2,
        take_profit_atr=1.0,
        stop_loss_atr=0.7,
        trailing_activate_atr=0.50,
        use_limit_entries=True,
        limit_offset_atr=0.10,
    )
    ga_cfg = GAConfig(population_size=6, generations=0, mutation_rate=0.15, crossover_rate=0.50)

    run_staged_experiment(
        mode="real",
        output=str(output),
        candle_root=args.candle_root,
        tick_root=args.tick_root,
        start=args.start,
        end=args.end,
        anchor_timeframe="M1",
        max_folds=args.max_folds,
        training_cfg=training_cfg,
        subnet_cfg=subnet_cfg,
        backtest_cfg=backtest_cfg,
        ga_cfg=ga_cfg,
        include_signal_only_tpo=False,
        max_workers=args.max_workers,
        device_override=args.device,
        logger=logger,
        status_file=str(status_file),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
