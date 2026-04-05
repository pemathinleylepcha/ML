from __future__ import annotations

import argparse
import json
from pathlib import Path

from staged_v4.config import BacktestConfig, GAConfig, TrainingConfig
from staged_v4.training.train_staged import run_staged_experiment
from staged_v4.utils.runtime_logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged_v4 on a fixed no-GA configuration.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-stage1", type=int, default=1)
    parser.add_argument("--epochs-stage2", type=int, default=1)
    parser.add_argument("--epochs-stage3", type=int, default=1)
    parser.add_argument("--base-entry-threshold", type=float, default=0.60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output = output_root / "report.json"
    train_log = output_root / "train.log"
    status_file = output_root / "status.json"

    logger = configure_logging("train_staged", str(train_log))
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
    backtest_cfg = BacktestConfig(
        base_entry_threshold=args.base_entry_threshold,
        threshold_volatility_coeff=10.0,
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
        use_limit_entries=True,
        limit_offset_atr=0.10,
    )
    ga_cfg = GAConfig(population_size=6, generations=0, mutation_rate=0.15, crossover_rate=0.50)

    report = run_staged_experiment(
        mode="real",
        output=str(output),
        cache_root=args.cache_root,
        anchor_timeframe="M1",
        max_folds=args.max_folds,
        training_cfg=training_cfg,
        backtest_cfg=backtest_cfg,
        ga_cfg=ga_cfg,
        include_signal_only_tpo=False,
        device_override=args.device,
        logger=logger,
        status_file=str(status_file),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "fold_count": len(report.get("fold_metrics", [])),
                "mean_sharpe": report.get("mean_sharpe"),
                "pbo": report.get("pbo"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
