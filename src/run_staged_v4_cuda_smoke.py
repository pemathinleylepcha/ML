from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod-oriented CUDA verification helpers for staged_v4.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanity = subparsers.add_parser("sanity", help="Print CUDA runtime facts and write a small JSON report.")
    sanity.add_argument("--output-root", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run a short synthetic staged_v4 smoke.")
    synthetic.add_argument("--output-root", required=True)
    synthetic.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    synthetic.add_argument("--anchor-timeframe", default="M1")
    synthetic.add_argument("--timeframes", default="M1,M5")
    synthetic.add_argument("--batch-size", type=int, default=16)
    synthetic.add_argument("--epochs-stage1", type=int, default=1)
    synthetic.add_argument("--epochs-stage2", type=int, default=1)
    synthetic.add_argument("--epochs-stage3", type=int, default=0)
    synthetic.add_argument("--max-folds", type=int, default=1)
    synthetic.add_argument("--synthetic-n-anchor", type=int, default=24)
    synthetic.add_argument("--max-workers", type=int, default=2)
    synthetic.add_argument("--torch-threads", type=int, default=4)
    synthetic.add_argument("--torch-interop-threads", type=int, default=1)
    synthetic.add_argument("--disable-torch-compile", action="store_true")
    synthetic.add_argument("--memory-guard-min-mb", type=float, default=4096.0)
    synthetic.add_argument("--memory-guard-critical-mb", type=float, default=2048.0)
    synthetic.add_argument("--gpu-memory-guard-min-mb", type=float, default=4096.0)
    synthetic.add_argument("--gpu-memory-guard-critical-mb", type=float, default=2048.0)
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_sanity(args: argparse.Namespace) -> int:
    import torch

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        ),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "compile_backends": torch._dynamo.list_backends() if hasattr(torch, "_dynamo") else [],
    }
    _write_json(output_root / "sanity.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


def _run_synthetic(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

    import torch

    from staged_v4.config import BacktestConfig, GAConfig, SubnetConfig, TrainingConfig
    from staged_v4.training.train_staged import run_staged_experiment
    from staged_v4.utils.runtime_logging import configure_logging

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(max(1, args.torch_interop_threads))
    except RuntimeError:
        pass

    train_log = output_root / "train.log"
    status_file = output_root / "status.json"
    report_file = output_root / "report.json"
    timeframe_order = tuple(part.strip() for part in args.timeframes.split(",") if part.strip())
    logger = configure_logging("runpod_cuda_smoke", str(train_log))
    logger.info(
        "state=smoke_start command=synthetic torch_threads=%d torch_interop_threads=%d inductor_compile_threads=%s",
        torch.get_num_threads(),
        getattr(torch, "get_num_interop_threads", lambda: args.torch_interop_threads)(),
        os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    )

    training_cfg = TrainingConfig(
        anchor_timeframe=args.anchor_timeframe,
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
        memory_guard_min_available_mb=args.memory_guard_min_mb,
        memory_guard_critical_available_mb=args.memory_guard_critical_mb,
        use_torch_compile=not args.disable_torch_compile,
        gpu_memory_guard_min_mb=args.gpu_memory_guard_min_mb,
        gpu_memory_guard_critical_mb=args.gpu_memory_guard_critical_mb,
    )
    subnet_cfg = SubnetConfig(
        timeframe_order=timeframe_order,
        exchange_every_k_batches=2,
        active_loss_boost=2.0,
        enable_entry_head=True,
    )
    backtest_cfg = BacktestConfig()
    ga_cfg = GAConfig(population_size=4, generations=0, mutation_rate=0.15, crossover_rate=0.5)

    report = run_staged_experiment(
        mode="synthetic",
        output=str(report_file),
        anchor_timeframe=args.anchor_timeframe,
        max_folds=args.max_folds,
        strict=False,
        training_cfg=training_cfg,
        subnet_cfg=subnet_cfg,
        backtest_cfg=backtest_cfg,
        ga_cfg=ga_cfg,
        synthetic_n_anchor=args.synthetic_n_anchor,
        include_signal_only_tpo=False,
        max_workers=args.max_workers,
        device_override=args.device,
        logger=logger,
        status_file=str(status_file),
    )
    _write_json(
        output_root / "smoke_summary.json",
        {
            "summary": report["summary"],
            "training_config": report["training_config"],
            "subnet_config": report["subnet_config"],
        },
    )
    logger.info("state=smoke_done output=%s folds=%s", report_file, report["summary"].get("folds"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "sanity":
        return _run_sanity(args)
    if args.command == "synthetic":
        return _run_synthetic(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
