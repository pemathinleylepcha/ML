"""RunPod v5.2 training launcher using pre-built cache.

Expects cache at /workspace/cache_v52_q4/ (SCP'd from 172 box).
GPU used from minute 1 — no panel loading, no TPO presweep.

Usage on RunPod:
    cd /workspace/Algo-C2-Codex
    nohup python scripts/run_v52_cached_runpod.py > runs/v52_q4_runpod/nohup.out 2>&1 &
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from staged_v5.config import SubnetConfig, TrainingConfig
from staged_v5.training.train_staged import run_staged_experiment
from staged_v5.utils.runtime_logging import configure_logging


CACHE_ROOT = "/workspace/cache_v52_q4"
TICK_ROOT = "/data/ea_training_tickroot_v52_q4_20251001_20251231_1000ms"
OUTPUT_ROOT = "/workspace/Algo-C2-Codex/runs/v52_q4_runpod"
STATUS_FILE = os.path.join(OUTPUT_ROOT, "status.json")
LOG_FILE = os.path.join(OUTPUT_ROOT, "train.log")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    logger = configure_logging("train_staged", LOG_FILE)

    training_cfg = TrainingConfig(
        batch_size=16,
        epochs_stage1=2,
        epochs_stage2=2,
        epochs_stage3=0,
        split_frequency="week",
        outer_holdout_blocks=0,
        min_train_blocks=2,
    )

    # Cache has M1,M5 — tick loaded via JIT if tick_root present
    timeframe_order = ("M1", "M5")
    tick_root = TICK_ROOT if os.path.isdir(TICK_ROOT) else None
    if tick_root:
        timeframe_order = ("tick", "M1", "M5")
        logger.info("tick_root found at %s, enabling tick timeframe", TICK_ROOT)
    else:
        logger.info("tick_root not found at %s, using M1/M5 only", TICK_ROOT)

    subnet_cfg = SubnetConfig(timeframe_order=timeframe_order)

    result = run_staged_experiment(
        mode="real",
        output=OUTPUT_ROOT,
        cache_root=CACHE_ROOT,
        tick_root=tick_root,
        max_folds=20,
        max_workers=8,
        device_override="auto",
        training_cfg=training_cfg,
        subnet_cfg=subnet_cfg,
        logger=logger,
        status_file=STATUS_FILE,
    )
    logger.info("Training complete: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
