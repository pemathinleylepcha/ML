"""Build v5.2 feature cache on 172 box (CPU only, no GPU cost).

Run on 172:
    cd D:/work/Algo-C2-Codex
    python scripts/build_cache_172.py

Output: D:/work/Algo-C2-Codex/cache_v52_q4/
Transfer to RunPod, then train with --cache-root.
"""
from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from staged_v5.data.cache import prepare_staged_cache
from staged_v5.utils.runtime_logging import configure_logging


CANDLE_ROOT = "D:/COLLECT-TICK-MT5/ea_training_bundle_6m_full44"
OUTPUT_ROOT = "D:/work/Algo-C2-Codex/cache_v52_q4"
STATUS_FILE = "D:/work/Algo-C2-Codex/cache_v52_q4/status.json"

# No tick root for cache build — tick features use JIT loader at training time
TICK_ROOT = None

TIMEFRAMES = ("M1", "M5")
ANCHOR_TIMEFRAME = "M1"
SPLIT_FREQUENCY = "week"
OUTER_HOLDOUT_BLOCKS = 0
MIN_TRAIN_BLOCKS = 2
PURGE_BARS = 6
MAX_WORKERS = 8


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    logger = configure_logging("build_cache", os.path.join(OUTPUT_ROOT, "build.log"))
    logger.info("=== v5.2 Cache Build on 172 ===")
    logger.info("candle_root=%s", CANDLE_ROOT)
    logger.info("output_root=%s", OUTPUT_ROOT)
    logger.info("timeframes=%s", TIMEFRAMES)

    t0 = time.perf_counter()
    manifest = prepare_staged_cache(
        output_root=OUTPUT_ROOT,
        mode="real",
        candle_root=CANDLE_ROOT,
        tick_root=TICK_ROOT,
        anchor_timeframe=ANCHOR_TIMEFRAME,
        timeframes=TIMEFRAMES,
        split_frequency=SPLIT_FREQUENCY,
        outer_holdout_blocks=OUTER_HOLDOUT_BLOCKS,
        min_train_blocks=MIN_TRAIN_BLOCKS,
        purge_bars=PURGE_BARS,
        max_workers=MAX_WORKERS,
        logger=logger,
        status_file=STATUS_FILE,
    )
    elapsed = time.perf_counter() - t0
    logger.info("=== Cache build complete in %.1f seconds ===", elapsed)
    logger.info("manifest=%s", manifest)

    # Print summary for quick check
    import json
    print(f"\nDone in {elapsed:.1f}s")
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
