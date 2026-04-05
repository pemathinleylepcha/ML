from __future__ import annotations

import argparse
import json
from pathlib import Path

from staged_v4.data.cache import prepare_staged_cache
from staged_v4.utils.runtime_logging import configure_logging, log_exception, stage_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare cached staged_v4 feature batches for faster real-data training.")
    parser.add_argument("--mode", choices=("synthetic", "real"), default="real")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candle-root", default=None)
    parser.add_argument("--tick-root", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--anchor-timeframe", default="M1")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--timeframes", default=None, help="Comma-separated timeframe subset")
    parser.add_argument("--synthetic-n-anchor", type=int, default=96)
    parser.add_argument("--tradeable-tpo-only", action="store_true")
    parser.add_argument("--split-frequency", choices=("week", "month"), default="week")
    parser.add_argument("--outer-holdout-blocks", type=int, default=1)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--purge-bars", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=0, help="Parallel symbol-load workers; 0 = auto")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)

    logger = configure_logging("prepare_staged_v4_cache", args.log_file)
    timeframes = tuple(tf.strip() for tf in args.timeframes.split(",")) if args.timeframes else None
    try:
        with stage_context(logger, args.status_file, "prepare_cache_main", mode=args.mode, output_root=args.output_root):
            manifest = prepare_staged_cache(
                output_root=args.output_root,
                mode=args.mode,
                candle_root=args.candle_root,
                tick_root=args.tick_root,
                start=args.start,
                end=args.end,
                anchor_timeframe=args.anchor_timeframe,
                strict=args.strict,
                timeframes=timeframes,
                synthetic_n_anchor=args.synthetic_n_anchor,
                include_signal_only_tpo=not args.tradeable_tpo_only,
                split_frequency=args.split_frequency,
                outer_holdout_blocks=args.outer_holdout_blocks,
                min_train_blocks=args.min_train_blocks,
                purge_bars=args.purge_bars,
                max_workers=args.max_workers,
                logger=logger,
                status_file=args.status_file,
            )
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        log_exception(logger, args.status_file, "prepare_cache_main", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
