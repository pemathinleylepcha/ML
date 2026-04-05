from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from staged_v4.utils.runtime_logging import configure_logging, log_exception, log_progress, stage_context, write_status
from tick_resampler import parse_tick_csv, resample_to_1000ms, stream_resample_tick_csv, write_output
from universe import ALL_INSTRUMENTS


def _match_instrument(path: Path, instruments: tuple[str, ...]) -> str | None:
    stem_upper = path.stem.upper()
    for instrument in sorted(instruments, key=len, reverse=True):
        if instrument.upper() in stem_upper:
            return instrument
    return None


def _discover_files(raw_root: Path, instruments: tuple[str, ...]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {instrument: [] for instrument in instruments}
    for path in raw_root.rglob("*.csv"):
        instrument = _match_instrument(path, instruments)
        if instrument is None:
            continue
        grouped[instrument].append(path)
    return {instrument: sorted(paths) for instrument, paths in grouped.items() if paths}


def build_tick_range_root(
    raw_root: Path,
    output_root: Path,
    start: str | None,
    end: str | None,
    instruments: tuple[str, ...],
    skip_existing: bool = False,
    logger: logging.Logger | None = None,
    status_file: str | None = None,
    stop_on_error: bool = False,
) -> dict[str, object]:
    logger = logger or logging.getLogger("build_tick_range_root")
    output_root.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    discovered = _discover_files(raw_root, instruments)
    written: dict[str, str] = {}
    skipped: list[str] = []
    failures: list[dict[str, str]] = []

    logger.info("discovered %d instruments with raw tick files", len(discovered))

    for idx, instrument in enumerate(instruments, start=1):
        log_progress(logger, status_file, "tick_range_build", idx - 1, len(instruments), instrument=instrument, written=len(written), failures=len(failures))
        files = discovered.get(instrument, [])
        if not files:
            continue
        existing_parquet = output_root / f"{instrument}_1000ms.parquet"
        existing_csv = output_root / f"{instrument}_1000ms.csv"
        if skip_existing and (existing_parquet.exists() or existing_csv.exists()):
            skipped.append(instrument)
            continue

        frames = []
        try:
            logger.info("instrument=%s files=%d state=start", instrument, len(files))
            if len(files) == 1 and files[0].stat().st_size >= 150_000_000:
                bars = stream_resample_tick_csv(
                    files[0],
                    instrument=instrument,
                    start=start_ts,
                    end=end_ts,
                )
            else:
                for csv_path in files:
                    ticks = parse_tick_csv(csv_path)
                    if start_ts is not None:
                        ticks = ticks[ticks["datetime"] >= start_ts]
                    if end_ts is not None:
                        ticks = ticks[ticks["datetime"] <= end_ts]
                    if len(ticks) == 0:
                        continue
                    frames.append(ticks)
                if not frames:
                    continue
                ticks = pd.concat(frames, ignore_index=True)
                ticks = ticks.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
                if len(ticks) == 0:
                    continue
                bars = resample_to_1000ms(ticks, instrument)
            if len(bars) == 0:
                continue
            written[instrument] = str(write_output(bars, instrument, output_root))
            logger.info("instrument=%s state=done bars=%d", instrument, len(bars))
        except Exception as exc:
            failures.append({"instrument": instrument, "error": f"{exc.__class__.__name__}: {exc}"})
            logger.exception("instrument=%s state=failed", instrument)
            if stop_on_error:
                raise
            continue

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "start": start,
        "end": end,
        "requested_instruments": list(instruments),
        "discovered_instruments": sorted(discovered.keys()),
        "written_count": len(written),
        "written": written,
        "skipped_existing": skipped,
        "failures": failures,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status(status_file, {"state": "completed", "stage": "tick_range_build", "summary": summary})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a ranged 1000ms tick root from recursive raw MT5 tick CSVs.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--instruments", default=None, help="Comma-separated subset; default is all instruments")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args(argv)

    logger = configure_logging("build_tick_range_root", args.log_file)
    instruments = tuple(i.strip() for i in args.instruments.split(",")) if args.instruments else tuple(ALL_INSTRUMENTS)
    try:
        with stage_context(logger, args.status_file, "tick_range_build_main", raw_root=args.raw_root, output_root=args.output_root, start=args.start, end=args.end):
            summary = build_tick_range_root(
                raw_root=Path(args.raw_root),
                output_root=Path(args.output_root),
                start=args.start,
                end=args.end,
                instruments=instruments,
                skip_existing=args.skip_existing,
                logger=logger,
                status_file=args.status_file,
                stop_on_error=args.stop_on_error,
            )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        log_exception(logger, args.status_file, "tick_range_build_main", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
