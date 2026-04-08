from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from research_dataset import _read_candles_csv
from staged_v4.data.dataset import _resample_frame


def _build_one(m1_path: Path, force: bool = False) -> dict[str, object]:
    m5_path = m1_path.with_name("candles_M5.csv")
    if m5_path.exists() and not force:
        return {
            "m1_path": str(m1_path),
            "m5_path": str(m5_path),
            "status": "exists",
            "rows": None,
        }

    frame = _read_candles_csv(m1_path).set_index("dt")
    frame["real"] = True
    resampled = _resample_frame(frame, "M5")
    if resampled is None or len(resampled) == 0:
        out = pd.DataFrame(
            columns=["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        )
        rows = 0
    else:
        out = resampled.reset_index().rename(
            columns={
                "dt": "bar_time",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "tk": "tick_volume",
                "sp": "spread",
            }
        )
        out["bar_time"] = out["bar_time"].dt.strftime("%Y.%m.%d %H:%M:%S")
        out["real_volume"] = 0.0
        out = out[["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
        rows = int(len(out))
    out.to_csv(m5_path, index=False)
    return {
        "m1_path": str(m1_path),
        "m5_path": str(m5_path),
        "status": "written",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build candles_M5.csv files from an EA M1 candle bundle.")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    bundle_root = Path(args.bundle_root)
    m1_files = sorted(bundle_root.rglob("candles_M1.csv"))
    results: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_build_one, path, args.force): path for path in m1_files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "m1_path": str(path),
                    "m5_path": str(path.with_name("candles_M5.csv")),
                    "status": "failed",
                    "error": str(exc),
                }
            results.append(result)

    summary = {
        "bundle_root": str(bundle_root),
        "workers": args.workers,
        "m1_count": len(m1_files),
        "written_count": sum(1 for item in results if item["status"] == "written"),
        "exists_count": sum(1 for item in results if item["status"] == "exists"),
        "failed_count": sum(1 for item in results if item["status"] == "failed"),
    }
    report = {"summary": summary, "results": sorted(results, key=lambda item: item["m1_path"])}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
