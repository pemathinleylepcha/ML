from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenBenchmarkReference:
    name: str
    report_path: str
    outer_sharpe: float
    pbo: float | None
    win_rate: float


def load_best_compact_benchmark(repo_root: str | Path | None = None) -> FrozenBenchmarkReference:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    report_path = root / "data" / "remote_clean_2025_runs" / "clean_2025_no_bridge_optk10_report.json"
    if not report_path.exists():
        return FrozenBenchmarkReference(
            name="compact_no_bridge_topk10",
            report_path=str(report_path),
            outer_sharpe=7.923213100413904,
            pbo=0.1104,
            win_rate=0.5269855199648968,
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    pbo_payload = payload.get("pbo") or {}
    outer = payload.get("outer_holdout") or {}
    sharpe = outer.get("strategy_sharpe", outer.get("sharpe"))
    return FrozenBenchmarkReference(
        name="compact_no_bridge_topk10",
        report_path=str(report_path),
        outer_sharpe=float(sharpe),
        pbo=None if pbo_payload.get("pbo") is None else float(pbo_payload["pbo"]),
        win_rate=float(outer["win_rate"]),
    )
