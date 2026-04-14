"""RunPod smoke test — verify v5.2 pipeline deps, CUDA, and numba before training."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def check_imports() -> list[str]:
    """Return list of failed imports."""
    failures = []
    for mod in ("numpy", "pandas", "torch", "scipy", "sklearn", "catboost", "psutil", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            failures.append(mod)
    return failures


def check_cuda() -> dict:
    import torch
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": False,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total_mb"] = round(getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**2)
        free, total = torch.cuda.mem_get_info(0)
        info["vram_free_mb"] = round(free / 1024**2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def check_memory() -> dict:
    from staged_v5.data.memory_budget import get_available_ram_mb, get_available_vram_mb
    return {
        "ram_available_mb": round(get_available_ram_mb()),
        "vram_available_mb": round(get_available_vram_mb("cuda")),
    }


def check_numba() -> dict:
    try:
        from staged_v5.data.vp_features import _HAS_NUMBA
        return {"numba_available": _HAS_NUMBA}
    except Exception as e:
        return {"numba_available": False, "error": str(e)}


def check_v52_imports() -> list[str]:
    """Check all v5.2 pipeline modules load."""
    failures = []
    modules = [
        "staged_v5.config",
        "staged_v5.contracts",
        "staged_v5.data",
        "staged_v5.data.tick_features",
        "staged_v5.data.vp_features",
        "staged_v5.data.of_features",
        "staged_v5.data.jit_tick_loader",
        "staged_v5.data.memory_budget",
        "staged_v5.data.tick_preflight",
        "staged_v5.execution_gate",
        "staged_v5.execution_gate.features",
        "staged_v5.models",
        "staged_v5.training.train_staged",
    ]
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            failures.append(f"{mod}: {e}")
    return failures


def check_gate_v2() -> dict:
    """Verify gate v2 state vector builds correctly."""
    import numpy as np
    from staged_v5.execution_gate.features import (
        build_gate_state_vector_v2,
        GATE_V2_STATE_FEATURE_NAMES,
    )
    from staged_v5.config import VP_FEATURE_NAMES, OF_FEATURE_NAMES

    vp = np.zeros(len(VP_FEATURE_NAMES), dtype=np.float32)
    of = np.zeros(len(OF_FEATURE_NAMES), dtype=np.float32)
    state = build_gate_state_vector_v2(
        prob_buy=0.5, prob_entry=0.5, atr=0.001, volatility=0.3,
        session_code=1, vp_features=vp, of_features=of,
    )
    return {
        "gate_v2_dim": int(state.shape[0]),
        "expected_dim": len(GATE_V2_STATE_FEATURE_NAMES),
        "match": state.shape[0] == len(GATE_V2_STATE_FEATURE_NAMES),
    }


def check_chunk_sizing() -> dict:
    """Simulate chunk sizing for RunPod A100."""
    from staged_v5.data.memory_budget import compute_tick_chunk_size
    ram = get_available_ram_mb_safe()
    # Simulate A100 80GB
    a100_vram = 81920.0
    ram_chunk = compute_tick_chunk_size(ram, n_nodes=43, n_features=22, budget_fraction=0.25)
    vram_chunk = compute_tick_chunk_size(a100_vram, n_nodes=43, n_features=22, budget_fraction=0.15)
    return {
        "ram_chunk_bars": ram_chunk,
        "a100_vram_chunk_bars": vram_chunk,
        "effective_chunk": max(ram_chunk, vram_chunk),
    }


def get_available_ram_mb_safe() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available / 1024**2)
    except ImportError:
        return 4096.0


def check_volume_sizes(
    tick_root: str | None = None,
    candle_root: str | None = None,
    cache_root: str | None = None,
    repo_root: str | None = None,
) -> dict:
    """Check disk usage for dataset dirs, repo, and free space on each mount."""
    from pathlib import Path
    import shutil

    repo_root = repo_root or os.path.join(os.path.dirname(__file__), "..")
    results: dict = {}

    def _dir_size_mb(path: str) -> float | None:
        p = Path(path)
        if not p.exists():
            return None
        total = 0
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return round(total / 1024 / 1024, 1)

    def _file_count(path: str) -> int | None:
        p = Path(path)
        if not p.exists():
            return None
        return sum(1 for f in p.rglob("*") if f.is_file())

    def _disk_free_mb(path: str) -> float | None:
        try:
            usage = shutil.disk_usage(path)
            return round(usage.free / 1024 / 1024, 1)
        except Exception:
            return None

    def _largest_files(path: str, n: int = 5) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        files = [(f, f.stat().st_size) for f in p.rglob("*") if f.is_file()]
        files.sort(key=lambda x: x[1], reverse=True)
        return [
            {"name": str(f.name), "mb": round(sz / 1024 / 1024, 1)}
            for f, sz in files[:n]
        ]

    # Repo size
    repo_size = _dir_size_mb(repo_root)
    results["repo"] = {
        "path": str(repo_root),
        "size_mb": repo_size,
        "files": _file_count(repo_root),
        "disk_free_mb": _disk_free_mb(repo_root),
    }

    # Src only
    src_path = os.path.join(repo_root, "src")
    results["src"] = {
        "path": src_path,
        "size_mb": _dir_size_mb(src_path),
        "files": _file_count(src_path),
    }

    # Tick root
    if tick_root:
        results["tick_root"] = {
            "path": tick_root,
            "exists": Path(tick_root).exists(),
            "size_mb": _dir_size_mb(tick_root),
            "files": _file_count(tick_root),
            "disk_free_mb": _disk_free_mb(tick_root),
            "largest": _largest_files(tick_root),
        }

    # Candle root
    if candle_root:
        results["candle_root"] = {
            "path": candle_root,
            "exists": Path(candle_root).exists(),
            "size_mb": _dir_size_mb(candle_root),
            "files": _file_count(candle_root),
            "disk_free_mb": _disk_free_mb(candle_root),
            "largest": _largest_files(candle_root),
        }

    # Feature cache (optional)
    if cache_root:
        results["cache_root"] = {
            "path": cache_root,
            "exists": Path(cache_root).exists(),
            "size_mb": _dir_size_mb(cache_root),
            "files": _file_count(cache_root),
            "disk_free_mb": _disk_free_mb(cache_root),
        }

    # Estimate minimum RunPod volume needed
    data_total = 0.0
    for key in ("tick_root", "candle_root", "cache_root"):
        if key in results and results[key].get("size_mb"):
            data_total += results[key]["size_mb"]
    overhead_mb = (repo_size or 50) + 2000  # repo + pip packages + scratch
    min_volume_gb = round((data_total + overhead_mb) / 1024 * 1.3, 1)  # 30% headroom
    results["volume_recommendation"] = {
        "data_total_mb": round(data_total, 1),
        "overhead_mb": round(overhead_mb, 1),
        "min_volume_gb": min_volume_gb,
        "recommended_gb": max(min_volume_gb, 50),  # at least 50GB
    }

    return results


def main():
    import json
    import argparse

    parser = argparse.ArgumentParser(description="RunPod v5.2 smoke test")
    parser.add_argument(
        "--tick-root", type=str, default=None,
        help="Path to tick root (default: auto-detect from known locations)",
    )
    parser.add_argument(
        "--candle-root", type=str, default=None,
        help="Path to candle root (default: auto-detect from known locations)",
    )
    parser.add_argument(
        "--cache-root", type=str, default=None,
        help="Path to feature cache (optional, speeds up training)",
    )
    args = parser.parse_args()

    # Auto-detect data paths from known locations (remote 172 box and RunPod mounts)
    _KNOWN_TICK_ROOTS = [
        "/data/ea_training_tickroot_v52_q4_20251001_20251231_1000ms",
        "/workspace/data/ea_training_tickroot_v52_q4_20251001_20251231_1000ms",
        "/runpod-volume/ea_training_tickroot_v52_q4_20251001_20251231_1000ms",
        "D:/COLLECT-TICK-MT5/ea_training_tickroot_v52_q4_20251001_20251231_1000ms",
    ]
    _KNOWN_CANDLE_ROOTS = [
        "/data/ea_training_bundle_6m_full44",
        "/workspace/data/ea_training_bundle_6m_full44",
        "/runpod-volume/ea_training_bundle_6m_full44",
        "D:/COLLECT-TICK-MT5/ea_training_bundle_6m_full44",
    ]
    _KNOWN_CACHE_ROOTS = [
        "/data/ea_training_cache_6m_full44_m1m5_v2",
        "/workspace/data/ea_training_cache_6m_full44_m1m5_v2",
        "/runpod-volume/ea_training_cache_6m_full44_m1m5_v2",
        "D:/COLLECT-TICK-MT5/ea_training_cache_6m_full44_m1m5_v2",
    ]

    def _auto_detect(explicit, candidates):
        if explicit:
            return explicit
        for path in candidates:
            if os.path.isdir(path):
                return path
        return None

    tick_root = _auto_detect(args.tick_root, _KNOWN_TICK_ROOTS)
    candle_root = _auto_detect(args.candle_root, _KNOWN_CANDLE_ROOTS)
    cache_root = _auto_detect(args.cache_root, _KNOWN_CACHE_ROOTS)

    results = {}

    print("=== RunPod v5.2 Smoke Test ===\n")

    # 0. Volume / disk sizes
    vol = check_volume_sizes(
        tick_root=tick_root,
        candle_root=candle_root,
        cache_root=cache_root,
    )
    results["volumes"] = vol
    print("--- Volume Sizes ---")
    print(f"  Repo: {vol['repo']['size_mb']} MB ({vol['repo']['files']} files) | disk free: {vol['repo']['disk_free_mb']} MB")
    print(f"  Src:  {vol['src']['size_mb']} MB ({vol['src']['files']} files)")
    if "tick_root" in vol:
        tr = vol["tick_root"]
        if tr["exists"]:
            print(f"  Tick root: {tr['size_mb']} MB ({tr['files']} files) | disk free: {tr['disk_free_mb']} MB")
            for f in tr.get("largest", [])[:3]:
                print(f"    {f['name']}: {f['mb']} MB")
        else:
            print(f"  Tick root: NOT FOUND at {tr['path']}")
    if "candle_root" in vol:
        cr = vol["candle_root"]
        if cr["exists"]:
            print(f"  Candle root: {cr['size_mb']} MB ({cr['files']} files) | disk free: {cr['disk_free_mb']} MB")
            for f in cr.get("largest", [])[:3]:
                print(f"    {f['name']}: {f['mb']} MB")
        else:
            print(f"  Candle root: NOT FOUND at {cr['path']}")
    if "cache_root" in vol:
        cc = vol["cache_root"]
        if cc["exists"]:
            print(f"  Feature cache: {cc['size_mb']} MB ({cc['files']} files)")
        else:
            print(f"  Feature cache: NOT FOUND at {cc['path']}")
    else:
        print("  Feature cache: not specified (will build from candles, slower)")
    rec = vol["volume_recommendation"]
    print(f"  Volume recommendation: min {rec['min_volume_gb']} GB, recommended {rec['recommended_gb']} GB")
    print()

    # 1. Core imports
    failed = check_imports()
    results["core_imports"] = {"failed": failed, "ok": len(failed) == 0}
    print(f"Core imports: {'PASS' if not failed else 'FAIL ' + str(failed)}")

    # 2. CUDA
    cuda = check_cuda()
    results["cuda"] = cuda
    status = "PASS" if cuda["cuda_available"] else "FAIL (CPU only)"
    print(f"CUDA: {status}")
    if cuda["cuda_available"]:
        print(f"  GPU: {cuda.get('device_name')} | VRAM: {cuda.get('vram_total_mb')}MB | bf16: {cuda.get('bf16_supported')}")

    # 3. Memory
    mem = check_memory()
    results["memory"] = mem
    print(f"RAM: {mem['ram_available_mb']}MB | VRAM: {mem['vram_available_mb']}MB")

    # 4. Numba
    nb = check_numba()
    results["numba"] = nb
    print(f"Numba: {'PASS' if nb.get('numba_available') else 'WARN (VP will use slow numpy path)'}")

    # 5. v5.2 imports
    v52_failed = check_v52_imports()
    results["v52_imports"] = {"failed": v52_failed, "ok": len(v52_failed) == 0}
    print(f"v5.2 modules: {'PASS' if not v52_failed else 'FAIL'}")
    for f in v52_failed:
        print(f"  FAIL: {f}")

    # 6. Gate v2
    gate = check_gate_v2()
    results["gate_v2"] = gate
    print(f"Gate v2: {'PASS' if gate['match'] else 'FAIL'} (dim={gate['gate_v2_dim']})")

    # 7. Chunk sizing
    chunk = check_chunk_sizing()
    results["chunk_sizing"] = chunk
    print(f"Chunk sizing: RAM={chunk['ram_chunk_bars']} VRAM(A100)={chunk['a100_vram_chunk_bars']} effective={chunk['effective_chunk']}")

    # Summary
    all_ok = (
        results["core_imports"]["ok"]
        and results["cuda"]["cuda_available"]
        and results["v52_imports"]["ok"]
        and results["gate_v2"]["match"]
    )
    print(f"\n{'ALL PASS' if all_ok else 'SOME CHECKS FAILED'}")
    print(json.dumps(results, indent=2, default=str))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
