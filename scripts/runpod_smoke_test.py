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
        info["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**2)
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


def main():
    import json
    results = {}

    print("=== RunPod v5.2 Smoke Test ===\n")

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
