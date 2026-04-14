from __future__ import annotations

import logging


_LOGGER = logging.getLogger(__name__)


def get_available_ram_mb() -> float:
    """Return available system RAM in MB.

    Falls back to a conservative 4096 MB if `psutil` is unavailable.
    """
    try:
        import psutil

        return float(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        _LOGGER.warning("psutil not available; assuming 4096 MB RAM")
        return 4096.0


def get_available_vram_mb(device_type: str = "cuda") -> float:
    """Return available GPU VRAM in MB.

    Returns 0.0 for non-CUDA devices or if CUDA is unavailable.
    """
    if device_type != "cuda":
        return 0.0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return float(free_bytes / (1024 * 1024))
    except Exception:
        return 0.0


def compute_tick_chunk_size(
    available_ram_mb: float,
    n_nodes: int,
    n_features: int,
    budget_fraction: float = 0.25,
    min_chunk_bars: int = 1_000,
    max_chunk_bars: int = 600_000,
) -> int:
    """Compute a safe chunk size for tick-bar loading.

    The estimate assumes float32 storage for the feature tensor itself and
    intentionally keeps the calculation simple; additional tensors and masks are
    handled by the budget fraction and clamp bounds.
    """
    if n_nodes <= 0 or n_features <= 0:
        return int(max_chunk_bars)
    budget_bytes = float(available_ram_mb) * float(budget_fraction) * 1024 * 1024
    bytes_per_bar = int(n_nodes) * int(n_features) * 4
    if bytes_per_bar <= 0:
        return int(max_chunk_bars)
    chunk_bars = int(budget_bytes / bytes_per_bar)
    return int(max(min_chunk_bars, min(chunk_bars, max_chunk_bars)))
