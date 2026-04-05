from __future__ import annotations

import numpy as np


def adaptive_entry_threshold(base_threshold: float, volatility: np.ndarray, coeff: float) -> np.ndarray:
    vol = np.asarray(volatility, dtype=np.float64)
    threshold = base_threshold + coeff * vol
    return np.clip(threshold, 0.50, 0.90).astype(np.float32)


def probability_spread_mask(p_buy: np.ndarray, p_sell: np.ndarray, threshold: float) -> np.ndarray:
    spread = np.abs(np.asarray(p_buy, dtype=np.float64) - np.asarray(p_sell, dtype=np.float64))
    return spread >= float(threshold)


def apply_cooldown(lock_state: np.ndarray, stop_events: np.ndarray, cooldown_bars: int) -> np.ndarray:
    lock_state = np.asarray(lock_state, dtype=np.int32).copy()
    stop_events = np.asarray(stop_events, dtype=np.bool_)
    for idx in range(len(lock_state)):
        if stop_events[idx]:
            lock_state[idx: idx + cooldown_bars] = np.maximum(lock_state[idx: idx + cooldown_bars], np.arange(cooldown_bars, 0, -1, dtype=np.int32))
    return lock_state


def enforce_correlation_exposure(
    proposed_pairs: list[str],
    active_pairs: list[str],
    max_group_exposure: int,
) -> list[str]:
    def _currencies(symbol: str) -> tuple[str, str]:
        return symbol[:3], symbol[3:6]

    accepted: list[str] = []
    active_groups = [_currencies(symbol) for symbol in active_pairs]
    for symbol in proposed_pairs:
        base, quote = _currencies(symbol)
        overlap = 0
        for active_base, active_quote in active_groups:
            if base in (active_base, active_quote) or quote in (active_base, active_quote):
                overlap += 1
        if overlap < max_group_exposure:
            accepted.append(symbol)
            active_groups.append((base, quote))
    return accepted
