from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BarState:
    bar_index: int
    node_idx: int
    prob_buy: float
    prob_entry: float | None
    high: float
    low: float
    close: float
    atr: float
    volatility: float
    session_code: int
    pair_name: str


@dataclass(slots=True)
class Order:
    node_idx: int
    pair_name: str
    direction: int
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float
    entry_atr: float
    signal_bar: int
    entry_bar: int


@dataclass(slots=True)
class EntryContext:
    timestamp: object | None = None
    entry_timestamp: object | None = None
    anchor_node_features: Any | None = None
    anchor_tpo_features: Any | None = None
    neural_gate_runtime: Any | None = None


@dataclass(slots=True)
class ExitDecision:
    exit_price: float
    reason: str


@dataclass(slots=True)
class OpenPosition:
    node_idx: int
    pair_name: str
    direction: int
    signal_bar: int
    entry_bar: int
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float
    entry_atr: float
    tp_extensions: int = 0


@dataclass(slots=True)
class RejectionCounters:
    bars_seen: int = 0
    total_evaluated: int = 0
    latency_skip: int = 0
    session_closed: int = 0
    max_positions_blocked: int = 0
    exposure_cooldown_blocked: int = 0
    already_open_blocked: int = 0
    cooldown_blocked: int = 0
    correlation_exposure_blocked: int = 0
    direction_threshold_failed: int = 0
    entry_head_failed: int = 0
    limit_no_fill: int = 0
    neural_gate_reject: int = 0
    neural_gate_no_fill: int = 0
    entries_created: int = 0
