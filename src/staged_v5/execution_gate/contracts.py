from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class NeuralGateAction(IntEnum):
    REJECT_WAIT = 0
    MARKET_NOW = 1
    LIMIT_2TICK = 2
    LIMIT_3TICK = 3
    LIMIT_HALF_ATR = 4
    PASSIVE_LIMIT_1TICK = 5


def direction_from_prob(prob_buy: float) -> int:
    return 1 if float(prob_buy) >= 0.5 else -1


def _default_action_counts() -> dict[str, int]:
    return {action.name.lower(): 0 for action in NeuralGateAction}


def _default_action_reward_sums() -> dict[str, float]:
    return {action.name.lower(): 0.0 for action in NeuralGateAction}


@dataclass(slots=True)
class GateEvent:
    timestamp: np.datetime64
    pair_name: str
    direction: int
    prob_buy: float
    prob_entry: float | None
    close: float
    atr: float
    volatility: float
    session_code: int
    spread: float
    tick_count: float
    tpo_features: np.ndarray
    vp_features: np.ndarray | None = None
    of_features: np.ndarray | None = None


@dataclass(slots=True)
class NeuralGateRuntime:
    model: Any
    config: Any
    tick_store: Any
    device: Any
    action_counts: dict[str, int] = field(default_factory=_default_action_counts)
    fill_counts: dict[str, int] = field(default_factory=_default_action_counts)
    no_fill_counts: dict[str, int] = field(default_factory=_default_action_counts)
    near_miss_counts: dict[str, int] = field(default_factory=_default_action_counts)
    reward_sums: dict[str, float] = field(default_factory=_default_action_reward_sums)
    reject_count: int = 0
    no_fill_count: int = 0
