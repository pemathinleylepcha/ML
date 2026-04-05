from __future__ import annotations

from dataclasses import dataclass, field

from universe import SUBNET_24x5, SUBNET_24x5_TRADEABLE, SUBNET_24x7


ALL_TIMEFRAMES = ("tick", "M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1")
TIMEFRAME_FREQ = {
    "tick": "1s",
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "H12": "12h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1ME",
}
LOWER_TIMEFRAME = {
    "M1": "tick",
    "M5": "M1",
    "M15": "M5",
    "M30": "M15",
    "H1": "M30",
    "H4": "H1",
    "H12": "H4",
    "D1": "H12",
    "W1": "D1",
    "MN1": "W1",
}
DEFAULT_SEQ_LENS = {
    "tick": 120,
    "M1": 90,
    "M5": 72,
    "M15": 48,
    "M30": 36,
    "H1": 24,
    "H4": 18,
    "H12": 12,
    "D1": 8,
    "W1": 6,
    "MN1": 4,
}
RAW_FEATURE_NAMES = (
    "o",
    "h",
    "l",
    "c",
    "sp",
    "tk",
    "ret_1",
    "ret_3",
    "atr_norm",
    "range_norm",
    "lap_residual",
    "regime_signal",
    "session_code",
    "session_transition",
)
TPO_FEATURE_NAMES = (
    "tpo_poc_distance_atr",
    "tpo_value_area_width_atr",
    "tpo_balance_score",
    "tpo_support_score",
    "tpo_resistance_score",
    "tpo_rejection_score",
    "tpo_poc_drift_atr",
    "tpo_value_area_overlap",
)
EDGE_TYPES = ("rolling_corr", "fundamental", "session")
BTC_NODE_NAMES = tuple(SUBNET_24x7)
FX_NODE_NAMES = tuple(SUBNET_24x5)
FX_TRADABLE_NAMES = tuple(SUBNET_24x5_TRADEABLE)
TPO_SOURCE_TIMEFRAME = {
    "tick": "M5",
    "M1": "M5",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "H12": "H12",
    "D1": "D1",
    "W1": "W1",
    "MN1": "MN1",
}


@dataclass(slots=True)
class STGNNBlockConfig:
    hidden_dim: int = 48
    output_dim: int = 48
    n_heads: int = 4
    ff_multiplier: int = 2
    dropout: float = 0.10
    tpo_gate_alpha: float = 6.0
    tpo_gate_threshold: float = 0.0015
    trainable_tpo_gate: bool = True


@dataclass(slots=True)
class SubnetConfig:
    timeframe_order: tuple[str, ...] = field(default_factory=lambda: ALL_TIMEFRAMES)
    exchange_every_k_batches: int = 4
    active_loss_boost: float = 2.0
    enable_entry_head: bool = True


@dataclass(slots=True)
class TrainingConfig:
    anchor_timeframe: str = "M1"
    batch_size: int = 8
    epochs_stage1: int = 2
    epochs_stage2: int = 2
    epochs_stage3: int = 1
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 3e-4
    purge_bars: int = 6
    split_frequency: str = "week"
    outer_holdout_blocks: int = 1
    min_train_blocks: int = 2


@dataclass(slots=True)
class BacktestConfig:
    base_entry_threshold: float = 0.60
    threshold_volatility_coeff: float = 12.0
    exit_threshold: float = 0.52
    probability_spread_threshold: float = 0.10
    latency_bars: int = 1
    cooldown_bars: int = 3
    max_positions: int = 6
    max_hold_bars: int = 6
    entry_gate_threshold: float = 0.50
    max_confidence_threshold: float = 0.70
    max_group_exposure: int = 2
    take_profit_atr: float = 1.00
    stop_loss_atr: float = 0.70
    use_limit_entries: bool = True
    limit_offset_atr: float = 0.10


@dataclass(slots=True)
class GAConfig:
    population_size: int = 8
    generations: int = 3
    mutation_rate: float = 0.15
    crossover_rate: float = 0.50
