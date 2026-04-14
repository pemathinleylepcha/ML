from __future__ import annotations

from dataclasses import dataclass, field, fields

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
VP_FEATURE_NAMES = (
    "vp_poc_distance_atr",
    "vp_value_area_width_atr",
    "vp_high_volume_nodes",
    "vp_poc_slope",
)
OF_FEATURE_NAMES = (
    "of_delta",
    "of_delta_cumulative_60",
    "of_delta_cumulative_300",
    "of_absorption_ratio",
    "of_imbalance_streak",
    "of_spread_compression",
)
ATR_MIN_THRESHOLD = 1e-6
ATR_NORM_CLIP = 50.0  # max abs value for ATR-normalized TPO features
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
    batch_size: int = 16
    epochs_stage1: int = 2
    epochs_stage2: int = 2
    epochs_stage3: int = 1
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 3e-4
    purge_bars: int = 6
    split_frequency: str = "week"
    outer_holdout_blocks: int = 1
    min_train_blocks: int = 2
    label_smooth: float = 0.10
    memory_guard_min_available_mb: float = 4096.0
    memory_guard_critical_available_mb: float = 2048.0
    memory_guard_check_interval: int = 25
    use_torch_compile: bool = True
    gpu_memory_guard_min_mb: float = 4096.0
    gpu_memory_guard_critical_mb: float = 2048.0


@dataclass(slots=True)
class TickChunkConfig:
    budget_fraction_ram: float = 0.25
    budget_fraction_vram: float = 0.15
    min_chunk_bars: int = 1_000
    max_chunk_bars: int = 2_000_000
    tick_seq_len: int = 120
    prefetch: bool = True


@dataclass(slots=True)
class EntryConfig:
    entry_type: str = "limit"
    base_entry_threshold: float = 0.65
    threshold_volatility_coeff: float = 12.0
    probability_spread_threshold: float = 0.15
    entry_gate_threshold: float = 0.55
    max_confidence_threshold: float = 1.01
    latency_bars: int = 1
    limit_offset_atr: float = 0.10
    slippage_atr: float = 0.01
    max_entry_atr_pct: float = 0.01


@dataclass(slots=True)
class NeuralGateConfig:
    enabled: bool = False
    action_temperature: float = 1.0
    reference_price_mode: str = "poc"
    mfe_horizon_bars: int = 120
    mae_horizon_bars: int = 120
    grpo_group_size: int = 8
    grpo_clip_epsilon: float = 0.2
    grpo_kl_beta: float = 0.01
    positive_fill_reward_boost: float = 2.0
    reject_wait_penalty: float = -5e-5
    market_order_slippage_ticks: float = 1.0
    limit_at_poc_near_miss_reward: float = 5e-5
    limit_at_poc_no_fill_penalty: float = -1e-5
    passive_limit_no_fill_penalty: float = -2.5e-5
    zero_variance_skip_epsilon: float = 1e-5


@dataclass(slots=True)
class ExitConfig:
    exit_type: str = "trailing_atr"
    take_profit_atr: float = 1.00
    stop_loss_atr: float = 0.70
    max_loss_pct_per_trade: float = 0.005
    trailing_activate_atr: float = 0.50
    max_hold_bars: int = 6
    exit_threshold: float = 0.52
    slippage_atr: float = 0.01
    close_before_weekend: bool = True
    weekend_close_hour_utc: int = 21
    enable_tp_shift: bool = False
    tp_shift_signal_threshold: float = 0.70
    tp_shift_atr: float = 0.50
    tp_shift_sl_lock_atr: float = 0.30
    tp_shift_max_extensions: int = 3


@dataclass(slots=True)
class PositionConfig:
    max_positions: int = 6
    cooldown_bars: int = 3
    max_group_exposure: int = 2


@dataclass
class BacktestConfig:
    entry: EntryConfig = field(default_factory=EntryConfig)
    neural_gate: NeuralGateConfig = field(default_factory=NeuralGateConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    ece_gate_threshold: float = 0.0

    @classmethod
    def from_flat(cls, d: dict) -> "BacktestConfig":
        entry_kwargs: dict = {}
        gate_kwargs: dict = {}
        exit_kwargs: dict = {}
        pos_kwargs: dict = {}
        ece_gate = d.get("ece_gate_threshold", 0.0)

        entry_keys = {
            "base_entry_threshold",
            "threshold_volatility_coeff",
            "probability_spread_threshold",
            "entry_gate_threshold",
            "max_confidence_threshold",
            "latency_bars",
            "limit_offset_atr",
            "max_entry_atr_pct",
        }
        exit_keys = {
            "take_profit_atr",
            "stop_loss_atr",
            "max_loss_pct_per_trade",
            "trailing_activate_atr",
            "max_hold_bars",
            "exit_threshold",
            "close_before_weekend",
            "weekend_close_hour_utc",
            "enable_tp_shift",
            "tp_shift_signal_threshold",
            "tp_shift_atr",
            "tp_shift_sl_lock_atr",
            "tp_shift_max_extensions",
        }
        pos_keys = {"max_positions", "cooldown_bars", "max_group_exposure"}
        gate_keys = {field.name for field in fields(NeuralGateConfig)}

        for key, value in d.items():
            if key in entry_keys:
                entry_kwargs[key] = value
            elif key in exit_keys:
                exit_kwargs[key] = value
            elif key in pos_keys:
                pos_kwargs[key] = value
            elif key.startswith("neural_gate_"):
                gate_key = key[len("neural_gate_") :]
                if gate_key in gate_keys:
                    gate_kwargs[gate_key] = value
            elif key in gate_keys:
                gate_kwargs[key] = value
        if "slippage_atr" in d:
            entry_kwargs["slippage_atr"] = d["slippage_atr"]
            exit_kwargs["slippage_atr"] = d["slippage_atr"]
        if "entry_type" in d:
            entry_kwargs["entry_type"] = d["entry_type"]
        elif "use_limit_entries" in d:
            entry_kwargs["entry_type"] = "limit" if d["use_limit_entries"] else "market"
        return cls(
            entry=EntryConfig(**entry_kwargs),
            neural_gate=NeuralGateConfig(**gate_kwargs),
            exit=ExitConfig(**exit_kwargs),
            position=PositionConfig(**pos_kwargs),
            ece_gate_threshold=ece_gate,
        )

    def to_flat(self) -> dict:
        flat: dict = {}
        for item in fields(self.entry):
            if item.name == "entry_type":
                continue
            flat[item.name] = getattr(self.entry, item.name)
        flat["entry_type"] = self.entry.entry_type
        flat["use_limit_entries"] = self.entry.entry_type == "limit"
        for item in fields(self.neural_gate):
            flat[f"neural_gate_{item.name}"] = getattr(self.neural_gate, item.name)
        for item in fields(self.exit):
            if item.name in ("exit_type", "slippage_atr"):
                continue
            flat[item.name] = getattr(self.exit, item.name)
        if "slippage_atr" not in flat:
            flat["slippage_atr"] = self.entry.slippage_atr
        for item in fields(self.position):
            flat[item.name] = getattr(self.position, item.name)
        flat["ece_gate_threshold"] = self.ece_gate_threshold
        return flat


GA_PARAM_SPACE: list[tuple[str, float, float]] = [
    ("entry.base_entry_threshold", 0.55, 0.90),
    ("entry.threshold_volatility_coeff", 4.0, 20.0),
    ("entry.probability_spread_threshold", 0.05, 0.25),
    ("entry.entry_gate_threshold", 0.50, 0.80),
    ("entry.limit_offset_atr", 0.0, 0.30),
    ("entry.latency_bars", 1, 3),
    ("exit.take_profit_atr", 0.50, 3.00),
    ("exit.stop_loss_atr", 0.30, 2.50),
    ("exit.max_loss_pct_per_trade", 0.002, 0.02),
    ("exit.trailing_activate_atr", 0.20, 1.50),
    ("exit.max_hold_bars", 3, 30),
    ("exit.exit_threshold", 0.50, 0.60),
    ("position.max_positions", 2, 10),
    ("position.cooldown_bars", 1, 6),
    ("position.max_group_exposure", 1, 4),
]

_INT_GA_PARAMS = {
    "entry.latency_bars",
    "exit.max_hold_bars",
    "position.max_positions",
    "position.cooldown_bars",
    "position.max_group_exposure",
}


def decode_ga_genome(genome: list[float], base_cfg: BacktestConfig) -> BacktestConfig:
    flat = base_cfg.to_flat()
    for gene_value, (param_path, lo, hi) in zip(genome, GA_PARAM_SPACE):
        key = param_path.split(".")[-1]
        value = lo + gene_value * (hi - lo)
        if param_path in _INT_GA_PARAMS:
            value = round(value)
        flat[key] = value
    return BacktestConfig.from_flat(flat)


@dataclass(slots=True)
class GAConfig:
    population_size: int = 30
    generations: int = 10
    mutation_rate: float = 0.15
    crossover_rate: float = 0.50
    ga_objective: str = "sharpe"
