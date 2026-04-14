from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research_dataset import SESSION_NAMES
from staged_v5.config import VP_FEATURE_NAMES, OF_FEATURE_NAMES


MICROSTRUCTURE_FEATURE_NAMES = (
    "tick_velocity_mean",
    "tick_velocity_std",
    "spread_z_mean",
    "spread_z_last",
    "bid_ask_imbalance_mean",
    "bid_ask_imbalance_abs_max",
    "price_velocity_last",
)

GATE_STATE_FEATURE_NAMES = (
    "prob_buy",
    "prob_entry",
    "atr",
    "volatility",
    "spread",
    "tick_count",
    *tuple(f"session_{name}" for name in SESSION_NAMES),
    "tpo_poc_distance_atr",
    "tpo_value_area_width_atr",
    "tpo_balance_score",
    "tpo_support_score",
    "tpo_resistance_score",
    "tpo_rejection_score",
    "tpo_poc_drift_atr",
    "tpo_value_area_overlap",
    *MICROSTRUCTURE_FEATURE_NAMES,
)

_REQUIRED_COLUMNS = (
    "dt",
    "o",
    "h",
    "l",
    "c",
    "sp",
    "tk",
    "tick_velocity",
    "spread_z",
    "bid_ask_imbalance",
    "price_velocity",
)


class TickProxyStore:
    def __init__(self, tick_root: str | Path):
        self.tick_root = Path(tick_root)
        self._cache: dict[str, pd.DataFrame] = {}

    def _candidate_paths(self, pair_name: str) -> list[Path]:
        upper = pair_name.upper()
        return [
            self.tick_root / f"{upper}_1000ms.parquet",
            self.tick_root / f"{upper}_1000ms.csv",
            *self.tick_root.rglob(f"{upper}_1000ms.parquet"),
            *self.tick_root.rglob(f"{upper}_1000ms.csv"),
        ]

    def find_path(self, pair_name: str) -> Path | None:
        key = pair_name.upper()
        return next((candidate for candidate in self._candidate_paths(key) if candidate.exists()), None)

    def missing_pairs(self, pair_names: list[str] | tuple[str, ...]) -> list[str]:
        return [pair_name for pair_name in pair_names if self.find_path(pair_name) is None]

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        rename = {
            "datetime": "dt",
            "bar_time": "dt",
            "time": "dt",
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "spread": "sp",
            "tick_volume": "tk",
            "volume": "tk",
        }
        frame = frame.rename(columns=rename)
        missing = [col for col in _REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required tick proxy columns {missing} in {path}")
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce", utc=False)
        frame = frame.dropna(subset=["dt"]).copy()
        for col in _REQUIRED_COLUMNS[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        frame = frame.sort_values("dt").drop_duplicates("dt")
        return frame.set_index("dt")[list(_REQUIRED_COLUMNS[1:])]

    def get_frame(self, pair_name: str) -> pd.DataFrame:
        key = pair_name.upper()
        if key not in self._cache:
            path = self.find_path(key)
            if path is None:
                raise FileNotFoundError(f"Could not find 1000ms proxy bars for {pair_name} under {self.tick_root}")
            self._cache[key] = self._read_frame(path)
        return self._cache[key]

    def get_pre_entry_window(self, pair_name: str, anchor_timestamp: object, lookback_seconds: int = 60) -> pd.DataFrame:
        anchor_ts = pd.Timestamp(anchor_timestamp)
        start_ts = anchor_ts - pd.Timedelta(seconds=lookback_seconds)
        frame = self.get_frame(pair_name)
        return frame[(frame.index >= start_ts) & (frame.index < anchor_ts)]

    def get_execution_window(self, pair_name: str, anchor_timestamp: object, horizon_bars: int) -> pd.DataFrame:
        anchor_ts = pd.Timestamp(anchor_timestamp)
        frame = self.get_frame(pair_name)
        window = frame[frame.index >= anchor_ts]
        if horizon_bars > 0:
            return window.iloc[:horizon_bars].copy()
        return window.copy()


def _session_one_hot(session_code: int) -> np.ndarray:
    one_hot = np.zeros(len(SESSION_NAMES), dtype=np.float32)
    if 0 <= int(session_code) < len(one_hot):
        one_hot[int(session_code)] = 1.0
    return one_hot


def summarize_microstructure_window(store: TickProxyStore, pair_name: str, anchor_timestamp: object) -> np.ndarray:
    window = store.get_pre_entry_window(pair_name, anchor_timestamp)
    if len(window) == 0:
        return np.zeros(len(MICROSTRUCTURE_FEATURE_NAMES), dtype=np.float32)
    return np.asarray(
        [
            float(window["tick_velocity"].mean()),
            float(window["tick_velocity"].std(ddof=0)),
            float(window["spread_z"].mean()),
            float(window["spread_z"].iloc[-1]),
            float(window["bid_ask_imbalance"].mean()),
            float(np.abs(window["bid_ask_imbalance"]).max()),
            float(window["price_velocity"].iloc[-1]),
        ],
        dtype=np.float32,
    )


def reference_price_from_tpo(close: float, atr: float, tpo_features: np.ndarray, reference_price_mode: str = "poc") -> float:
    if reference_price_mode != "poc" or len(tpo_features) == 0:
        return float(close)
    return float(close - float(tpo_features[0]) * max(float(atr), 1e-8))


def build_gate_state_vector(
    *,
    prob_buy: float,
    prob_entry: float | None,
    atr: float,
    volatility: float,
    spread: float,
    tick_count: float,
    session_code: int,
    tpo_features: np.ndarray,
    pair_name: str,
    anchor_timestamp: object,
    tick_store: TickProxyStore,
) -> np.ndarray:
    tpo = np.asarray(tpo_features, dtype=np.float32)
    if tpo.shape != (8,):
        raise ValueError(f"Expected 8 TPO features, got shape {tpo.shape}")
    micro = summarize_microstructure_window(tick_store, pair_name, anchor_timestamp)
    state = np.concatenate(
        [
            np.asarray(
                [
                    float(prob_buy),
                    float(0.5 if prob_entry is None else prob_entry),
                    float(atr),
                    float(volatility),
                    float(spread),
                    float(tick_count),
                ],
                dtype=np.float32,
            ),
            _session_one_hot(session_code),
            tpo,
            micro,
        ]
    )
    expected_dim = len(GATE_STATE_FEATURE_NAMES)
    if state.shape != (expected_dim,):
        raise ValueError(f"Expected gate state dim {expected_dim}, got {state.shape}")
    return state


# ---------------------------------------------------------------------------
# v5.2 gate state vector — VP + OF replacing TPO + microstructure
# ---------------------------------------------------------------------------

GATE_V2_STATE_FEATURE_NAMES = (
    "prob_buy",
    "prob_entry",
    "atr",
    "volatility",
    *tuple(f"session_{name}" for name in SESSION_NAMES),
    *VP_FEATURE_NAMES,
    *OF_FEATURE_NAMES,
)


def build_gate_state_vector_v2(
    *,
    prob_buy: float,
    prob_entry: float | None,
    atr: float,
    volatility: float,
    session_code: int,
    vp_features: np.ndarray,
    of_features: np.ndarray,
) -> np.ndarray:
    """Build v5.2 gate state vector with VP + OF features.

    Replaces v5.1's ``build_gate_state_vector`` which used TPO + microstructure.
    """
    vp = np.asarray(vp_features, dtype=np.float32)
    of = np.asarray(of_features, dtype=np.float32)
    if vp.shape != (len(VP_FEATURE_NAMES),):
        raise ValueError(f"Expected {len(VP_FEATURE_NAMES)} VP features, got shape {vp.shape}")
    if of.shape != (len(OF_FEATURE_NAMES),):
        raise ValueError(f"Expected {len(OF_FEATURE_NAMES)} OF features, got shape {of.shape}")

    state = np.concatenate(
        [
            np.asarray(
                [
                    float(prob_buy),
                    float(0.5 if prob_entry is None else prob_entry),
                    float(atr),
                    float(volatility),
                ],
                dtype=np.float32,
            ),
            _session_one_hot(session_code),
            vp,
            of,
        ]
    )
    expected_dim = len(GATE_V2_STATE_FEATURE_NAMES)
    if state.shape != (expected_dim,):
        raise ValueError(f"Expected gate v2 state dim {expected_dim}, got {state.shape}")
    return state
