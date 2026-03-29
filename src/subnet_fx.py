"""
Algo C2 v2 — 24x5 FX Subnet Feature Pipeline

Handles all 28 tradeable FX pairs plus 14 signal-only nodes.
Session-gated: Monday 00:00 UTC – Friday 23:59 UTC.
Produces the feature vector for CatBoost-FX.

Feature vector per tradeable pair (per bar):
  - Per-pair technical indicators (16): from compute_pair_features()
  - Per-pair tick microstructure (4): vel_z, spread_z, OFI proxy, price_vel
  - Per-pair Laplacian residual (1): from 43-node graph
  - Residual persistence score (1): consecutive bars in same residual direction
  - Signal-only cross-asset features (14 × 7 = 98): from compute_signal_only_features()
  - TDA + spectral gap (4): β₀, β₁, H₁ lifespan, λ₂
  - Regime label (1): encoded 0-4
  - BTC->FX bridge features (8): from bridge.py
  - Shared graph features (5): from bridge.py
  Total per pair: 16 + 4 + 1 + 1 + 98 + 4 + 1 + 8 + 5 = 138 features
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from feature_engine import (
    compute_pair_features, compute_signal_only_features,
    PAIRS_ALL_V2, EPS,
)
from bridge import BridgeState

try:
    from universe import (
        TRADEABLE, SIGNAL_ONLY, NODE_IDX,
        REGIONAL_CORRELATION_MAP, SESSION_TOKYO, SESSION_LONDON, SESSION_NY,
    )
    _HAS_UNIVERSE = True
except ImportError:
    _HAS_UNIVERSE = False
    TRADEABLE = []
    SIGNAL_ONLY = []
    NODE_IDX = {}

_REGIME_ORDER = ["LOW_VOL", "NORMAL", "TRANSITIONAL", "HIGH_STRESS", "FRAGMENTED"]
_MICRO_WINDOW = 60


@dataclass
class FXBarFeatures:
    """Feature vector for one tradeable FX pair at one bar."""
    pair: str = ""
    timestamp: str = ""
    features: dict = field(default_factory=dict)
    valid: bool = False
    direction_hint: int = 0   # +1 long, -1 short, 0 flat (from Laplacian residual)

    def to_array(self, feature_names: list[str]) -> np.ndarray:
        return np.array([self.features.get(k, 0.0) for k in feature_names], dtype=np.float32)


class FXSubnet:
    """
    24x5 FX subnet feature pipeline.

    Maintains per-pair rolling state and builds per-pair feature vectors
    for CatBoost-FX consumption. Signal-only nodes contribute 98 cross-asset
    features that are appended identically to every tradeable pair's vector.
    """

    def __init__(self, tradeable: list[str] = None, signal_only: list[str] = None):
        self.tradeable = tradeable or (TRADEABLE[1:] if _HAS_UNIVERSE else [])   # exclude BTCUSD
        self.signal_only = signal_only or (SIGNAL_ONLY if _HAS_UNIVERSE else [])

        # Per-pair rolling buffers for microstructure z-scores
        self._vel_bufs: dict[str, deque] = {p: deque(maxlen=_MICRO_WINDOW) for p in self.tradeable}
        self._spread_bufs: dict[str, deque] = {p: deque(maxlen=_MICRO_WINDOW) for p in self.tradeable}

        # Per-pair residual persistence tracking
        self._residual_streak: dict[str, int] = {p: 0 for p in self.tradeable}
        self._residual_sign: dict[str, int] = {p: 0 for p in self.tradeable}

        # Feature name registry (populated on first compute_all() call)
        self._feature_names: list[str] | None = None
        self._n_features: int = 0

    def _z_score(self, value: float, buf: deque) -> float:
        arr = np.array(buf) if buf else np.array([value])
        mean = float(arr.mean())
        std = float(arr.std()) + EPS
        z = (value - mean) / std
        buf.append(value)
        return z

    def _update_residual_streak(self, pair: str, residual: float,
                                dead_zone: float = 1e-4) -> int:
        """Track consecutive bars where residual has same sign. Returns current streak."""
        if abs(residual) < dead_zone:
            self._residual_streak[pair] = 0
            self._residual_sign[pair] = 0
            return 0

        sign = 1 if residual > 0 else -1
        if sign == self._residual_sign[pair]:
            self._residual_streak[pair] += 1
        else:
            self._residual_streak[pair] = 1
            self._residual_sign[pair] = sign
        return self._residual_streak[pair]

    def _compute_signal_only_block(self,
                                   ohlc_all: dict[str, dict],
                                   bar_idx: int,
                                   math_state,
                                   lookback: int) -> dict:
        """
        Compute all signal-only cross-asset features (14 × 7 = 98 features).
        This block is identical for every tradeable pair in the same bar.
        """
        block: dict = {}
        for inst in self.signal_only:
            if inst not in ohlc_all:
                # Instrument data not yet available — fill with zeros
                block.update({
                    f"{inst}_log_ret": 0.0,
                    f"{inst}_rsi_14": 50.0,
                    f"{inst}_macd_hist": 0.0,
                    f"{inst}_bb_bandwidth": 0.0,
                    f"{inst}_atr_14": 0.0,
                    f"{inst}_cci_20": 0.0,
                    f"{inst}_laplacian_residual": 0.0,
                })
                continue

            # Get Laplacian residual for this signal-only node
            lap_res = 0.0
            if math_state is not None and math_state.valid and _HAS_UNIVERSE:
                idx = NODE_IDX.get(inst, -1)
                if 0 <= idx < len(math_state.residuals):
                    lap_res = float(math_state.residuals[idx])

            feats = compute_signal_only_features(
                ohlc_all[inst], inst, bar_idx,
                lookback=lookback, laplacian_residual=lap_res,
            )
            # Replace NaN with neutral defaults
            for k, v in feats.items():
                if isinstance(v, float) and np.isnan(v):
                    feats[k] = 0.0
            block.update(feats)
        return block

    def compute_all(self,
                    ohlc_all: dict[str, dict],
                    bar_idx: int,
                    math_state,
                    bridge_state: BridgeState,
                    lookback: int = 120) -> dict[str, FXBarFeatures]:
        """
        Compute feature vectors for all tradeable FX pairs in a single bar.

        The signal-only block and bridge/graph features are computed once
        and appended to every pair's vector.

        Args:
            ohlc_all: {instrument: ohlc_dict} for all 43 instruments
            bar_idx: current bar index (same across all instruments after alignment)
            math_state: MathState from the full 43-node engine
            bridge_state: current BridgeState from BridgeComputer
            lookback: indicator lookback window

        Returns:
            {pair: FXBarFeatures} for all tradeable FX pairs
        """
        # ── Shared: signal-only block (computed once, appended to all pairs) ──
        signal_block = self._compute_signal_only_block(ohlc_all, bar_idx, math_state, lookback)

        # ── Shared: TDA + spectral ─────────────────────────────────────────
        if math_state is not None and math_state.valid:
            b0 = float(math_state.beta_0)
            b1 = float(math_state.beta_1)
            h1 = float(math_state.h1_lifespan)
            sg = float(math_state.spectral_gap)
            regime_str = math_state.regime
        else:
            b0, b1, h1, sg = 1.0, 0.0, 0.0, 0.0
            regime_str = "NORMAL"
        regime_enc = float(_REGIME_ORDER.index(regime_str)) if regime_str in _REGIME_ORDER else 2.0

        # ── Shared: bridge features ────────────────────────────────────────
        btc_to_fx = bridge_state.btc_to_fx_features()
        shared_graph = bridge_state.shared_features()

        # ── Per-pair features ─────────────────────────────────────────────
        results: dict[str, FXBarFeatures] = {}

        for pair in self.tradeable:
            if pair not in ohlc_all:
                results[pair] = FXBarFeatures(pair=pair, valid=False)
                continue

            ohlc = ohlc_all[pair]
            if bar_idx >= len(ohlc["c"]):
                results[pair] = FXBarFeatures(pair=pair, valid=False)
                continue

            dt = ohlc["dt"][bar_idx] if "dt" in ohlc else ""

            # Technical indicators (16)
            tech = compute_pair_features(ohlc, pair, bar_idx, lookback=lookback)
            for k, v in tech.items():
                if isinstance(v, float) and np.isnan(v):
                    tech[k] = 0.0

            # Tick microstructure (4)
            tk = float(ohlc["tk"][bar_idx]) if "tk" in ohlc else 0.0
            sp = float(ohlc["sp"][bar_idx]) if "sp" in ohlc else 0.0
            vel_z = self._z_score(tk, self._vel_bufs[pair])
            spread_z = self._z_score(sp, self._spread_bufs[pair])
            log_ret = tech.get(f"{pair}_log_ret", 0.0)
            ofi_proxy = float(tk * np.sign(log_ret)) if log_ret != 0 else 0.0
            prev_c = ohlc["c"][bar_idx - 1] if bar_idx > 0 else ohlc["c"][bar_idx]
            price_vel = float((ohlc["c"][bar_idx] - prev_c) / (prev_c + EPS))

            # Laplacian residual (1) + persistence (1)
            if math_state is not None and math_state.valid and _HAS_UNIVERSE:
                pair_idx = NODE_IDX.get(pair, -1)
                lap_res = float(math_state.residuals[pair_idx]) if 0 <= pair_idx < len(math_state.residuals) else 0.0
            else:
                lap_res = 0.0
            streak = self._update_residual_streak(pair, lap_res)
            direction_hint = self._residual_sign.get(pair, 0)

            # Assemble full feature vector
            features: dict = {}
            features.update(tech)                        # 16 per-pair technical
            features[f"{pair}_tick_vel_z"]   = vel_z
            features[f"{pair}_spread_z"]     = spread_z
            features[f"{pair}_ofi_proxy"]    = ofi_proxy
            features[f"{pair}_price_vel"]    = price_vel
            features[f"{pair}_lap_residual"] = lap_res
            features[f"{pair}_res_streak"]   = float(streak)
            features.update(signal_block)               # 98 signal-only features
            features["tda_beta_0"]    = b0
            features["tda_beta_1"]    = b1
            features["tda_h1_life"]   = h1
            features["tda_spec_gap"]  = sg
            features["tda_regime"]    = regime_enc
            features.update(btc_to_fx)                  # 8 BTC->FX bridge
            features.update(shared_graph)               # 5 shared graph

            if self._feature_names is None:
                self._feature_names = list(features.keys())
                self._n_features = len(self._feature_names)

            results[pair] = FXBarFeatures(
                pair=pair,
                timestamp=dt,
                features=features,
                valid=True,
                direction_hint=direction_hint,
            )

        return results

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Call compute_all() at least once before accessing feature_names")
        return self._feature_names

    @property
    def n_features(self) -> int:
        return self._n_features

    def get_tradeable_pairs(self) -> list[str]:
        return list(self.tradeable)

    def is_session_active(self, pair: str, hour: int) -> bool:
        """Returns True if the pair is typically active in the given UTC hour."""
        if not _HAS_UNIVERSE:
            return True
        if 2 <= hour < 9 and pair in SESSION_TOKYO:
            return True
        if 8 <= hour < 17 and pair in SESSION_LONDON:
            return True
        if 13 <= hour < 22 and pair in SESSION_NY:
            return True
        return False


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from math_engine import MathEngine
    from bridge import BridgeComputer, BridgeState

    print("FXSubnet smoke test...")
    rng = np.random.default_rng(42)
    n = 300

    try:
        from universe import TRADEABLE, SIGNAL_ONLY
        tradeable_fx = [p for p in TRADEABLE if p != "BTCUSD"]
        signal_only = SIGNAL_ONLY
    except ImportError:
        tradeable_fx = ["EURUSD", "GBPUSD", "USDJPY"]
        signal_only = ["XAUUSD", "US30"]

    # Synthetic data for all instruments
    all_inst = tradeable_fx + signal_only
    ohlc_all: dict = {}
    for inst in all_inst:
        base = 1.1 + rng.random() * 100
        c = base + np.cumsum(rng.normal(0, base * 0.001, n))
        ohlc_all[inst] = {
            "o":  c + rng.normal(0, base * 0.0003, n),
            "h":  c + np.abs(rng.normal(0, base * 0.0005, n)),
            "l":  c - np.abs(rng.normal(0, base * 0.0005, n)),
            "c":  c,
            "sp": np.abs(rng.normal(0.0002, 0.0001, n)),
            "tk": rng.integers(5, 100, n).astype(float),
            "dt": [f"2026-03-02 {i//60:02d}:{i%60:02d}" for i in range(n)],
        }

    n_nodes = len(all_inst)
    engine = MathEngine(n_pairs=n_nodes)
    bridge = BridgeComputer()
    subnet = FXSubnet(tradeable=tradeable_fx[:3], signal_only=signal_only[:2])

    for t in range(n):
        returns = np.zeros(n_nodes)
        for i, inst in enumerate(all_inst):
            c = ohlc_all[inst]["c"]
            if t > 0 and c[t - 1] > EPS:
                returns[i] = np.log(c[t] / c[t - 1])
        ms = engine.update(returns)
        bs = bridge.update(
            dt_str=f"2026-03-02 {t//60:02d}:{t%60:02d}",
            btc_close=50000.0,
            btc_tick_vel=10.0, btc_spread=5.0, btc_atr=500.0, btc_h1_lifespan=0.1,
            btc_close_history=np.array([50000.0]),
            math_state=ms,
            fx_closes={p: ohlc_all[p]["c"][t] for p in tradeable_fx[:3]},
            fx_log_rets={p: returns[i] for i, p in enumerate(all_inst[:3])},
            fx_spreads={p: float(ohlc_all[p]["sp"][t]) for p in tradeable_fx[:3]},
        )

    bar_feats = subnet.compute_all(ohlc_all, n - 1, ms, bs, lookback=60)

    valid_pairs = [p for p, f in bar_feats.items() if f.valid]
    print(f"Valid pairs: {len(valid_pairs)}/{len(tradeable_fx[:3])}")
    print(f"n_features per pair: {subnet.n_features}")
    print(f"Feature names (first 5): {subnet.feature_names[:5]}")
    sample_pair = valid_pairs[0]
    sf = bar_feats[sample_pair]
    print(f"Sample [{sample_pair}]: lap_residual={sf.features.get(f'{sample_pair}_lap_residual', 'N/A'):.6f}")
    print(f"  direction_hint={sf.direction_hint}, streak={sf.features.get(f'{sample_pair}_res_streak', 0):.0f}")
    print("OK")
