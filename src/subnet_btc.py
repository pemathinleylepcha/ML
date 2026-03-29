"""
Algo C2 v2 — 24x7 BTC Subnet Feature Pipeline

Handles BTCUSD continuous tick stream including Saturday/Sunday.
Produces the feature vector for CatBoost-BTC.

Feature vector composition (per bar):
  - BTCUSD technical indicators (16): from compute_pair_features()
  - BTCUSD tick microstructure (4): velocity z, spread z, OFI proxy, price velocity
  - Laplacian residual (1): BTCUSD node residual from 43-node graph
  - TDA + spectral gap (4): β₀, β₁, H₁ lifespan, λ₂
  - Regime label (1): encoded 0-4
  - FX->BTC bridge features (6): from bridge.py
  - Shared graph features (5): from bridge.py
  Total: 37 features per bar
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from feature_engine import compute_pair_features, EPS
from bridge import BridgeComputer, BridgeState

# ── Constants ────────────────────────────────────────────────────────────────

_REGIME_ORDER = ["LOW_VOL", "NORMAL", "TRANSITIONAL", "HIGH_STRESS", "FRAGMENTED"]
_MICRO_WINDOW = 60   # 1h at M1 for z-score windows


@dataclass
class BTCFeatures:
    """Feature vector for one BTC bar, ready for CatBoost-BTC input."""
    timestamp: str = ""
    features: dict = field(default_factory=dict)
    valid: bool = False

    def to_array(self, feature_names: list[str]) -> np.ndarray:
        """Convert to numpy array in canonical feature order."""
        return np.array([self.features.get(k, 0.0) for k in feature_names], dtype=np.float32)


class BTCSubnet:
    """
    24x7 BTC subnet feature pipeline.

    Maintains rolling state for microstructure z-scores and builds the
    full BTC feature vector each bar for CatBoost-BTC consumption.
    """

    def __init__(self):
        # Rolling z-score buffers (60 bars = 1h at M1)
        self._vel_buf: deque[float] = deque(maxlen=_MICRO_WINDOW)
        self._spread_buf: deque[float] = deque(maxlen=_MICRO_WINDOW)
        self._close_history: deque[float] = deque(maxlen=300)  # 5h for 4h return

        # Feature name registry (populated on first call)
        self._feature_names: list[str] | None = None

    def _z_score(self, value: float, buf: deque[float]) -> float:
        """Compute z-score of value relative to buffer, then push."""
        arr = np.array(buf) if buf else np.array([value])
        mean = float(arr.mean())
        std = float(arr.std()) + EPS
        z = (value - mean) / std
        buf.append(value)
        return z

    def compute(self,
                ohlc: dict,
                bar_idx: int,
                math_state,
                bridge_state: BridgeState,
                lookback: int = 120) -> BTCFeatures:
        """
        Compute the full BTC feature vector for a single bar.

        Args:
            ohlc: dict with 'o','h','l','c','sp','tk' arrays (BTCUSD only)
            bar_idx: current bar index
            math_state: MathState from the full 43-node engine (node 0 = BTCUSD)
            bridge_state: current BridgeState from BridgeComputer
            lookback: indicator lookback window

        Returns:
            BTCFeatures with 37-feature dict
        """
        c = ohlc["c"]
        if bar_idx < 1 or bar_idx >= len(c):
            return BTCFeatures(valid=False)

        dt = ohlc["dt"][bar_idx] if "dt" in ohlc else ""

        # ── Technical indicators (16) ──────────────────────────────────────
        tech = compute_pair_features(ohlc, "BTCUSD", bar_idx, lookback=lookback)

        # ── Tick microstructure (4) ────────────────────────────────────────
        tk = float(ohlc["tk"][bar_idx]) if "tk" in ohlc else 0.0
        sp = float(ohlc["sp"][bar_idx]) if "sp" in ohlc else 0.0

        tick_vel_z = self._z_score(tk, self._vel_buf)
        spread_z = self._z_score(sp, self._spread_buf)

        # OFI proxy: tick velocity × price direction
        log_ret = tech.get("BTCUSD_log_ret", 0.0)
        ofi_proxy = float(tk * np.sign(log_ret)) if log_ret != 0 else 0.0

        # Price velocity: fractional close-to-close
        prev_c = c[bar_idx - 1]
        price_vel = float((c[bar_idx] - prev_c) / (prev_c + EPS))

        # ── Laplacian residual (1) — BTCUSD is node 0 ─────────────────────
        if math_state is not None and math_state.valid and len(math_state.residuals) > 0:
            lap_residual = float(math_state.residuals[0])
        else:
            lap_residual = 0.0

        # ── TDA + spectral (4) ─────────────────────────────────────────────
        if math_state is not None and math_state.valid:
            b0 = float(math_state.beta_0)
            b1 = float(math_state.beta_1)
            h1 = float(math_state.h1_lifespan)
            sg = float(math_state.spectral_gap)
        else:
            b0, b1, h1, sg = 1.0, 0.0, 0.0, 0.0

        # ── Regime label (1) ──────────────────────────────────────────────
        regime_str = math_state.regime if (math_state and math_state.valid) else "NORMAL"
        regime_enc = float(_REGIME_ORDER.index(regime_str)) if regime_str in _REGIME_ORDER else 2.0

        # ── FX->BTC bridge features (6) ───────────────────────────────────
        fx_to_btc = bridge_state.fx_to_btc_features()

        # ── Shared graph features (5) ─────────────────────────────────────
        shared = bridge_state.shared_features()

        # ── Assemble ──────────────────────────────────────────────────────
        features: dict = {}
        features.update(tech)                    # 16 technical features
        features["BTCUSD_tick_vel_z"]   = tick_vel_z
        features["BTCUSD_spread_z"]     = spread_z
        features["BTCUSD_ofi_proxy"]    = ofi_proxy
        features["BTCUSD_price_vel"]    = price_vel
        features["BTCUSD_lap_residual"] = lap_residual
        features["BTCUSD_beta_0"]       = b0
        features["BTCUSD_beta_1"]       = b1
        features["BTCUSD_h1_lifespan"]  = h1
        features["BTCUSD_spectral_gap"] = sg
        features["BTCUSD_regime"]       = regime_enc
        features.update(fx_to_btc)               # 6 FX->BTC bridge
        features.update(shared)                  # 5 shared graph

        if self._feature_names is None:
            self._feature_names = list(features.keys())

        return BTCFeatures(timestamp=dt, features=features, valid=True)

    @property
    def feature_names(self) -> list[str]:
        """Returns canonical feature name list after first compute() call."""
        if self._feature_names is None:
            raise RuntimeError("Call compute() at least once before accessing feature_names")
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names) if self._feature_names else 0


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from math_engine import MathEngine
    from bridge import BridgeComputer

    print("BTCSubnet smoke test...")
    rng = np.random.default_rng(42)
    n = 300

    # Synthetic BTCUSD data
    base = 50000.0
    closes = base + np.cumsum(rng.normal(0, 100, n))
    ohlc = {
        "o":  closes + rng.normal(0, 50, n),
        "h":  closes + np.abs(rng.normal(0, 100, n)),
        "l":  closes - np.abs(rng.normal(0, 100, n)),
        "c":  closes,
        "sp": np.abs(rng.normal(5, 2, n)),
        "tk": rng.integers(5, 100, n).astype(float),
        "dt": [f"2026-03-02 {i//60:02d}:{i%60:02d}" for i in range(n)],
    }

    engine = MathEngine(n_pairs=43)
    bridge = BridgeComputer()
    subnet = BTCSubnet()

    # Warm up
    all_returns = np.zeros(43)
    for t in range(n):
        if t > 0 and closes[t - 1] > 0:
            all_returns[0] = np.log(closes[t] / closes[t - 1])
        ms = engine.update(all_returns.copy())
        bs = bridge.update(
            dt_str=ohlc["dt"][t],
            btc_close=closes[t],
            btc_tick_vel=ohlc["tk"][t],
            btc_spread=ohlc["sp"][t],
            btc_atr=500.0,
            btc_h1_lifespan=ms.h1_lifespan if ms.valid else 0.0,
            btc_close_history=closes[:t + 1],
            math_state=ms,
            fx_closes={},
            fx_log_rets={},
            fx_spreads={},
        )
        if t == n - 1:
            feat = subnet.compute(ohlc, t, ms, bs)

    print(f"Features: {feat.n_features if hasattr(feat, 'n_features') else len(feat.features)}")
    print(f"Feature names ({subnet.n_features}): {subnet.feature_names[:5]}...")
    print(f"Valid: {feat.valid}")
    print(f"Sample: tick_vel_z={feat.features.get('BTCUSD_tick_vel_z', 'N/A'):.4f}")
    print("OK")
