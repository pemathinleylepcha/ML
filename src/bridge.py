"""
Algo C2 v2 — Cross-learning bridge
Bidirectional feature exchange between 24x7 BTC subnet and 24x5 FX subnet.

Implemented as appended feature vectors (not a separate neural network) so
CatBoost's anti-overfitting properties are fully preserved.

BTC -> FX (8 features): weekend regime carry, volatility percentile, risk
    sentiment, spectral gap delta, tick velocity z-score, spread z-score,
    H1 lifespan.

FX -> BTC (6 features): session phase, regime summary, DXY proxy, average
    spread z-score, JPY cross mean return, spectral gap.

Shared/bidirectional (9 features): Betti numbers, H1 lifespan, spectral gap,
    regime label (from full 43-node graph), prev_regime, bars_in_regime,
    session_transition, regime_ema.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── DXY weights (approximate ICE basket) ────────────────────────────────────
# EUR=57.6%, JPY=13.6%, GBP=11.9%, CAD=9.1%, SEK=4.2%, CHF=3.6%
# SEK not in universe, distribute its weight proportionally.
_DXY_WEIGHTS = {
    "EURUSD": -0.576,   # negative: EUR/USD up → DXY down
    "USDJPY":  0.136,
    "GBPUSD": -0.119,
    "USDCAD":  0.091,
    "USDCHF":  0.036,
    "AUDUSD": -0.042,   # substitute for SEK (not in universe)
    "NZDUSD": -0.032,
}

# FX session phase encoding
SESSION_PHASES = {
    "pre_tokyo": 0,
    "tokyo":     1,
    "london":    2,
    "ny":        3,
    "sydney":    4,
    "closed":    5,
}

# JPY crosses for risk proxy
_JPY_CROSSES = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]

EPS = 1e-10


# ── Rolling statistics helper ────────────────────────────────────────────────

class _RollingStats:
    """Maintains rolling mean and std over a fixed window."""

    def __init__(self, window: int = 60):
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> tuple[float, float]:
        """Push a value, return (mean, std) of current window."""
        self._buf.append(value)
        arr = np.array(self._buf)
        return float(arr.mean()), float(arr.std() + EPS)

    def z_score(self, value: float) -> float:
        """Return z-score of value relative to current window, then push."""
        arr = np.array(self._buf) if self._buf else np.array([value])
        mean = arr.mean()
        std = arr.std() + EPS
        z = (value - mean) / std
        self._buf.append(value)
        return float(z)


# ── Weekend regime cache ─────────────────────────────────────────────────────

@dataclass
class WeekendRegimeCache:
    """
    Stores BTC regime computed at Sunday 23:59 UTC.
    Frozen until the following Monday — provides 'weekend carry' to FX model.
    """
    regime_at_close: str = "NORMAL"          # regime at Sunday 23:59
    regime_at_open: str = "NORMAL"           # regime at Saturday 00:00
    regime_shift: int = 0                    # 1=deteriorated, -1=improved, 0=stable
    spectral_gap_at_close: float = 0.0
    last_update_date: str = ""               # ISO date string of last Sunday

    def update(self, dt_str: str, regime: str, spectral_gap: float):
        """Call each bar with current datetime and BTC regime."""
        if len(dt_str) < 16:
            return
        # Parse weekday: Monday=0, Sunday=6
        try:
            from datetime import datetime
            dt = datetime.strptime(dt_str[:16], "%Y-%m-%d %H:%M")
            weekday = dt.weekday()
            hour = dt.hour
            minute = dt.minute
        except ValueError:
            return

        # Capture Saturday open (for shift calculation)
        if weekday == 5 and hour == 0 and minute < 5:
            self.regime_at_open = regime

        # Capture Sunday close
        if weekday == 6 and hour == 23 and minute >= 55:
            date_str = dt_str[:10]
            if date_str != self.last_update_date:
                self.regime_at_close = regime
                self.spectral_gap_at_close = spectral_gap
                self.last_update_date = date_str
                # Compute shift: did weekend make things worse or better?
                _order = ["LOW_VOL", "NORMAL", "TRANSITIONAL", "HIGH_STRESS", "FRAGMENTED"]
                try:
                    idx_open = _order.index(self.regime_at_open)
                    idx_close = _order.index(self.regime_at_close)
                    self.regime_shift = idx_close - idx_open  # >0 = deteriorated
                except ValueError:
                    self.regime_shift = 0

    def to_features(self) -> dict:
        """Return weekend carry features (frozen until next Monday update)."""
        _order = ["LOW_VOL", "NORMAL", "TRANSITIONAL", "HIGH_STRESS", "FRAGMENTED"]
        regime_idx = _order.index(self.regime_at_close) if self.regime_at_close in _order else 2
        return {
            "btc_weekend_regime": float(regime_idx),
            "btc_weekend_regime_shift": float(self.regime_shift),
        }


# ── Bridge state (updated every bar) ────────────────────────────────────────

@dataclass
class BridgeState:
    """
    Current cross-learning features from both subnets.
    Updated every bar by BridgeComputer.update().
    """
    # BTC -> FX (8 features + gate = 9 total)
    btc_weekend_regime: float = 2.0         # 0=LOW_VOL..4=FRAGMENTED
    btc_weekend_regime_shift: float = 0.0
    btc_vol_percentile: float = 0.5
    btc_risk_sentiment: float = 0.0         # 4h log return
    btc_spectral_gap_delta: float = 0.0     # Δλ₂ over last 1h
    btc_tick_velocity_z: float = 0.0
    btc_spread_z: float = 0.0
    btc_h1_lifespan: float = 0.0
    btc_bridge_gate: float = 1.0            # conditional gate ∈ [0,1]

    # FX -> BTC (6 features)
    fx_session_phase: float = 5.0           # encoded: 0-5
    fx_regime_summary: float = 0.0         # mean log return across majors
    fx_dxy_proxy: float = 0.0
    fx_avg_spread_z: float = 0.0
    fx_jpy_cross_mean_ret: float = 0.0
    fx_spectral_gap: float = 0.0

    # Shared / bidirectional (9 features)
    graph_betti_0: float = 1.0
    graph_betti_1: float = 0.0
    graph_max_h1_life: float = 0.0
    graph_spectral_gap: float = 0.0
    graph_regime: float = 2.0               # encoded: 0=LOW_VOL..4=FRAGMENTED
    # Regime memory features (GPT session-regime fixes)
    prev_regime: float = 2.0               # regime at previous bar
    bars_in_regime: float = 0.0            # bars elapsed in current regime (log-scaled)
    session_transition: float = 0.0        # 1.0 when session changed this bar, else 0
    regime_ema: float = 2.0               # EWMA of regime (α=0.1), soft persistence

    def btc_to_fx_features(self) -> dict:
        """Return the 9 BTC->FX bridge features (8 gated values + gate itself).

        Each of the 8 BTC features is multiplied by btc_bridge_gate so the FX
        model sees zeroed-out BTC context when the gate is closed (closed session
        or fragmented BTC regime).  The gate value is also passed explicitly so
        the model can distinguish gate=0 (no signal) from btc_feature=0 (neutral).
        """
        g = self.btc_bridge_gate
        return {
            "btc_weekend_regime":       self.btc_weekend_regime * g,
            "btc_weekend_regime_shift": self.btc_weekend_regime_shift * g,
            "btc_vol_percentile":       self.btc_vol_percentile * g,
            "btc_risk_sentiment":       self.btc_risk_sentiment * g,
            "btc_spectral_gap_delta":   self.btc_spectral_gap_delta * g,
            "btc_tick_velocity_z":      self.btc_tick_velocity_z * g,
            "btc_spread_z":             self.btc_spread_z * g,
            "btc_h1_lifespan":          self.btc_h1_lifespan * g,
            "btc_bridge_gate":          g,
        }

    def fx_to_btc_features(self) -> dict:
        """Return the 6 FX->BTC bridge features."""
        return {
            "fx_session_phase":      self.fx_session_phase,
            "fx_regime_summary":     self.fx_regime_summary,
            "fx_dxy_proxy":          self.fx_dxy_proxy,
            "fx_avg_spread_z":       self.fx_avg_spread_z,
            "fx_jpy_cross_mean_ret": self.fx_jpy_cross_mean_ret,
            "fx_spectral_gap":       self.fx_spectral_gap,
        }

    def shared_features(self) -> dict:
        """Return the 9 shared graph + regime-memory features."""
        return {
            "graph_betti_0":       self.graph_betti_0,
            "graph_betti_1":       self.graph_betti_1,
            "graph_max_h1_life":   self.graph_max_h1_life,
            "graph_spectral_gap":  self.graph_spectral_gap,
            "graph_regime":        self.graph_regime,
            "prev_regime":         self.prev_regime,
            "bars_in_regime":      self.bars_in_regime,
            "session_transition":  self.session_transition,
            "regime_ema":          self.regime_ema,
        }


# ── Bridge computer ──────────────────────────────────────────────────────────

class BridgeComputer:
    """
    Maintains stateful cross-learning bridge between BTC and FX subnets.

    Call update() each bar with current OHLC snapshots and graph state.
    Access bridge.state for the latest BridgeState.
    """

    # ATR window for volatility percentile
    _ATR_HISTORY_BARS = 30 * 1440   # 30 days at M1

    def __init__(self):
        self.state = BridgeState()
        self._weekend_cache = WeekendRegimeCache()

        # Rolling z-score trackers (60-bar = 1h at M1)
        self._btc_tick_vel_stats = _RollingStats(window=60)
        self._btc_spread_stats = _RollingStats(window=60)
        self._fx_spread_stats = _RollingStats(window=60)

        # BTC ATR history for percentile computation
        self._btc_atr_history: deque[float] = deque(maxlen=self._ATR_HISTORY_BARS)

        # Spectral gap history for delta computation (60 bars)
        self._spectral_gap_history: deque[float] = deque(maxlen=60)

        # Regime encoding
        self._regime_order = ["LOW_VOL", "NORMAL", "TRANSITIONAL", "HIGH_STRESS", "FRAGMENTED"]

        # Regime memory state
        self._prev_regime: float = 2.0      # NORMAL default
        self._prev_session: float = 5.0     # closed default
        self._bars_in_regime: int = 0       # bars elapsed in current regime
        self._regime_ema: float = 2.0       # EWMA of regime (α=0.1)

    @staticmethod
    def _compute_bridge_gate(session_phase: float, graph_regime: float,
                              is_fx_weekend: bool = False) -> float:
        """
        Conditional BTC->FX bridge gate ∈ [0, 1].

        Session gate: scales by FX session activity.
          - weekend: 0.0 — FX closed, BTC context irrelevant
          - closed (5): 0.0 — FX session closed
          - sydney (4): 0.4 — low BTC/FX overlap
          - pre_tokyo (0): 0.5
          - tokyo  (1): 0.6 — moderate overlap
          - london (2): 0.8 — high overlap
          - ny     (3): 1.0 — peak BTC/FX correlation (US equity hours)

        Regime gate: BTC noisy in extreme regimes.
          - FRAGMENTED (4): 0.3 — BTC graph disconnected, signal unreliable
          - HIGH_STRESS (3): 0.7 — elevated but still informative
          - otherwise: 1.0
        """
        if is_fx_weekend:
            return 0.0

        # Session gate
        _session_gate = {0.0: 0.5, 1.0: 0.6, 2.0: 0.8, 3.0: 1.0, 4.0: 0.4, 5.0: 0.0}
        s_gate = _session_gate.get(float(int(session_phase)), 0.5)

        # Regime gate
        if graph_regime >= 4.0:
            r_gate = 0.3
        elif graph_regime >= 3.0:
            r_gate = 0.7
        else:
            r_gate = 1.0

        return float(s_gate * r_gate)

    def _encode_regime(self, regime: str) -> float:
        try:
            return float(self._regime_order.index(regime))
        except ValueError:
            return 2.0  # NORMAL as default

    def _get_session_phase(self, hour: int) -> float:
        """Encode FX session phase from UTC hour."""
        if 0 <= hour < 2:
            return float(SESSION_PHASES["sydney"])
        elif 2 <= hour < 9:
            return float(SESSION_PHASES["tokyo"])
        elif 7 <= hour < 9:
            return float(SESSION_PHASES["tokyo"])   # Tokyo/London overlap
        elif 9 <= hour < 13:
            return float(SESSION_PHASES["london"])
        elif 13 <= hour < 17:
            return float(SESSION_PHASES["ny"])      # London/NY overlap
        elif 17 <= hour < 22:
            return float(SESSION_PHASES["ny"])
        elif 22 <= hour < 24:
            return float(SESSION_PHASES["pre_tokyo"])
        return float(SESSION_PHASES["closed"])

    def update(self,
               dt_str: str,
               btc_close: float,
               btc_tick_vel: float,
               btc_spread: float,
               btc_atr: float,
               btc_h1_lifespan: float,
               btc_close_history: np.ndarray,   # last 240 bars (4h at M1)
               math_state,                       # MathState from 43-node engine
               fx_closes: dict[str, float],      # {pair: close_price} for key FX
               fx_log_rets: dict[str, float],    # {pair: log_return}
               fx_spreads: dict[str, float],     # {pair: spread_in_price}
               fx_spectral_gap: float = 0.0) -> BridgeState:
        """
        Update bridge state for the current bar.

        Args:
            dt_str: datetime string "YYYY-MM-DD HH:MM"
            btc_close: BTCUSD close price
            btc_tick_vel: tick count this bar (proxy for velocity)
            btc_spread: BTCUSD spread in price
            btc_atr: BTCUSD ATR(14)
            btc_h1_lifespan: H1 max lifespan from TDA
            btc_close_history: last 240+ close prices for 4h return
            math_state: MathState from the full 43-node engine
            fx_closes: {pair: close} for DXY proxy computation
            fx_log_rets: {pair: log_return} for session summary
            fx_spreads: {pair: spread} for avg spread z-score
            fx_spectral_gap: λ₂ from FX-only subgraph (or full graph)

        Returns:
            Updated BridgeState (also stored in self.state)
        """
        # ── Weekend regime cache (BTC -> FX) ──────────────────────────────
        if math_state is not None and math_state.valid:
            self._weekend_cache.update(dt_str, math_state.regime, math_state.spectral_gap)
        weekend_feats = self._weekend_cache.to_features()

        # ── BTC volatility percentile ──────────────────────────────────────
        if btc_atr > 0:
            self._btc_atr_history.append(btc_atr)
        if len(self._btc_atr_history) > 1:
            arr = np.array(self._btc_atr_history)
            vol_pct = float(np.searchsorted(np.sort(arr), btc_atr) / len(arr))
        else:
            vol_pct = 0.5

        # ── BTC 4h risk sentiment (log return over trailing 240 M1 bars) ──
        if len(btc_close_history) >= 2:
            anchor = btc_close_history[max(0, len(btc_close_history) - 241)]
            risk_sentiment = float(np.log(btc_close / anchor)) if anchor > EPS else 0.0
        else:
            risk_sentiment = 0.0

        # ── BTC spectral gap delta (change over last 60 bars = 1h) ─────────
        if math_state is not None and math_state.valid:
            self._spectral_gap_history.append(math_state.spectral_gap)
        sg_now = math_state.spectral_gap if (math_state and math_state.valid) else 0.0
        sg_delta = 0.0
        if len(self._spectral_gap_history) >= 60:
            sg_old = list(self._spectral_gap_history)[0]
            sg_delta = float(sg_now - sg_old)

        # ── BTC tick velocity z-score ──────────────────────────────────────
        tick_vel_z = self._btc_tick_vel_stats.z_score(float(btc_tick_vel))

        # ── BTC spread z-score ─────────────────────────────────────────────
        spread_z = self._btc_spread_stats.z_score(float(btc_spread))

        # ── FX session phase (UTC hour) + weekend check ───────────────────
        try:
            hour = int(dt_str[11:13]) if len(dt_str) >= 13 else 12
        except (ValueError, IndexError):
            hour = 12
        session_phase = self._get_session_phase(hour)
        # Weekday check for bridge gate: Saturday=5, Sunday=6
        is_fx_weekend = False
        try:
            from datetime import datetime
            _dt = datetime.strptime(dt_str[:16], "%Y-%m-%d %H:%M")
            is_fx_weekend = _dt.weekday() >= 5
        except (ValueError, ImportError):
            pass

        # ── FX regime summary (mean log return across majors) ─────────────
        majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]
        major_rets = [fx_log_rets.get(p, 0.0) for p in majors if p in fx_log_rets]
        fx_regime_summary = float(np.mean(major_rets)) if major_rets else 0.0

        # ── DXY proxy (weighted USD composite) ────────────────────────────
        dxy = 0.0
        total_w = 0.0
        for pair, w in _DXY_WEIGHTS.items():
            if pair in fx_log_rets:
                dxy += w * fx_log_rets[pair]
                total_w += abs(w)
        fx_dxy_proxy = float(dxy / total_w) if total_w > EPS else 0.0

        # ── FX average spread z-score ──────────────────────────────────────
        avg_spread = np.mean(list(fx_spreads.values())) if fx_spreads else 0.0
        fx_spread_z = self._fx_spread_stats.z_score(float(avg_spread))

        # ── JPY cross mean return (risk proxy) ─────────────────────────────
        jpy_rets = [fx_log_rets.get(p, 0.0) for p in _JPY_CROSSES if p in fx_log_rets]
        jpy_mean_ret = float(np.mean(jpy_rets)) if jpy_rets else 0.0

        # ── Shared graph features ──────────────────────────────────────────
        if math_state is not None and math_state.valid:
            b0 = float(math_state.beta_0)
            b1 = float(math_state.beta_1)
            h1_life = float(math_state.h1_lifespan)
            sg = float(math_state.spectral_gap)
            regime_enc = self._encode_regime(math_state.regime)
        else:
            b0, b1, h1_life, sg, regime_enc = 1.0, 0.0, 0.0, 0.0, 2.0

        # ── Conditional bridge gate (critics.md item 4) ───────────────────
        bridge_gate = self._compute_bridge_gate(session_phase, regime_enc, is_fx_weekend)

        # ── Regime memory ──────────────────────────────────────────────────
        # Track how long we've been in the current regime
        if regime_enc == self._prev_regime:
            self._bars_in_regime += 1
        else:
            self._bars_in_regime = 0
        # Log-scale: regime of 1 bar vs 100 bars is not linear in importance
        bars_in_regime_log = float(math.log1p(self._bars_in_regime))

        # EWMA smoothed regime (α=0.1) — prevents instant regime flip features
        _REGIME_ALPHA = 0.1
        self._regime_ema = _REGIME_ALPHA * regime_enc + (1.0 - _REGIME_ALPHA) * self._regime_ema

        # Session transition flag: 1.0 on the first bar of a new session
        session_transition = 1.0 if session_phase != self._prev_session else 0.0

        # Snapshot prev values before updating
        prev_regime_snap = self._prev_regime
        self._prev_regime = regime_enc
        self._prev_session = session_phase

        # ── Assemble state ─────────────────────────────────────────────────
        self.state = BridgeState(
            btc_weekend_regime=weekend_feats["btc_weekend_regime"],
            btc_weekend_regime_shift=weekend_feats["btc_weekend_regime_shift"],
            btc_vol_percentile=vol_pct,
            btc_risk_sentiment=risk_sentiment,
            btc_spectral_gap_delta=sg_delta,
            btc_tick_velocity_z=tick_vel_z,
            btc_spread_z=spread_z,
            btc_h1_lifespan=btc_h1_lifespan,
            btc_bridge_gate=bridge_gate,
            fx_session_phase=session_phase,
            fx_regime_summary=fx_regime_summary,
            fx_dxy_proxy=fx_dxy_proxy,
            fx_avg_spread_z=fx_spread_z,
            fx_jpy_cross_mean_ret=jpy_mean_ret,
            fx_spectral_gap=fx_spectral_gap,
            graph_betti_0=b0,
            graph_betti_1=b1,
            graph_max_h1_life=h1_life,
            graph_spectral_gap=sg,
            graph_regime=regime_enc,
            prev_regime=prev_regime_snap,
            bars_in_regime=bars_in_regime_log,
            session_transition=session_transition,
            regime_ema=self._regime_ema,
        )
        return self.state

    def get_btc_feature_names(self) -> list[str]:
        """All feature names for the BTC subnet (FX->BTC + shared)."""
        return list(self.state.fx_to_btc_features().keys()) + list(self.state.shared_features().keys())

    def get_fx_feature_names(self) -> list[str]:
        """All feature names for the FX subnet (BTC->FX + shared)."""
        return list(self.state.btc_to_fx_features().keys()) + list(self.state.shared_features().keys())


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Bridge smoke test...")
    bc = BridgeComputer()

    rng = np.random.default_rng(42)
    btc_prices = 50000.0 + np.cumsum(rng.normal(0, 100, 300))
    fx_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]

    for t in range(300):
        dt = f"2026-03-{(t // 1440) + 2:02d} {(t % 1440) // 60:02d}:{t % 60:02d}"
        fx_log_rets = {p: float(rng.normal(0, 0.0001)) for p in fx_pairs}
        fx_spreads = {p: float(rng.uniform(0.00005, 0.0002)) for p in fx_pairs}
        fx_closes = {p: 1.1 + rng.random() * 0.2 for p in fx_pairs}
        state = bc.update(
            dt_str=dt,
            btc_close=btc_prices[t],
            btc_tick_vel=float(rng.integers(5, 50)),
            btc_spread=float(rng.uniform(1, 10)),
            btc_atr=500.0 + rng.random() * 200,
            btc_h1_lifespan=float(rng.random() * 0.5),
            btc_close_history=btc_prices[:t + 1],
            math_state=None,
            fx_closes=fx_closes,
            fx_log_rets=fx_log_rets,
            fx_spreads=fx_spreads,
        )

    print(f"BTC->FX features ({len(state.btc_to_fx_features())}): {list(state.btc_to_fx_features().keys())}")
    print(f"FX->BTC features ({len(state.fx_to_btc_features())}): {list(state.fx_to_btc_features().keys())}")
    print(f"Shared features  ({len(state.shared_features())}): {list(state.shared_features().keys())}")
    print(f"Session phase: {state.fx_session_phase}, DXY proxy: {state.fx_dxy_proxy:.6f}")
    print(f"BTC vol pct: {state.btc_vol_percentile:.3f}, risk sentiment: {state.btc_risk_sentiment:.6f}")
    print("OK")
