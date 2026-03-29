"""
live_engine.py -- Real-time M5 bar signal engine for Algo C2 v2

Connects trained CatBoost models to a live bar stream.
Call on_bar() each time an M5 bar closes; receive per-instrument signals.

Usage:
    engine = LiveEngine("D:/dataset-ml/models/catboost_v2_full3")
    signal = engine.on_bar("2026-03-27 10:00", {
        "BTCUSD": {"o": 87000, "h": 87200, "l": 86900, "c": 87100, "sp": 25, "tk": 120},
        "EURUSD": {"o": 1.0800, "h": 1.0812, "l": 1.0795, "c": 1.0805, "sp": 0.8, "tk": 45},
        ...
    })
    for pair, sig in signal.items():
        print(pair, sig.direction, sig.size, sig.p_buy, sig.p_sell)
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("catboost not installed — run: pip install catboost")

# Local imports
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from universe import ALL_INSTRUMENTS, FX_PAIRS
from math_engine import MathEngine
from bridge import BridgeComputer
from subnet_btc import BTCSubnet
from subnet_fx import FXSubnet
from signal_filter import HysteresisFilter, PositionSizer

# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Per-instrument signal from one bar."""
    pair:      str
    direction: int    # +1=BUY, -1=SELL, 0=FLAT
    size:      float  # signed position size (base_risk units)
    p_buy:     float
    p_sell:    float
    p_hold:    float
    regime:    str    # current graph regime
    gate:      float  # BTC->FX bridge gate (0=closed, 1=fully open)


# ---------------------------------------------------------------------------
# Live engine
# ---------------------------------------------------------------------------

class LiveEngine:
    """
    Stateful M5 bar signal engine.

    Maintains rolling OHLCV buffers for all instruments, re-runs the full
    feature pipeline on each new bar, and applies per-pair hysteresis +
    position sizing before emitting signals.

    Warmup: first WARMUP_BARS bars are consumed silently (returns empty dict).
    """

    WARMUP_BARS = 200
    BUFFER_SIZE = 370   # WARMUP(200) + LOOKBACK(120) + slack(50)

    def __init__(
        self,
        model_dir: str,
        entry_threshold: float = 0.45,
        exit_threshold:  float = 0.35,
        base_risk: float = 1.0,
        lookback: int = 120,
    ):
        model_dir = Path(model_dir)

        # ── Load models ────────────────────────────────────────────────────
        btc_path  = model_dir / "catboost_btc_v2.cbm"
        fx_path   = model_dir / "catboost_fx_v2.cbm"
        meta_path = model_dir / "catboost_v2_feature_names.json"

        for p in [btc_path, fx_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing model file: {p}")

        self.btc_model = CatBoostClassifier()
        self.btc_model.load_model(str(btc_path))

        self.fx_model = CatBoostClassifier()
        self.fx_model.load_model(str(fx_path))

        with open(meta_path) as f:
            meta = json.load(f)

        # Feature names after training-time selection (may be subset of all features)
        self.btc_feat_names: list[str] = meta.get("btc", [])
        # FX names: last entry is "pair_id" — keep separate
        fx_meta_names: list[str] = meta.get("fx", [])
        self.fx_feat_names  = [n for n in fx_meta_names if n != "pair_id"]
        self.fx_pairs: list[str] = meta.get("fx_pairs", list(FX_PAIRS))
        self.lookback = meta.get("lookback", lookback)

        # ── Pipeline components ────────────────────────────────────────────
        self.inst_ordered = [i for i in ALL_INSTRUMENTS]
        self.n_inst = len(self.inst_ordered)
        self._inst_idx = {inst: i for i, inst in enumerate(self.inst_ordered)}

        self.math_engine  = MathEngine(n_pairs=self.n_inst)
        self.bridge       = BridgeComputer()
        self.btc_subnet   = BTCSubnet()
        self.fx_subnet    = FXSubnet()

        # ── Rolling OHLCV buffers ──────────────────────────────────────────
        self.buffers: dict[str, dict[str, deque]] = {
            inst: {
                "o":  deque(maxlen=self.BUFFER_SIZE),
                "h":  deque(maxlen=self.BUFFER_SIZE),
                "l":  deque(maxlen=self.BUFFER_SIZE),
                "c":  deque(maxlen=self.BUFFER_SIZE),
                "sp": deque(maxlen=self.BUFFER_SIZE),
                "tk": deque(maxlen=self.BUFFER_SIZE),
            }
            for inst in self.inst_ordered
        }
        self._last_close: dict[str, float] = {}   # for log-return computation

        # ── Per-pair signal filters ────────────────────────────────────────
        self.btc_filter = HysteresisFilter(entry_threshold, exit_threshold)
        self.fx_filters = {
            pair: HysteresisFilter(entry_threshold, exit_threshold)
            for pair in self.fx_pairs
        }
        self.sizer = PositionSizer(base_risk, entry_threshold)

        self._bar_count = 0
        self._last_regime = "NORMAL"

    # ── Public API ─────────────────────────────────────────────────────────

    def on_bar(self, dt_str: str, bars: dict[str, dict]) -> dict[str, Signal]:
        """
        Process one M5 bar close.

        Args:
            dt_str: bar close time "YYYY-MM-DD HH:MM"
            bars:   {instrument: {o, h, l, c, sp, tk}}
                    Missing instruments are forward-filled from last known close.

        Returns:
            {pair: Signal} for BTC and all FX pairs.
            Empty dict during warmup (first 200 bars).
        """
        self._bar_count += 1

        # ── 1. Update rolling buffers (forward-fill missing instruments) ──
        node_rets = np.zeros(self.n_inst, dtype=np.float64)
        for i, inst in enumerate(self.inst_ordered):
            if inst in bars:
                b = bars[inst]
                c = float(b.get("c", self._last_close.get(inst, 1.0)))
            else:
                c = self._last_close.get(inst, 0.0)
                b = {"o": c, "h": c, "l": c, "c": c, "sp": 0.0, "tk": 0.0}

            buf = self.buffers[inst]
            buf["o"].append(float(b.get("o", c)))
            buf["h"].append(float(b.get("h", c)))
            buf["l"].append(float(b.get("l", c)))
            buf["c"].append(c)
            buf["sp"].append(float(b.get("sp", 0.0)))
            buf["tk"].append(float(b.get("tk", 0.0)))

            # Log return for this bar
            prev = self._last_close.get(inst, c)
            if prev > 1e-10:
                node_rets[i] = np.log(max(c, 1e-10) / prev)
            self._last_close[inst] = c

        # ── 2. Update MathEngine with this bar's node returns ─────────────
        math_state = self.math_engine.update(node_rets)
        if math_state.valid:
            self._last_regime = math_state.regime

        # During warmup: accumulate state, emit nothing
        if self._bar_count < self.WARMUP_BARS:
            return {}

        # ── 3. Convert buffers to ohlc arrays ─────────────────────────────
        ohlc_live: dict[str, dict] = {}
        for inst in self.inst_ordered:
            buf = self.buffers[inst]
            if len(buf["c"]) < 2:
                continue
            ohlc_live[inst] = {k: np.array(list(buf[k]), dtype=np.float32) for k in buf}

        if "BTCUSD" not in ohlc_live:
            return {}

        btc_ohlc = ohlc_live["BTCUSD"]
        bar_idx  = len(btc_ohlc["c"]) - 1   # always the last bar

        # ── 4. Bridge state ───────────────────────────────────────────────
        btc_close = float(btc_ohlc["c"][bar_idx])
        btc_hist  = btc_ohlc["c"][max(0, bar_idx - 240): bar_idx + 1]
        btc_tk    = float(btc_ohlc["tk"][bar_idx])
        btc_sp    = float(btc_ohlc["sp"][bar_idx])
        btc_lr_w  = np.diff(np.log(np.maximum(btc_ohlc["c"][max(0, bar_idx-14):bar_idx+1], 1e-10)))
        btc_atr   = float(np.std(btc_lr_w) * btc_close * 1.414) if len(btc_lr_w) > 1 else btc_close * 0.002

        fx_closes  = {p: float(ohlc_live[p]["c"][len(ohlc_live[p]["c"])-1])
                      for p in self.fx_pairs if p in ohlc_live}
        fx_lr_now  = {}
        fx_spreads = {}
        for p in self.fx_pairs:
            if p in ohlc_live:
                c_arr = ohlc_live[p]["c"]
                t = len(c_arr) - 1
                fx_lr_now[p]  = float(np.log(max(c_arr[t], 1e-10) / max(c_arr[t-1], 1e-10))) if t > 0 else 0.0
                fx_spreads[p] = float(ohlc_live[p]["sp"][t])

        bridge_state = self.bridge.update(
            dt_str=dt_str,
            btc_close=btc_close,
            btc_tick_vel=btc_tk,
            btc_spread=btc_sp,
            btc_atr=btc_atr,
            btc_h1_lifespan=math_state.h1_lifespan if math_state.valid else 0.0,
            btc_close_history=btc_hist,
            math_state=math_state,
            fx_closes=fx_closes,
            fx_log_rets=fx_lr_now,
            fx_spreads=fx_spreads,
        )

        signals: dict[str, Signal] = {}
        gate = bridge_state.btc_bridge_gate

        # ── 5. BTC signal ─────────────────────────────────────────────────
        bf = self.btc_subnet.compute(btc_ohlc, bar_idx, math_state, bridge_state,
                                     lookback=self.lookback)
        if bf.valid and self.btc_feat_names:
            row = bf.to_array(self.btc_feat_names).reshape(1, -1)
            proba = self.btc_model.predict_proba(row)[0]
            p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])
            direction = self.btc_filter.step(p_sell, p_buy)
            size = self.sizer.size_from_proba(
                proba, direction, regime=self._last_regime
            ) * (1 if direction >= 0 else -1)
            signals["BTCUSD"] = Signal(
                pair="BTCUSD", direction=direction, size=size,
                p_buy=p_buy, p_sell=p_sell, p_hold=p_hold,
                regime=self._last_regime, gate=1.0,
            )

        # ── 6. FX signals ─────────────────────────────────────────────────
        if self.fx_feat_names:
            fx_results = self.fx_subnet.compute_all(
                ohlc_live, bar_idx, math_state, bridge_state, lookback=self.lookback
            )
            for pid, pair in enumerate(self.fx_pairs):
                ff = fx_results.get(pair)
                if not ff or not ff.valid:
                    continue
                feat_row = ff.to_array(self.fx_feat_names)
                # Build DataFrame with integer columns matching training layout
                df = pd.DataFrame(feat_row.reshape(1, -1).astype(np.float32))
                cat_col = df.shape[1]   # pair_id appended as last column
                df[cat_col] = str(pid)
                proba = self.fx_model.predict_proba(df)[0]
                p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])
                filt = self.fx_filters.get(pair, self.fx_filters.get(self.fx_pairs[0]))
                direction = filt.step(p_sell, p_buy)
                size = self.sizer.size_from_proba(
                    proba, direction, regime=self._last_regime
                ) * (1 if direction >= 0 else -1)
                signals[pair] = Signal(
                    pair=pair, direction=direction, size=size,
                    p_buy=p_buy, p_sell=p_sell, p_hold=p_hold,
                    regime=self._last_regime, gate=gate,
                )

        return signals

    def reset_filters(self) -> None:
        """Reset all hysteresis filters to FLAT (e.g. after session close)."""
        self.btc_filter.reset()
        for f in self.fx_filters.values():
            f.reset()

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def is_warmed_up(self) -> bool:
        return self._bar_count >= self.WARMUP_BARS


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="D:/dataset-ml/models/catboost_v2_full3")
    p.add_argument("--n-bars",    type=int, default=250)
    args = p.parse_args()

    print(f"LiveEngine smoke test ({args.n_bars} synthetic bars)...")
    engine = LiveEngine(args.model_dir)
    print(f"  Models loaded. BTC feat={len(engine.btc_feat_names)}  "
          f"FX feat={len(engine.fx_feat_names)}  pairs={len(engine.fx_pairs)}")

    rng  = np.random.default_rng(42)
    btc_price = 85000.0
    fx_prices = {p: rng.uniform(0.6, 1.5) for p in engine.fx_pairs}

    n_signals = 0
    for i in range(args.n_bars):
        dt = f"2026-03-{(i // 288) + 1:02d} {(i % 288) * 5 // 60:02d}:{(i % 12) * 5:02d}"
        btc_price *= np.exp(rng.normal(0, 0.002))
        bars = {"BTCUSD": {"o": btc_price*0.999, "h": btc_price*1.001,
                            "l": btc_price*0.998, "c": btc_price,
                            "sp": 25.0, "tk": int(rng.integers(50, 200))}}
        for pair in engine.fx_pairs:
            fx_prices[pair] *= np.exp(rng.normal(0, 0.0003))
            bars[pair] = {"o": fx_prices[pair]*0.9999, "h": fx_prices[pair]*1.0001,
                           "l": fx_prices[pair]*0.9998, "c": fx_prices[pair],
                           "sp": 0.0001, "tk": int(rng.integers(10, 80))}

        sigs = engine.on_bar(dt, bars)
        if sigs:
            n_signals += 1

    print(f"  Bars processed: {args.n_bars}  |  bars with signals: {n_signals}")
    if sigs:
        print(f"  Sample signals (last bar):")
        for pair, sig in list(sigs.items())[:5]:
            print(f"    {pair:12s}  dir={sig.direction:+d}  size={sig.size:.3f}  "
                  f"P(B/H/S)={sig.p_buy:.3f}/{sig.p_hold:.3f}/{sig.p_sell:.3f}  "
                  f"regime={sig.regime}  gate={sig.gate:.2f}")
    print("OK")
