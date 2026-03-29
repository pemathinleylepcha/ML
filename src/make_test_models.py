"""
make_test_models.py -- Generate tiny synthetic CatBoost models for smoke-testing live_engine.py

Run once to create test model artifacts in D:/Algo-C2/models/test_v2/
Then: python src/live_engine.py --model-dir D:/Algo-C2/models/test_v2
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from catboost import CatBoostClassifier
from universe import FX_PAIRS
from math_engine import MathEngine
from bridge import BridgeComputer
from subnet_btc import BTCSubnet
from subnet_fx import FXSubnet

OUT_DIR = Path("D:/Algo-C2/models/test_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 120
SEED_BARS = 160   # enough to warm up subnets

rng = np.random.default_rng(42)

# ── Build one pass to discover actual feature names ────────────────────────
n_inst = 43  # ALL_INSTRUMENTS count
from universe import ALL_INSTRUMENTS
FX_PAIRS_LIST = list(FX_PAIRS)

math_engine   = MathEngine(n_pairs=n_inst)
bridge        = BridgeComputer()
btc_subnet    = BTCSubnet()
fx_subnet     = FXSubnet()

inst_ordered = list(ALL_INSTRUMENTS)
inst_idx     = {inst: i for i, inst in enumerate(inst_ordered)}

# Rolling buffers
from collections import deque
BUFFER_SIZE = 370
buffers = {
    inst: {k: deque(maxlen=BUFFER_SIZE) for k in ("o","h","l","c","sp","tk")}
    for inst in inst_ordered
}
last_close = {}

btc_price = 85000.0
fx_prices = {p: rng.uniform(0.6, 1.5) for p in FX_PAIRS_LIST}

print(f"Warming up {SEED_BARS} bars to discover feature names...")
btc_feats = None
fx_feats_map = {}

for bar_i in range(SEED_BARS):
    node_rets = np.zeros(n_inst)

    btc_price *= np.exp(rng.normal(0, 0.002))
    bars = {
        "BTCUSD": {"o": btc_price*0.999, "h": btc_price*1.001,
                   "l": btc_price*0.998, "c": btc_price, "sp": 25.0,
                   "tk": float(rng.integers(50, 200))}
    }
    for p in FX_PAIRS_LIST:
        fx_prices[p] *= np.exp(rng.normal(0, 0.0003))
        bars[p] = {"o": fx_prices[p]*0.9999, "h": fx_prices[p]*1.0001,
                   "l": fx_prices[p]*0.9998, "c": fx_prices[p],
                   "sp": 0.0001, "tk": float(rng.integers(10, 80))}

    for i, inst in enumerate(inst_ordered):
        if inst in bars:
            b = bars[inst]
            c = float(b.get("c", last_close.get(inst, 1.0)))
        else:
            c = last_close.get(inst, 1.0)
            b = {"o": c, "h": c, "l": c, "c": c, "sp": 0.0, "tk": 0.0}
        buf = buffers[inst]
        buf["o"].append(float(b.get("o", c)))
        buf["h"].append(float(b.get("h", c)))
        buf["l"].append(float(b.get("l", c)))
        buf["c"].append(c)
        buf["sp"].append(float(b.get("sp", 0.0)))
        buf["tk"].append(float(b.get("tk", 0.0)))
        prev = last_close.get(inst, c)
        if prev > 1e-10:
            node_rets[i] = np.log(max(c, 1e-10) / prev)
        last_close[inst] = c

    math_state = math_engine.update(node_rets)

    ohlc_live = {}
    for inst in inst_ordered:
        buf = buffers[inst]
        if len(buf["c"]) >= 2:
            ohlc_live[inst] = {k: np.array(list(buf[k]), dtype=np.float32) for k in buf}

    if "BTCUSD" not in ohlc_live:
        continue

    btc_ohlc = ohlc_live["BTCUSD"]
    bar_idx  = len(btc_ohlc["c"]) - 1
    btc_close = float(btc_ohlc["c"][bar_idx])
    btc_hist  = btc_ohlc["c"][max(0, bar_idx-240):bar_idx+1]
    btc_tk    = float(btc_ohlc["tk"][bar_idx])
    btc_sp    = float(btc_ohlc["sp"][bar_idx])
    btc_lr_w  = np.diff(np.log(np.maximum(btc_ohlc["c"][max(0, bar_idx-14):bar_idx+1], 1e-10)))
    btc_atr   = float(np.std(btc_lr_w) * btc_close * 1.414) if len(btc_lr_w) > 1 else btc_close * 0.002

    fx_closes  = {p: float(ohlc_live[p]["c"][-1]) for p in FX_PAIRS_LIST if p in ohlc_live}
    fx_lr_now  = {}
    fx_spreads = {}
    for p in FX_PAIRS_LIST:
        if p in ohlc_live:
            c_arr = ohlc_live[p]["c"]
            t = len(c_arr) - 1
            fx_lr_now[p]  = float(np.log(max(c_arr[t], 1e-10) / max(c_arr[t-1], 1e-10))) if t > 0 else 0.0
            fx_spreads[p] = float(ohlc_live[p]["sp"][t])

    dt_str = f"2026-03-{(bar_i // 288)+1:02d} {(bar_i % 288)*5//60:02d}:{(bar_i % 12)*5:02d}"
    bridge_state = bridge.update(
        dt_str=dt_str,
        btc_close=btc_close, btc_tick_vel=btc_tk, btc_spread=btc_sp, btc_atr=btc_atr,
        btc_h1_lifespan=math_state.h1_lifespan if math_state.valid else 0.0,
        btc_close_history=btc_hist, math_state=math_state,
        fx_closes=fx_closes, fx_log_rets=fx_lr_now, fx_spreads=fx_spreads,
    )

    bf = btc_subnet.compute(btc_ohlc, bar_idx, math_state, bridge_state, lookback=LOOKBACK)
    if bf.valid and btc_feats is None:
        btc_feats = list(bf.features.keys())

    fx_results = fx_subnet.compute_all(ohlc_live, bar_idx, math_state, bridge_state, lookback=LOOKBACK)
    for p in FX_PAIRS_LIST:
        ff = fx_results.get(p)
        if ff and ff.valid and p not in fx_feats_map:
            fx_feats_map[p] = list(ff.features.keys())

if btc_feats is None:
    raise RuntimeError("BTCSubnet never produced valid features — increase SEED_BARS")

# Use first available FX pair's feature names
fx_pair0 = FX_PAIRS_LIST[0]
if fx_pair0 not in fx_feats_map:
    raise RuntimeError(f"FXSubnet never produced valid features for {fx_pair0}")
fx_feats = fx_feats_map[fx_pair0]

# Trim to a small subset for test models (first 15 BTC, first 20 FX)
BTC_K = min(15, len(btc_feats))
FX_K  = min(20, len(fx_feats))
btc_feat_sel = btc_feats[:BTC_K]
fx_feat_sel  = fx_feats[:FX_K]
fx_feat_with_id = fx_feat_sel + ["pair_id"]

print(f"BTC features: {len(btc_feats)} total, using {BTC_K}")
print(f"FX  features: {len(fx_feats)} total, using {FX_K} + pair_id")

# ── Synthetic training data ────────────────────────────────────────────────
N = 300
X_btc = rng.standard_normal((N, BTC_K)).astype(np.float32)
y_btc = rng.integers(0, 3, N)

import pandas as pd
X_fx_arr = rng.standard_normal((N, FX_K)).astype(np.float32)
df_fx = pd.DataFrame(X_fx_arr, columns=fx_feat_sel)
df_fx["pair_id"] = [str(rng.integers(0, len(FX_PAIRS_LIST))) for _ in range(N)]
y_fx = rng.integers(0, 3, N)

# ── Train tiny CatBoost models ─────────────────────────────────────────────
print("Training synthetic BTC model...")
btc_model = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1,
                                 loss_function="MultiClass", classes_count=3,
                                 random_seed=42, verbose=0)
btc_model.fit(X_btc, y_btc)
btc_path = OUT_DIR / "catboost_btc_v2.cbm"
btc_model.save_model(str(btc_path))
print(f"  Saved: {btc_path}")

print("Training synthetic FX model...")
fx_model = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1,
                                loss_function="MultiClass", classes_count=3,
                                cat_features=["pair_id"],
                                random_seed=42, verbose=0)
fx_model.fit(df_fx, y_fx)
fx_path = OUT_DIR / "catboost_fx_v2.cbm"
fx_model.save_model(str(fx_path))
print(f"  Saved: {fx_path}")

# ── Save feature names JSON ─────────────────────────────────────────────────
meta = {
    "btc": btc_feat_sel,
    "fx":  fx_feat_with_id,
    "fx_pairs": FX_PAIRS_LIST,
    "horizon": 3,
    "lookback": LOOKBACK,
}
meta_path = OUT_DIR / "catboost_v2_feature_names.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: {meta_path}")
print("\nDone. Run:")
print(f"  python src/live_engine.py --model-dir {OUT_DIR} --n-bars 250")
