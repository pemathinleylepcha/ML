"""Diagnose avg_cb variability across bars for AlphaNet."""
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "C:/Algo-C2/src")
from feature_engine import compute_pair_features, extract_graph_features
from math_engine import MathEngine
from market_neutral_model import AlphaNet, predict_weights, weights_to_confidence

print("Loading data...")
with open("C:/Algo-C2/data/algo_c2_5day_data.json") as f:
    data = json.load(f)

print("Loading model...")
ckpt = torch.load("C:/Algo-C2/models/alpha_net.pt", map_location="cpu", weights_only=False)
model = AlphaNet(n_features=ckpt["n_features"], n_pairs=ckpt["n_pairs"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scaler_mean = np.array(ckpt["scaler_mean"])
scaler_std = np.array(ckpt["scaler_std"])
alpha_pairs = ckpt["pairs"]

pairs = list(data.keys())
arrays = {}
for pair, bars in data.items():
    arrays[pair] = {
        "o": np.array([b["o"] for b in bars], dtype=float),
        "h": np.array([b["h"] for b in bars], dtype=float),
        "l": np.array([b["l"] for b in bars], dtype=float),
        "c": np.array([b["c"] for b in bars], dtype=float),
        "sp": np.array([b["sp"] for b in bars], dtype=float),
        "tk": np.array([b["tk"] for b in bars], dtype=float),
        "dt": [b["dt"] for b in bars],
        "tp_paths": [b.get("tp", []) for b in bars],
    }

T = len(next(iter(arrays.values()))["c"])
n_pairs = len(pairs)
print(f"T={T}, sampling every 100 bars from bar 200 to {min(T, 5000)}")

engine = MathEngine(n_pairs=n_pairs)
for t in range(200):
    ret = np.zeros(n_pairs)
    for i, pair in enumerate(pairs):
        c = arrays[pair]["c"]
        if t > 0 and c[t-1] > 0:
            ret[i] = np.log(c[t] / c[t-1])
    engine.update(ret)

avg_cbs_s5 = []
avg_cbs_s10 = []
max_cbs_s10 = []
weight_stds = []

for t in range(200, min(T, 5000), 20):  # sample every 20 bars = 240 samples
    ret = np.zeros(n_pairs)
    for i, pair in enumerate(pairs):
        c = arrays[pair]["c"]
        if t > 0 and c[t-1] > 0:
            ret[i] = np.log(c[t] / c[t-1])
    math_state = engine.update(ret)
    if not math_state.valid:
        continue

    feat_vec = []
    for pair in alpha_pairs:
        if pair not in arrays:
            feat_vec.extend([0.0] * 16)
            continue
        pf = compute_pair_features(arrays[pair], pair, t, lookback=120)
        feat_vec.extend([
            v if not (isinstance(v, float) and np.isnan(v)) else 0.0
            for v in pf.values()
        ])
    gf = extract_graph_features(math_state)
    feat_vec.extend([
        gf.get("graph_residual_mean", 0.0),
        gf.get("graph_residual_std", 0.0),
        gf.get("graph_residual_max", 0.0),
        gf.get("graph_spectral_gap", 0.0),
        gf.get("graph_betti_h0", 0.0),
        gf.get("graph_betti_h1", 0.0),
        gf.get("graph_avg_correlation", 0.0),
        gf.get("graph_laplacian_trace", 0.0),
    ])

    feat_arr = np.array(feat_vec, dtype=np.float32)
    w = predict_weights(model, feat_arr, scaler_mean, scaler_std)

    conf5 = 1.0 / (1.0 + np.exp(-np.abs(w) * 5.0))
    conf10 = 1.0 / (1.0 + np.exp(-np.abs(w) * 10.0))

    avg_cbs_s5.append(np.mean(conf5))
    avg_cbs_s10.append(np.mean(conf10))
    max_cbs_s10.append(np.max(conf10))
    weight_stds.append(np.std(w))

avg_cbs_s5 = np.array(avg_cbs_s5)
avg_cbs_s10 = np.array(avg_cbs_s10)
max_cbs_s10 = np.array(max_cbs_s10)
weight_stds = np.array(weight_stds)

print(f"\nSamples: {len(avg_cbs_s5)}")
print(f"\nscale=5: avg_cb min={avg_cbs_s5.min():.4f}, max={avg_cbs_s5.max():.4f}, mean={avg_cbs_s5.mean():.4f}, std={avg_cbs_s5.std():.4f}")
print(f"  pct >= 0.53: {(avg_cbs_s5 >= 0.53).mean():.1%}")
print(f"  pct >= 0.54: {(avg_cbs_s5 >= 0.54).mean():.1%}")
print(f"  pct >= 0.55: {(avg_cbs_s5 >= 0.55).mean():.1%}")
print(f"  pct >= 0.56: {(avg_cbs_s5 >= 0.56).mean():.1%}")
print(f"  pct >= 0.57: {(avg_cbs_s5 >= 0.57).mean():.1%}")
print(f"  pct >= 0.58: {(avg_cbs_s5 >= 0.58).mean():.1%}")

print(f"\nscale=10: avg_cb min={avg_cbs_s10.min():.4f}, max={avg_cbs_s10.max():.4f}, mean={avg_cbs_s10.mean():.4f}, std={avg_cbs_s10.std():.4f}")
print(f"  pct >= 0.55: {(avg_cbs_s10 >= 0.55).mean():.1%}")
print(f"  pct >= 0.57: {(avg_cbs_s10 >= 0.57).mean():.1%}")
print(f"  pct >= 0.58: {(avg_cbs_s10 >= 0.58).mean():.1%}")
print(f"  pct >= 0.59: {(avg_cbs_s10 >= 0.59).mean():.1%}")
print(f"  pct >= 0.60: {(avg_cbs_s10 >= 0.60).mean():.1%}")
print(f"  pct >= 0.62: {(avg_cbs_s10 >= 0.62).mean():.1%}")

print(f"\nmax_cb (scale=10): min={max_cbs_s10.min():.4f}, max={max_cbs_s10.max():.4f}, mean={max_cbs_s10.mean():.4f}")

print(f"\nweight_std: min={weight_stds.min():.4f}, max={weight_stds.max():.4f}, mean={weight_stds.mean():.4f}, std={weight_stds.std():.4f}")
print(f"  For G2 using weight_std >= 0.03: {(weight_stds >= 0.03).mean():.1%}")
print(f"  For G2 using weight_std >= 0.035: {(weight_stds >= 0.035).mean():.1%}")
print(f"  For G2 using weight_std >= 0.04: {(weight_stds >= 0.04).mean():.1%}")
print(f"  For G2 using weight_std >= 0.045: {(weight_stds >= 0.045).mean():.1%}")
print(f"  For G2 using weight_std >= 0.05: {(weight_stds >= 0.05).mean():.1%}")
