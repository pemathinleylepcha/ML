"""
test_catboost_v2_models.py -- Sanity check for trained CatBoost v2 baseline models.

Checks:
  1. Models load without error
  2. Class order (SELL=0, HOLD=1, BUY=2)
  3. Feature names match training metadata
  4. BTC prediction shape and probability sum
  5. FX prediction shape, pair_id dtype handling, probability sum
  6. Prediction distribution (are we collapsing to HOLD?)
  7. Signal direction distribution across all 28 pairs

Usage:
    python src/test_catboost_v2_models.py
    python src/test_catboost_v2_models.py --model-dir D:/dataset-ml/models/catboost_v2_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("ERROR: catboost not installed. Run: pip install catboost")
    sys.exit(1)

# ── Class constants (must match train_catboost_v2.py) ──────────────────────
SELL_CLASS = 0
HOLD_CLASS = 1
BUY_CLASS  = 2
CLASS_NAMES = {SELL_CLASS: "SELL", HOLD_CLASS: "HOLD", BUY_CLASS: "BUY"}

N_BTC_FEATURES = 37
N_FX_FEATURES  = 138   # + 1 pair_id = 139 total columns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="D:/dataset-ml/models/catboost_v2_baseline",
                   help="Directory containing the .cbm and .json files")
    return p.parse_args()


def load_models(model_dir: Path):
    btc_path  = model_dir / "catboost_btc_v2.cbm"
    fx_path   = model_dir / "catboost_fx_v2.cbm"
    meta_path = model_dir / "catboost_v2_feature_names.json"

    print(f"\n[1] Loading models from {model_dir}")
    for p in [btc_path, fx_path, meta_path]:
        if not p.exists():
            print(f"  ERROR: missing {p}")
            sys.exit(1)
        print(f"  OK  {p.name}  ({p.stat().st_size // 1024} KB)")

    btc_model = CatBoostClassifier()
    btc_model.load_model(str(btc_path))

    fx_model = CatBoostClassifier()
    fx_model.load_model(str(fx_path))

    with open(meta_path) as f:
        meta = json.load(f)

    return btc_model, fx_model, meta


def check_class_order(btc_model, fx_model):
    print("\n[2] Class order check")
    for name, model in [("BTC", btc_model), ("FX", fx_model)]:
        classes = list(model.classes_)
        expected = [SELL_CLASS, HOLD_CLASS, BUY_CLASS]
        ok = classes == expected
        status = "OK" if ok else "MISMATCH"
        print(f"  {name}: classes={classes}  expected={expected}  [{status}]")
        if not ok:
            print(f"  WARNING: proba[:,0]={CLASS_NAMES.get(classes[0],'?')}  "
                  f"[:,1]={CLASS_NAMES.get(classes[1],'?')}  "
                  f"[:,2]={CLASS_NAMES.get(classes[2],'?')}")


def check_feature_names(btc_model, fx_model, meta):
    print("\n[3] Feature name check")
    btc_names_model = btc_model.feature_names_
    btc_names_meta  = meta.get("btc", [])
    # BTC was trained on raw numpy -> model stores integer column names ("0","1",...)
    # Count match is what matters for inference; name mismatch is expected.
    btc_count_ok = len(btc_names_model) == len(btc_names_meta)
    btc_uses_idx = all(n.isdigit() for n in btc_names_model[:3])
    print(f"  BTC: model={len(btc_names_model)} feats  meta={len(btc_names_meta)} feats  "
          f"[{'OK' if btc_count_ok else 'COUNT MISMATCH'}]"
          f"{'  (model uses integer col indices, expected)' if btc_uses_idx else ''}")

    fx_names_model = fx_model.feature_names_
    fx_names_meta  = meta.get("fx", [])
    fx_ok = len(fx_names_model) == len(fx_names_meta)
    print(f"  FX:  model={len(fx_names_model)} feats  meta={len(fx_names_meta)} feats  "
          f"[{'OK' if fx_ok else 'MISMATCH'}]")

    fx_pairs = meta.get("fx_pairs", [])
    print(f"  FX pairs in meta: {len(fx_pairs)}  -> {fx_pairs[:5]} ...")


def check_btc_predictions(btc_model, meta):
    print("\n[4] BTC prediction check (100 synthetic bars)")
    rng = np.random.default_rng(42)
    n_feat = len(meta.get("btc", [])) or N_BTC_FEATURES
    X = rng.standard_normal((100, n_feat)).astype(np.float32)

    proba = btc_model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    print(f"  Output shape:  {proba.shape}  (expected (100, 3))")
    print(f"  Prob sums:     min={proba.sum(axis=1).min():.6f}  max={proba.sum(axis=1).max():.6f}  (should be ~1.0)")
    print(f"  Prob range:    [{proba.min():.4f}, {proba.max():.4f}]")

    dist = {CLASS_NAMES[c]: int((preds == c).sum()) for c in [SELL_CLASS, HOLD_CLASS, BUY_CLASS]}
    print(f"  Pred dist:     {dist}")

    mean_p = proba.mean(axis=0)
    print(f"  Mean proba:    SELL={mean_p[SELL_CLASS]:.4f}  HOLD={mean_p[HOLD_CLASS]:.4f}  BUY={mean_p[BUY_CLASS]:.4f}")

    if dist.get("HOLD", 0) > 90:
        print("  WARNING: model predicts HOLD >90% of the time — likely HOLD-collapse")


def check_fx_predictions(fx_model, meta):
    print("\n[5] FX prediction check (100 synthetic bars × 28 pairs)")
    rng = np.random.default_rng(42)
    fx_pairs = meta.get("fx_pairs", [f"PAIR{i}" for i in range(28)])
    n_pairs  = len(fx_pairs)
    n_feat   = (len(meta.get("fx", [])) - 1) if meta.get("fx") else N_FX_FEATURES

    all_preds = []
    all_proba = []

    for pid in range(n_pairs):
        X_raw = rng.standard_normal((100, n_feat)).astype(np.float32)
        pair_col = np.full((100, 1), float(pid), dtype=np.float32)
        X = np.hstack([X_raw, pair_col])

        # Same DataFrame wrapping required as in training
        df = pd.DataFrame(X.astype(np.float32))
        df[n_feat] = df[n_feat].astype(int).astype(str)

        proba = fx_model.predict_proba(df)
        preds = np.argmax(proba, axis=1)
        all_preds.append(preds)
        all_proba.append(proba)

    all_preds = np.concatenate(all_preds)
    all_proba = np.vstack(all_proba)

    print(f"  Output shape:  {all_proba.shape}  (expected ({100*n_pairs}, 3))")
    print(f"  Prob sums:     min={all_proba.sum(axis=1).min():.6f}  max={all_proba.sum(axis=1).max():.6f}")

    dist = {CLASS_NAMES[c]: int((all_preds == c).sum()) for c in [SELL_CLASS, HOLD_CLASS, BUY_CLASS]}
    total = sum(dist.values())
    dist_pct = {k: f"{v/total*100:.1f}%" for k, v in dist.items()}
    print(f"  Pred dist:     {dist}  {dist_pct}")

    mean_p = all_proba.mean(axis=0)
    print(f"  Mean proba:    SELL={mean_p[SELL_CLASS]:.4f}  HOLD={mean_p[HOLD_CLASS]:.4f}  BUY={mean_p[BUY_CLASS]:.4f}")

    if dist.get("HOLD", 0) / total > 0.90:
        print("  WARNING: FX model predicts HOLD >90% — likely HOLD-collapse from class imbalance")

    # Per-pair signal balance
    print("\n[6] Per-pair prediction breakdown")
    print(f"  {'Pair':<12} {'SELL':>6} {'HOLD':>6} {'BUY':>6}  {'signal_mean':>12}")
    for pid, pair in enumerate(fx_pairs):
        p = all_proba[pid*100:(pid+1)*100]
        pred = np.argmax(p, axis=1)
        sell_n = (pred == SELL_CLASS).sum()
        hold_n = (pred == HOLD_CLASS).sum()
        buy_n  = (pred == BUY_CLASS).sum()
        signal_mean = (p[:, BUY_CLASS] - p[:, SELL_CLASS]).mean()
        print(f"  {pair:<12} {sell_n:>6} {hold_n:>6} {buy_n:>6}  {signal_mean:>+12.4f}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    btc_model, fx_model, meta = load_models(model_dir)
    check_class_order(btc_model, fx_model)
    check_feature_names(btc_model, fx_model, meta)
    check_btc_predictions(btc_model, meta)
    check_fx_predictions(fx_model, meta)

    print("\n=== Sanity check complete ===")


if __name__ == "__main__":
    main()
