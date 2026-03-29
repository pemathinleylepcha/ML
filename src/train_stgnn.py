"""
Train STGNN on real FX tick data.
Separate launcher so multiprocessing workers never re-import this file.
"""
import multiprocessing
multiprocessing.freeze_support()

import argparse
import glob as _glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stgnn_model import (
    MultiTimeframePreprocessor, STGNNTrainer,
    N_FEATURES, TIMEFRAME_NAMES,
)


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


def main():
    parser = argparse.ArgumentParser(description="STGNN Real Data Training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("STGNN -- Real Data Training")
    print(f"  Data dir : {data_dir}")
    print(f"  Model dir: {model_dir}")
    print(f"  Folds={args.n_folds}, Epochs={args.epochs}, Batch={args.batch_size}")
    print("=" * 60)

    # [1] Discover CSV files
    csv_files = sorted(_glob.glob(str(data_dir / "*.csv")))
    print(f"[1/5] Found {len(csv_files)} CSV files")
    if not csv_files:
        print("ERROR: No CSV files found.")
        sys.exit(1)

    # [2] Load tick CSVs → M1 OHLC → multi-timeframe tensors
    t_step = time.time()
    print("[2/5] Loading & resampling tick CSVs to OHLC M1 bars...")
    preprocessor = MultiTimeframePreprocessor()
    json_data = {}

    for f in csv_files:
        pair = Path(f).stem.split("_")[0]
        try:
            df = pd.read_csv(f, sep="\t", header=0)
            df.columns = [c.strip("<>").lower() for c in df.columns]
            if "date" in df.columns and "time" in df.columns:
                df["dt"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    format="%Y.%m.%d %H:%M:%S.%f", errors="coerce"
                )
                df = df.dropna(subset=["dt"]).set_index("dt")
                mid = (df["bid"] + df["ask"]) / 2.0
                spread = df["ask"] - df["bid"]
                vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
                m1 = pd.DataFrame({
                    "o": mid.resample("1min").first(),
                    "h": mid.resample("1min").max(),
                    "l": mid.resample("1min").min(),
                    "c": mid.resample("1min").last(),
                    "sp": spread.resample("1min").mean(),
                    "tk": vol.resample("1min").sum(),
                }).dropna(subset=["c"]).reset_index()
                m1 = m1.rename(columns={m1.columns[0]: "dt"})
                json_data[pair] = m1.to_dict("records")
                print(f"  {pair}: {len(json_data[pair])} M1 bars")
        except Exception as e:
            print(f"  Skipping {pair}: {e}")

    print(f"  Loaded {len(json_data)} pairs")
    print(f"  Resampling to all 10 timeframes...")
    tf_data = preprocessor.resample_from_json(json_data)
    features = preprocessor.extract_features(tf_data)
    laplacians = preprocessor.compute_laplacians(tf_data)
    for tf, arr in features.items():
        print(f"    {tf}: {arr.shape}")
    print(f"  [step 2 done in {_elapsed(t_step)}]")

    # [3] Targets and labels
    t_step = time.time()
    print("[3/5] Computing targets and labels...")
    m1_arr = features.get("M1")
    if m1_arr is None:
        print("ERROR: No M1 data.")
        sys.exit(1)
    n_bars = m1_arr.shape[0]
    close_idx = 3

    # 10-bar forward return: more signal than 1-bar, less noise
    FWD_BARS = 10
    closes = m1_arr[:, 0, close_idx].astype("float64")
    targets = np.zeros(n_bars, dtype="float32")
    targets[:n_bars - FWD_BARS] = closes[FWD_BARS:] - closes[:n_bars - FWD_BARS]

    # ATR-based threshold: label only moves > 0.5 * ATR(14)
    highs = m1_arr[:, 0, 2].astype("float64")   # feature index 2 = high
    lows  = m1_arr[:, 0, 1].astype("float64")   # feature index 1 = low
    tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows  - np.roll(closes, 1)))
    tr[0] = highs[0] - lows[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values.astype("float32")
    half_atr = 0.5 * atr14

    labels = np.ones(n_bars, dtype="int64")         # hold by default
    labels[targets >  half_atr] = 2                 # buy
    labels[targets < -half_atr] = 0                 # sell
    print(f"  Labels: sell={int((labels==0).sum())}, hold={int((labels==1).sum())}, buy={int((labels==2).sum())}")
    print(f"  ATR half-threshold mean={half_atr.mean():.6f}, fwd_bars={FWD_BARS}")
    print(f"  [step 3 done in {_elapsed(t_step)}]")

    # [4] Train
    t_step = time.time()
    print("[4/5] Training STGNN...")
    trainer = STGNNTrainer(
        n_pairs=m1_arr.shape[1],
        n_features=N_FEATURES,
        temporal_hidden=256,
        spatial_out=32,
        seq_len=60,
        n_folds=args.n_folds,
        epochs=args.epochs,
        patience=15,
        batch_size=args.batch_size,
    )
    results = trainer.train_full(features, laplacians, targets, labels, verbose=True)
    print(f"  [step 4 training done in {_elapsed(t_step)}]")

    # [5] Save
    t_step = time.time()
    print("[5/5] Saving models...")
    stgnn_path = model_dir / "stgnn_weights.pt"
    cb_path = model_dir / "catboost_exec.cbm"
    trainer.save(str(stgnn_path), str(cb_path))
    print(f"  STGNN    : {stgnn_path}")
    print(f"  CatBoost : {cb_path}")
    print(f"  [step 5 done in {_elapsed(t_step)}]")

    print("\n--- Cross-Validation Results ---")
    for fold in results.get("fold_results", []):
        conf = fold.get('conf_acc', fold['accuracy'])
        print(f"  Fold {fold['fold']}: MSE={fold['mse']:.6f}, R2={fold['r2']:.4f}, "
              f"Acc={fold['accuracy']*100:.1f}%, ConfAcc(top30%)={conf*100:.1f}%")
    print("=" * 60)
    print(f"Training complete.  Total wall-clock: {_elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
