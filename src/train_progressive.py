"""
train_progressive.py -- Progressive STGNN training across data phases.

Phases:
  phase1_macro:    H1 data, 2018-2020  (macro structure, train from scratch)
  phase2_intraday: M15 data, 2022-2023 (intraday patterns, warm-start phase1)
  phase3_recent:   M5 data, 2024-2025  (recent market, warm-start phase2)
  phase3b_m1:      M1 data, 2025-Q4 to 2026 (finest resolution, warm-start phase3)

Usage:
  python train_progressive.py                        # all phases
  python train_progressive.py --phases 1,2
  python train_progressive.py --phases 3b --data-dir D:/dataset-ml/DataExtractor
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from src/ or from the project root
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from universe import ALL_INSTRUMENTS, TIMEFRAME_FREQ, TIMEFRAME_MINUTES
from stgnn_model import (
    MultiTimeframePreprocessor, STGNNTrainer,
    TIMEFRAME_NAMES, N_FEATURES,
)
from pbo_analysis import compute_pbo, print_pbo_report

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ── Phase definitions ──────────────────────────────────────────────────────────

PHASES = [
    dict(
        name="phase1_macro",
        key="1",
        tf="H1",
        start="2018-01-01",
        end="2020-12-31",
        epochs=60,
        lr=1e-4,
        n_folds=4,
        pretrained=None,
    ),
    dict(
        name="phase2_intraday",
        key="2",
        tf="M15",
        start="2022-01-01",
        end="2023-12-31",
        epochs=40,
        lr=5e-5,
        n_folds=3,
        pretrained="phase1_macro",
    ),
    dict(
        name="phase3_recent",
        key="3",
        tf="M5",
        start="2024-10-01",
        end="2025-09-30",
        epochs=30,
        lr=2e-5,
        n_folds=3,
        pretrained="phase2_intraday",
    ),
    dict(
        name="phase3b_m1",
        key="3b",
        tf="M1",
        start="2025-10-01",
        end="2026-03-26",
        epochs=20,
        lr=1e-5,
        n_folds=2,
        pretrained="phase3_recent",
    ),
]

PHASE_BY_KEY = {p["key"]: p for p in PHASES}

# TF rank by resolution: M1 = 0 (finest), MN1 = 9 (coarsest)
_TF_RANK = {tf: i for i, tf in enumerate(TIMEFRAME_NAMES)}

# ── Data loading ───────────────────────────────────────────────────────────────

_MT5_FMT  = "%Y.%m.%d %H:%M:%S"
_CSV_COLS = ["bar_time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
_RENAME   = {
    "bar_time": "dt", "open": "o", "high": "h", "low": "l",
    "close": "c", "tick_volume": "tk", "spread": "sp",
}


def load_candles(
    data_root: Path,
    symbol: str,
    tf: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Load candle CSVs from DataExtractor layout: {YEAR}/{Qn}/{SYMBOL}/candles_{TF}.csv

    Returns DataFrame with columns o,h,l,c,sp,tk indexed by datetime, or None.
    """
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    frames   = []

    for year in range(start_dt.year, end_dt.year + 1):
        for q in ("Q1", "Q2", "Q3", "Q4"):
            csv_path = data_root / str(year) / q / symbol / f"candles_{tf}.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, header=None, names=_CSV_COLS)
                df = df.rename(columns=_RENAME)
                df["dt"] = pd.to_datetime(df["dt"], format=_MT5_FMT, errors="coerce")
                df = df.dropna(subset=["dt"])
                df = df[["dt", "o", "h", "l", "c", "sp", "tk"]]
                frames.append(df)
            except Exception as exc:
                print(f"    [warn] {csv_path}: {exc}")

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("dt").drop_duplicates("dt")
    out = out[(out["dt"] >= start_dt) & (out["dt"] <= end_dt)]
    out = out.set_index("dt")
    for col in ("o", "h", "l", "c", "sp", "tk"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sp"] = out["sp"].fillna(0)
    out["tk"] = out["tk"].fillna(0)
    out = out.dropna(subset=["o", "h", "l", "c"])
    return out if len(out) > 0 else None


def build_tf_data(
    data_root: Path,
    pairs: list[str],
    primary_tf: str,
    start: str,
    end: str,
) -> tuple[dict, list[str]]:
    """
    Load primary-TF candles for all pairs, then resample upward to coarser TFs.

    Finer TFs (below primary) are intentionally omitted; callers zero-fill them.

    Returns:
        tf_data:       {tf_name: {pair: DataFrame(o,h,l,c,sp,tk with dt column)}}
        available_tfs: TF names actually populated, primary + coarser only
    """
    primary_rank = _TF_RANK[primary_tf]
    coarser_tfs  = [tf for tf in TIMEFRAME_NAMES if _TF_RANK[tf] >= primary_rank]

    # Load primary TF for every pair
    raw: dict[str, pd.DataFrame] = {}
    for symbol in pairs:
        df = load_candles(data_root, symbol, primary_tf, start, end)
        if df is not None:
            raw[symbol] = df

    if not raw:
        raise ValueError(
            f"No candle data found for any pair at {primary_tf} ({start} -> {end})"
        )

    print(f"  Loaded {len(raw)}/{len(pairs)} pairs at {primary_tf} "
          f"({start} -> {end}, {len(next(iter(raw.values())))} bars avg)")

    # Resample up to each coarser TF
    tf_data: dict[str, dict[str, pd.DataFrame]] = {}
    for tf_name in coarser_tfs:
        freq = TIMEFRAME_FREQ[tf_name]
        tf_data[tf_name] = {}
        for symbol, df in raw.items():
            if tf_name == primary_tf:
                tf_data[tf_name][symbol] = df.reset_index()
                continue

            resampled = df.resample(freq).agg(
                {"o": "first", "h": "max", "l": "min",
                 "c": "last", "sp": "mean", "tk": "sum"}
            ).dropna(subset=["c"])

            for col in ("o", "h", "l", "c", "sp"):
                if resampled[col].isna().any():
                    resampled[col] = resampled[col].interpolate(
                        method="linear", limit_direction="both"
                    )
            resampled["tk"] = resampled["tk"].fillna(0)
            tf_data[tf_name][symbol] = resampled.reset_index()

    return tf_data, coarser_tfs


# ── Targets / labels (primary-TF aware) ───────────────────────────────────────

def _compute_targets_from_tf(
    tf_data: dict,
    pairs: list[str],
    primary_tf: str,
    target_pair: str = "EURUSD",
    threshold: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute next-bar log returns and 3-class labels from the primary TF data.
    Falls back to the first available pair if target_pair is absent.
    """
    pair_data = tf_data.get(primary_tf, {})
    if target_pair not in pair_data:
        target_pair = next((p for p in pairs if p in pair_data), None)
    if target_pair is None:
        raise ValueError("No pair data found for target computation")

    df = pair_data[target_pair]
    c  = df["c"].values.astype(float)
    returns = np.zeros(len(c), dtype=np.float32)
    for t in range(len(c) - 1):
        if c[t] > 0:
            returns[t] = float(np.log(c[t + 1] / c[t]))

    labels = np.ones(len(returns), dtype=np.int64)   # hold
    labels[returns >  threshold] = 2                  # buy
    labels[returns < -threshold] = 0                  # sell
    return returns, labels


# ── Weight I/O ────────────────────────────────────────────────────────────────

def load_weights(model_dir: Path, phase_name: str) -> dict | None:
    """Return STGNN state_dict from a prior phase checkpoint, or None."""
    if not HAS_TORCH:
        return None
    pt_path = model_dir / f"{phase_name}_stgnn.pt"
    if not pt_path.exists():
        print(f"  [warm-start] no checkpoint at {pt_path} — starting fresh")
        return None
    ckpt = torch.load(str(pt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    print(f"  [warm-start] loaded {pt_path.name} ({len(state)} tensors)")
    return state


def save_weights(trainer: STGNNTrainer, model_dir: Path, phase_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    stgnn_path = str(model_dir / f"{phase_name}_stgnn.pt")
    cb_path    = str(model_dir / f"{phase_name}_catboost.cbm")
    trainer.save(stgnn_path, cb_path)
    print(f"  Saved: {stgnn_path}")
    print(f"         {cb_path}")


# ── Warm-start injection ───────────────────────────────────────────────────────

def _make_warmstart_stage1(original_stage1, warmstart_state: dict, verbose: bool):
    """
    Return a bound method that injects warm-start weights into the model
    before calling the original train_stage1.
    """
    def _stage1_with_warmstart(self, model, features, laplacians, targets,
                               train_end, val_end, verbose=True):
        try:
            model.load_state_dict(warmstart_state, strict=False)
            if verbose:
                print("    [warm-start] weights injected (strict=False)")
        except Exception as exc:
            print(f"    [warm-start] inject failed: {exc}")
        return original_stage1(self, model, features, laplacians, targets,
                               train_end, val_end, verbose)
    return _stage1_with_warmstart


# ── Phase runner ──────────────────────────────────────────────────────────────

def run_phase(
    phase: dict,
    data_root: Path,
    model_dir: Path,
    n_pairs: int = 43,
    seq_len: int = 60,
) -> None:
    name       = phase["name"]
    primary_tf = phase["tf"]
    start      = phase["start"]
    end        = phase["end"]

    print(f"\n{'='*60}")
    print(f"  PHASE : {name}")
    print(f"  TF    : {primary_tf}   {start} -> {end}")
    print(f"  Folds : {phase['n_folds']}   Epochs: {phase['epochs']}   LR: {phase['lr']}")
    print(f"{'='*60}")

    pairs = ALL_INSTRUMENTS[:n_pairs]

    # ── 1. Load & resample ─────────────────────────────────────────────────────
    tf_data, available_tfs = build_tf_data(data_root, pairs, primary_tf, start, end)

    # ── 2. Feature extraction ──────────────────────────────────────────────────
    print(f"\n  Extracting features ({len(available_tfs)} timeframes available)...")
    preprocessor = MultiTimeframePreprocessor(pairs=pairs)
    features     = preprocessor.extract_features(tf_data)

    # Zero-fill feature tensors for finer TFs (below primary_tf)
    primary_rank = _TF_RANK[primary_tf]
    finer_tfs    = [tf for tf in TIMEFRAME_NAMES if _TF_RANK[tf] < primary_rank]
    if finer_tfs:
        ref = features[primary_tf]  # (T, N, F)
        T, N, F = ref.shape
        for tf_name in finer_tfs:
            features[tf_name] = np.zeros((T, N, F), dtype=np.float32)
        print(f"  Zero-filled {len(finer_tfs)} finer TFs: {finer_tfs}")

    # ── 3. Laplacians ──────────────────────────────────────────────────────────
    print(f"\n  Computing Laplacians...")
    laplacians = preprocessor.compute_laplacians(tf_data)

    # Identity laplacians for finer TFs
    if finer_tfs and laplacians:
        ref_lap  = laplacians[primary_tf]
        lap_size = ref_lap[0].shape[0] if ref_lap else n_pairs
        identity = np.eye(lap_size, dtype=np.float32)
        for tf_name in finer_tfs:
            laplacians[tf_name] = [identity.copy() for _ in ref_lap]

    # ── 4. Targets & labels ────────────────────────────────────────────────────
    print(f"\n  Computing targets & labels...")
    targets, labels = _compute_targets_from_tf(
        tf_data, pairs, primary_tf, target_pair="EURUSD"
    )

    # Align to primary-TF bar count
    n_bars  = len(features[primary_tf])
    targets = targets[:n_bars] if len(targets) >= n_bars \
              else np.pad(targets, (0, n_bars - len(targets)))
    labels  = labels[:n_bars]  if len(labels)  >= n_bars \
              else np.pad(labels,  (0, n_bars - len(labels)))

    buy  = int((labels == 2).sum())
    hold = int((labels == 1).sum())
    sell = int((labels == 0).sum())
    print(f"  Bars: {n_bars}  |  Buy: {buy}  Hold: {hold}  Sell: {sell}")

    # ── 5. Build trainer ───────────────────────────────────────────────────────
    trainer = STGNNTrainer(
        n_pairs    = n_pairs,
        n_features = N_FEATURES,
        seq_len    = seq_len,
        n_folds    = phase["n_folds"],
        epochs     = phase["epochs"],
        lr         = phase["lr"],
        patience   = 15,
        batch_size = 64,
    )

    # Warm-start: patch train_stage1 at instance level to inject prior weights
    if HAS_TORCH and phase.get("pretrained"):
        prev_state = load_weights(model_dir, phase["pretrained"])
        if prev_state is not None:
            original = STGNNTrainer.train_stage1
            patched  = _make_warmstart_stage1(original, prev_state, verbose=True)
            trainer.train_stage1 = types.MethodType(patched, trainer)

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print(f"\n  Training...")
    results = trainer.train_full(features, laplacians, targets, labels, verbose=True)

    # ── 7. Save ────────────────────────────────────────────────────────────────
    save_weights(trainer, model_dir, name)

    # ── Summary ────────────────────────────────────────────────────────────────
    fold_results = results.get("fold_results", [])
    if fold_results:
        accs    = [f["accuracy"]         for f in fold_results if "accuracy"         in f]
        mses    = [f["mse"]              for f in fold_results if "mse"              in f]
        sharpes = [f["strategy_sharpe"]  for f in fold_results if "strategy_sharpe"  in f]
        if accs:
            print(f"\n  Val accuracy:  mean={np.mean(accs):.4f}  best={max(accs):.4f}")
        if mses:
            print(f"  Val MSE:       mean={np.mean(mses):.6f}")
        if sharpes:
            print(f"  OOS Sharpe:    mean={np.mean(sharpes):.4f}  best={max(sharpes):.4f}")

    # ── 8. PBO / overfitting analysis ──────────────────────────────────────────
    pbo_result = compute_pbo(fold_results)
    print_pbo_report(pbo_result, phase_name=name)

    print(f"\n  Phase {name} complete.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progressive STGNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {p['key']:4s}  {p['name']:22s}  {p['tf']:4s}  {p['start']} -> {p['end']}"
            for p in PHASES
        ),
    )
    parser.add_argument(
        "--data-dir", default="D:/dataset-ml/DataExtractor",
        help="Root DataExtractor directory (default: D:/dataset-ml/DataExtractor)",
    )
    parser.add_argument(
        "--model-dir", default="D:/dataset-ml/models",
        help="Output directory for model weights (default: D:/dataset-ml/models)",
    )
    parser.add_argument(
        "--phases", default="1,2,3,3b",
        help="Phase keys to run, comma-separated: 1,2,3,3b  (default: all)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=43,
        help="Number of instruments from ALL_INSTRUMENTS (default: 43)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=60,
        help="STGNN sequence length in bars (default: 60)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    keys = [k.strip() for k in args.phases.split(",")]
    phases_to_run = []
    for k in keys:
        if k not in PHASE_BY_KEY:
            print(f"[error] Unknown phase key '{k}'. Valid: {list(PHASE_BY_KEY)}")
            sys.exit(1)
        phases_to_run.append(PHASE_BY_KEY[k])

    print(f"\nProgressive STGNN Training")
    print(f"  Data dir  : {data_root}")
    print(f"  Model dir : {model_dir}")
    print(f"  Phases    : {[p['name'] for p in phases_to_run]}")
    print(f"  Pairs     : {args.n_pairs}/{len(ALL_INSTRUMENTS)}")
    print(f"  Seq len   : {args.seq_len}")

    for phase in phases_to_run:
        run_phase(
            phase     = phase,
            data_root = data_root,
            model_dir = model_dir,
            n_pairs   = args.n_pairs,
            seq_len   = args.seq_len,
        )

    print("\nAll phases complete.")


if __name__ == "__main__":
    main()
