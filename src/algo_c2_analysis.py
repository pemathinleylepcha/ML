"""
Algo C2 -- Phase 6: Analysis Orchestrator
Runs the full pipeline: data loading -> math engine -> feature matrix ->
6 ensemble models -> outlier detection -> results export.

Usage:
    python algo_c2_analysis.py --data algo_c2_5day_data.json --output analysis_results.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from math_engine import MathEngine
from feature_engine import (
    build_feature_matrix, extract_graph_features,
    PAIRS_ALL, PIP_SIZES,
)
from market_neutral_model import (
    WalkForwardTrainer, PortfolioBacktester,
    predict_weights, weights_to_confidence, HAS_TORCH,
)
from stgnn_model import (
    MultiTimeframePreprocessor, HierarchicalSTGNN, STGNNTrainer,
    CatBoostExecutionHead, TIMEFRAME_NAMES, N_PAIRS, N_FEATURES,
    HAS_TORCH as STGNN_HAS_TORCH,
)

warnings.filterwarnings("ignore", category=UserWarning)


# -- Model Suite -------------------------------------------------------------

def build_models():
    """Build the 6-model ensemble suite with exact hyperparameters from spec."""
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", oob_score=True, n_jobs=-1, random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100, learning_rate=0.1, random_state=42,
        ),
        "BaggingTrees": BaggingClassifier(
            n_estimators=100, max_features=0.8,
            oob_score=True, n_jobs=-1, random_state=42,
        ),
    }


# -- Training Pipeline -------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, feature_cols: list[str]):
    """
    Train all 6 models using walk-forward split (80/20).
    Returns dict of model results.
    """
    df_clean = df.dropna(subset=["label"]).copy()

    # Forward-fill then zero-fill NaN features
    X = df_clean[feature_cols].ffill().fillna(0).values
    y = df_clean["label"].values.astype(int)

    # Walk-forward split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) < 20 or len(X_test) < 5:
        print("Warning: insufficient data for train/test split")
        return {}

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Label balance (train): {np.mean(y_train):.3f}")

    models = build_models()
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            # Predict probabilities for AUC
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = 0.5
            else:
                auc = 0.5

            # Feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = np.zeros(len(feature_cols))

            # OOB score if available
            oob = getattr(model, "oob_score_", None)

            results[name] = {
                "auc": round(float(auc), 4),
                "oob_score": round(float(oob), 4) if oob is not None else None,
                "importances": importances,
            }
            print(f"  {name}: AUC={auc:.4f}" +
                  (f" OOB={oob:.4f}" if oob is not None else ""))

        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results[name] = {"auc": 0.5, "importances": np.zeros(len(feature_cols))}

    return results


def aggregate_importances(results: dict, feature_cols: list[str],
                          top_n: int = 20) -> list[dict]:
    """Aggregate feature importances across all models."""
    if not results:
        return []

    all_imp = np.zeros(len(feature_cols))
    count = 0
    for name, res in results.items():
        imp = res.get("importances")
        if imp is not None and len(imp) == len(feature_cols):
            all_imp += imp
            count += 1

    if count > 0:
        all_imp /= count

    # Sort and return top N
    ranked = sorted(enumerate(all_imp), key=lambda x: x[1], reverse=True)
    top = []
    for idx, score in ranked[:top_n]:
        top.append({
            "rank": len(top) + 1,
            "feature": feature_cols[idx],
            "importance": round(float(score), 6),
        })
    return top


# -- Outlier Detection -------------------------------------------------------

def detect_outliers(X: np.ndarray, timestamps: list, contamination: float = 0.02):
    """
    Run 3-method outlier detection with consensus rule.
    Returns dict with per-method and consensus results.
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for EllipticEnvelope (needs n_features <= n_samples)
    n_components = min(20, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    results = {}

    # 1. Isolation Forest
    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso_labels = iso.fit_predict(X_scaled)
        iso_outliers = np.where(iso_labels == -1)[0]
        results["isolation_forest"] = {
            "count": int(len(iso_outliers)),
            "indices": iso_outliers.tolist(),
        }
    except Exception as e:
        print(f"  IsolationForest failed: {e}")
        iso_outliers = np.array([])
        results["isolation_forest"] = {"count": 0, "indices": []}

    # 2. Local Outlier Factor
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_labels = lof.fit_predict(X_scaled)
        lof_outliers = np.where(lof_labels == -1)[0]
        results["local_outlier_factor"] = {
            "count": int(len(lof_outliers)),
            "indices": lof_outliers.tolist(),
        }
    except Exception as e:
        print(f"  LOF failed: {e}")
        lof_outliers = np.array([])
        results["local_outlier_factor"] = {"count": 0, "indices": []}

    # 3. Elliptic Envelope (on PCA-reduced data)
    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        ee_labels = ee.fit_predict(X_pca)
        ee_outliers = np.where(ee_labels == -1)[0]
        results["elliptic_envelope"] = {
            "count": int(len(ee_outliers)),
            "indices": ee_outliers.tolist(),
        }
    except Exception as e:
        print(f"  EllipticEnvelope failed: {e}")
        ee_outliers = np.array([])
        results["elliptic_envelope"] = {"count": 0, "indices": []}

    # Consensus: flagged by >= 2 of 3
    all_indices = set()
    for method_result in results.values():
        all_indices.update(method_result["indices"])

    consensus = []
    for idx in sorted(all_indices):
        vote_count = sum(
            1 for method_result in results.values()
            if idx in method_result["indices"]
        )
        if vote_count >= 2:
            ts = timestamps[idx] if idx < len(timestamps) else "unknown"
            consensus.append({
                "index": int(idx),
                "timestamp": ts,
                "votes": int(vote_count),
            })

    results["consensus"] = {
        "count": len(consensus),
        "pct": round(len(consensus) / len(X) * 100, 2) if len(X) > 0 else 0,
        "outliers": consensus,
    }

    return results


# -- Laplacian Analysis Summary -----------------------------------------------

def run_laplacian_summary(data: dict, pairs: list[str], window: int = 60):
    """
    Run math engine over all data and return summary of last window.
    Returns eigenspectrum, spectral gap, Betti numbers, top mispricings.
    """
    n_pairs = len(pairs)
    engine = MathEngine(n_pairs=n_pairs)

    # Convert to close arrays
    close_arrays = {}
    for pair in pairs:
        if pair not in data:
            continue
        close_arrays[pair] = np.array([b["c"] for b in data[pair]], dtype=float)

    available = [p for p in pairs if p in close_arrays]
    min_bars = min(len(close_arrays[p]) for p in available)

    last_state = None
    for t in range(min_bars):
        returns = np.zeros(n_pairs)
        for i, pair in enumerate(available):
            c = close_arrays[pair]
            if t > 0 and c[t - 1] > 0:
                returns[i] = np.log(c[t] / c[t - 1])
        state = engine.update(returns)
        if state.valid:
            last_state = state

    if last_state is None:
        return {}

    # Top mispricings
    mispricings = []
    for i, pair in enumerate(available):
        res = float(last_state.residuals[i])
        if abs(res) > 0.02:
            signal = "LONG" if res > 0 else "SHORT"
        else:
            signal = "FLAT"
        mispricings.append({
            "pair": pair,
            "residual": round(res, 6),
            "signal": signal,
        })
    mispricings.sort(key=lambda x: abs(x["residual"]), reverse=True)

    return {
        "eigenspectrum": [round(float(e), 4) for e in last_state.eigenvalues[:10]],
        "spectral_gap": round(float(last_state.spectral_gap), 4),
        "beta_0": int(last_state.beta_0),
        "beta_1": int(last_state.beta_1),
        "h1_lifespan": round(float(last_state.h1_lifespan), 4),
        "regime": last_state.regime,
        "sigma": round(float(last_state.sigma), 4),
        "top_mispricings": mispricings[:10],
    }


# -- Alpha Model Training ----------------------------------------------------

def train_alpha_model(df: pd.DataFrame, data: dict, pairs: list[str],
                      feature_cols: list[str], n_folds: int = 5,
                      epochs: int = 100, model_path: str = "models/alpha_net.pt"):
    """
    Train the market-neutral AlphaNet on the feature matrix.
    Returns training results dict or None if PyTorch unavailable.
    """
    if not HAS_TORCH:
        print("  PyTorch not installed -- skipping alpha model training")
        return None

    n_features = len(feature_cols)
    trainer = WalkForwardTrainer(
        n_features=n_features, n_pairs=len(pairs),
        n_folds=n_folds, epochs=epochs,
    )

    # Prepare aligned features + forward returns
    features, forward_returns, timestamps = trainer.prepare_data(df, data, pairs)

    print(f"  Feature matrix: {features.shape}")
    print(f"  Forward returns: {forward_returns.shape}")

    # Train walk-forward
    result = trainer.train(features, forward_returns, verbose=True)

    if result["best_fold"] < 0:
        print("  No valid fold found")
        return None

    # Save model with scaler from best fold
    best_fold_data = result["fold_results"][result["best_fold"]]
    trainer.save_model(
        result["model"], model_path,
        scaler_mean=best_fold_data["scaler_mean"],
        scaler_std=best_fold_data["scaler_std"],
        pairs=pairs,
        metadata={
            "best_sharpe": result["best_sharpe"],
            "best_fold": result["best_fold"],
            "n_folds": n_folds,
            "n_features": n_features,
        },
    )

    return result


# -- Main Orchestrator -------------------------------------------------------

def train_stgnn(data: dict, pairs: list[str],
                 n_folds: int = 3, epochs: int = 50,
                 model_dir: str = "models/stgnn") -> dict:
    """
    Train the Hierarchical STGNN + CatBoost pipeline on real data.

    Steps:
        1. Resample 1-min JSON to 10 timeframes
        2. Extract 16 features per pair per bar per timeframe
        3. Compute graph Laplacians per timeframe
        4. Walk-forward train STGNN (Stage 1) + CatBoost (Stage 2)
        5. Return cross-validation metrics
    """
    if not STGNN_HAS_TORCH:
        print("  PyTorch not available -- skipping STGNN training")
        return {"error": "no_torch"}

    import torch
    from pathlib import Path as P

    print("  [1/5] Resampling to 10 timeframes...")
    preprocessor = MultiTimeframePreprocessor(pairs=pairs)
    tf_data = preprocessor.resample_from_json(data)

    print("  [2/5] Extracting features...")
    features = preprocessor.extract_features(tf_data)

    print("  [3/5] Computing graph Laplacians...")
    laplacians = preprocessor.compute_laplacians(tf_data)

    print("  [4/5] Computing targets and labels...")
    trainer = STGNNTrainer(
        n_pairs=len(pairs), n_features=N_FEATURES,
        n_folds=n_folds, epochs=epochs, patience=10,
        batch_size=16, lr=1e-3,
    )
    targets = trainer.compute_targets(tf_data, pairs)
    labels = trainer.compute_labels(targets)

    print(f"  Label distribution: sell={np.sum(labels==0)}, "
          f"hold={np.sum(labels==1)}, buy={np.sum(labels==2)}")

    print("  [5/5] Walk-forward training...")
    results = trainer.train_full(features, laplacians, targets, labels, verbose=True)

    # Save model
    model_path = P(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    if results.get("model_state"):
        torch.save(results["model_state"], model_path / "stgnn_weights.pt")
        print(f"  Model saved to {model_path / 'stgnn_weights.pt'}")
    if results.get("catboost_model"):
        cb = results["catboost_model"]
        # CatBoostExecutionHead wraps the model; access inner model
        inner = getattr(cb, 'model', cb)
        if hasattr(inner, 'save_model'):
            inner.save_model(str(model_path / "catboost_exec.cbm"))
        print(f"  CatBoost saved to {model_path / 'catboost_exec.cbm'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Algo C2 Analysis Pipeline")
    parser.add_argument("--data", required=True, help="Path to JSON data file")
    parser.add_argument("--output", default="analysis_results.json", help="Output path")
    parser.add_argument("--window", type=int, default=120, help="Feature window size")
    parser.add_argument("--step", type=int, default=15, help="Feature step size")
    parser.add_argument("--train-alpha-model", action="store_true",
                        help="Train market-neutral AlphaNet model")
    parser.add_argument("--model-path", default="models/alpha_net.pt",
                        help="Path to save/load alpha model")
    parser.add_argument("--train-stgnn", action="store_true",
                        help="Train Hierarchical STGNN + CatBoost pipeline")
    parser.add_argument("--stgnn-epochs", type=int, default=50,
                        help="STGNN training epochs per fold")
    parser.add_argument("--stgnn-folds", type=int, default=3,
                        help="STGNN walk-forward folds")
    parser.add_argument("--stgnn-model-dir", default="models/stgnn",
                        help="Directory to save STGNN model artifacts")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)

    pairs = [p for p in PAIRS_ALL if p in data]
    print(f"Found {len(pairs)} pairs")

    # Step 1: Laplacian analysis
    print("\n--- Laplacian Analysis ---")
    laplacian = run_laplacian_summary(data, pairs)
    if laplacian:
        print(f"  Spectral gap L2 = {laplacian['spectral_gap']}")
        print(f"  B0 = {laplacian['beta_0']}, B1 = {laplacian['beta_1']}")
        print(f"  Regime: {laplacian['regime']}")
        if laplacian.get("top_mispricings"):
            print(f"  Top mispricing: {laplacian['top_mispricings'][0]['pair']} "
                  f"eps={laplacian['top_mispricings'][0]['residual']}")

    # Step 2: Feature matrix
    print(f"\n--- Feature Matrix (window={args.window}, step={args.step}) ---")
    df = build_feature_matrix(data, pairs=pairs, window=args.window, step=args.step)

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    print(f"Features: {len(feature_cols)}")

    # Step 3: Model training
    print("\n--- Model Training ---")
    model_results = train_and_evaluate(df, feature_cols)

    # Step 4: Feature importances
    print("\n--- Feature Importances (Top 10) ---")
    top_features = aggregate_importances(model_results, feature_cols, top_n=20)
    for f in top_features[:10]:
        print(f"  {f['rank']:2d}. {f['feature']:30s} {f['importance']:.4f}")

    # Step 5: Outlier detection
    print("\n--- Outlier Detection ---")
    df_clean = df.dropna(subset=["label"])
    X_outlier = df_clean[feature_cols].ffill().fillna(0).values
    timestamps = df_clean["timestamp"].tolist() if "timestamp" in df_clean.columns else []
    outlier_results = detect_outliers(X_outlier, timestamps)
    c = outlier_results["consensus"]
    print(f"  Consensus outliers: {c['count']} ({c['pct']:.1f}%)")

    # Step 6: Alpha model (optional)
    alpha_result = None
    if args.train_alpha_model:
        print("\n--- Alpha Model Training ---")
        alpha_result = train_alpha_model(
            df, data, pairs, feature_cols,
            model_path=args.model_path,
        )

    # Step 6b: Portfolio backtest with alpha model
    portfolio_result = None
    if alpha_result is not None and alpha_result["best_fold"] >= 0:
        print("\n--- Portfolio Backtest (direct weight allocation) ---")
        loaded = WalkForwardTrainer.load_model(args.model_path)
        trainer_tmp = WalkForwardTrainer(
            n_features=len(feature_cols), n_pairs=len(pairs),
        )
        features, forward_returns, ts = trainer_tmp.prepare_data(df, data, pairs)

        pbt = PortfolioBacktester(
            init_capital=10000, leverage=2.0,
            transaction_cost_bps=3.0, rebalance_freq=1,
        )
        portfolio_result = pbt.run(
            loaded["model"], features, forward_returns,
            scaler_mean=loaded["scaler_mean"],
            scaler_std=loaded["scaler_std"],
            timestamps=ts, pairs=pairs,
        )
        PortfolioBacktester.print_report(portfolio_result)

    # Step 7: STGNN training (optional)
    stgnn_result = None
    if args.train_stgnn:
        print("\n--- STGNN Hierarchical Training ---")
        stgnn_result = train_stgnn(
            data, pairs,
            n_folds=args.stgnn_folds,
            epochs=args.stgnn_epochs,
            model_dir=args.stgnn_model_dir,
        )
        if stgnn_result and "cv_metrics" in stgnn_result:
            cv = stgnn_result["cv_metrics"]
            print(f"  Cross-val MAD: {cv.get('mad', 'N/A')}")
            print(f"  Cross-val MSE: {cv.get('mse', 'N/A')}")
            print(f"  Cross-val RMSE: {cv.get('rmse', 'N/A')}")
            print(f"  Cross-val R2: {cv.get('r2', 'N/A')}")

    # Step 8: Compile and export
    output = {
        "laplacian": laplacian,
        "models": {
            name: {"auc": r["auc"], "oob_score": r.get("oob_score")}
            for name, r in model_results.items()
        },
        "feature_importances": top_features,
        "outliers": {
            k: {kk: vv for kk, vv in v.items() if kk != "indices"}
            if isinstance(v, dict) else v
            for k, v in outlier_results.items()
        },
        "data_stats": {
            "pairs": len(pairs),
            "pair_names": pairs,
            "total_bars": max(len(data[p]) for p in pairs),
            "feature_matrix_shape": list(df.shape),
        },
        "alpha_model": {
            "trained": alpha_result is not None,
            "best_sharpe": alpha_result["best_sharpe"] if alpha_result else None,
            "best_fold": alpha_result["best_fold"] if alpha_result else None,
            "fold_results": [
                {k: v for k, v in fr.items() if k not in ("scaler_mean", "scaler_std")}
                for fr in alpha_result["fold_results"]
            ] if alpha_result else [],
        } if args.train_alpha_model else None,
        "portfolio_backtest": {
            "metrics": portfolio_result["metrics"],
            "pair_pnl": portfolio_result["pair_pnl"],
        } if portfolio_result else None,
        "stgnn": {
            "trained": stgnn_result is not None and "error" not in stgnn_result,
            "cv_metrics": stgnn_result.get("cv_metrics") if stgnn_result else None,
            "fold_results": [
                {k: v for k, v in fr.items()
                 if k not in ("model_state", "catboost_model")}
                for fr in stgnn_result.get("fold_results", [])
            ] if stgnn_result else [],
        } if args.train_stgnn else None,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")
    print("Done.")


if __name__ == "__main__":
    # If no args provided, run smoke test with synthetic data
    if len(sys.argv) == 1:
        print("Smoke test (no data file provided, using synthetic)...")
        import numpy as np

        rng = np.random.default_rng(42)
        n_bars = 300
        pairs = PAIRS_ALL[:5]

        data = {}
        for pair in pairs:
            base = 1.0 + rng.random() * 0.5
            closes = base + np.cumsum(rng.normal(0, 0.001, n_bars))
            data[pair] = []
            for i in range(n_bars):
                c = closes[i]
                data[pair].append({
                    "dt": f"2026-03-02 {(i // 60) % 24:02d}:{i % 60:02d}",
                    "o": round(c + rng.normal(0, 0.0003), 5),
                    "h": round(c + abs(rng.normal(0, 0.0005)), 5),
                    "l": round(c - abs(rng.normal(0, 0.0005)), 5),
                    "c": round(c, 5),
                    "sp": round(rng.uniform(0.5, 3.0), 2),
                    "tk": int(rng.integers(5, 100)),
                })

        # Run mini analysis
        laplacian = run_laplacian_summary(data, pairs)
        print(f"Laplacian: spectral_gap={laplacian.get('spectral_gap', 'N/A')}, "
              f"regime={laplacian.get('regime', 'N/A')}")

        df = build_feature_matrix(data, pairs=pairs, window=60, step=10,
                                  target_pair=pairs[0])
        feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
        print(f"Feature matrix: {df.shape}")

        model_results = train_and_evaluate(df, feature_cols)
        top = aggregate_importances(model_results, feature_cols, top_n=5)
        print(f"Top feature: {top[0]['feature'] if top else 'none'}")
        print("OK")
    else:
        main()
