from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from calibration import expected_calibration_error
from pbo_analysis import compute_pbo
from research_backtester import run_probability_backtest
from research_bridge import BridgeContextEncoder
from research_dataset import (
    CanonicalResearchDataset,
    build_split_metadata,
    load_canonical_research_dataset,
    make_triple_barrier_labels,
)
from research_features import CompactFeatureExtractor
from universe import FX_PAIRS

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


@dataclass(slots=True)
class FoldResult:
    fold: int
    auc: float
    log_loss: float
    ece: float
    strategy_sharpe: float
    trade_count: int
    confidence_hit_rate: float
    stage1_val_loss: float
    n_bars: int


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    if not HAS_CATBOOST:
        idx = np.arange(min(top_k, X.shape[1]))
        return idx, [feature_names[i] for i in idx]

    selector = CatBoostClassifier(
        loss_function="Logloss",
        iterations=100,
        depth=4,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
        auto_class_weights="Balanced",
    )
    selector.fit(X, y, verbose=False)
    importances = selector.get_feature_importance()
    selected = np.argsort(importances)[::-1][:top_k]
    selected.sort()
    return selected.astype(np.int32), [feature_names[i] for i in selected]


def build_fx_rows(
    dataset: CanonicalResearchDataset,
    use_bridge: bool = True,
    binary_labels: bool = True,
) -> dict:
    bridge_encoder = BridgeContextEncoder()
    extractor = CompactFeatureExtractor(["BTCUSD", *FX_PAIRS], bridge_encoder)
    states = extractor.build_state_series(dataset)

    rows = []
    targets = []
    pair_ids = []
    timestamps = []
    sessions = []
    regimes = []
    forward_returns = []
    quarter_ids = []
    names = None

    labels_by_pair = {}
    valid_by_pair = {}
    for pair in FX_PAIRS:
        if pair not in dataset.tf_data[dataset.base_timeframe]:
            continue
        frame = dataset.tf_data[dataset.base_timeframe][pair]
        labels, valid = make_triple_barrier_labels(
            frame["c"], frame["h"], frame["l"], frame["sp"], pair, binary=binary_labels
        )
        labels_by_pair[pair] = labels
        valid_by_pair[pair] = valid

    empty_bridge = type("EmptyBridge", (), {"features": {}})

    for bar_idx in range(32, dataset.n_bars - 6):
        math_state = states[bar_idx]
        bridge_context = bridge_encoder.encode(dataset, bar_idx) if use_bridge else empty_bridge()
        for pid, pair in enumerate(FX_PAIRS):
            if pair not in labels_by_pair:
                continue
            if bar_idx >= len(valid_by_pair[pair]):
                continue
            if not bool(valid_by_pair[pair][bar_idx]):
                continue
            frame = dataset.tf_data[dataset.base_timeframe][pair]
            if bar_idx >= len(frame["c"]) - 1:
                continue
            if not bool(frame["real"][bar_idx]):
                continue
            features = extractor.extract_pair_row(dataset, pair, bar_idx, math_state, bridge_context)
            if names is None:
                names = list(features.keys())
            rows.append([features[name] for name in names])
            targets.append(int(labels_by_pair[pair][bar_idx]))
            pair_ids.append(pid)
            timestamps.append(str(dataset.base_timestamps[bar_idx]))
            sessions.append(int(dataset.session_codes[bar_idx]))
            regimes.append(float(features["local_regime"]))
            quarter_ids.append(str(dataset.quarter_ids[bar_idx]))
            fwd = np.log(max(float(frame["c"][bar_idx + 1]), 1e-10) / max(float(frame["c"][bar_idx]), 1e-10))
            forward_returns.append(float(fwd))

    return {
        "X": np.array(rows, dtype=np.float32),
        "y": np.array(targets, dtype=np.int32),
        "pair_id": np.array(pair_ids, dtype=np.int32),
        "timestamps": np.array(timestamps, dtype=object),
        "sessions": np.array(sessions, dtype=np.int8),
        "regimes": np.array(regimes, dtype=np.float32),
        "quarter_ids": np.array(quarter_ids, dtype=object),
        "forward_returns": np.array(forward_returns, dtype=np.float32),
        "feature_names": names or [],
    }


def _fit_binary_catboost(X_train: np.ndarray, y_train: np.ndarray) -> CatBoostClassifier:
    if not HAS_CATBOOST:
        raise RuntimeError("catboost not installed")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=300,
        depth=5,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _quarter_based_splits(
    quarter_ids: np.ndarray,
    timestamps: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray | None]:
    unique_quarters = [q for q in sorted(np.unique(quarter_ids).tolist()) if q not in set(outer_holdout_quarters)]
    if len(unique_quarters) < 2:
        holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) if outer_holdout_quarters else None
        return [], holdout_mask

    unique_ts = np.array(sorted(np.unique(timestamps).tolist()), dtype=object)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_idx in range(1, len(unique_quarters)):
        train_quarters = unique_quarters[:split_idx]
        val_quarters = [unique_quarters[split_idx]]
        train_mask = np.isin(quarter_ids, train_quarters)
        val_mask = np.isin(quarter_ids, val_quarters)
        if not train_mask.any() or not val_mask.any():
            continue
        if purge_bars > 0:
            val_start = min(timestamps[val_mask])
            val_pos = int(np.searchsorted(unique_ts, val_start, side="left"))
            purge_ts = set(unique_ts[max(0, val_pos - purge_bars):val_pos].tolist())
            if purge_ts:
                train_mask &= ~np.isin(timestamps, list(purge_ts))
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) if outer_holdout_quarters else None
    return splits, holdout_mask


def walk_forward_binary(
    X: np.ndarray,
    y: np.ndarray,
    forward_returns: np.ndarray,
    sessions: np.ndarray,
    regimes: np.ndarray,
    quarter_ids: np.ndarray,
    timestamps: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    top_k: int,
    feature_names: list[str],
    purge_bars: int = 6,
) -> tuple[list[FoldResult], np.ndarray, list[str]]:
    quarter_splits, _ = _quarter_based_splits(quarter_ids, timestamps, outer_holdout_quarters, purge_bars)
    fold_results: list[FoldResult] = []
    selected_idx = None
    selected_names = feature_names

    if quarter_splits:
        split_iter = quarter_splits
    else:
        split_iter = list(TimeSeriesSplit(n_splits=4).split(X))

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
        if fold_idx == 0:
            selected_idx, selected_names = select_top_features(X[tr_idx], y[tr_idx], feature_names, top_k=top_k)
        Xtr = X[tr_idx][:, selected_idx]
        Xva = X[va_idx][:, selected_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        model = _fit_binary_catboost(Xtr, ytr)
        p_train = model.predict_proba(Xtr)[:, 1]
        p_val_raw = model.predict_proba(Xva)[:, 1]

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(p_train, ytr)
        p_val = isotonic.transform(p_val_raw)
        proba = np.column_stack([1.0 - p_val, p_val])

        auc = roc_auc_score(yva, p_val) if len(np.unique(yva)) > 1 else 0.5
        ll = log_loss(yva, proba, labels=[0, 1])
        ece = expected_calibration_error(np.column_stack([1.0 - p_val, np.zeros_like(p_val), p_val]), yva * 2, n_bins=10)
        backtest = run_probability_backtest(
            p_buy=p_val,
            p_sell=1.0 - p_val,
            forward_returns=forward_returns[va_idx],
            regime_codes=regimes[va_idx],
            session_codes=sessions[va_idx],
            entry_threshold=0.60,
            exit_threshold=0.52,
            confidence_threshold=0.12,
        )
        fold_results.append(FoldResult(
            fold=fold_idx,
            auc=float(auc),
            log_loss=float(ll),
            ece=float(ece),
            strategy_sharpe=backtest.sharpe,
            trade_count=backtest.trade_count,
            confidence_hit_rate=backtest.confidence_hit_rate,
            stage1_val_loss=float(ll),
            n_bars=len(va_idx),
        ))

    return fold_results, selected_idx, selected_names


def save_final_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    selected_idx: np.ndarray,
    selected_names: list[str],
    model_dir: str,
    config: dict,
) -> dict:
    model_path = None
    isotonic_path = None
    meta_path = None
    if not HAS_CATBOOST:
        return {"model_path": model_path, "isotonic_path": isotonic_path, "meta_path": meta_path}

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = max(100, int(len(X) * 0.85))
    split = min(split, max(len(X) - 1, 1))
    X_train = X[:split][:, selected_idx]
    y_train = y[:split]
    X_cal = X[split:][:, selected_idx]
    y_cal = y[split:]
    if len(X_cal) == 0:
        X_train = X[:, selected_idx]
        y_train = y
        X_cal = X_train
        y_cal = y_train

    model = _fit_binary_catboost(X_train, y_train)
    p_cal_raw = model.predict_proba(X_cal)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal_raw, y_cal)

    model_path = out_dir / "research_compact_fx.cbm"
    isotonic_path = out_dir / "research_compact_fx_isotonic.pkl"
    meta_path = out_dir / "research_compact_fx_meta.json"
    model.save_model(str(model_path))
    joblib.dump(isotonic, isotonic_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_names_selected": selected_names,
            "n_features": len(selected_names),
            **config,
        }, fh, indent=2)
    return {
        "model_path": str(model_path),
        "isotonic_path": str(isotonic_path),
        "meta_path": str(meta_path),
    }


def evaluate_outer_holdout(
    X: np.ndarray,
    y: np.ndarray,
    forward_returns: np.ndarray,
    sessions: np.ndarray,
    regimes: np.ndarray,
    quarter_ids: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    selected_idx: np.ndarray,
) -> dict | None:
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters))
    train_mask = ~holdout_mask
    if not holdout_mask.any() or not train_mask.any() or not HAS_CATBOOST:
        return None

    Xtr = X[train_mask][:, selected_idx]
    ytr = y[train_mask]
    Xho = X[holdout_mask][:, selected_idx]
    yho = y[holdout_mask]
    model = _fit_binary_catboost(Xtr, ytr)

    p_train = model.predict_proba(Xtr)[:, 1]
    p_holdout_raw = model.predict_proba(Xho)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_train, ytr)
    p_holdout = isotonic.transform(p_holdout_raw)
    proba = np.column_stack([1.0 - p_holdout, p_holdout])

    backtest = run_probability_backtest(
        p_buy=p_holdout,
        p_sell=1.0 - p_holdout,
        forward_returns=forward_returns[holdout_mask],
        regime_codes=regimes[holdout_mask],
        session_codes=sessions[holdout_mask],
        entry_threshold=0.60,
        exit_threshold=0.52,
        confidence_threshold=0.12,
    )
    auc = roc_auc_score(yho, p_holdout) if len(np.unique(yho)) > 1 else 0.5
    ll = log_loss(yho, proba, labels=[0, 1])
    ece = expected_calibration_error(np.column_stack([1.0 - p_holdout, np.zeros_like(p_holdout), p_holdout]), yho * 2, n_bins=10)
    return {
        "quarters": list(outer_holdout_quarters),
        "n_rows": int(holdout_mask.sum()),
        "auc": float(auc),
        "log_loss": float(ll),
        "ece": float(ece),
        "strategy_sharpe": float(backtest.sharpe),
        "trade_count": int(backtest.trade_count),
        "confidence_hit_rate": float(backtest.confidence_hit_rate),
    }


def run_pipeline(args) -> dict:
    dataset = load_canonical_research_dataset(
        args.data_dir,
        symbols=["BTCUSD", *FX_PAIRS],
        start=args.start,
        end=args.end,
    )
    split_meta = build_split_metadata(dataset.quarter_ids, dataset.outer_holdout_quarters, n_inner_folds=4)
    rows = build_fx_rows(dataset, use_bridge=args.with_bridge, binary_labels=not args.multiclass)
    if len(rows["X"]) == 0:
        raise ValueError("No FX rows were produced; check the dataset and label filters.")

    fold_results, selected_idx, selected_names = walk_forward_binary(
        rows["X"],
        rows["y"],
        rows["forward_returns"],
        rows["sessions"],
        rows["regimes"],
        rows["quarter_ids"],
        rows["timestamps"],
        dataset.outer_holdout_quarters,
        top_k=args.fx_top_k,
        feature_names=rows["feature_names"],
    )

    pbo_report = compute_pbo([asdict(fr) for fr in fold_results])
    outer_holdout = evaluate_outer_holdout(
        rows["X"],
        rows["y"],
        rows["forward_returns"],
        rows["sessions"],
        rows["regimes"],
        rows["quarter_ids"],
        dataset.outer_holdout_quarters,
        selected_idx,
    )
    artifacts = save_final_artifacts(
        rows["X"],
        rows["y"],
        selected_idx,
        selected_names,
        model_dir=args.model_dir,
        config={
            "with_bridge": args.with_bridge,
            "binary_labels": not args.multiclass,
            "confidence_threshold": 0.12,
            "entry_threshold": 0.60,
            "exit_threshold": 0.52,
        },
    )
    payload = {
        "config": {
            "with_bridge": args.with_bridge,
            "fx_top_k": args.fx_top_k,
            "binary_labels": not args.multiclass,
        },
        "dataset": {
            "n_bars": int(dataset.n_bars),
            "outer_holdout_quarters": list(dataset.outer_holdout_quarters),
            "split_metadata": split_meta,
            "n_rows": int(len(rows["X"])),
        },
        "feature_names_raw": rows["feature_names"],
        "feature_names_selected": selected_names,
        "artifacts": artifacts,
        "outer_holdout": outer_holdout,
        "folds": [asdict(fr) for fr in fold_results],
        "pbo": pbo_report,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Compact FX research pipeline")
    parser.add_argument("--data-dir", required=True, help="DataExtractor root")
    parser.add_argument("--output", default="data/research_compact_fx.json")
    parser.add_argument("--model-dir", default="models/research_compact_fx")
    parser.add_argument("--fx-top-k", type=int, default=25)
    parser.add_argument("--with-bridge", action="store_true")
    parser.add_argument("--multiclass", action="store_true", help="Keep HOLD labels instead of binary barrier-only labels")
    parser.add_argument("--start", help="Optional inclusive start timestamp/date, e.g. 2025-10-01")
    parser.add_argument("--end", help="Optional inclusive end timestamp/date, e.g. 2026-03-28")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args)
    print(f"Saved compact research report to {args.output}")
    print(f"Rows={result['dataset']['n_rows']}  selected_features={len(result['feature_names_selected'])}")
