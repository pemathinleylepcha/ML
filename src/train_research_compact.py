from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import asdict, dataclass
import os
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
    SESSION_CODES,
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
    win_rate: float
    confidence_hit_rate: float
    stage1_val_loss: float
    n_bars: int


@dataclass(slots=True)
class RuntimeProfile:
    logical_cpus: int
    total_memory_gb: float | None
    io_workers: int
    catboost_threads: int
    catboost_ram_limit: str | None


def _detect_total_memory_gb() -> float | None:
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024 ** 3), 2)
            return None

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 2)
    except Exception:
        return None


def detect_runtime_profile(
    io_workers_override: int | None = None,
    catboost_threads_override: int | None = None,
    catboost_ram_gb_override: int | None = None,
) -> RuntimeProfile:
    logical_cpus = max(1, os.cpu_count() or 1)
    total_memory_gb = _detect_total_memory_gb()
    io_workers = io_workers_override or min(12, max(4, logical_cpus // 4))
    catboost_threads = catboost_threads_override or min(logical_cpus, max(1, logical_cpus - 4))
    if catboost_ram_gb_override is not None:
        ram_gb = max(2, int(catboost_ram_gb_override))
    elif total_memory_gb is not None:
        ram_gb = max(4, int(total_memory_gb * 0.72))
    else:
        ram_gb = 8
    return RuntimeProfile(
        logical_cpus=logical_cpus,
        total_memory_gb=total_memory_gb,
        io_workers=io_workers,
        catboost_threads=catboost_threads,
        catboost_ram_limit=f"{ram_gb}gb",
    )


def _catboost_runtime_kwargs(runtime: RuntimeProfile) -> dict:
    kwargs = {
        "task_type": "CPU",
        "thread_count": runtime.catboost_threads,
        "allow_writing_files": False,
    }
    if runtime.catboost_ram_limit:
        kwargs["used_ram_limit"] = runtime.catboost_ram_limit
    return kwargs


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int,
    runtime: RuntimeProfile,
) -> tuple[np.ndarray, list[str]]:
    if not HAS_CATBOOST:
        idx = np.arange(min(top_k, X.shape[1]))
        return idx, [feature_names[i] for i in idx]

    selector_kwargs = dict(
        loss_function="Logloss",
        iterations=100,
        depth=4,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
        auto_class_weights="Balanced",
    )
    selector_kwargs.update(_catboost_runtime_kwargs(runtime))
    selector = CatBoostClassifier(**selector_kwargs)
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
    state_features = extractor.build_state_features(dataset)
    extractor.prepare_static_feature_cache(dataset, state_features)
    bridge_matrix = bridge_encoder.build_feature_matrix(dataset) if use_bridge else None
    feature_names = extractor.all_feature_names(use_bridge)

    row_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    pair_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    session_blocks: list[np.ndarray] = []
    regime_blocks: list[np.ndarray] = []
    forward_return_blocks: list[np.ndarray] = []
    quarter_blocks: list[np.ndarray] = []

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

    valid_bars = np.zeros(dataset.n_bars, dtype=np.bool_)
    if dataset.n_bars > 38:
        valid_bars[32: dataset.n_bars - 6] = True

    for pid, pair in enumerate(FX_PAIRS):
        if pair not in labels_by_pair:
            continue
        frame = dataset.tf_data[dataset.base_timeframe][pair]
        pair_mask = valid_bars.copy()
        pair_valid = valid_by_pair[pair]
        pair_real = frame["real"].astype(np.bool_, copy=False)
        pair_limit = min(dataset.n_bars, len(pair_valid), len(pair_real), max(len(frame["c"]) - 1, 0))
        if pair_limit <= 0:
            continue

        pair_mask[pair_limit:] = False
        pair_mask[:pair_limit] &= pair_valid[:pair_limit]
        pair_mask[:pair_limit] &= pair_real[:pair_limit]

        # Require a real-history warmup for late-starting pairs so early padded bars never enter training.
        real_history = np.cumsum(pair_real[:pair_limit].astype(np.int32))
        pair_mask[:pair_limit] &= real_history >= 32

        row_idx = np.flatnonzero(pair_mask)
        if len(row_idx) == 0:
            continue

        row_blocks.append(extractor.build_pair_matrix(
            dataset=dataset,
            symbol=pair,
            bar_indices=row_idx,
            state_features=state_features,
            bridge_matrix=bridge_matrix,
        ))
        target_blocks.append(labels_by_pair[pair][row_idx].astype(np.int32, copy=False))
        pair_blocks.append(np.full(len(row_idx), pid, dtype=np.int32))
        timestamp_blocks.append(dataset.base_timestamps[row_idx])
        session_blocks.append(dataset.session_codes[row_idx].astype(np.int8, copy=False))
        regime_blocks.append(state_features.regime_codes[row_idx].astype(np.float32, copy=False))
        quarter_blocks.append(dataset.quarter_ids[row_idx])

        close = np.maximum(frame["c"].astype(np.float64), 1e-10)
        forward_return_blocks.append(np.log(close[row_idx + 1] / close[row_idx]).astype(np.float32))

    if row_blocks:
        X = np.vstack(row_blocks).astype(np.float32, copy=False)
        y = np.concatenate(target_blocks).astype(np.int32, copy=False)
        pair_id = np.concatenate(pair_blocks).astype(np.int32, copy=False)
        timestamps = np.concatenate(timestamp_blocks)
        sessions = np.concatenate(session_blocks).astype(np.int8, copy=False)
        regimes = np.concatenate(regime_blocks).astype(np.float32, copy=False)
        quarter_ids = np.concatenate(quarter_blocks)
        forward_returns = np.concatenate(forward_return_blocks).astype(np.float32, copy=False)

        order = np.lexsort((pair_id, timestamps.astype("datetime64[ns]").astype(np.int64)))
        X = X[order]
        y = y[order]
        pair_id = pair_id[order]
        timestamps = timestamps[order]
        sessions = sessions[order]
        regimes = regimes[order]
        quarter_ids = quarter_ids[order]
        forward_returns = forward_returns[order]
    else:
        X = np.zeros((0, len(feature_names)), dtype=np.float32)
        y = np.zeros(0, dtype=np.int32)
        pair_id = np.zeros(0, dtype=np.int32)
        timestamps = np.zeros(0, dtype="datetime64[ns]")
        sessions = np.zeros(0, dtype=np.int8)
        regimes = np.zeros(0, dtype=np.float32)
        quarter_ids = np.zeros(0, dtype=object)
        forward_returns = np.zeros(0, dtype=np.float32)

    return {
        "X": X,
        "y": y,
        "pair_id": pair_id,
        "timestamps": timestamps,
        "sessions": sessions,
        "regimes": regimes,
        "quarter_ids": quarter_ids,
        "forward_returns": forward_returns,
        "feature_names": feature_names,
    }


def _fit_binary_catboost(X_train: np.ndarray, y_train: np.ndarray, runtime: RuntimeProfile) -> CatBoostClassifier:
    if not HAS_CATBOOST:
        raise RuntimeError("catboost not installed")

    model_kwargs = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=300,
        depth=5,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
        border_count=254,
        bootstrap_type="Bayesian",
        bagging_temperature=0.8,
    )
    model_kwargs.update(_catboost_runtime_kwargs(runtime))
    model = CatBoostClassifier(**model_kwargs)
    model.fit(X_train, y_train, verbose=False)
    return model


def _backtest_kwargs() -> dict:
    return {
        "entry_threshold": 0.60,
        "exit_threshold": 0.52,
        "confidence_threshold": 0.12,
        "persistence_threshold": 0.64,
        "tp_points": 2.0,
        "sl_points": 1.0,
        "trail_points": 1.0,
        "point_lookback": 24,
    }


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

    ts64 = timestamps.astype("datetime64[ns]")
    unique_ts = np.unique(ts64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_idx in range(1, len(unique_quarters)):
        train_quarters = unique_quarters[:split_idx]
        val_quarters = [unique_quarters[split_idx]]
        train_mask = np.isin(quarter_ids, train_quarters)
        val_mask = np.isin(quarter_ids, val_quarters)
        if not train_mask.any() or not val_mask.any():
            continue
        if purge_bars > 0:
            val_start = ts64[val_mask].min()
            val_pos = int(np.searchsorted(unique_ts, val_start, side="left"))
            purge_ts = unique_ts[max(0, val_pos - purge_bars):val_pos]
            if len(purge_ts) > 0:
                train_mask &= ~np.isin(ts64, purge_ts)
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) if outer_holdout_quarters else None
    return splits, holdout_mask


def _apply_time_purge(
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    timestamps: np.ndarray,
    purge_bars: int,
) -> np.ndarray:
    if purge_bars <= 0 or not val_mask.any():
        return train_mask

    ts64 = timestamps.astype("datetime64[ns]")
    unique_ts = np.unique(ts64)
    if len(unique_ts) == 0:
        return train_mask

    val_start = ts64[val_mask].min()
    val_pos = int(np.searchsorted(unique_ts, val_start, side="left"))
    purge_ts = unique_ts[max(0, val_pos - purge_bars):val_pos]
    if len(purge_ts) == 0:
        return train_mask
    return train_mask & ~np.isin(ts64, purge_ts)


def _overlap_day_based_splits(
    quarter_ids: np.ndarray,
    timestamps: np.ndarray,
    sessions: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    overlap_fold_days: int,
    min_train_blocks: int,
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    ts64 = timestamps.astype("datetime64[ns]")
    dates = ts64.astype("datetime64[D]")
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) if outer_holdout_quarters else np.zeros(len(ts64), dtype=np.bool_)
    inner_mask = ~holdout_mask

    overlap_days = np.unique(dates[inner_mask & (sessions == SESSION_CODES["overlap"])])
    if len(overlap_days) == 0:
        overlap_days = np.unique(dates[inner_mask])

    blocks = [overlap_days[start:start + overlap_fold_days] for start in range(0, len(overlap_days), overlap_fold_days)]
    blocks = [block for block in blocks if len(block) > 0]
    if len(blocks) <= min_train_blocks:
        return [], {
            "mode": "overlap",
            "overlap_fold_days": int(overlap_fold_days),
            "min_train_blocks": int(min_train_blocks),
            "inner_folds": [],
            "outer_holdout_quarters": list(outer_holdout_quarters),
        }

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    meta_folds: list[dict] = []
    for block_idx in range(min_train_blocks, len(blocks)):
        train_days = np.concatenate(blocks[:block_idx])
        val_days = blocks[block_idx]
        train_mask = inner_mask & np.isin(dates, train_days)
        val_mask = inner_mask & np.isin(dates, val_days)
        if not train_mask.any() or not val_mask.any():
            continue
        train_mask = _apply_time_purge(train_mask, val_mask, timestamps, purge_bars)
        if not train_mask.any():
            continue
        splits.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))
        meta_folds.append({
            "fold": len(meta_folds),
            "train_start": str(train_days.min()),
            "train_end": str(train_days.max()),
            "val_start": str(val_days.min()),
            "val_end": str(val_days.max()),
            "val_overlap_days": int(len(val_days)),
        })

    return splits, {
        "mode": "overlap",
        "overlap_fold_days": int(overlap_fold_days),
        "min_train_blocks": int(min_train_blocks),
        "inner_folds": meta_folds,
        "outer_holdout_quarters": list(outer_holdout_quarters),
    }


def _monthly_walk_forward_splits(
    quarter_ids: np.ndarray,
    timestamps: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    min_train_blocks: int,
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    ts64 = timestamps.astype("datetime64[ns]")
    months = ts64.astype("datetime64[M]")
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) if outer_holdout_quarters else np.zeros(len(ts64), dtype=np.bool_)
    inner_mask = ~holdout_mask
    unique_months = np.unique(months[inner_mask])
    if len(unique_months) <= min_train_blocks:
        return [], {
            "mode": "month",
            "min_train_blocks": int(min_train_blocks),
            "inner_folds": [],
            "outer_holdout_quarters": list(outer_holdout_quarters),
        }

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    meta_folds: list[dict] = []
    for month_idx in range(min_train_blocks, len(unique_months)):
        train_months = unique_months[:month_idx]
        val_month = unique_months[month_idx]
        train_mask = inner_mask & np.isin(months, train_months)
        val_mask = inner_mask & (months == val_month)
        if not train_mask.any() or not val_mask.any():
            continue
        train_mask = _apply_time_purge(train_mask, val_mask, timestamps, purge_bars)
        if not train_mask.any():
            continue
        splits.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))
        meta_folds.append({
            "fold": len(meta_folds),
            "train_months": [np.datetime_as_string(month, unit="M") for month in train_months],
            "val_months": [np.datetime_as_string(val_month, unit="M")],
        })

    return splits, {
        "mode": "month",
        "min_train_blocks": int(min_train_blocks),
        "inner_folds": meta_folds,
        "outer_holdout_quarters": list(outer_holdout_quarters),
    }


def _build_split_plan(
    quarter_ids: np.ndarray,
    timestamps: np.ndarray,
    sessions: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    split_mode: str,
    overlap_fold_days: int,
    min_train_blocks: int,
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    if split_mode == "quarter":
        splits, _ = _quarter_based_splits(quarter_ids, timestamps, outer_holdout_quarters, purge_bars)
        meta = build_split_metadata(quarter_ids, outer_holdout_quarters, n_inner_folds=4)
        meta["mode"] = "quarter"
        return splits, meta
    if split_mode == "overlap":
        return _overlap_day_based_splits(
            quarter_ids=quarter_ids,
            timestamps=timestamps,
            sessions=sessions,
            outer_holdout_quarters=outer_holdout_quarters,
            overlap_fold_days=overlap_fold_days,
            min_train_blocks=min_train_blocks,
            purge_bars=purge_bars,
        )
    if split_mode == "month":
        return _monthly_walk_forward_splits(
            quarter_ids=quarter_ids,
            timestamps=timestamps,
            outer_holdout_quarters=outer_holdout_quarters,
            min_train_blocks=min_train_blocks,
            purge_bars=purge_bars,
        )
    raise ValueError(f"Unsupported split_mode={split_mode!r}")


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
    runtime: RuntimeProfile,
    split_mode: str = "overlap",
    overlap_fold_days: int = 10,
    min_train_blocks: int = 2,
    purge_bars: int = 6,
) -> tuple[list[FoldResult], np.ndarray, list[str], dict]:
    split_plan, split_meta = _build_split_plan(
        quarter_ids=quarter_ids,
        timestamps=timestamps,
        sessions=sessions,
        outer_holdout_quarters=outer_holdout_quarters,
        split_mode=split_mode,
        overlap_fold_days=overlap_fold_days,
        min_train_blocks=min_train_blocks,
        purge_bars=purge_bars,
    )
    fold_results: list[FoldResult] = []
    selected_idx = None
    selected_names = feature_names

    if split_plan:
        split_iter = split_plan
    else:
        split_iter = list(TimeSeriesSplit(n_splits=4).split(X))
        split_meta = {
            "mode": "timeseriessplit_fallback",
            "inner_folds": [{"fold": fold_idx} for fold_idx, _ in enumerate(split_iter)],
            "outer_holdout_quarters": list(outer_holdout_quarters),
        }

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
        if fold_idx == 0:
            selected_idx, selected_names = select_top_features(
                X[tr_idx],
                y[tr_idx],
                feature_names,
                top_k=top_k,
                runtime=runtime,
            )
        Xtr = X[tr_idx][:, selected_idx]
        Xva = X[va_idx][:, selected_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        model = _fit_binary_catboost(Xtr, ytr, runtime=runtime)
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
            **_backtest_kwargs(),
        )
        fold_results.append(FoldResult(
            fold=fold_idx,
            auc=float(auc),
            log_loss=float(ll),
            ece=float(ece),
            strategy_sharpe=backtest.sharpe,
            trade_count=backtest.trade_count,
            win_rate=backtest.win_rate,
            confidence_hit_rate=backtest.confidence_hit_rate,
            stage1_val_loss=float(ll),
            n_bars=len(va_idx),
        ))

    return fold_results, selected_idx, selected_names, split_meta


def save_final_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    selected_idx: np.ndarray,
    selected_names: list[str],
    model_dir: str,
    config: dict,
    runtime: RuntimeProfile,
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

    model = _fit_binary_catboost(X_train, y_train, runtime=runtime)
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
    runtime: RuntimeProfile,
) -> dict | None:
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters))
    train_mask = ~holdout_mask
    if not holdout_mask.any() or not train_mask.any() or not HAS_CATBOOST:
        return None

    Xtr = X[train_mask][:, selected_idx]
    ytr = y[train_mask]
    Xho = X[holdout_mask][:, selected_idx]
    yho = y[holdout_mask]
    model = _fit_binary_catboost(Xtr, ytr, runtime=runtime)

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
        **_backtest_kwargs(),
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
        "win_rate": float(backtest.win_rate),
        "confidence_hit_rate": float(backtest.confidence_hit_rate),
    }


def run_pipeline(args) -> dict:
    runtime = detect_runtime_profile(
        io_workers_override=args.io_workers,
        catboost_threads_override=args.catboost_threads,
        catboost_ram_gb_override=args.catboost_ram_gb,
    )
    print(
        f"[runtime] logical_cpus={runtime.logical_cpus} "
        f"total_memory_gb={runtime.total_memory_gb} "
        f"io_workers={runtime.io_workers} "
        f"catboost_threads={runtime.catboost_threads} "
        f"catboost_ram_limit={runtime.catboost_ram_limit}"
    )
    print(f"[stage] loading dataset from {args.data_dir}")
    dataset = load_canonical_research_dataset(
        args.data_dir,
        symbols=["BTCUSD", *FX_PAIRS],
        start=args.start,
        end=args.end,
        io_workers=runtime.io_workers,
    )
    print(f"[stage] building pooled FX rows (bridge={args.with_bridge})")
    rows = build_fx_rows(dataset, use_bridge=args.with_bridge, binary_labels=not args.multiclass)
    if len(rows["X"]) == 0:
        raise ValueError("No FX rows were produced; check the dataset and label filters.")

    print(f"[stage] walk-forward cv on {len(rows['X'])} rows")
    fold_results, selected_idx, selected_names, split_meta = walk_forward_binary(
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
        runtime=runtime,
        split_mode=args.split_mode,
        overlap_fold_days=args.overlap_fold_days,
        min_train_blocks=args.min_train_blocks,
        purge_bars=args.purge_bars,
    )

    pbo_report = compute_pbo([asdict(fr) for fr in fold_results])
    print("[stage] evaluating outer holdout")
    outer_holdout = evaluate_outer_holdout(
        rows["X"],
        rows["y"],
        rows["forward_returns"],
        rows["sessions"],
        rows["regimes"],
        rows["quarter_ids"],
        dataset.outer_holdout_quarters,
        selected_idx,
        runtime=runtime,
    )
    print(f"[stage] saving final artifacts to {args.model_dir}")
    artifacts = save_final_artifacts(
        rows["X"],
        rows["y"],
        selected_idx,
        selected_names,
        model_dir=args.model_dir,
        config={
            "with_bridge": args.with_bridge,
            "binary_labels": not args.multiclass,
            "split_mode": args.split_mode,
            "overlap_fold_days": args.overlap_fold_days,
            "min_train_blocks": args.min_train_blocks,
            "purge_bars": args.purge_bars,
            **_backtest_kwargs(),
        },
        runtime=runtime,
    )
    payload = {
        "config": {
            "with_bridge": args.with_bridge,
            "fx_top_k": args.fx_top_k,
            "binary_labels": not args.multiclass,
            "split_mode": args.split_mode,
            "overlap_fold_days": args.overlap_fold_days,
            "min_train_blocks": args.min_train_blocks,
            "purge_bars": args.purge_bars,
            "backtest": _backtest_kwargs(),
        },
        "runtime": asdict(runtime),
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
    parser.add_argument("--split-mode", choices=("quarter", "overlap", "month"), default="overlap")
    parser.add_argument("--overlap-fold-days", type=int, default=10)
    parser.add_argument("--min-train-blocks", type=int, default=2)
    parser.add_argument("--purge-bars", type=int, default=6)
    parser.add_argument("--io-workers", type=int, help="Override auto IO worker count")
    parser.add_argument("--catboost-threads", type=int, help="Override CatBoost thread count")
    parser.add_argument("--catboost-ram-gb", type=int, help="Override CatBoost RAM limit in GB")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args)
    print(f"Saved compact research report to {args.output}")
    print(f"Rows={result['dataset']['n_rows']}  selected_features={len(result['feature_names_selected'])}")
