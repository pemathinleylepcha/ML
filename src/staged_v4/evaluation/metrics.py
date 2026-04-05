from __future__ import annotations

import math

import numpy as np

from pbo_analysis import compute_pbo


def binary_log_loss(prob_buy: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    mask = np.ones(len(prob_buy), dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    p = np.clip(prob_buy[mask], 1e-6, 1.0 - 1e-6)
    y = y_true[mask].astype(np.float64)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def binary_brier(prob_buy: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    mask = np.ones(len(prob_buy), dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    y = y_true[mask].astype(np.float64)
    p = prob_buy[mask].astype(np.float64)
    return float(np.mean((p - y) ** 2))


def binary_ece(prob_buy: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray | None = None, n_bins: int = 10) -> float:
    mask = np.ones(len(prob_buy), dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    p = prob_buy[mask]
    y = y_true[mask]
    pred = (p >= 0.5).astype(np.int32)
    correct = (pred == y).astype(np.float64)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        bin_mask = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if bin_mask.sum() == 0:
            continue
        acc = float(correct[bin_mask].mean())
        conf = float(np.maximum(p[bin_mask], 1.0 - p[bin_mask]).mean())
        ece += (bin_mask.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def binary_accuracy(prob_buy: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    mask = np.ones(len(prob_buy), dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    pred = (prob_buy[mask] >= 0.5).astype(np.int32)
    return float((pred == y_true[mask]).mean())


def binary_auc(prob_buy: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    mask = np.ones(len(prob_buy), dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.5
    scores = prob_buy[mask].astype(np.float64)
    labels = y_true[mask].astype(np.int32)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def summarize_fold_metrics(fold_results: list[dict]) -> dict:
    pbo = compute_pbo(fold_results)
    if not fold_results:
        return {"folds": 0, "pbo": pbo}
    summary = {
        "folds": len(fold_results),
        "mean_auc": float(np.mean([fr.get("auc", 0.5) for fr in fold_results])),
        "mean_log_loss": float(np.mean([fr.get("log_loss", 0.0) for fr in fold_results])),
        "mean_ece": float(np.mean([fr.get("ece", 0.0) for fr in fold_results])),
        "mean_directional_accuracy": float(np.mean([fr.get("directional_accuracy", 0.0) for fr in fold_results])),
        "mean_sharpe": float(np.mean([fr.get("strategy_sharpe", 0.0) for fr in fold_results])),
        "worst_fold_sharpe": float(np.min([fr.get("strategy_sharpe", 0.0) for fr in fold_results])),
        "fold_dispersion": float(np.std([fr.get("strategy_sharpe", 0.0) for fr in fold_results])),
        "mean_win_rate": float(np.mean([fr.get("win_rate", 0.0) for fr in fold_results])),
        "mean_trade_count": float(np.mean([fr.get("trade_count", 0.0) for fr in fold_results])),
        "pbo": pbo,
    }
    return summary
