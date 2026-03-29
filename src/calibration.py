"""
calibration.py -- Probability calibration metrics for Algo C2 v2

Metrics:
  - Log-loss (cross-entropy) on validation set
  - ECE  (Expected Calibration Error): how far mean confidence deviates from accuracy per bin
  - Brier score: mean squared probability error (lower = better)
  - Calibration curve: per-class confidence buckets vs actual frequency

Used by:
  - train_catboost_v2.py: per-fold calibration reporting
  - live_engine.py (future): online calibration monitoring

Class order assumed: SELL=0, HOLD=1, BUY=2
"""

from __future__ import annotations

import numpy as np

SELL_IDX = 0
HOLD_IDX = 1
BUY_IDX  = 2
CLASS_NAMES = {SELL_IDX: "SELL", HOLD_IDX: "HOLD", BUY_IDX: "BUY"}

EPS = 1e-7


def log_loss(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy loss on validation probabilities."""
    n = len(y_true)
    if n == 0:
        return 1.0
    p = np.clip(proba[np.arange(n), y_true], EPS, 1 - EPS)
    return float(-np.mean(np.log(p)))


def brier_score(proba: np.ndarray, y_true: np.ndarray, n_classes: int = 3) -> float:
    """
    Multi-class Brier score: mean squared error between probability vector and one-hot.

    Range [0, 2]; 0 = perfect, 2 = maximally wrong.
    """
    y_oh = np.eye(n_classes)[y_true]
    return float(np.mean((proba - y_oh) ** 2))


def expected_calibration_error(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE using max-probability confidence bucketing.

    For each bin b of predicted confidence:
      ECE += (|bin_b| / N) * |accuracy_b - mean_confidence_b|

    Well-calibrated model: ECE near 0.
    Overconfident model: ECE > 0 (model says 0.9 but is right only 0.6 of the time).
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct     = (predictions == y_true).astype(np.float64)
    n           = len(y_true)
    ece         = 0.0

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def calibration_curve(
    proba: np.ndarray,
    y_true: np.ndarray,
    class_idx: int,
    n_bins: int = 5,
) -> list[tuple[float, float, int]]:
    """
    Per-class calibration curve: (mean_predicted_prob, actual_freq, count) per bin.

    Bins predicted P(class) into n_bins equal-width buckets, computes the
    fraction of bars where class_idx was the true label.

    Returns list of (mean_pred, actual_freq, count) for non-empty bins.
    """
    p_cls  = proba[:, class_idx]
    actual = (y_true == class_idx).astype(np.float64)
    result = []

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (p_cls >= lo) & (p_cls < hi)
        if b == n_bins - 1:
            mask = (p_cls >= lo) & (p_cls <= hi)  # include right edge on last bin
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_pred  = float(p_cls[mask].mean())
        actual_freq = float(actual[mask].mean())
        result.append((mean_pred, actual_freq, cnt))

    return result


def calibration_report(
    proba: np.ndarray,
    y_true: np.ndarray,
    label: str = "",
    n_bins: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Compute and optionally print full calibration report.

    Returns dict with keys: log_loss, brier, ece, curves (per class).
    """
    ll    = log_loss(proba, y_true)
    bs    = brier_score(proba, y_true)
    ece   = expected_calibration_error(proba, y_true, n_bins=10)
    curves = {
        cls: calibration_curve(proba, y_true, cls, n_bins=n_bins)
        for cls in [SELL_IDX, HOLD_IDX, BUY_IDX]
    }

    if verbose:
        tag = f"  [{label}] " if label else "  "
        print(f"{tag}Calibration  val_ll={ll:.4f}  brier={bs:.4f}  ECE={ece:.4f}")
        for cls, curve in curves.items():
            if not curve:
                continue
            parts = []
            for mean_pred, actual_freq, cnt in curve:
                gap = actual_freq - mean_pred
                parts.append(f"{mean_pred:.2f}->{actual_freq:.2f}({gap:+.2f})")
            print(f"    {CLASS_NAMES[cls]:4s}: {' | '.join(parts)}")

    return {"log_loss": ll, "brier": bs, "ece": ece, "curves": curves}


def print_calibration_summary(
    fold_metrics: list[dict],
    label: str = "",
) -> None:
    """
    Print mean calibration metrics across folds.

    fold_metrics: list of dicts returned by calibration_report() per fold.
    """
    if not fold_metrics:
        return
    mean_ll    = float(np.mean([m["log_loss"] for m in fold_metrics]))
    mean_brier = float(np.mean([m["brier"]    for m in fold_metrics]))
    mean_ece   = float(np.mean([m["ece"]      for m in fold_metrics]))
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Mean calibration across {len(fold_metrics)} folds: "
          f"val_ll={mean_ll:.4f}  brier={mean_brier:.4f}  ECE={mean_ece:.4f}")
    if mean_ece > 0.05:
        print(f"  {tag}WARNING: ECE={mean_ece:.4f} > 0.05 -- model may be miscalibrated")
