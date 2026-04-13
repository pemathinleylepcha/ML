from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from staged_v5.contracts import CalibrationArtifact


def fit_platt_scaler(logits: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray | None = None) -> CalibrationArtifact:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        logits = logits[mask]
        labels = labels[mask]
    if len(logits) == 0:
        return CalibrationArtifact(coef=1.0, intercept=0.0)
    if len(np.unique(labels)) < 2:
        return CalibrationArtifact(coef=1.0, intercept=0.0)
    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(logits, labels)
    return CalibrationArtifact(coef=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))


def apply_platt_scaler(artifact: CalibrationArtifact, logits: np.ndarray) -> np.ndarray:
    return artifact.transform_logits(logits)
