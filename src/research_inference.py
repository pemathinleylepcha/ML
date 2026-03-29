from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

try:
    from catboost import CatBoostClassifier
except Exception as exc:
    raise RuntimeError("catboost is required for research_inference") from exc


class CompactFXInference:
    def __init__(self, model_dir: str):
        model_root = Path(model_dir)
        meta_path = model_root / "research_compact_fx_meta.json"
        model_path = model_root / "research_compact_fx.cbm"
        iso_path = model_root / "research_compact_fx_isotonic.pkl"

        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.feature_names = list(self.meta["feature_names_selected"])

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        self.calibrator = joblib.load(iso_path)

        model_features = list(self.model.feature_names_)
        if model_features and all(name.isdigit() for name in model_features):
            if len(model_features) != len(self.feature_names):
                raise ValueError("Saved model feature count does not match metadata feature count")
        elif model_features != self.feature_names:
            raise ValueError("Saved model feature names do not match metadata feature names")

    def predict_proba(self, feature_rows: np.ndarray) -> np.ndarray:
        raw_buy = self.model.predict_proba(feature_rows)[:, 1]
        buy = self.calibrator.transform(raw_buy)
        return np.column_stack([1.0 - buy, buy])
