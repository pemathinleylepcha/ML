from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import SubnetBatch, SubnetState, SystemOutput
from research_dataset import SESSION_CODES


OPEN_IDX = 0
HIGH_IDX = 1
LOW_IDX = 2
CLOSE_IDX = 3
SPREAD_IDX = 4
TICK_IDX = 5
REGIME_IDX = 9
EPS = 1e-6
TRADE_SESSION_CODES = (
    SESSION_CODES["london"],
    SESSION_CODES["newyork"],
    SESSION_CODES["overlap"],
)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def torch_sigmoid_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return torch.sigmoid(value).detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))


@dataclass
class MetaFeatureMatrix:
    X: np.ndarray
    feature_names: list[str]
    references: list[dict[str, Any]]
    y: np.ndarray | None = None


CORE_META_FEATURES = {
    "macro_prob",
    "micro_prob",
    "macro_entry_prob",
    "micro_entry_prob",
    "macro_confidence",
    "micro_confidence",
    "trade_session",
    "open_ready",
    "open_progress",
    "opening_range_width",
    "range_position",
    "breakout_signed",
    "momentum_score",
    "momentum_alignment",
    "volatility_ratio",
    "spread_stress",
    "tick_shock",
    "regime_signal",
    "lap_neighbor_momentum",
    "lap_momentum_residual",
    "lap_neighbor_volatility",
    "lap_vol_laggard",
    "lap_laggard_score",
    "lap_relation_strength",
}


class ScalpContextLayer:
    feature_names = [
        "trade_session",
        "open_ready",
        "open_progress",
        "opening_range_width",
        "range_position",
        "breakout_signed",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "momentum_score",
        "momentum_alignment",
        "volatility_6",
        "volatility_12",
        "volatility_ratio",
        "spread_stress",
        "tick_shock",
        "regime_signal",
    ]

    def __init__(
        self,
        opening_range_bars: int = 3,
        trade_session_codes: tuple[int, ...] = TRADE_SESSION_CODES,
    ):
        self.opening_range_bars = max(1, int(opening_range_bars))
        self.trade_session_codes = tuple(int(code) for code in trade_session_codes)

    def _session_segment_start(self, session_codes: np.ndarray) -> int:
        start = len(session_codes) - 1
        current = int(session_codes[-1])
        while start > 0 and int(session_codes[start - 1]) == current:
            start -= 1
        return start

    def _log_return(self, closes: np.ndarray, steps: int) -> float:
        if len(closes) <= steps:
            return 0.0
        current = float(max(closes[-1], EPS))
        previous = float(max(closes[-steps - 1], EPS))
        return float(np.log(current / previous))

    def _realized_volatility(self, closes: np.ndarray, lookback: int) -> float:
        if len(closes) < 3:
            return 0.0
        log_close = np.log(np.maximum(closes.astype(np.float64, copy=False), EPS))
        returns = np.diff(log_close)
        if len(returns) == 0:
            return 0.0
        tail = returns[-lookback:]
        return float(np.sqrt(np.mean(np.square(tail)))) if len(tail) > 0 else 0.0

    def _tick_shock(self, ticks: np.ndarray) -> float:
        if len(ticks) < 4:
            return 0.0
        baseline = ticks[-13:-1] if len(ticks) >= 13 else ticks[:-1]
        if len(baseline) < 2:
            return 0.0
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        if std < EPS:
            return 0.0
        return float((ticks[-1] - mean) / std)

    def build_row(
        self,
        micro_features: np.ndarray,
        micro_valid_mask: np.ndarray,
        micro_session_codes: np.ndarray,
    ) -> list[float]:
        valid_seq = micro_valid_mask.astype(bool, copy=False)
        session_seq = micro_session_codes.astype(np.int32, copy=False)
        current_session = int(session_seq[-1]) if len(session_seq) > 0 else SESSION_CODES["closed"]
        trade_session = float(current_session in self.trade_session_codes)

        valid_positions = np.flatnonzero(valid_seq)
        if len(valid_positions) == 0:
            return [trade_session, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        close_values = micro_features[valid_positions, CLOSE_IDX]
        high_values = micro_features[:, HIGH_IDX]
        low_values = micro_features[:, LOW_IDX]
        spread_values = micro_features[valid_positions, SPREAD_IDX]
        tick_values = micro_features[valid_positions, TICK_IDX]
        regime_values = micro_features[valid_positions, REGIME_IDX]
        last_close = float(max(close_values[-1], EPS))

        session_start = self._session_segment_start(session_seq)
        session_positions = np.arange(session_start, len(session_seq), dtype=np.int32)
        valid_session_positions = session_positions[valid_seq[session_positions]]
        open_seen = int(len(valid_session_positions))
        open_progress = float(min(open_seen / float(self.opening_range_bars), 2.0)) if trade_session > 0.0 else 0.0
        open_ready = float(trade_session > 0.0 and open_seen >= self.opening_range_bars)

        opening_range_width = 0.0
        range_position = 0.0
        breakout_signed = 0.0
        if open_seen > 0:
            opening_positions = valid_session_positions[: self.opening_range_bars]
            or_high = float(np.max(high_values[opening_positions]))
            or_low = float(np.min(low_values[opening_positions]))
            width = max(or_high - or_low, last_close * EPS)
            opening_range_width = float(width / last_close)
            range_position = float(np.clip((last_close - or_low) / width, -3.0, 3.0))
            or_mid = 0.5 * (or_high + or_low)
            breakout_signed = float(np.clip((last_close - or_mid) / width, -4.0, 4.0))

        momentum_3 = self._log_return(close_values, 3)
        momentum_6 = self._log_return(close_values, 6)
        momentum_12 = self._log_return(close_values, 12)
        volatility_6 = self._realized_volatility(close_values, 6)
        volatility_12 = self._realized_volatility(close_values, 12)
        volatility_ratio = float(volatility_6 / max(volatility_12, EPS)) if volatility_12 > 0.0 else 0.0
        momentum_score = float(abs(momentum_3) / max(volatility_12, EPS)) if volatility_12 > 0.0 else 0.0
        momentum_signs = [np.sign(value) for value in (momentum_3, momentum_6, momentum_12) if abs(value) > EPS]
        momentum_alignment = float(abs(sum(momentum_signs)) / len(momentum_signs)) if momentum_signs else 0.0
        spread_stress = float((spread_values[-1] / last_close) / max(volatility_6, EPS)) if len(spread_values) > 0 else 0.0
        tick_shock = self._tick_shock(tick_values)
        regime_signal = float(regime_values[-1]) if len(regime_values) > 0 else 0.0

        return [
            trade_session,
            open_ready,
            open_progress,
            opening_range_width,
            range_position,
            breakout_signed,
            float(momentum_3),
            float(momentum_6),
            float(momentum_12),
            float(momentum_score),
            float(momentum_alignment),
            float(volatility_6),
            float(volatility_12),
            float(volatility_ratio),
            float(spread_stress),
            float(tick_shock),
            float(regime_signal),
        ]


class LaplacianLaggardLayer:
    feature_names = [
        "lap_neighbor_momentum",
        "lap_momentum_residual",
        "lap_neighbor_volatility",
        "lap_vol_laggard",
        "lap_laggard_score",
        "lap_relation_strength",
    ]

    def __init__(self, corr_shrinkage: float = 0.15, min_overlap: int = 6):
        self.corr_shrinkage = float(corr_shrinkage)
        self.min_overlap = int(min_overlap)

    def _build_returns(self, closes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_steps, n_nodes = closes.shape
        returns = np.full((max(0, n_steps - 1), n_nodes), np.nan, dtype=np.float32)
        ret_valid = valid[1:] & valid[:-1]
        if n_steps > 1:
            safe_prev = np.maximum(closes[:-1].astype(np.float64, copy=False), EPS)
            safe_next = np.maximum(closes[1:].astype(np.float64, copy=False), EPS)
            raw = np.log(safe_next / safe_prev).astype(np.float32)
            returns[ret_valid] = raw[ret_valid]
        return returns, ret_valid

    def _pairwise_corr(self, returns: np.ndarray, ret_valid: np.ndarray) -> np.ndarray:
        n_nodes = returns.shape[1]
        corr = np.eye(n_nodes, dtype=np.float32)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                mask = ret_valid[:, i] & ret_valid[:, j]
                if int(mask.sum()) < self.min_overlap:
                    value = 0.0
                else:
                    lhs = returns[mask, i].astype(np.float64, copy=False)
                    rhs = returns[mask, j].astype(np.float64, copy=False)
                    lhs_std = float(lhs.std(ddof=0))
                    rhs_std = float(rhs.std(ddof=0))
                    if lhs_std < EPS or rhs_std < EPS:
                        value = 0.0
                    else:
                        value = float(np.corrcoef(lhs, rhs)[0, 1])
                        if not np.isfinite(value):
                            value = 0.0
                corr[i, j] = value
                corr[j, i] = value
        corr = (1.0 - self.corr_shrinkage) * corr + self.corr_shrinkage * np.eye(n_nodes, dtype=np.float32)
        np.fill_diagonal(corr, 0.0)
        return corr

    def _latest_sum(self, values: np.ndarray, valid: np.ndarray, lookback: int) -> np.ndarray:
        out = np.zeros(values.shape[1], dtype=np.float32)
        for col in range(values.shape[1]):
            tail = values[valid[:, col], col]
            if len(tail) == 0:
                continue
            out[col] = float(np.sum(tail[-lookback:]))
        return out

    def _latest_vol(self, returns: np.ndarray, ret_valid: np.ndarray, lookback: int) -> np.ndarray:
        out = np.zeros(returns.shape[1], dtype=np.float32)
        for col in range(returns.shape[1]):
            tail = returns[ret_valid[:, col], col]
            if len(tail) <= 1:
                continue
            used = tail[-lookback:].astype(np.float64, copy=False)
            out[col] = float(np.sqrt(np.mean(np.square(used))))
        return out

    def build_batch(
        self,
        micro_node_features: np.ndarray,
        micro_valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        batch_size, _, n_nodes, _ = micro_node_features.shape
        payload = {name: np.zeros((batch_size, n_nodes), dtype=np.float32) for name in self.feature_names}

        close = micro_node_features[:, :, :, CLOSE_IDX]
        valid = micro_valid_mask.astype(bool, copy=False)

        for batch_idx in range(batch_size):
            closes = close[batch_idx]
            node_valid = valid[batch_idx]
            returns, ret_valid = self._build_returns(closes, node_valid)
            if returns.size == 0:
                continue

            corr = self._pairwise_corr(returns, ret_valid)
            abs_adj = np.abs(corr)
            degree = abs_adj.sum(axis=1)
            safe_degree = np.maximum(degree, EPS)
            relation_strength = (degree / max(1.0, float(n_nodes - 1))).astype(np.float32)

            rv6 = self._latest_vol(returns, ret_valid, lookback=6)
            rv12 = self._latest_vol(returns, ret_valid, lookback=12)
            vol_ratio = rv6 / np.maximum(rv12, EPS)

            momentum_raw = self._latest_sum(returns, ret_valid, lookback=3)
            momentum_norm = momentum_raw / np.maximum(rv12 * np.sqrt(3.0), EPS)

            neighbor_vol = (abs_adj @ vol_ratio) / safe_degree
            signed_adj = corr / safe_degree[:, None]
            neighbor_momentum = signed_adj @ momentum_norm

            momentum_residual = momentum_norm - neighbor_momentum
            vol_laggard = neighbor_vol - vol_ratio
            laggard_score = np.clip(vol_laggard, 0.0, None) * np.clip(np.abs(neighbor_momentum), 0.0, 3.0)

            payload["lap_neighbor_momentum"][batch_idx] = neighbor_momentum.astype(np.float32, copy=False)
            payload["lap_momentum_residual"][batch_idx] = momentum_residual.astype(np.float32, copy=False)
            payload["lap_neighbor_volatility"][batch_idx] = neighbor_vol.astype(np.float32, copy=False)
            payload["lap_vol_laggard"][batch_idx] = vol_laggard.astype(np.float32, copy=False)
            payload["lap_laggard_score"][batch_idx] = laggard_score.astype(np.float32, copy=False)
            payload["lap_relation_strength"][batch_idx] = relation_strength

        return payload


class MetaFeatureBuilder:
    def __init__(self, macro_timeframe: str = "M5", micro_timeframe: str = "tick", opening_range_bars: int = 3):
        self.macro_timeframe = macro_timeframe
        self.micro_timeframe = micro_timeframe
        self.scalp_layer = ScalpContextLayer(opening_range_bars=opening_range_bars)
        self.laplacian_laggard_layer = LaplacianLaggardLayer()

    def build(self, system_output: SystemOutput, fx_batch: SubnetBatch) -> MetaFeatureMatrix:
        fx_state = system_output.fx_state
        macro_state = fx_state.timeframe_states[self.macro_timeframe]
        micro_state = fx_state.timeframe_states[self.micro_timeframe]
        tradable_indices = fx_batch.tradable_indices
        micro_batch = fx_batch.timeframe_batches[self.micro_timeframe]

        macro_logits = torch.nan_to_num(macro_state.directional_logits[:, tradable_indices], nan=0.0, posinf=0.0, neginf=0.0)
        micro_logits = torch.nan_to_num(micro_state.directional_logits[:, tradable_indices], nan=0.0, posinf=0.0, neginf=0.0)
        macro_prob = torch_sigmoid_numpy(macro_logits)
        micro_prob = torch_sigmoid_numpy(micro_logits)
        macro_entry = (
            torch_sigmoid_numpy(torch.nan_to_num(macro_state.entry_logits[:, tradable_indices], nan=0.0, posinf=0.0, neginf=0.0))
            if macro_state.entry_logits is not None
            else np.zeros_like(macro_prob)
        )
        micro_entry = (
            torch_sigmoid_numpy(torch.nan_to_num(micro_state.entry_logits[:, tradable_indices], nan=0.0, posinf=0.0, neginf=0.0))
            if micro_state.entry_logits is not None
            else np.zeros_like(micro_prob)
        )

        macro_embed = np.nan_to_num(to_numpy(macro_state.node_embeddings[:, tradable_indices]), nan=0.0, posinf=0.0, neginf=0.0)
        micro_embed = np.nan_to_num(to_numpy(micro_state.node_embeddings[:, tradable_indices]), nan=0.0, posinf=0.0, neginf=0.0)
        micro_node_features = np.nan_to_num(to_numpy(micro_batch.node_features), nan=0.0, posinf=0.0, neginf=0.0)
        micro_valid_mask = to_numpy(micro_batch.valid_mask).astype(bool, copy=False)
        micro_session_codes = to_numpy(micro_batch.session_codes).astype(np.int32, copy=False)
        lap_batch = self.laplacian_laggard_layer.build_batch(
            micro_node_features[:, :, tradable_indices, :],
            micro_valid_mask[:, :, tradable_indices],
        )

        rows = []
        refs: list[dict[str, Any]] = []
        y = []
        target = fx_batch.timeframe_batches[self.macro_timeframe].target_direction
        target_valid = fx_batch.timeframe_batches[self.macro_timeframe].target_valid
        target_np = None if target is None else to_numpy(target)[:, tradable_indices]
        valid_np = None if target_valid is None else to_numpy(target_valid)[:, tradable_indices].astype(bool, copy=False)
        feature_names = [
            "macro_prob",
            "micro_prob",
            "macro_entry_prob",
            "micro_entry_prob",
            "macro_confidence",
            "micro_confidence",
        ]
        embed_dim = macro_embed.shape[-1]
        feature_names.extend([f"macro_emb_{idx}" for idx in range(embed_dim)])
        feature_names.extend([f"micro_emb_{idx}" for idx in range(embed_dim)])
        feature_names.extend(self.scalp_layer.feature_names)
        feature_names.extend(self.laplacian_laggard_layer.feature_names)

        for batch_idx in range(macro_prob.shape[0]):
            for local_idx, node_idx in enumerate(tradable_indices):
                if valid_np is not None and not bool(valid_np[batch_idx, local_idx]):
                    continue
                macro_p = float(macro_prob[batch_idx, local_idx])
                micro_p = float(micro_prob[batch_idx, local_idx])
                row = [
                    macro_p,
                    micro_p,
                    float(macro_entry[batch_idx, local_idx]),
                    float(micro_entry[batch_idx, local_idx]),
                    abs(macro_p - 0.5) * 2.0,
                    abs(micro_p - 0.5) * 2.0,
                ]
                row.extend(macro_embed[batch_idx, local_idx].astype(np.float32).tolist())
                row.extend(micro_embed[batch_idx, local_idx].astype(np.float32).tolist())
                row.extend(
                    self.scalp_layer.build_row(
                        micro_node_features[batch_idx, :, node_idx, :],
                        micro_valid_mask[batch_idx, :, node_idx],
                        micro_session_codes[batch_idx],
                    )
                )
                row.extend(float(lap_batch[name][batch_idx, local_idx]) for name in self.laplacian_laggard_layer.feature_names)
                rows.append(row)
                refs.append(
                    {
                        "base_index": None if fx_batch.base_indices is None else int(to_numpy(fx_batch.base_indices)[batch_idx]),
                        "sample_index": batch_idx,
                        "node_index": int(node_idx),
                        "symbol": fx_batch.node_names[node_idx],
                    }
                )
                if target_np is not None:
                    y.append(float(target_np[batch_idx, local_idx]))

        return MetaFeatureMatrix(
            X=np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            feature_names=feature_names,
            references=refs,
            y=None if target_np is None else np.asarray(y, dtype=np.float32),
        )


class CatBoostMetaClassifier:
    def __init__(self, **kwargs: Any):
        from catboost import CatBoostClassifier

        default_kwargs = {
            "loss_function": "Logloss",
            "allow_writing_files": False,
            "verbose": False,
            "random_seed": 42,
        }
        self.max_features = kwargs.pop("max_features", None)
        if self.max_features is not None and int(self.max_features) <= 0:
            self.max_features = None
        default_kwargs.update(kwargs)
        self.model = CatBoostClassifier(**default_kwargs)
        self.fit_mode = "unfit"
        self.fallback_probability: float | None = None
        self.selected_feature_indices: np.ndarray | None = None

    def _select_feature_indices(self, matrix: MetaFeatureMatrix, nonconstant: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.max_features is None or int(self.max_features) <= 0 or len(nonconstant) <= int(self.max_features):
            return nonconstant.astype(np.int32, copy=False)
        y_centered = y.astype(np.float64, copy=False) - float(np.mean(y))
        y_scale = float(np.sqrt(np.mean(np.square(y_centered))))
        if y_scale < EPS:
            return nonconstant[: int(self.max_features)].astype(np.int32, copy=False)
        scores: list[tuple[float, int]] = []
        for idx in nonconstant.tolist():
            col = np.nan_to_num(matrix.X[:, idx].astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            col_centered = col - float(np.mean(col))
            col_scale = float(np.sqrt(np.mean(np.square(col_centered))))
            if col_scale < EPS:
                score = 0.0
            else:
                score = float(abs(np.mean(col_centered * y_centered) / (col_scale * y_scale)))
            scores.append((score, idx))
        core = [idx for idx in nonconstant.tolist() if idx < len(matrix.feature_names) and matrix.feature_names[idx] in CORE_META_FEATURES]
        selected = list(dict.fromkeys(core))
        remaining_slots = max(0, int(self.max_features) - len(selected))
        if remaining_slots > 0:
            ranked = sorted(
                ((score, idx) for score, idx in scores if idx not in selected),
                key=lambda item: item[0],
                reverse=True,
            )
            selected.extend(idx for _, idx in ranked[:remaining_slots])
        return np.asarray(selected, dtype=np.int32)

    def fit(self, matrix: MetaFeatureMatrix) -> "CatBoostMetaClassifier":
        if matrix.y is None:
            raise ValueError("MetaFeatureMatrix.y is required for fitting")
        x = np.asarray(matrix.X, dtype=np.float32)
        y = np.asarray(matrix.y, dtype=np.float32)
        if x.size == 0 or len(y) == 0:
            self.fit_mode = "empty_fallback"
            self.fallback_probability = 0.5
            self.selected_feature_indices = None
            return self
        variances = np.nanvar(x, axis=0)
        nonconstant = np.flatnonzero(variances > 1e-12)
        self.selected_feature_indices = self._select_feature_indices(matrix, nonconstant, y)
        if len(np.unique(y)) < 2 or len(nonconstant) == 0:
            self.fit_mode = "prior_fallback"
            self.fallback_probability = float(np.clip(y.mean() if len(y) > 0 else 0.5, 1e-6, 1.0 - 1e-6))
            return self
        self.fit_mode = "catboost"
        self.fallback_probability = None
        selected = self.selected_feature_indices
        if selected is None or len(selected) == 0:
            self.fit_mode = "prior_fallback"
            self.fallback_probability = float(np.clip(y.mean() if len(y) > 0 else 0.5, 1e-6, 1.0 - 1e-6))
            return self
        self.model.fit(x[:, selected], y)
        return self

    def predict_proba(self, matrix: MetaFeatureMatrix) -> np.ndarray:
        x = np.asarray(matrix.X, dtype=np.float32)
        if self.fit_mode != "catboost":
            prob = float(self.fallback_probability if self.fallback_probability is not None else 0.5)
            return np.column_stack([
                np.full(len(x), 1.0 - prob, dtype=np.float32),
                np.full(len(x), prob, dtype=np.float32),
            ])
        selected = self.selected_feature_indices
        if selected is None or len(selected) == 0:
            prob = 0.5
            return np.column_stack([
                np.full(len(x), 1.0 - prob, dtype=np.float32),
                np.full(len(x), prob, dtype=np.float32),
            ])
        return np.asarray(self.model.predict_proba(x[:, selected]), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if self.fit_mode == "catboost":
            self.model.save_model(str(target))
            return
        payload = {
            "fit_mode": self.fit_mode,
            "fallback_probability": self.fallback_probability,
            "selected_feature_indices": None if self.selected_feature_indices is None else self.selected_feature_indices.tolist(),
        }
        target.write_text(str(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostMetaClassifier":
        from catboost import CatBoostClassifier

        instance = cls()
        payload_path = Path(path)
        text = payload_path.read_text(encoding="utf-8", errors="ignore")
        if text.startswith("{'fit_mode'") or text.startswith('{"fit_mode"'):
            payload = ast.literal_eval(text)
            instance.fit_mode = payload.get("fit_mode", "prior_fallback")
            instance.fallback_probability = payload.get("fallback_probability")
            selected = payload.get("selected_feature_indices")
            instance.selected_feature_indices = None if selected is None else np.asarray(selected, dtype=np.int32)
            return instance
        model = CatBoostClassifier()
        model.load_model(str(payload_path))
        instance.model = model
        instance.fit_mode = "catboost"
        return instance
