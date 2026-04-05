from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from calibration import expected_calibration_error
from export_research_stgnn_pack import FEATURE_ORDER, build_tensor_pack
from pbo_analysis import compute_pbo
from research_backtester import run_probability_backtest
from research_dataset import (
    SESSION_CODES,
    build_split_metadata,
    load_canonical_research_dataset,
    make_triple_barrier_labels,
)
from train_research_compact import RuntimeProfile, detect_runtime_profile
from universe import ALL_INSTRUMENTS, FX_PAIRS

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = object
    Dataset = object

if HAS_TORCH:
    from stgnn_model import SpatialBlock
else:
    SpatialBlock = object

VALIDITY_IDX = FEATURE_ORDER.index("validity")
FAST_TIMEFRAMES = ("M5", "M15")
SLOW_TIMEFRAMES = ("H1", "H4")
DEFAULT_SEQ_LENS = {"M5": 48, "M15": 32, "H1": 24, "H4": 16}


@dataclass(slots=True)
class DualFoldResult:
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
    n_rows: int


def _split_csv(raw: str, expected_keys: tuple[str, ...]) -> dict[str, int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} comma-separated values, got {raw!r}")
    return {key: int(value) for key, value in zip(expected_keys, values)}


def _fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> IsotonicRegression | None:
    if len(probabilities) == 0 or len(np.unique(labels)) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, labels)
    return calibrator


def _apply_isotonic(calibrator: IsotonicRegression | None, probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probabilities.astype(np.float32, copy=False)
    return calibrator.transform(probabilities).astype(np.float32, copy=False)


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


def _compute_static_laplacian(close_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    safe_close = np.maximum(np.nan_to_num(close_values.astype(np.float64, copy=False), nan=np.nan), 1e-8)
    log_close = np.log(safe_close)
    returns = np.diff(log_close, axis=0)
    valid_ret = valid_mask[1:] & valid_mask[:-1] & np.isfinite(returns)
    returns = np.where(valid_ret, returns, np.nan)

    n_nodes = close_values.shape[1]
    corr = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        corr[i, i] = 1.0
        for j in range(i + 1, n_nodes):
            joint = np.isfinite(returns[:, i]) & np.isfinite(returns[:, j])
            if joint.sum() < 16:
                value = 0.0
            else:
                ri = returns[joint, i]
                rj = returns[joint, j]
                std_i = float(np.std(ri))
                std_j = float(np.std(rj))
                if std_i < 1e-8 or std_j < 1e-8:
                    value = 0.0
                else:
                    value = float(np.corrcoef(ri, rj)[0, 1])
            corr[i, j] = corr[j, i] = value

    adjacency = np.abs(np.nan_to_num(corr, nan=0.0))
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    norm = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    laplacian = np.eye(n_nodes, dtype=np.float32) - norm.astype(np.float32)
    return laplacian


def _compute_global_regime_codes(dataset) -> np.ndarray:
    close_stack = []
    valid_stack = []
    for symbol in FX_PAIRS:
        if symbol not in dataset.tf_data[dataset.base_timeframe]:
            continue
        frame = dataset.tf_data[dataset.base_timeframe][symbol]
        close_stack.append(frame["c"].astype(np.float64, copy=False))
        valid_stack.append(frame["real"].astype(np.bool_, copy=False))

    if not close_stack:
        return np.ones(dataset.n_bars, dtype=np.float32)

    close_matrix = np.column_stack(close_stack)
    valid_matrix = np.column_stack(valid_stack)
    safe_close = np.maximum(np.nan_to_num(close_matrix, nan=np.nan), 1e-8)
    log_close = np.log(safe_close)
    returns = np.zeros_like(log_close, dtype=np.float64)
    returns[1:] = np.diff(log_close, axis=0)
    valid_ret = valid_matrix.copy()
    valid_ret[1:] &= valid_matrix[:-1]
    returns = np.where(valid_ret, returns, np.nan)

    cross_asset_stress = np.nanmedian(np.abs(returns), axis=1)
    finite = np.isfinite(cross_asset_stress)
    if not finite.any():
        return np.ones(dataset.n_bars, dtype=np.float32)

    q50, q75, q90 = np.quantile(cross_asset_stress[finite], [0.5, 0.75, 0.9])
    regime = np.ones(dataset.n_bars, dtype=np.float32)
    regime[cross_asset_stress >= q50] = 2.0
    regime[cross_asset_stress >= q75] = 3.0
    regime[cross_asset_stress >= q90] = 4.0
    regime[~finite] = 1.0
    return regime.astype(np.float32, copy=False)


def build_panel_targets(dataset) -> dict[str, np.ndarray]:
    n_bars = dataset.n_bars
    n_pairs = len(FX_PAIRS)
    labels = np.zeros((n_bars, n_pairs), dtype=np.float32)
    valid = np.zeros((n_bars, n_pairs), dtype=np.bool_)
    forward_returns = np.zeros((n_bars, n_pairs), dtype=np.float32)

    regime_codes = _compute_global_regime_codes(dataset)

    for pair_idx, pair in enumerate(FX_PAIRS):
        if pair not in dataset.tf_data[dataset.base_timeframe]:
            continue

        frame = dataset.tf_data[dataset.base_timeframe][pair]
        pair_labels, pair_valid = make_triple_barrier_labels(
            frame["c"],
            frame["h"],
            frame["l"],
            frame["sp"],
            pair,
            binary=True,
        )
        pair_real = frame["real"].astype(np.bool_, copy=False)
        pair_limit = min(dataset.n_bars, len(pair_valid), len(pair_real), max(len(frame["c"]) - 1, 0))
        if pair_limit <= 0:
            continue

        pair_mask = np.zeros(dataset.n_bars, dtype=np.bool_)
        if dataset.n_bars > 38:
            pair_mask[32: dataset.n_bars - 6] = True
        pair_mask[pair_limit:] = False
        pair_mask[:pair_limit] &= pair_valid[:pair_limit]
        pair_mask[:pair_limit] &= pair_real[:pair_limit]

        real_history = np.cumsum(pair_real[:pair_limit].astype(np.int32))
        pair_mask[:pair_limit] &= real_history >= 32

        labels[:pair_limit, pair_idx] = pair_labels[:pair_limit].astype(np.float32, copy=False)
        valid[:pair_limit, pair_idx] = pair_mask[:pair_limit]

        close = np.maximum(frame["c"].astype(np.float64), 1e-10)
        forward_returns[:pair_limit, pair_idx] = np.log(close[1: pair_limit + 1] / close[:pair_limit]).astype(np.float32)

    return {
        "labels": labels,
        "valid": valid,
        "forward_returns": forward_returns,
        "regime_codes": regime_codes,
    }


def build_overlap_day_splits(
    base_timestamps: np.ndarray,
    quarter_ids: np.ndarray,
    session_codes: np.ndarray,
    valid_panel: np.ndarray,
    outer_holdout_quarters: tuple[str, ...],
    overlap_fold_days: int,
    min_train_blocks: int,
    purge_bars: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    base_ts = base_timestamps.astype("datetime64[ns]")
    base_dates = base_ts.astype("datetime64[D]")
    eligible_base = valid_panel.any(axis=1)
    holdout_mask = np.isin(quarter_ids, list(outer_holdout_quarters)) & eligible_base
    inner_mask = (~np.isin(quarter_ids, list(outer_holdout_quarters))) & eligible_base

    overlap_days = np.unique(base_dates[inner_mask & (session_codes == SESSION_CODES["overlap"])])
    if len(overlap_days) == 0:
        overlap_days = np.unique(base_dates[inner_mask])

    blocks = []
    for start in range(0, len(overlap_days), overlap_fold_days):
        chunk = overlap_days[start: start + overlap_fold_days]
        if len(chunk) == 0:
            continue
        blocks.append(chunk)

    if len(blocks) <= min_train_blocks:
        return [], holdout_mask

    unique_ts = np.unique(base_ts[inner_mask])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block_idx in range(min_train_blocks, len(blocks)):
        train_days = np.concatenate(blocks[:block_idx])
        val_days = blocks[block_idx]

        train_mask = inner_mask & np.isin(base_dates, train_days)
        val_mask = inner_mask & np.isin(base_dates, val_days)
        if not train_mask.any() or not val_mask.any():
            continue

        if purge_bars > 0:
            val_start = base_ts[val_mask].min()
            val_pos = int(np.searchsorted(unique_ts, val_start, side="left"))
            purge_ts = unique_ts[max(0, val_pos - purge_bars):val_pos]
            if len(purge_ts) > 0:
                train_mask &= ~np.isin(base_ts, purge_ts)
        if train_mask.any() and val_mask.any():
            splits.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))

    return splits, holdout_mask


class PanelSequenceDataset(Dataset):
    def __init__(
        self,
        pack: dict[str, np.ndarray],
        tf_index_for_base: dict[str, np.ndarray],
        base_indices: np.ndarray,
        labels: np.ndarray,
        valid: np.ndarray,
        seq_lens: dict[str, int],
    ):
        self.pack = pack
        self.tf_index_for_base = tf_index_for_base
        self.base_indices = np.asarray(base_indices, dtype=np.int32)
        self.labels = labels
        self.valid = valid
        self.seq_lens = seq_lens

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.base_indices[idx])
        features: dict[str, torch.Tensor] = {}
        for tf_name, tensor in self.pack.items():
            end = int(self.tf_index_for_base[tf_name][base_idx])
            seq_len = int(self.seq_lens[tf_name])
            start = max(0, end + 1 - seq_len)
            chunk = tensor[start: end + 1]
            if len(chunk) < seq_len:
                pad = np.zeros((seq_len - len(chunk), tensor.shape[1], tensor.shape[2]), dtype=np.float32)
                chunk = np.concatenate([pad, chunk], axis=0)
            features[tf_name] = torch.from_numpy(np.nan_to_num(chunk, nan=0.0).astype(np.float32, copy=False))

        target = torch.from_numpy(self.labels[base_idx].astype(np.float32, copy=False))
        valid = torch.from_numpy(self.valid[base_idx].astype(np.float32, copy=False))
        return features, target, valid, base_idx


def collate_panel_sequences(batch):
    tf_names = tuple(batch[0][0].keys())
    features = {
        tf_name: torch.stack([item[0][tf_name] for item in batch], dim=0)
        for tf_name in tf_names
    }
    targets = torch.stack([item[1] for item in batch], dim=0)
    valid = torch.stack([item[2] for item in batch], dim=0)
    base_indices = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return features, targets, valid, base_indices


class TimeframeGraphEncoder(nn.Module):
    def __init__(self, n_features: int, n_tradeable: int, spatial_hidden: int, spatial_out: int, temporal_hidden: int, dropout: float):
        super().__init__()
        self.spatial = SpatialBlock(n_features, spatial_hidden, spatial_out, K=3, dropout=dropout)
        self.temporal = nn.GRU(
            input_size=(n_tradeable * spatial_out) + spatial_out,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor, tradeable_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _, _ = x.shape
        seq_inputs: list[torch.Tensor] = []
        last_tradeable = None
        for step in range(x.size(1)):
            step_x = x[:, step]
            step_valid = step_x[:, :, VALIDITY_IDX:VALIDITY_IDX + 1]
            spatial = self.spatial(step_x, laplacian)
            spatial = spatial * step_valid
            tradeable = spatial.index_select(1, tradeable_indices)
            denom = step_valid.sum(dim=1).clamp_min(1.0)
            global_ctx = spatial.sum(dim=1) / denom
            seq_inputs.append(torch.cat([tradeable.reshape(batch_size, -1), global_ctx], dim=-1))
            last_tradeable = tradeable

        seq_tensor = torch.stack(seq_inputs, dim=1)
        _, hidden = self.temporal(seq_tensor)
        return hidden[-1], last_tradeable


class BranchAggregator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

    def forward(self, timeframe_hidden: list[torch.Tensor], timeframe_pairs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.stack(timeframe_hidden, dim=1)
        _, hidden = self.fusion(seq)
        return hidden[-1], torch.cat(timeframe_pairs, dim=-1)


class DualBranchSTGNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_tradeable: int,
        tradeable_node_indices: list[int],
        spatial_hidden: int = 24,
        spatial_out: int = 12,
        temporal_hidden: int = 48,
        head_hidden: int = 64,
        pair_embedding_dim: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("tradeable_node_indices", torch.tensor(tradeable_node_indices, dtype=torch.long))
        self.register_buffer("pair_output_indices", torch.arange(n_tradeable, dtype=torch.long))
        self.fast_encoders = nn.ModuleDict({
            tf_name: TimeframeGraphEncoder(n_features, n_tradeable, spatial_hidden, spatial_out, temporal_hidden, dropout)
            for tf_name in FAST_TIMEFRAMES
        })
        self.slow_encoders = nn.ModuleDict({
            tf_name: TimeframeGraphEncoder(n_features, n_tradeable, spatial_hidden, spatial_out, temporal_hidden, dropout)
            for tf_name in SLOW_TIMEFRAMES
        })
        self.fast_branch = BranchAggregator(temporal_hidden)
        self.slow_branch = BranchAggregator(temporal_hidden)
        self.pair_embedding = nn.Embedding(n_tradeable, pair_embedding_dim)
        pair_repr_dim = spatial_out * (len(FAST_TIMEFRAMES) + len(SLOW_TIMEFRAMES))
        context_dim = temporal_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(context_dim + pair_repr_dim + pair_embedding_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, features: dict[str, torch.Tensor], laplacians: dict[str, torch.Tensor]) -> torch.Tensor:
        fast_hidden, fast_pairs = [], []
        slow_hidden, slow_pairs = [], []

        for tf_name, encoder in self.fast_encoders.items():
            hidden, pair_repr = encoder(features[tf_name], laplacians[tf_name], self.tradeable_node_indices)
            fast_hidden.append(hidden)
            fast_pairs.append(pair_repr)

        for tf_name, encoder in self.slow_encoders.items():
            hidden, pair_repr = encoder(features[tf_name], laplacians[tf_name], self.tradeable_node_indices)
            slow_hidden.append(hidden)
            slow_pairs.append(pair_repr)

        fast_ctx, fast_pair_repr = self.fast_branch(fast_hidden, fast_pairs)
        slow_ctx, slow_pair_repr = self.slow_branch(slow_hidden, slow_pairs)

        context = torch.cat([fast_ctx, slow_ctx], dim=-1)
        pair_repr = torch.cat([fast_pair_repr, slow_pair_repr], dim=-1)
        pair_embed = self.pair_embedding(self.pair_output_indices).unsqueeze(0).expand(context.size(0), -1, -1)
        context_expand = context.unsqueeze(1).expand(-1, pair_repr.size(1), -1)
        logits = self.head(torch.cat([context_expand, pair_repr, pair_embed], dim=-1)).squeeze(-1)
        return logits


def _train_epoch(model, loader, optimizer, laplacians, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for features, targets, valid, _ in loader:
        features = {tf: tensor.to(device) for tf, tensor in features.items()}
        targets = targets.to(device)
        valid = valid.to(device)

        logits = model(features, laplacians)
        loss_matrix = criterion(logits, targets)
        loss = (loss_matrix * valid).sum() / valid.sum().clamp_min(1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _collect_predictions(model, loader, laplacians, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    row_refs: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for features, targets, valid, base_idx in loader:
            features = {tf: tensor.to(device) for tf, tensor in features.items()}
            targets = targets.to(device)
            valid = valid.to(device)

            logits = model(features, laplacians)
            loss_matrix = criterion(logits, targets)
            loss = (loss_matrix * valid).sum() / valid.sum().clamp_min(1.0)
            total_loss += float(loss.item())
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = targets.cpu().numpy()
            valid_np = valid.cpu().numpy() > 0.5
            base_np = base_idx.cpu().numpy().astype(np.int32, copy=False)

            for row_idx in range(len(base_np)):
                mask = valid_np[row_idx]
                if not mask.any():
                    continue
                pair_ids = np.flatnonzero(mask).astype(np.int32, copy=False)
                refs = np.column_stack([
                    np.full(mask.sum(), base_np[row_idx], dtype=np.int32),
                    pair_ids,
                ])
                row_refs.append(refs)
                probs_parts.append(probs[row_idx, mask].astype(np.float32, copy=False))
                labels_parts.append(labels[row_idx, mask].astype(np.int32, copy=False))

    if probs_parts:
        probabilities = np.concatenate(probs_parts).astype(np.float32, copy=False)
        labels = np.concatenate(labels_parts).astype(np.int32, copy=False)
        refs = np.vstack(row_refs).astype(np.int32, copy=False)
    else:
        probabilities = np.zeros(0, dtype=np.float32)
        labels = np.zeros(0, dtype=np.int32)
        refs = np.zeros((0, 2), dtype=np.int32)

    return probabilities, labels, refs, total_loss / max(n_batches, 1)


def _build_metric_payload(
    probabilities: np.ndarray,
    labels: np.ndarray,
    refs: np.ndarray,
    panel: dict[str, np.ndarray],
    dataset,
) -> dict:
    if len(probabilities) == 0:
        return {
            "auc": 0.5,
            "log_loss": 0.6931471805599453,
            "ece": 0.0,
            "strategy_sharpe": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "confidence_hit_rate": 0.0,
            "n_rows": 0,
        }

    calibrated = np.clip(np.nan_to_num(probabilities.astype(np.float32, copy=False), nan=0.5), 1e-6, 1.0 - 1e-6)
    proba = np.column_stack([1.0 - calibrated, calibrated])
    auc = roc_auc_score(labels, calibrated) if len(np.unique(labels)) > 1 else 0.5
    ll = log_loss(labels, proba, labels=[0, 1])
    ece = expected_calibration_error(np.column_stack([1.0 - calibrated, np.zeros_like(calibrated), calibrated]), labels * 2, n_bins=10)

    base_idx = refs[:, 0]
    pair_idx = refs[:, 1]
    forward_returns = np.nan_to_num(panel["forward_returns"][base_idx, pair_idx], nan=0.0).astype(np.float32, copy=False)
    session_codes = dataset.session_codes[base_idx]
    regime_codes = panel["regime_codes"][base_idx]
    backtest = run_probability_backtest(
        p_buy=calibrated,
        p_sell=1.0 - calibrated,
        forward_returns=forward_returns,
        regime_codes=regime_codes,
        session_codes=session_codes,
        **_backtest_kwargs(),
    )
    return {
        "auc": float(auc),
        "log_loss": float(ll),
        "ece": float(ece),
        "strategy_sharpe": float(backtest.sharpe),
        "trade_count": int(backtest.trade_count),
        "win_rate": float(backtest.win_rate),
        "confidence_hit_rate": float(backtest.confidence_hit_rate),
        "n_rows": int(len(probabilities)),
    }


def _fit_fold_laplacians(pack: dict[str, np.ndarray], tf_index_for_base: dict[str, np.ndarray], train_base_idx: np.ndarray) -> dict[str, np.ndarray]:
    laplacians: dict[str, np.ndarray] = {}
    close_idx = FEATURE_ORDER.index("c")
    for tf_name, tensor in pack.items():
        tf_rows = np.unique(tf_index_for_base[tf_name][train_base_idx])
        close_values = tensor[tf_rows, :, close_idx]
        valid_values = tensor[tf_rows, :, VALIDITY_IDX] > 0.5
        laplacians[tf_name] = _compute_static_laplacian(close_values, valid_values)
    return laplacians


def _train_one_fold(
    pack: dict[str, np.ndarray],
    dataset,
    panel: dict[str, np.ndarray],
    seq_lens: dict[str, int],
    train_base_idx: np.ndarray,
    val_base_idx: np.ndarray,
    runtime: RuntimeProfile,
    args,
) -> tuple[DualFoldResult, dict[str, np.ndarray], IsotonicRegression | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PanelSequenceDataset(pack, dataset.tf_index_for_base, train_base_idx, panel["labels"], panel["valid"], seq_lens)
    val_ds = PanelSequenceDataset(pack, dataset.tf_index_for_base, val_base_idx, panel["labels"], panel["valid"], seq_lens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_panel_sequences)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_panel_sequences)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_panel_sequences)

    lap_np = _fit_fold_laplacians(pack, dataset.tf_index_for_base, train_base_idx)
    lap_tensors = {tf: torch.from_numpy(lap).to(device) for tf, lap in lap_np.items()}

    tradeable_node_indices = [ALL_INSTRUMENTS.index(pair) for pair in FX_PAIRS]
    model = DualBranchSTGNN(
        n_features=len(FEATURE_ORDER),
        n_tradeable=len(FX_PAIRS),
        tradeable_node_indices=tradeable_node_indices,
        spatial_hidden=args.spatial_hidden,
        spatial_out=args.spatial_out,
        temporal_hidden=args.temporal_hidden,
        head_hidden=args.head_hidden,
        pair_embedding_dim=args.pair_embedding_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_loss = float("inf")
    patience_left = args.patience

    for _epoch in range(args.epochs):
        _train_epoch(model, train_loader, optimizer, lap_tensors, device)
        _, _, _, val_loss = _collect_predictions(model, val_loader, lap_tensors, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_probs, train_labels, _, _ = _collect_predictions(model, train_eval_loader, lap_tensors, device)
    calibrator = _fit_isotonic(train_probs, train_labels)

    val_probs_raw, val_labels, val_refs, _ = _collect_predictions(model, val_loader, lap_tensors, device)
    val_probs = _apply_isotonic(calibrator, val_probs_raw)
    metrics = _build_metric_payload(val_probs, val_labels, val_refs, panel, dataset)
    result = DualFoldResult(
        fold=0,
        auc=metrics["auc"],
        log_loss=metrics["log_loss"],
        ece=metrics["ece"],
        strategy_sharpe=metrics["strategy_sharpe"],
        trade_count=metrics["trade_count"],
        win_rate=metrics["win_rate"],
        confidence_hit_rate=metrics["confidence_hit_rate"],
        stage1_val_loss=float(best_val_loss),
        n_bars=int(len(val_base_idx)),
        n_rows=metrics["n_rows"],
    )
    artifacts = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "laplacians": lap_np,
    }
    return result, artifacts, calibrator


def _save_artifacts(model_state: dict, calibrator: IsotonicRegression | None, laplacians: dict[str, np.ndarray], out_dir: Path, config: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "research_stgnn_dual.pt"
    iso_path = out_dir / "research_stgnn_dual_isotonic.pkl"
    meta_path = out_dir / "research_stgnn_dual_meta.json"

    torch.save({"model_state_dict": model_state, "config": config}, weights_path)
    if calibrator is not None:
        joblib.dump(calibrator, iso_path)
        iso_payload = str(iso_path)
    else:
        iso_payload = None
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_order": list(FEATURE_ORDER),
            "laplacian_timeframes": list(laplacians.keys()),
            **config,
        }, fh, indent=2)
    return {
        "weights_path": str(weights_path),
        "isotonic_path": iso_payload,
        "meta_path": str(meta_path),
    }


def run_pipeline(args) -> dict:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for train_research_stgnn_dual.py")

    runtime = detect_runtime_profile(
        io_workers_override=args.io_workers,
        catboost_threads_override=None,
        catboost_ram_gb_override=None,
    )
    print(
        f"[runtime] logical_cpus={runtime.logical_cpus} "
        f"total_memory_gb={runtime.total_memory_gb} "
        f"io_workers={runtime.io_workers}"
    )
    print(f"[stage] loading masked dataset from {args.data_dir}")
    dataset = load_canonical_research_dataset(
        args.data_dir,
        symbols=ALL_INSTRUMENTS,
        start=args.start,
        end=args.end,
        io_workers=runtime.io_workers,
        fill_policy="mask",
    )
    print("[stage] building masked tensor pack")
    pack, _ = build_tensor_pack(dataset)
    print("[stage] building target panel")
    panel = build_panel_targets(dataset)
    split_meta = build_split_metadata(dataset.quarter_ids, dataset.outer_holdout_quarters, n_inner_folds=4)

    seq_lens = DEFAULT_SEQ_LENS.copy()
    seq_lens.update(_split_csv(args.seq_lens, ("M5", "M15", "H1", "H4")))

    print(f"[stage] building overlap-day folds ({args.overlap_fold_days} days per block)")
    fold_splits, holdout_mask = build_overlap_day_splits(
        dataset.base_timestamps,
        dataset.quarter_ids,
        dataset.session_codes,
        panel["valid"],
        dataset.outer_holdout_quarters,
        overlap_fold_days=args.overlap_fold_days,
        min_train_blocks=args.min_train_blocks,
        purge_bars=args.purge_bars,
    )
    if not fold_splits:
        raise ValueError("No overlap-day folds were produced; try lowering overlap_fold_days or min_train_blocks.")

    fold_results: list[DualFoldResult] = []
    for fold_idx, (train_base_idx, val_base_idx) in enumerate(fold_splits):
        print(f"[fold {fold_idx}] train_bars={len(train_base_idx)} val_bars={len(val_base_idx)}")
        result, _, _ = _train_one_fold(
            pack=pack,
            dataset=dataset,
            panel=panel,
            seq_lens=seq_lens,
            train_base_idx=train_base_idx,
            val_base_idx=val_base_idx,
            runtime=runtime,
            args=args,
        )
        result.fold = fold_idx
        fold_results.append(result)

    pbo_report = compute_pbo([asdict(result) for result in fold_results])

    train_base_idx = np.flatnonzero((~holdout_mask) & panel["valid"].any(axis=1))
    holdout_base_idx = np.flatnonzero(holdout_mask)
    print("[stage] training final model for outer holdout")
    outer_result, final_artifacts, final_calibrator = _train_one_fold(
        pack=pack,
        dataset=dataset,
        panel=panel,
        seq_lens=seq_lens,
        train_base_idx=train_base_idx,
        val_base_idx=holdout_base_idx,
        runtime=runtime,
        args=args,
    )

    artifact_paths = _save_artifacts(
        model_state=final_artifacts["model_state"],
        calibrator=final_calibrator,
        laplacians=final_artifacts["laplacians"],
        out_dir=Path(args.model_dir),
        config={
            "seq_lens": seq_lens,
            "overlap_fold_days": args.overlap_fold_days,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "spatial_hidden": args.spatial_hidden,
            "spatial_out": args.spatial_out,
            "temporal_hidden": args.temporal_hidden,
            "head_hidden": args.head_hidden,
        },
    )

    outer_payload = {
        "quarters": list(dataset.outer_holdout_quarters),
        "n_bars": int(len(holdout_base_idx)),
        "n_rows": int(outer_result.n_rows),
        "auc": float(outer_result.auc),
        "log_loss": float(outer_result.log_loss),
        "ece": float(outer_result.ece),
        "strategy_sharpe": float(outer_result.strategy_sharpe),
        "trade_count": int(outer_result.trade_count),
        "win_rate": float(outer_result.win_rate),
        "confidence_hit_rate": float(outer_result.confidence_hit_rate),
    }

    payload = {
        "config": {
            "model_type": "dual_branch_stgnn",
            "fast_timeframes": list(FAST_TIMEFRAMES),
            "slow_timeframes": list(SLOW_TIMEFRAMES),
            "seq_lens": seq_lens,
            "overlap_fold_days": args.overlap_fold_days,
            "min_train_blocks": args.min_train_blocks,
            "purge_bars": args.purge_bars,
            "start": args.start,
            "end": args.end,
            "backtest": _backtest_kwargs(),
        },
        "runtime": asdict(runtime),
        "dataset": {
            "n_bars": int(dataset.n_bars),
            "n_output_rows": int(panel["valid"].sum()),
            "outer_holdout_quarters": list(dataset.outer_holdout_quarters),
            "split_metadata": split_meta,
            "n_overlap_folds": int(len(fold_splits)),
        },
        "artifacts": artifact_paths,
        "folds": [asdict(result) for result in fold_results],
        "outer_holdout": outer_payload,
        "pbo": pbo_report,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-branch STGNN research trainer")
    parser.add_argument("--data-dir", required=True, help="DataExtractor root")
    parser.add_argument("--output", default="data/research_stgnn_dual.json")
    parser.add_argument("--model-dir", default="models/research_stgnn_dual")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--spatial-hidden", type=int, default=24)
    parser.add_argument("--spatial-out", type=int, default=12)
    parser.add_argument("--temporal-hidden", type=int, default=48)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--pair-embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--overlap-fold-days", type=int, default=5)
    parser.add_argument("--min-train-blocks", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=6)
    parser.add_argument("--seq-lens", default="48,32,24,16", help="Comma-separated M5,M15,H1,H4 sequence lengths")
    parser.add_argument("--io-workers", type=int, help="Override dataset IO worker count")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args)
    print(f"Saved dual-branch STGNN report to {args.output}")
    print(f"Output rows={result['dataset']['n_output_rows']}  folds={len(result['folds'])}")
