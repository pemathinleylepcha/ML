"""
Algo C2 -- Market-Neutral Alpha Model
PyTorch neural network that learns dollar-neutral portfolio weights
across 35 pairs. Trained via backpropagation on differentiable Sharpe
ratio using the feature matrix from feature_engine.py.

Replaces the CatBoost confidence proxy in signal_pipeline.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as _Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    _Dataset = object  # fallback base class

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PAIRS = 35
EPS = 1e-8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_BATCH_SIZE = 32
LAMBDA_TURNOVER = 0.01
LAMBDA_CONCENTRATION = 0.005


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AlphaDataset(_Dataset):
    """Sequential dataset of (features, forward_returns) pairs."""

    def __init__(self, features: np.ndarray, forward_returns: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.R = torch.tensor(forward_returns, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.R[idx]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class AlphaNet(nn.Module if HAS_TORCH else object):
    """
    Market-neutral portfolio weight predictor.

    Input:  (batch, n_features)  -- 568 features per bar
    Output: (batch, 35)          -- dollar-neutral weights (sum ~ 0, |sum| = 1)
    """

    def __init__(self, n_features: int = 568, n_pairs: int = N_PAIRS,
                 hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        in_dim = n_features
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.1))
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout * (1 - i * 0.15)))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1], n_pairs)
        self.n_pairs = n_pairs

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns dollar-neutral weights: sum -> 0, gross exposure -> 1."""
        h = self.backbone(x)
        raw = self.head(h)  # (batch, 35)

        # Market-neutral: subtract mean so weights sum to 0
        weights = raw - raw.mean(dim=-1, keepdim=True)

        # Normalize gross exposure to 1
        gross = weights.abs().sum(dim=-1, keepdim=True) + EPS
        weights = weights / gross

        return weights


# ---------------------------------------------------------------------------
# Loss: Differentiable Sharpe + regularization
# ---------------------------------------------------------------------------

def sharpe_loss(weights, forward_returns, prev_weights=None,
                lambda_turnover=LAMBDA_TURNOVER,
                lambda_concentration=LAMBDA_CONCENTRATION,
                annualize=252.0):
    """
    Differentiable negative Sharpe ratio with turnover + concentration penalties.

    Args:
        weights: (batch, 35) dollar-neutral weights
        forward_returns: (batch, 35) next-bar log returns
        prev_weights: (batch, 35) previous bar weights for turnover calc
        lambda_turnover: penalty weight for portfolio turnover
        lambda_concentration: penalty for max single-pair weight
        annualize: annualization factor (252 for daily)
    """
    # Portfolio returns: weighted sum of pair returns
    port_ret = (weights * forward_returns).sum(dim=-1)  # (batch,)

    # Sharpe ratio (negative because we minimize)
    mean_ret = port_ret.mean()
    std_ret = port_ret.std() + EPS
    neg_sharpe = -(mean_ret / std_ret) * (annualize ** 0.5)

    # Turnover penalty
    turnover = torch.tensor(0.0)
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=-1).mean()

    # Concentration penalty (penalize max weight magnitude)
    max_weight = weights.abs().max(dim=-1).values.mean()

    loss = neg_sharpe + lambda_turnover * turnover + lambda_concentration * max_weight
    return loss


# ---------------------------------------------------------------------------
# Walk-Forward Trainer
# ---------------------------------------------------------------------------

class WalkForwardTrainer:
    """
    Walk-forward training with expanding window, early stopping,
    and fold-based evaluation.
    """

    def __init__(self, n_features: int = 568, n_pairs: int = N_PAIRS,
                 n_folds: int = 5, lr: float = DEFAULT_LR,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 epochs: int = DEFAULT_EPOCHS,
                 patience: int = DEFAULT_PATIENCE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str = None):
        self.n_features = n_features
        self.n_pairs = n_pairs
        self.n_folds = n_folds
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def prepare_data(self, feature_df, data: dict, pairs: list[str]):
        """
        Build aligned features + forward returns from feature DataFrame
        and raw OHLC data.

        Args:
            feature_df: DataFrame from build_feature_matrix() with 'timestamp' + 'label' + 568 features
            data: dict of pair -> list of bar dicts
            pairs: ordered list of pair names (must match math engine ordering)

        Returns:
            features: (N, n_features) numpy array
            forward_returns: (N, n_pairs) numpy array
            timestamps: list of bar timestamps
        """
        feature_cols = [c for c in feature_df.columns if c not in ("timestamp", "label")]
        features = feature_df[feature_cols].ffill().fillna(0).values
        timestamps = feature_df["timestamp"].tolist() if "timestamp" in feature_df.columns else []

        # Build close price arrays for all pairs
        close_arrays = {}
        for pair in pairs:
            if pair in data:
                close_arrays[pair] = np.array([b["c"] for b in data[pair]], dtype=float)

        # We need to map feature matrix rows to bar indices
        # Feature matrix has a step (e.g. 15 bars), so each row corresponds
        # to a specific bar index. We compute forward returns for the NEXT bar.
        n_rows = len(features)
        forward_returns = np.zeros((n_rows, len(pairs)), dtype=np.float64)

        # Get min bars across all pairs
        min_bars = min(len(close_arrays[p]) for p in pairs if p in close_arrays)

        # Map timestamps to bar indices (approximate: use the step pattern)
        # The feature matrix is built with a window + step, so rows correspond
        # to evenly-spaced bars. We use the timestamp to find the bar index.
        dt_to_idx = {}
        ref_pair = pairs[0] if pairs[0] in data else list(data.keys())[0]
        for idx, bar in enumerate(data[ref_pair]):
            dt_to_idx[bar["dt"]] = idx

        for row_idx in range(n_rows):
            ts = timestamps[row_idx] if row_idx < len(timestamps) else None
            bar_idx = dt_to_idx.get(ts)

            if bar_idx is not None and bar_idx + 1 < min_bars:
                for j, pair in enumerate(pairs):
                    if pair in close_arrays:
                        c = close_arrays[pair]
                        if c[bar_idx] > 0:
                            forward_returns[row_idx, j] = np.log(
                                c[bar_idx + 1] / c[bar_idx]
                            )

        return features, forward_returns, timestamps

    def train(self, features: np.ndarray, forward_returns: np.ndarray,
              verbose: bool = True) -> dict:
        """
        Walk-forward training across folds.

        Returns:
            dict with:
                'model': trained AlphaNet (best fold)
                'fold_results': list of per-fold metrics
                'best_fold': index of best fold
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")

        n_samples = len(features)
        fold_size = n_samples // (self.n_folds + 1)

        fold_results = []
        best_sharpe = -float("inf")
        best_model_state = None
        best_fold = -1

        for fold in range(self.n_folds):
            train_end = fold_size * (fold + 2)  # expanding window
            val_end = min(train_end + fold_size, n_samples)

            if val_end <= train_end or train_end < self.batch_size:
                continue

            X_train = features[:train_end]
            R_train = forward_returns[:train_end]
            X_val = features[train_end:val_end]
            R_val = forward_returns[train_end:val_end]

            if len(X_val) < 10:
                continue

            # Standardize using train stats
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + EPS
            X_train_norm = (X_train - mean) / std
            X_val_norm = (X_val - mean) / std

            # Build model
            model = AlphaNet(
                n_features=self.n_features,
                n_pairs=self.n_pairs,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=7
            )

            train_ds = AlphaDataset(X_train_norm, R_train)
            # Sequential loading — no shuffle for time series
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size,
                shuffle=False, drop_last=True
            )

            val_X_t = torch.tensor(X_val_norm, dtype=torch.float32).to(self.device)
            val_R_t = torch.tensor(R_val, dtype=torch.float32).to(self.device)

            # Training loop with early stopping
            best_val_loss = float("inf")
            patience_counter = 0
            best_epoch_state = None

            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                prev_w = None

                for batch_X, batch_R in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_R = batch_R.to(self.device)

                    weights = model(batch_X)
                    loss = sharpe_loss(weights, batch_R, prev_weights=prev_w)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    prev_w = weights.detach()

                avg_train_loss = epoch_loss / max(n_batches, 1)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_w = model(val_X_t)
                    val_loss = sharpe_loss(val_w, val_R_t).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Fold {fold}: early stop at epoch {epoch}")
                    break

            # Restore best epoch
            if best_epoch_state is not None:
                model.load_state_dict(best_epoch_state)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_w = model(val_X_t)
                port_ret = (val_w * val_R_t).sum(dim=-1)
                val_sharpe = float(
                    (port_ret.mean() / (port_ret.std() + EPS)) * (252 ** 0.5)
                )
                val_mean_ret = float(port_ret.mean())
                val_std_ret = float(port_ret.std())
                avg_turnover = float(
                    (val_w[1:] - val_w[:-1]).abs().sum(dim=-1).mean()
                ) if len(val_w) > 1 else 0.0
                max_weight = float(val_w.abs().max())

            fold_results.append({
                "fold": fold,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "val_sharpe": round(val_sharpe, 4),
                "val_mean_ret": round(val_mean_ret, 6),
                "val_std_ret": round(val_std_ret, 6),
                "avg_turnover": round(avg_turnover, 4),
                "max_weight": round(max_weight, 4),
                "best_epoch": epoch - patience_counter,
                "best_val_loss": round(best_val_loss, 6),
                "scaler_mean": mean,
                "scaler_std": std,
            })

            if verbose:
                print(f"  Fold {fold}: Sharpe={val_sharpe:.4f}, "
                      f"mean_ret={val_mean_ret:.6f}, "
                      f"turnover={avg_turnover:.4f}, "
                      f"max_w={max_weight:.4f}")

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_fold = fold

        # Build final model from best fold
        final_model = AlphaNet(
            n_features=self.n_features,
            n_pairs=self.n_pairs,
        )
        if best_model_state is not None:
            final_model.load_state_dict(best_model_state)

        if verbose and best_fold >= 0:
            print(f"\nBest fold: {best_fold}, Sharpe={best_sharpe:.4f}")

        return {
            "model": final_model,
            "fold_results": fold_results,
            "best_fold": best_fold,
            "best_sharpe": best_sharpe,
        }

    def save_model(self, model: "AlphaNet", path: str,
                   scaler_mean: np.ndarray = None, scaler_std: np.ndarray = None,
                   pairs: list[str] = None, metadata: dict = None):
        """Save model + scaler + metadata to a checkpoint."""
        if not HAS_TORCH:
            return
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "n_features": model.backbone[0].in_features,
            "n_pairs": model.n_pairs,
        }
        if scaler_mean is not None:
            checkpoint["scaler_mean"] = scaler_mean
        if scaler_std is not None:
            checkpoint["scaler_std"] = scaler_std
        if pairs is not None:
            checkpoint["pairs"] = pairs
        if metadata is not None:
            checkpoint["metadata"] = metadata

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: str, device: str = "cpu") -> dict:
        """Load model + scaler from checkpoint."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = AlphaNet(
            n_features=checkpoint["n_features"],
            n_pairs=checkpoint["n_pairs"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "model": model,
            "scaler_mean": checkpoint.get("scaler_mean"),
            "scaler_std": checkpoint.get("scaler_std"),
            "pairs": checkpoint.get("pairs"),
            "metadata": checkpoint.get("metadata"),
        }


# ---------------------------------------------------------------------------
# Inference helper (for signal_pipeline integration)
# ---------------------------------------------------------------------------

def predict_weights(model: "AlphaNet", features: np.ndarray,
                    scaler_mean: np.ndarray = None,
                    scaler_std: np.ndarray = None,
                    device: str = "cpu") -> np.ndarray:
    """
    Run inference on a single feature snapshot.

    Args:
        model: trained AlphaNet
        features: (n_features,) or (1, n_features) feature vector
        scaler_mean: training mean for normalization
        scaler_std: training std for normalization

    Returns:
        weights: (35,) dollar-neutral portfolio weights
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Normalize
    if scaler_mean is not None and scaler_std is not None:
        features = (features - scaler_mean) / (scaler_std + EPS)

    x = torch.tensor(features, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        weights = model(x)

    return weights.cpu().numpy().squeeze()


def weights_to_confidence(weights: np.ndarray, scale: float = 5.0) -> dict:
    """
    Convert model weights to confidence scores for signal_pipeline.

    Args:
        weights: (35,) dollar-neutral weights
        scale: sigmoid scaling factor

    Returns:
        dict with:
            'confidence': (35,) array in [0, 1] — gate G2 score
            'direction': (35,) array of +1/-1/0
    """
    # Confidence = sigmoid(|weight| * scale), so larger weights → higher confidence
    confidence = 1.0 / (1.0 + np.exp(-np.abs(weights) * scale))

    # Direction from weight sign
    direction = np.sign(weights)

    return {
        "confidence": confidence,
        "direction": direction.astype(int),
    }


# ---------------------------------------------------------------------------
# Portfolio-Level Backtester (direct weight allocation, no gates)
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """
    Backtest the AlphaNet model using direct portfolio allocation.

    Instead of routing through the 6-gate signal pipeline, this backtester
    applies the model's dollar-neutral weights directly each bar:
        portfolio_return_t = sum(w_i * r_i) for all 35 pairs

    This captures the model's true alpha separate from the discrete
    trade execution system.
    """

    def __init__(self, init_capital: float = 10000.0,
                 leverage: float = 1.0,
                 transaction_cost_bps: float = 2.0,
                 rebalance_freq: int = 1,
                 max_position_pct: float = 0.15):
        """
        Args:
            init_capital: starting capital in USD
            leverage: gross leverage multiplier (1.0 = unleveraged)
            transaction_cost_bps: round-trip cost in basis points
            rebalance_freq: rebalance every N bars (1 = every bar)
            max_position_pct: max weight per single pair (clamp)
        """
        self.init_capital = init_capital
        self.leverage = leverage
        self.tc_bps = transaction_cost_bps
        self.rebalance_freq = rebalance_freq
        self.max_pos_pct = max_position_pct

    def run(self, model, features: np.ndarray, forward_returns: np.ndarray,
            scaler_mean: np.ndarray = None, scaler_std: np.ndarray = None,
            timestamps: list = None, pairs: list = None) -> dict:
        """
        Run portfolio backtest bar-by-bar.

        Args:
            model: trained AlphaNet
            features: (N, n_features) feature matrix
            forward_returns: (N, n_pairs) next-bar log returns
            scaler_mean/std: feature normalization parameters
            timestamps: optional bar timestamps
            pairs: optional pair names

        Returns:
            dict with equity curve, metrics, weight history, etc.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed")

        n_bars = len(features)
        n_pairs = forward_returns.shape[1]

        # Normalize features
        if scaler_mean is not None and scaler_std is not None:
            features_norm = (features - scaler_mean) / (scaler_std + EPS)
        else:
            features_norm = features

        # Pre-compute all weights in batch
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(features_norm, dtype=torch.float32)
            all_weights = model(X_t).numpy()  # (N, n_pairs)

        # Clamp individual positions
        all_weights = np.clip(all_weights, -self.max_pos_pct, self.max_pos_pct)
        # Re-normalize to maintain dollar-neutrality after clamping
        all_weights = all_weights - all_weights.mean(axis=-1, keepdims=True)
        gross = np.abs(all_weights).sum(axis=-1, keepdims=True) + EPS
        all_weights = all_weights / gross

        # Apply leverage
        all_weights = all_weights * self.leverage

        # Bar-by-bar simulation
        capital = self.init_capital
        equity_curve = [capital]
        returns_series = []
        weight_history = []
        turnover_series = []
        prev_weights = np.zeros(n_pairs)
        long_exposure = []
        short_exposure = []
        tc_total = 0.0

        for t in range(n_bars):
            w = all_weights[t]

            # Only rebalance at frequency
            if t % self.rebalance_freq != 0 and t > 0:
                w = prev_weights

            # Transaction costs from turnover
            turnover = np.abs(w - prev_weights).sum()
            tc = turnover * (self.tc_bps / 10000.0) * capital
            tc_total += tc

            # Portfolio return (weighted sum of pair returns)
            port_ret = float(np.sum(w * forward_returns[t]))

            # Update capital
            pnl = capital * port_ret - tc
            capital += pnl

            # Record
            equity_curve.append(capital)
            returns_series.append(port_ret)
            weight_history.append(w.copy())
            turnover_series.append(turnover)
            long_exposure.append(float(w[w > 0].sum()))
            short_exposure.append(float(np.abs(w[w < 0].sum())))
            prev_weights = w.copy()

        # Compute metrics
        returns_arr = np.array(returns_series)
        equity_arr = np.array(equity_curve)

        # Sharpe ratio
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = float(
                (returns_arr.mean() / returns_arr.std()) * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation only)
        downside = returns_arr[returns_arr < 0]
        if len(downside) > 1:
            sortino = float(
                (returns_arr.mean() / downside.std()) * np.sqrt(252)
            )
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        dd = (peak - equity_arr) / (peak + EPS)
        max_dd = float(dd.max())

        # Calmar ratio
        total_return = (capital - self.init_capital) / self.init_capital
        calmar = total_return / max_dd if max_dd > 0 else 0.0

        # Hit rate (bars with positive return)
        hit_rate = float((returns_arr > 0).mean()) if len(returns_arr) > 0 else 0.0

        # Average turnover
        avg_turnover = float(np.mean(turnover_series))

        # Average gross/net exposure
        avg_long_exp = float(np.mean(long_exposure))
        avg_short_exp = float(np.mean(short_exposure))
        avg_gross_exp = avg_long_exp + avg_short_exp
        avg_net_exp = avg_long_exp - avg_short_exp

        # Per-pair attribution
        pair_pnl = {}
        weight_arr = np.array(weight_history)  # (N, n_pairs)
        for j in range(n_pairs):
            pair_ret = float(np.sum(weight_arr[:, j] * forward_returns[:, j]))
            name = pairs[j] if pairs and j < len(pairs) else f"pair_{j}"
            pair_pnl[name] = round(pair_ret * self.init_capital, 2)

        # Sort by contribution
        pair_pnl_sorted = dict(
            sorted(pair_pnl.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        metrics = {
            "init_capital": self.init_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(total_return * 100, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "hit_rate": round(hit_rate, 4),
            "total_bars": n_bars,
            "avg_turnover": round(avg_turnover, 4),
            "total_tc_usd": round(tc_total, 2),
            "avg_long_exposure": round(avg_long_exp, 4),
            "avg_short_exposure": round(avg_short_exp, 4),
            "avg_gross_exposure": round(avg_gross_exp, 4),
            "avg_net_exposure": round(avg_net_exp, 4),
            "n_pairs": n_pairs,
            "leverage": self.leverage,
            "rebalance_freq": self.rebalance_freq,
        }

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "returns": returns_series,
            "weight_history": weight_history,
            "turnover": turnover_series,
            "pair_pnl": pair_pnl_sorted,
            "timestamps": timestamps,
        }

    @staticmethod
    def print_report(result: dict):
        """Print a formatted backtest report."""
        m = result["metrics"]
        print(f"\n{'=' * 50}")
        print(f"  PORTFOLIO BACKTEST REPORT")
        print(f"{'=' * 50}")
        print(f"  Capital:     ${m['init_capital']:,.0f} -> ${m['final_capital']:,.2f}")
        print(f"  Return:      {m['total_return_pct']:+.2f}%")
        print(f"  Sharpe:      {m['sharpe']:.4f}")
        print(f"  Sortino:     {m['sortino']:.4f}")
        print(f"  Calmar:      {m['calmar']:.4f}")
        print(f"  Max DD:      {m['max_drawdown_pct']:.2f}%")
        print(f"  Hit rate:    {m['hit_rate']:.1%}")
        print(f"  Bars:        {m['total_bars']}")
        print(f"  Avg turnover:{m['avg_turnover']:.4f}")
        print(f"  Total TC:    ${m['total_tc_usd']:.2f}")
        print(f"  Long exp:    {m['avg_long_exposure']:.4f}")
        print(f"  Short exp:   {m['avg_short_exposure']:.4f}")
        print(f"  Net exp:     {m['avg_net_exposure']:.4f}")
        print(f"  Gross exp:   {m['avg_gross_exposure']:.4f}")

        # Top pair contributors
        pair_pnl = result.get("pair_pnl", {})
        if pair_pnl:
            items = list(pair_pnl.items())
            print(f"\n  Top contributors (USD):")
            for name, pnl in items[:5]:
                sign = "+" if pnl >= 0 else ""
                print(f"    {name:<12s} {sign}{pnl:>8.2f}")
            print(f"  Bottom contributors:")
            for name, pnl in items[-3:]:
                sign = "+" if pnl >= 0 else ""
                print(f"    {name:<12s} {sign}{pnl:>8.2f}")

        print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping smoke test.")
        print("Install with: pip install torch")
        sys.exit(0)

    print("Market-neutral model smoke test...")
    rng = np.random.default_rng(42)

    # Synthetic data: 200 samples, 568 features, 35 pairs
    n_samples = 200
    n_features = 568
    n_pairs = 35

    features = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)
    forward_returns = rng.normal(0, 0.001, (n_samples, n_pairs)).astype(np.float32)

    # Test model forward pass
    print("  Testing AlphaNet forward pass...")
    model = AlphaNet(n_features=n_features, n_pairs=n_pairs)
    x = torch.tensor(features[:10], dtype=torch.float32)
    w = model(x)
    print(f"    Output shape: {w.shape}")
    print(f"    Weight sum (should be ~0): {w.sum(dim=-1).tolist()}")
    print(f"    Gross exposure (should be ~1): {w.abs().sum(dim=-1).tolist()}")
    assert w.shape == (10, 35)
    assert all(abs(s) < 1e-5 for s in w.sum(dim=-1).tolist()), "Weights not dollar-neutral"

    # Test loss
    print("  Testing Sharpe loss...")
    r = torch.tensor(forward_returns[:10], dtype=torch.float32)
    loss = sharpe_loss(w, r)
    print(f"    Loss: {loss.item():.6f}")

    # Test walk-forward trainer
    print("  Testing walk-forward training (2 folds, 10 epochs)...")
    trainer = WalkForwardTrainer(
        n_features=n_features, n_pairs=n_pairs,
        n_folds=2, epochs=10, patience=5, batch_size=16,
    )
    result = trainer.train(features, forward_returns, verbose=True)
    print(f"    Best fold: {result['best_fold']}")
    print(f"    Best Sharpe: {result['best_sharpe']:.4f}")

    # Test inference
    print("  Testing inference...")
    weights = predict_weights(result["model"], features[0])
    print(f"    Weights shape: {weights.shape}")
    print(f"    Sum: {weights.sum():.8f}")

    # Test confidence mapping
    conf = weights_to_confidence(weights)
    print(f"    Confidence range: [{conf['confidence'].min():.4f}, {conf['confidence'].max():.4f}]")
    print(f"    Long/Short/Flat: "
          f"{(conf['direction'] > 0).sum()}/{(conf['direction'] < 0).sum()}/{(conf['direction'] == 0).sum()}")

    # Test save/load
    print("  Testing save/load...")
    test_path = "models/test_alpha_net.pt"
    trainer.save_model(
        result["model"], test_path,
        scaler_mean=features.mean(axis=0),
        scaler_std=features.std(axis=0),
        pairs=[f"PAIR{i}" for i in range(n_pairs)],
        metadata={"best_sharpe": result["best_sharpe"]},
    )
    loaded = WalkForwardTrainer.load_model(test_path)
    w2 = predict_weights(loaded["model"], features[0],
                         loaded["scaler_mean"], loaded["scaler_std"])
    print(f"    Loaded weights match: {np.allclose(weights, w2, atol=1e-4)}")

    # Test portfolio backtester
    print("  Testing PortfolioBacktester...")
    pbt = PortfolioBacktester(
        init_capital=10000, leverage=2.0,
        transaction_cost_bps=3.0, rebalance_freq=1,
    )
    pbt_result = pbt.run(
        result["model"], features, forward_returns,
        scaler_mean=features.mean(axis=0),
        scaler_std=features.std(axis=0),
        pairs=[f"PAIR{i}" for i in range(n_pairs)],
    )
    PortfolioBacktester.print_report(pbt_result)
    assert len(pbt_result["equity_curve"]) == n_samples + 1
    assert "sharpe" in pbt_result["metrics"]

    # Cleanup
    Path(test_path).unlink(missing_ok=True)

    print("OK")
