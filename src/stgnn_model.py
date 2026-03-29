"""
Algo C2 -- Hierarchical Spatial-Temporal Graph Neural Network (STGNN)

Multi-timeframe financial forecasting with:
1. Data preprocessing: 10 timeframe tensors (M1..MN1)
2. Spatial graph convolution: ChebNet polynomial Laplacian filter
3. Temporal tracking: per-timeframe GRU encoder
4. Hierarchical memory exchange: Structural-RNN (bottom-up + top-down)
5. CatBoost execution head: probabilistic buy/sell signals
6. Cross-validation: MAD, MSE, RMSE, R^2
"""

import gc
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Pre-define so module-level class bodies never get NameError even in worker re-imports
import types as _t_stub
nn = _t_stub.SimpleNamespace(
    Module=object, GRU=object, GRUCell=object, Linear=object,
    LayerNorm=object, ModuleList=list, MSELoss=object, Parameter=object,
    Dropout=object,
)
del _t_stub

try:
    import torch
    import torch.nn as nn  # overwrites stub if torch is available
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    # Catch ALL exceptions -- CUDA workers on Windows can throw OSError/RuntimeError
    # when re-importing the module in a spawned process
    HAS_TORCH = False
    import types as _types, contextlib as _ctx
    torch = _types.SimpleNamespace(
        cuda=_types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            get_device_name=lambda i: "none",
            get_device_properties=lambda i: _types.SimpleNamespace(total_memory=0),
            amp=_types.SimpleNamespace(
                GradScaler=lambda **kw: _types.SimpleNamespace(
                    scale=lambda x: x, unscale_=lambda o: None,
                    step=lambda o: o.step(), update=lambda: None,
                ),
                autocast=lambda enabled=False: _ctx.nullcontext(),
            ),
        ),
        no_grad=_ctx.nullcontext,
        backends=_types.SimpleNamespace(
            cuda=_types.SimpleNamespace(matmul=_types.SimpleNamespace(allow_tf32=True)),
            cudnn=_types.SimpleNamespace(allow_tf32=True, benchmark=True),
        ),
        set_num_threads=lambda n: None,
        set_num_interop_threads=lambda n: None,
    )
    nn = _types.SimpleNamespace(
        Module=object, GRU=object, GRUCell=object, Linear=object,
        LayerNorm=object, ModuleList=list, MSELoss=object, Parameter=object,
        Dropout=object,
    )
    F = None
    Dataset = object
    DataLoader = object

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from math_engine import MathEngine
from feature_engine import (
    compute_rsi, compute_macd, compute_bollinger, compute_atr,
    compute_stochastic, compute_cci, compute_williams_r,
    PIP_SIZES, PAIRS_ALL,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Hardware Profile: AMD Ryzen 9 9950X3D / 32 GB DDR5-5600 + GTX 1050 Ti 4GB
# ---------------------------------------------------------------------------
import os as _os

HW_CORES = 16
HW_THREADS = 32
HW_RAM_GB = 32
HW_VCACHE_MB = 144   # L3 -- favours large batch GRU / matmul
HW_GPU_VRAM_GB = 4   # GTX 1050 Ti -- used for STGNN forward/backward
                     # CatBoost stays on CPU (32 threads faster for trees)

# PyTorch thread tuning -- leave 2 threads for OS / data loading
_TORCH_THREADS = max(1, HW_THREADS - 2)   # 30
_INTEROP_THREADS = max(1, HW_CORES // 2)  # 8
# Windows multiprocessing spawn requires freeze_support() -- use 0 workers to avoid
# the re-import crash; GPU transfers are fast enough that workers aren't needed
_DATALOADER_WORKERS = 0 if _os.name == "nt" else max(1, HW_CORES - 2)

if HAS_TORCH:
    torch.set_num_threads(_TORCH_THREADS)
    torch.set_num_interop_threads(_INTEROP_THREADS)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark=False prevents CUDA autotuner from spawning subprocesses on Windows
    torch.backends.cudnn.benchmark = False

# OMP / MKL env (set before any numpy heavy lifting)
_os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
_os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

# Detect best available device
def _detect_device() -> str:
    if HAS_TORCH and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        print(f"  [GPU] {name} ({vram:.1f} GB VRAM) -- STGNN on CUDA")
        return "cuda"
    print("  [CPU] No CUDA device -- STGNN on CPU")
    return "cpu"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-8
N_PAIRS = 35
N_FEATURES = 16  # per-pair indicators
N_TIMEFRAMES = 10

TIMEFRAME_NAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]
TIMEFRAME_FREQS = ["1min", "5min", "15min", "30min", "1h", "4h", "12h", "1D", "1W", "1ME"]
TIMEFRAME_WINDOWS = [60, 60, 40, 30, 30, 20, 15, 15, 10, 8]  # rolling window per tf


# ---------------------------------------------------------------------------
# Component 1: Multi-Timeframe Data Preprocessor
# ---------------------------------------------------------------------------

class MultiTimeframePreprocessor:
    """
    Resample 1-min bars to 10 timeframes and extract 16 features per node.
    Output: dict of {tf_name: (T, N_pairs, N_features)} numpy arrays.
    """

    def __init__(self, pairs: list[str] = None):
        self.pairs = pairs or sorted(PAIRS_ALL)
        self.n_pairs = len(self.pairs)

    def resample_from_json(self, data: dict) -> dict:
        """
        Convert 1-min JSON data to multi-timeframe OHLC DataFrames.

        Args:
            data: {pair: [{dt, o, h, l, c, sp, tk}, ...]}

        Returns:
            {tf_name: {pair: DataFrame(dt, o, h, l, c, sp, tk)}}
        """
        result = {}

        for tf_name, freq in zip(TIMEFRAME_NAMES, TIMEFRAME_FREQS):
            result[tf_name] = {}

            for pair in self.pairs:
                if pair not in data:
                    continue

                bars = data[pair]
                df = pd.DataFrame(bars)
                df["dt"] = pd.to_datetime(df["dt"])
                df = df.set_index("dt").sort_index()

                if tf_name == "M1":
                    # Already 1-min, just use as-is
                    resampled = df.copy()
                else:
                    resampled = df.resample(freq).agg({
                        "o": "first", "h": "max", "l": "min", "c": "last",
                        "sp": "mean", "tk": "sum",
                    }).dropna(subset=["c"])

                # Fill missing values by averaging neighbors
                for col in ["o", "h", "l", "c", "sp"]:
                    if resampled[col].isna().any():
                        resampled[col] = resampled[col].interpolate(
                            method="linear", limit_direction="both"
                        )

                resampled["tk"] = resampled["tk"].fillna(0)
                result[tf_name][pair] = resampled.reset_index()

            print(f"  {tf_name}: {len(result[tf_name])} pairs, "
                  f"{min(len(v) for v in result[tf_name].values()) if result[tf_name] else 0} bars min")

        return result

    def extract_features(self, tf_data: dict) -> dict:
        """
        Extract 16 features per pair per bar for each timeframe.

        Args:
            tf_data: {tf_name: {pair: DataFrame}}

        Returns:
            {tf_name: np.array(T, N_pairs, 16)}
        """
        tensors = {}

        for tf_name in TIMEFRAME_NAMES:
            if tf_name not in tf_data:
                continue

            pair_data = tf_data[tf_name]
            min_bars = min(len(pair_data[p]) for p in self.pairs if p in pair_data)

            features = np.zeros((min_bars, self.n_pairs, N_FEATURES), dtype=np.float32)

            for j, pair in enumerate(self.pairs):
                if pair not in pair_data:
                    continue

                df = pair_data[pair]
                c = df["c"].values[:min_bars].astype(float)
                h = df["h"].values[:min_bars].astype(float)
                l = df["l"].values[:min_bars].astype(float)
                o = df["o"].values[:min_bars].astype(float)
                sp = df["sp"].values[:min_bars].astype(float)
                tk = df["tk"].values[:min_bars].astype(float)
                pip = PIP_SIZES.get(pair, 0.0001)

                for t in range(min_bars):
                    # Slice up to current bar
                    cs = c[:t + 1]
                    hs = h[:t + 1]
                    ls = l[:t + 1]

                    # Indicators (with safe lookback)
                    rsi = compute_rsi(cs[-30:]) if len(cs) >= 14 else 50.0
                    macd_l, macd_h_val = compute_macd(cs[-50:]) if len(cs) >= 26 else (0.0, 0.0)
                    bb_pct, bb_bw = compute_bollinger(cs[-30:]) if len(cs) >= 20 else (0.5, 0.0)
                    atr = compute_atr(hs[-20:], ls[-20:], cs[-20:]) if len(cs) >= 14 else 0.0
                    sk, sd = compute_stochastic(hs[-30:], ls[-30:], cs[-30:]) if len(cs) >= 14 else (50.0, 50.0)
                    cci = compute_cci(hs[-25:], ls[-25:], cs[-25:]) if len(cs) >= 20 else 0.0
                    wr = compute_williams_r(hs[-20:], ls[-20:], cs[-20:]) if len(cs) >= 14 else -50.0

                    log_ret = float(np.log(cs[-1] / cs[-2])) if len(cs) >= 2 and cs[-2] > 0 else 0.0
                    tick_vel = tk[t]
                    spread = sp[t]
                    bar_range = (h[t] - l[t]) / pip if pip > 0 else 0.0
                    body = abs(c[t] - o[t])
                    full = h[t] - l[t]
                    body_ratio = body / full if full > 1e-10 else 0.0

                    if len(cs) >= 6:
                        mom_raw = cs[-1] - cs[-6]
                        mom_std = float(np.std(np.diff(cs[-6:]))) + EPS
                        mom5 = mom_raw / mom_std
                    else:
                        mom5 = 0.0

                    features[t, j] = [
                        rsi if not np.isnan(rsi) else 50.0,
                        macd_l if not np.isnan(macd_l) else 0.0,
                        macd_h_val if not np.isnan(macd_h_val) else 0.0,
                        bb_pct if not np.isnan(bb_pct) else 0.5,
                        bb_bw if not np.isnan(bb_bw) else 0.0,
                        float(atr / pip) if not np.isnan(atr) else 0.0,
                        sk if not np.isnan(sk) else 50.0,
                        sd if not np.isnan(sd) else 50.0,
                        cci if not np.isnan(cci) else 0.0,
                        wr if not np.isnan(wr) else -50.0,
                        log_ret, tick_vel, spread, bar_range,
                        float(body_ratio), float(mom5),
                    ]

            # Replace any remaining NaN
            features = np.nan_to_num(features, nan=0.0)
            tensors[tf_name] = features
            print(f"  {tf_name}: tensor shape {features.shape}")

        return tensors

    def compute_laplacians(self, tf_data: dict) -> dict:
        """
        Compute graph Laplacians per timeframe using MathEngine.

        Returns:
            {tf_name: list of (35, 35) Laplacian matrices, one per bar}
        """
        laplacians = {}

        for tf_idx, tf_name in enumerate(TIMEFRAME_NAMES):
            if tf_name not in tf_data:
                continue

            pair_data = tf_data[tf_name]
            min_bars = min(len(pair_data[p]) for p in self.pairs if p in pair_data)

            engine = MathEngine(
                n_pairs=self.n_pairs,
                rolling_window=TIMEFRAME_WINDOWS[tf_idx],
            )

            lap_list = []
            for t in range(min_bars):
                returns = np.zeros(self.n_pairs)
                for j, pair in enumerate(self.pairs):
                    if pair not in pair_data:
                        continue
                    c = pair_data[pair]["c"].values
                    if t > 0 and c[t - 1] > 0:
                        returns[j] = np.log(c[t] / c[t - 1])

                state = engine.update(returns)
                if state.valid:
                    lap_list.append(state.laplacian_matrix.astype(np.float32))
                else:
                    lap_list.append(np.eye(self.n_pairs, dtype=np.float32))

            laplacians[tf_name] = lap_list
            print(f"  {tf_name}: {len(lap_list)} Laplacians")

        return laplacians


# ---------------------------------------------------------------------------
# Component 2: Chebyshev Graph Convolution
# ---------------------------------------------------------------------------

class ChebGraphConv(nn.Module):
    """
    K-order Chebyshev polynomial graph convolution.

    Applies polynomial filter on the graph Laplacian to blend
    cross-asset features and remove noise at each time step.

    T_0(L) = I
    T_1(L) = L_hat
    T_k(L) = 2 * L_hat * T_{k-1} - T_{k-2}

    where L_hat = 2L/lambda_max - I  (scaled to [-1, 1])
    """

    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features

        # One weight matrix per Chebyshev order
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self._init_params()

    def _init_params(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, F_in) node features
            laplacian: (N, N) normalized Laplacian

        Returns:
            (batch, N, F_out) filtered features
        """
        N = laplacian.size(0)

        # Scale Laplacian to [-1, 1]: L_hat = 2L/lambda_max - I
        # Use approximate lambda_max = 2.0 for normalized Laplacian
        L_hat = laplacian - torch.eye(N, device=laplacian.device)

        # Chebyshev recurrence
        # T_0 = I, so T_0 * x = x
        # x: (batch, N, F_in), L_hat: (N, N)
        # Graph multiply: L @ X per batch -> einsum 'ij,bjf->bif'
        T_0 = x  # (batch, N, F_in)
        out = torch.matmul(T_0, self.weights[0])  # (batch, N, F_out)

        if self.K > 1:
            T_1 = torch.einsum('ij,bjf->bif', L_hat, x)
            out = out + torch.matmul(T_1, self.weights[1])

            for k in range(2, self.K):
                T_2 = 2 * torch.einsum('ij,bjf->bif', L_hat, T_1) - T_0
                out = out + torch.matmul(T_2, self.weights[k])
                T_0, T_1 = T_1, T_2

        return out + self.bias


class SpatialBlock(nn.Module):
    """ChebConv + LayerNorm + Dropout + activation for one timeframe snapshot."""

    def __init__(self, in_features: int, hidden: int, out_features: int,
                 K: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ChebGraphConv(in_features, hidden, K=K)
        self.conv2 = ChebGraphConv(hidden, out_features, K=K)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, F_in)
            lap: (N, N)
        Returns:
            (batch, N, F_out)
        """
        h = F.leaky_relu(self.norm1(self.conv1(x, lap)), 0.1)
        h = self.drop(h)
        h = F.leaky_relu(self.norm2(self.conv2(h, lap)), 0.1)
        return h


# ---------------------------------------------------------------------------
# Component 3: Temporal GRU Encoder (per timeframe)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    GRU over spatially-filtered snapshots for one timeframe.

    Input: sequence of graph-filtered features (batch, T, N * F_spatial)
    Output: hidden state (batch, hidden_dim) — temporal memory
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, T, input_dim) — flattened spatial features over time

        Returns:
            output: (batch, T, hidden_dim)
            h_n: (n_layers, batch, hidden_dim) — final hidden state
        """
        output, h_n = self.gru(x)
        return output, h_n


# ---------------------------------------------------------------------------
# Component 4: Hierarchical Memory Exchange (Structural-RNN)
# ---------------------------------------------------------------------------

class HierarchicalExchange(nn.Module):
    """
    Bottom-up then top-down hidden state cascade across 10 timeframes.

    Bottom-up: M1 -> M5 -> M15 -> ... -> MN1
    Top-down:  MN1 -> W1 -> D1 -> ... -> M1

    Each step: h_target' = GRUCell(h_source, h_target)
    After both passes, every timeframe carries context from all others.
    """

    def __init__(self, hidden_dim: int = 256, n_timeframes: int = N_TIMEFRAMES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_timeframes = n_timeframes

        # Bottom-up GRU cells (micro -> macro)
        self.up_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(n_timeframes - 1)
        ])

        # Top-down GRU cells (macro -> micro)
        self.down_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(n_timeframes - 1)
        ])

    def forward(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            hidden_states: list of 10 tensors, each (batch, hidden_dim)
                           ordered M1, M5, M15, M30, H1, H4, H12, D1, W1, MN1

        Returns:
            enriched: list of 10 tensors (batch, hidden_dim), same order
        """
        h = [s.clone() for s in hidden_states]

        # Bottom-up: cascade from M1 (idx 0) to MN1 (idx 9)
        for i in range(self.n_timeframes - 1):
            # Source = lower timeframe, target = higher timeframe
            h[i + 1] = self.up_cells[i](h[i], h[i + 1])

        # Top-down: cascade from MN1 (idx 9) to M1 (idx 0)
        for i in range(self.n_timeframes - 2, -1, -1):
            # Source = higher timeframe, target = lower timeframe
            h[i] = self.down_cells[i](h[i + 1], h[i])

        return h


# ---------------------------------------------------------------------------
# Component 5: Full STGNN Model
# ---------------------------------------------------------------------------

class HierarchicalSTGNN(nn.Module):
    """
    Complete Hierarchical Spatial-Temporal GNN.

    Pipeline per bar:
    1. Spatial: ChebConv filters node features using Laplacian (per timeframe)
    2. Temporal: GRU encodes filtered sequences (per timeframe)
    3. Hierarchical: bottom-up + top-down memory exchange
    4. Output: concatenated hidden states for CatBoost
    """

    def __init__(self, n_pairs: int = N_PAIRS, n_features: int = N_FEATURES,
                 n_timeframes: int = N_TIMEFRAMES, spatial_hidden: int = 64,
                 spatial_out: int = 32, temporal_hidden: int = 256,
                 temporal_layers: int = 3, cheb_K: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_pairs = n_pairs
        self.n_timeframes = n_timeframes
        self.temporal_hidden = temporal_hidden

        # Per-timeframe spatial blocks
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(n_features, spatial_hidden, spatial_out, K=cheb_K, dropout=dropout)
            for _ in range(n_timeframes)
        ])

        # Per-timeframe temporal encoders
        temporal_input = n_pairs * spatial_out  # flatten spatial output
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(temporal_input, temporal_hidden, temporal_layers, dropout)
            for _ in range(n_timeframes)
        ])

        # Hierarchical exchange
        self.hierarchy = HierarchicalExchange(temporal_hidden, n_timeframes)

        # Output projection (for MSE regression loss during Stage 1)
        self.output_proj = nn.Linear(temporal_hidden * n_timeframes, n_pairs)

    def forward(self, features: dict, laplacians: dict,
                seq_len: int = 1) -> dict:
        """
        Args:
            features: {tf_name: (batch, T, N, F)} tensors per timeframe
            laplacians: {tf_name: (N, N)} Laplacian per timeframe
            seq_len: number of temporal steps to process

        Returns:
            dict with:
                'prediction': (batch, N_pairs) next-bar return predictions
                'hidden_states': list of 10 (batch, hidden_dim) enriched states
                'features_for_catboost': (batch, 10 * hidden_dim) for CatBoost
        """
        tf_hidden_states = []

        for tf_idx, tf_name in enumerate(TIMEFRAME_NAMES[:self.n_timeframes]):
            if tf_name not in features:
                # Pad with zeros if timeframe not available
                batch_size = 1
                for v in features.values():
                    batch_size = v.size(0)
                    break
                tf_hidden_states.append(
                    torch.zeros(batch_size, self.temporal_hidden, device=next(self.parameters()).device)
                )
                continue

            feat = features[tf_name]  # (batch, T, N, F)
            lap = laplacians[tf_name]  # (N, N)

            batch_size = feat.size(0)
            T = feat.size(1)

            # Apply spatial convolution per time step
            spatial_out_list = []
            for t in range(T):
                x_t = feat[:, t, :, :]  # (batch, N, F)
                s_t = self.spatial_blocks[tf_idx](x_t, lap)  # (batch, N, F_out)
                spatial_out_list.append(s_t.reshape(batch_size, -1))  # (batch, N*F_out)

            # Stack temporal sequence
            temporal_input = torch.stack(spatial_out_list, dim=1)  # (batch, T, N*F_out)

            # GRU encoding
            _, h_n = self.temporal_encoders[tf_idx](temporal_input)
            # h_n: (n_layers, batch, hidden) -> take last layer
            tf_hidden_states.append(h_n[-1])  # (batch, hidden)

        # Hierarchical exchange
        enriched = self.hierarchy(tf_hidden_states)

        # Concatenate all timeframe hidden states
        cat_features = torch.cat(enriched, dim=-1)  # (batch, 10 * hidden)

        # Regression prediction (for Stage 1 training)
        prediction = self.output_proj(cat_features)  # (batch, N_pairs)

        return {
            "prediction": prediction,
            "hidden_states": enriched,
            "features_for_catboost": cat_features.detach(),
        }


# ---------------------------------------------------------------------------
# Component 6: CatBoost Execution Head
# ---------------------------------------------------------------------------

class CatBoostExecutionHead:
    """
    Trains CatBoost on frozen STGNN hidden features for buy/sell classification.
    """

    def __init__(self, iterations: int = 1000, depth: int = 8,
                 learning_rate: float = 0.03, verbose: bool = False):
        if not HAS_CATBOOST:
            raise RuntimeError("CatBoost not installed. Run: pip install catboost")

        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=42,
            verbose=verbose,
            task_type="CPU",
            thread_count=HW_THREADS,  # 32 threads on 9950X3D
            l2_leaf_reg=3.0,
            border_count=254,  # max histogram bins -- V-Cache handles this
            bootstrap_type="Bayesian",
            bagging_temperature=0.8,
        )
        self.is_fitted = False

    def train(self, features: np.ndarray, labels: np.ndarray,
              eval_features: np.ndarray = None, eval_labels: np.ndarray = None):
        """
        Train CatBoost on extracted STGNN hidden features.

        Args:
            features: (N, 10*hidden_dim) — concatenated hidden states
            labels: (N,) — 0=sell, 1=hold, 2=buy
        """
        eval_set = None
        if eval_features is not None and eval_labels is not None:
            eval_set = (eval_features, eval_labels)

        self.model.fit(features, labels, eval_set=eval_set, early_stopping_rounds=100)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict buy/sell/hold probabilities.

        Returns:
            dict with 'classes': (N,), 'probabilities': (N, 3)
        """
        if not self.is_fitted:
            raise RuntimeError("CatBoost not trained yet")

        proba = self.model.predict_proba(features)
        classes = self.model.predict(features).flatten().astype(int)

        return {
            "classes": classes,
            "probabilities": proba,
        }


# ---------------------------------------------------------------------------
# Component 7: Training Pipeline
# ---------------------------------------------------------------------------

class STGNNDataset(Dataset):
    """Dataset for multi-timeframe STGNN training."""

    def __init__(self, features: dict, laplacians: dict,
                 targets: np.ndarray, seq_len: int = 20):
        """
        Args:
            features: {tf_name: (T, N, F)} numpy arrays
            laplacians: {tf_name: list of (N, N) matrices}
            targets: (T,) next-bar returns for target pair
            seq_len: temporal window for GRU
        """
        self.features = features
        self.laplacians = laplacians
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

        # Use M1 length as the base timeline for indexing
        # After proportional slicing, M1 has the most bars
        self.base_len = len(features.get("M1", next(iter(features.values()))))
        self.min_len = self.base_len  # for ratio computation in __getitem__
        self.n_samples = max(0, self.base_len - seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len

        sample_features = {}
        sample_laps = {}

        for tf_name in self.features:
            tf_feat = self.features[tf_name]
            # Map M1 index to this timeframe's index
            # For higher timeframes, fewer bars exist. Use proportional mapping.
            tf_len = len(tf_feat)
            ratio = tf_len / self.min_len if self.min_len > 0 else 1.0
            tf_start = max(0, int(start * ratio))
            tf_end = min(tf_len, max(tf_start + 1, int(end * ratio)))

            chunk = tf_feat[tf_start:tf_end]
            sample_features[tf_name] = torch.tensor(chunk, dtype=torch.float32)

            # Use last available Laplacian in this window
            lap_idx = min(tf_end - 1, len(self.laplacians[tf_name]) - 1)
            sample_laps[tf_name] = torch.tensor(
                self.laplacians[tf_name][max(0, lap_idx)], dtype=torch.float32
            )

        target = self.targets[min(end, len(self.targets) - 1)]
        return sample_features, sample_laps, target


def collate_stgnn(batch):
    """Custom collate for variable-length timeframe sequences."""
    features_batch = {tf: [] for tf in TIMEFRAME_NAMES}
    laps_batch = {tf: [] for tf in TIMEFRAME_NAMES}
    targets = []

    for feat, lap, tgt in batch:
        for tf in feat:
            features_batch[tf].append(feat[tf])
            laps_batch[tf].append(lap[tf])
        targets.append(tgt)

    # Pad sequences to same length within each timeframe
    result_features = {}
    result_laps = {}

    for tf in TIMEFRAME_NAMES:
        if not features_batch[tf]:
            continue

        # Find max sequence length in this batch for this timeframe
        max_t = max(f.size(0) for f in features_batch[tf])
        n_nodes = features_batch[tf][0].size(1)
        n_feat = features_batch[tf][0].size(2)

        padded = torch.zeros(len(features_batch[tf]), max_t, n_nodes, n_feat)
        for i, f in enumerate(features_batch[tf]):
            padded[i, :f.size(0)] = f

        result_features[tf] = padded
        # Laplacians: just stack (they're all N x N)
        result_laps[tf] = torch.stack(laps_batch[tf])[0]  # Use first (shared graph)

    targets = torch.stack(targets)
    return result_features, result_laps, targets


class STGNNTrainer:
    """
    Two-stage walk-forward trainer with memory management.

    Stage 1: Train STGNN end-to-end (spatial + temporal + hierarchy) on MSE
    Stage 2: Freeze STGNN, extract features, train CatBoost on labels
    """

    def __init__(self, n_pairs: int = N_PAIRS, n_features: int = N_FEATURES,
                 temporal_hidden: int = 256, spatial_out: int = 32,
                 seq_len: int = 60, n_folds: int = 5,
                 epochs: int = 80, patience: int = 15,
                 batch_size: int = 64, lr: float = 1e-4,
                 device: str = None):
        self.n_pairs = n_pairs
        self.n_features = n_features
        self.temporal_hidden = temporal_hidden
        self.spatial_out = spatial_out
        self.seq_len = seq_len
        self.n_folds = n_folds
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr

        if device is None:
            self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        # Mixed precision scaler -- active only on CUDA (fp16 halves VRAM on 1050 Ti)
        self.use_amp = (self.device == "cuda") if HAS_TORCH else False
        if HAS_TORCH:
            self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_amp)
        else:
            self.device = device

    def compute_targets(self, tf_data: dict, pairs: list[str],
                        target_pair: str = "EURUSD") -> np.ndarray:
        """Compute next-bar log returns for target pair from M1 data."""
        if target_pair not in tf_data.get("M1", {}):
            target_pair = pairs[0]

        df = tf_data["M1"][target_pair]
        c = df["c"].values.astype(float)
        returns = np.zeros(len(c))
        for t in range(len(c) - 1):
            if c[t] > 0:
                returns[t] = np.log(c[t + 1] / c[t])
        return returns

    def compute_labels(self, returns: np.ndarray, threshold: float = 0.0001) -> np.ndarray:
        """Convert returns to 3-class labels: 0=sell, 1=hold, 2=buy."""
        labels = np.ones(len(returns), dtype=np.int64)  # default: hold
        labels[returns > threshold] = 2   # buy
        labels[returns < -threshold] = 0  # sell
        return labels

    def _proportional_slice(self, data: dict, n_m1: int,
                            start_m1: int, end_m1: int) -> dict:
        """Slice each timeframe proportionally based on M1 indices."""
        result = {}
        for tf_name, arr in data.items():
            tf_len = len(arr)
            ratio = tf_len / n_m1 if n_m1 > 0 else 1.0
            s = max(0, int(start_m1 * ratio))
            e = min(tf_len, max(s + 1, int(end_m1 * ratio)))
            result[tf_name] = arr[s:e]
        return result

    def train_stage1(self, model: HierarchicalSTGNN, features: dict,
                     laplacians: dict, targets: np.ndarray,
                     train_end: int, val_end: int,
                     verbose: bool = True) -> dict:
        """
        Stage 1: End-to-end STGNN training on MSE loss.
        train_end/val_end are M1-based indices.

        Returns dict with train/val metrics.
        """
        model = model.to(self.device)
        n_m1 = len(features.get("M1", next(iter(features.values()))))

        # Build datasets with proportional slicing
        train_feat = self._proportional_slice(features, n_m1, 0, train_end)
        train_lap = self._proportional_slice(laplacians, n_m1, 0, train_end)
        val_feat = self._proportional_slice(features, n_m1, train_end, val_end)
        val_lap = self._proportional_slice(laplacians, n_m1, train_end, val_end)

        train_ds = STGNNDataset(train_feat, train_lap, targets[:train_end], self.seq_len)
        val_ds = STGNNDataset(val_feat, val_lap, targets[train_end:val_end], self.seq_len)

        if len(train_ds) < 2 or len(val_ds) < 2:
            return {"train_loss": float("inf"), "val_loss": float("inf")}

        _pin = (self.device == "cuda")
        _nw = _DATALOADER_WORKERS
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_stgnn, drop_last=True,
            num_workers=_nw, pin_memory=_pin,
            persistent_workers=(_nw > 0),
            prefetch_factor=(4 if _nw > 0 else None),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_stgnn, drop_last=True,
            num_workers=min(4, _nw), pin_memory=_pin,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batch = 0
            for batch_feat, batch_lap, batch_tgt in train_loader:
                # Move to device
                for tf in batch_feat:
                    batch_feat[tf] = batch_feat[tf].to(self.device)
                    batch_lap[tf] = batch_lap[tf].to(self.device)
                batch_tgt = batch_tgt.to(self.device)

                # AMP forward pass (fp16 on CUDA, fp32 on CPU)
                with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                    out = model(batch_feat, batch_lap)
                    pred_mean = out["prediction"].mean(dim=-1)
                    loss = criterion(pred_mean, batch_tgt)

                # Skip NaN/Inf batches (can occur in early AMP training)
                if not torch.isfinite(loss):
                    del out, loss
                    continue

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item()
                n_batch += 1

                # Memory management
                del out, loss
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            avg_train = train_loss / max(n_batch, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_feat, batch_lap, batch_tgt in val_loader:
                    for tf in batch_feat:
                        batch_feat[tf] = batch_feat[tf].to(self.device)
                        batch_lap[tf] = batch_lap[tf].to(self.device)
                    batch_tgt = batch_tgt.to(self.device)

                    out = model(batch_feat, batch_lap)
                    pred_mean = out["prediction"].mean(dim=-1)
                    loss = criterion(pred_mean, batch_tgt)
                    val_loss += loss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            scheduler.step(epoch)

            if avg_val < best_val:
                best_val = avg_val
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1

            if verbose and epoch % 5 == 0:
                print(f"    Epoch {epoch}: train_mse={avg_train:.6f}, val_mse={avg_val:.6f}")

            if patience_count >= self.patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        return {"train_loss": avg_train, "val_loss": best_val}

    def extract_features(self, model: HierarchicalSTGNN, features: dict,
                         laplacians: dict, n_samples: int) -> np.ndarray:
        """
        Extract frozen hidden features from STGNN for CatBoost (batched).

        Returns: (n_valid_samples, 10 * hidden_dim) numpy array
        """
        model.eval()
        model = model.to(self.device)

        # Use DataLoader for efficient batched extraction
        # Create a dummy target array (not used, just for dataset compat)
        dummy_targets = np.zeros(n_samples, dtype=np.float32)
        ds = STGNNDataset(features, laplacians, dummy_targets, self.seq_len)

        if len(ds) == 0:
            return np.zeros((0, self.temporal_hidden * N_TIMEFRAMES))

        _nw = _DATALOADER_WORKERS
        loader = DataLoader(
            ds, batch_size=self.batch_size * 4,  # larger batch for inference
            shuffle=False, collate_fn=collate_stgnn, drop_last=False,
            num_workers=_nw, pin_memory=(self.device == "cuda"),
            prefetch_factor=(4 if _nw > 0 else None),
        )

        all_features = []
        with torch.no_grad():
            for batch_feat, batch_lap, _ in loader:
                for tf in batch_feat:
                    batch_feat[tf] = batch_feat[tf].to(self.device)
                    batch_lap[tf] = batch_lap[tf].to(self.device)

                out = model(batch_feat, batch_lap)
                all_features.append(out["features_for_catboost"].cpu().numpy())

                del out
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if all_features:
            return np.concatenate(all_features, axis=0)
        return np.zeros((0, self.temporal_hidden * N_TIMEFRAMES))

    def train_full(self, features: dict, laplacians: dict,
                   targets: np.ndarray, labels: np.ndarray,
                   verbose: bool = True) -> dict:
        """
        Full walk-forward training: Stage 1 (STGNN) + Stage 2 (CatBoost).

        Returns dict with fold results and final metrics.
        """
        # Use M1 length as the base for walk-forward splits
        # Higher timeframes have fewer bars but map proportionally
        n_m1 = len(features.get("M1", next(iter(features.values()))))
        n_samples = n_m1

        # Expanding-window walk-forward: reserve 20% for validation per fold
        val_ratio = 0.2
        val_size = max(self.seq_len * 2, int(n_samples * val_ratio / self.n_folds))

        fold_results = []

        for fold in range(self.n_folds):
            # Training grows with each fold, validation is a fixed window after
            train_end = int(n_samples * (0.5 + 0.5 * fold / max(self.n_folds, 1)))
            train_end = max(train_end, self.seq_len * 3)  # minimum training data
            val_end = min(train_end + val_size, n_samples)

            if val_end <= train_end + self.seq_len:
                continue

            if verbose:
                print(f"\n  Fold {fold}: train={train_end}, val={val_end - train_end}")

            # Stage 1: STGNN
            model = HierarchicalSTGNN(
                n_pairs=self.n_pairs, n_features=self.n_features,
                temporal_hidden=self.temporal_hidden,
                spatial_out=self.spatial_out,
            )

            # torch.compile for ~15-30% speedup on Zen5 AVX-512
            # Requires C++ compiler (cl.exe on Windows, gcc/clang on Linux)
            _can_compile = (
                hasattr(torch, "compile")
                and self.device == "cpu"
                and sys.platform != "win32"  # Windows needs MSVC cl.exe in PATH
            )
            if _can_compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    if verbose:
                        print("    [torch.compile enabled]")
                except Exception:
                    pass  # fallback to eager if compile fails

            s1 = self.train_stage1(
                model, features, laplacians, targets,
                train_end, val_end, verbose=verbose,
            )

            # Stage 2: Extract features + CatBoost
            if verbose:
                print(f"    Extracting STGNN features...")

            n_m1 = len(features.get("M1", next(iter(features.values()))))
            train_feats = self.extract_features(
                model,
                self._proportional_slice(features, n_m1, 0, train_end),
                self._proportional_slice(laplacians, n_m1, 0, train_end),
                train_end,
            )
            val_feats = self.extract_features(
                model,
                self._proportional_slice(features, n_m1, train_end, val_end),
                self._proportional_slice(laplacians, n_m1, train_end, val_end),
                val_end - train_end,
            )

            # Align labels
            train_labels = labels[self.seq_len:self.seq_len + len(train_feats)]
            val_labels = labels[train_end + self.seq_len:train_end + self.seq_len + len(val_feats)]

            if len(train_feats) < 10 or len(val_feats) < 5:
                if verbose:
                    print(f"    Insufficient features (train={len(train_feats)}, val={len(val_feats)})")
                continue

            # Trim to match
            min_train = min(len(train_feats), len(train_labels))
            min_val = min(len(val_feats), len(val_labels))
            train_feats = train_feats[:min_train]
            train_labels = train_labels[:min_train]
            val_feats = val_feats[:min_val]
            val_labels = val_labels[:min_val]

            if verbose:
                print(f"    CatBoost: train={len(train_feats)}, val={len(val_feats)}")

            # Train CatBoost
            if HAS_CATBOOST and len(np.unique(train_labels)) > 1:
                cb_head = CatBoostExecutionHead(iterations=1000, depth=8, verbose=False)
                cb_head.train(train_feats, train_labels, val_feats, val_labels)
                cb_pred = cb_head.predict(val_feats)

                # Compute metrics
                val_returns = targets[train_end + self.seq_len:train_end + self.seq_len + len(val_feats)]
                metrics = self._compute_metrics(
                    val_returns[:min_val], cb_pred["classes"][:min_val],
                    val_labels, cb_pred["probabilities"][:min_val],
                )
            else:
                metrics = {"mad": 0, "mse": 0, "rmse": 0, "r2": 0, "accuracy": 0, "strategy_sharpe": 0, "n_bars": 0}

            fold_results.append({
                "fold": fold,
                "stage1_val_loss": s1["val_loss"],
                **metrics,
            })

            if verbose:
                print(f"    Fold {fold}: MSE={metrics['mse']:.6f}, "
                      f"R2={metrics['r2']:.4f}, Acc={metrics['accuracy']:.1%}, "
                      f"ConfAcc(top30%)={metrics['conf_acc']:.1%}")

            # Keep last fold's model and CatBoost
            last_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            last_cb = cb_head if (HAS_CATBOOST and len(np.unique(train_labels)) > 1) else None
            self._last_model_state = last_model_state
            self._last_cb = last_cb
            self._last_model_arch = {"n_pairs": self.n_pairs, "n_features": self.n_features,
                                     "temporal_hidden": self.temporal_hidden, "spatial_out": self.spatial_out}

            # Cleanup
            del model, train_feats, val_feats
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Aggregate cross-validation metrics
        cv_metrics = {}
        if fold_results:
            for key in ["mad", "mse", "rmse", "r2", "accuracy", "conf_acc", "strategy_sharpe"]:
                vals = [fr[key] for fr in fold_results if key in fr]
                cv_metrics[key] = float(np.mean(vals)) if vals else 0.0

        return {
            "fold_results": fold_results,
            "cv_metrics": cv_metrics,
            "model_state": last_model_state if fold_results else None,
            "catboost_model": last_cb if fold_results else None,
        }

    def save(self, stgnn_path: str, catboost_path: str) -> None:
        """Save STGNN weights and CatBoost model from the last trained fold."""
        model_state = getattr(self, "_last_model_state", None)
        cb = getattr(self, "_last_cb", None)
        arch = getattr(self, "_last_model_arch", {})

        if model_state is None:
            raise RuntimeError("No trained model found. Call train_full() first.")

        torch.save({"model_state_dict": model_state, "arch": arch}, stgnn_path)

        if cb is not None:
            cb.model.save_model(catboost_path)
        else:
            # Write a placeholder so the path exists
            with open(catboost_path, "w") as f:
                f.write("# CatBoost not available or not trained\n")

    @staticmethod
    def _compute_metrics(returns: np.ndarray, predictions: np.ndarray,
                         labels: np.ndarray,
                         probabilities: np.ndarray = None) -> dict:
        """Compute MAD, MSE, RMSE, R^2, accuracy, and confidence-filtered accuracy."""
        n = min(len(returns), len(predictions))
        returns = returns[:n]
        predictions = predictions[:n]
        labels = labels[:n]

        # For regression metrics, map predictions to approximate returns
        pred_returns = (predictions.astype(float) - 1.0) * 0.001  # scale class to return proxy

        errors = returns - pred_returns
        mad = float(np.mean(np.abs(errors)))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((returns - returns.mean()) ** 2) + EPS
        r2 = float(1.0 - ss_res / ss_tot)

        accuracy = float(np.mean(predictions == labels))

        # Confidence-filtered accuracy: top 30% most confident predictions
        conf_acc = accuracy  # fallback
        if probabilities is not None and len(probabilities) > 10:
            proba = probabilities[:n]
            confidence = proba.max(axis=1)  # max class probability = confidence
            thresh = np.percentile(confidence, 70)  # top 30%
            mask = confidence >= thresh
            if mask.sum() > 5:
                conf_acc = float(np.mean(predictions[mask] == labels[mask]))

        # Strategy Sharpe: signal (+1 buy, 0 hold, -1 sell) × actual return
        signal = predictions.astype(float) - 1.0
        strat_rets = signal * returns
        sr_std = float(strat_rets.std()) + 1e-8
        strategy_sharpe = float(strat_rets.mean() / sr_std * np.sqrt(max(n, 1)))

        return {
            "mad": round(mad, 8),
            "mse": round(mse, 10),
            "rmse": round(rmse, 8),
            "r2": round(r2, 4),
            "accuracy": round(accuracy, 4),
            "conf_acc": round(conf_acc, 4),
            "strategy_sharpe": round(strategy_sharpe, 4),
            "n_bars": n,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing guard -- must be first line inside __main__
    import multiprocessing
    multiprocessing.freeze_support()
    import argparse

    parser = argparse.ArgumentParser(description="STGNN -- train on real data or run smoke test")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to CSV data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Where to save trained models")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="Force smoke test even if --data-dir given")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch not installed. Skipping.")
        sys.exit(0)

    # --- Real data training mode ---
    if args.data_dir and not args.smoke:
        import glob as _glob

        data_dir = Path(args.data_dir)
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        import time as _time

        def _elapsed(t0: float) -> str:
            s = int(_time.time() - t0)
            return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"

        t_total = _time.time()

        print(f"STGNN -- Real Data Training")
        print(f"  Data dir : {data_dir}")
        print(f"  Model dir: {model_dir}")
        print(f"  Folds={args.n_folds}, Epochs={args.epochs}, Batch={args.batch_size}")
        print("=" * 60)

        # Load CSV files
        csv_files = sorted(_glob.glob(str(data_dir / "*.csv")))
        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            sys.exit(1)
        print(f"[1/5] Found {len(csv_files)} CSV files")

        # Preprocess multi-timeframe data
        t_step = _time.time()
        print("[2/5] Loading & resampling tick CSVs to OHLC M1 bars...")
        preprocessor = MultiTimeframePreprocessor()

        # Build JSON-style dict: {pair: [{dt, o, h, l, c, sp, tk}]}
        json_data = {}
        for f in csv_files:
            pair = Path(f).stem.split("_")[0]
            try:
                df = pd.read_csv(f, sep="\t", header=0)
                df.columns = [c.strip("<>").lower() for c in df.columns]
                # Combine DATE + TIME into datetime
                if "date" in df.columns and "time" in df.columns:
                    df["dt"] = pd.to_datetime(
                        df["date"].astype(str) + " " + df["time"].astype(str),
                        format="%Y.%m.%d %H:%M:%S.%f", errors="coerce"
                    )
                    df = df.dropna(subset=["dt"])
                    # Resample ticks to M1 OHLC — set DatetimeIndex first
                    df = df.set_index("dt")
                    mid = (df["bid"] + df["ask"]) / 2.0
                    spread = df["ask"] - df["bid"]
                    vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
                    m1 = pd.DataFrame({
                        "o": mid.resample("1min").first(),
                        "h": mid.resample("1min").max(),
                        "l": mid.resample("1min").min(),
                        "c": mid.resample("1min").last(),
                        "sp": spread.resample("1min").mean(),
                        "tk": vol.resample("1min").sum(),
                    }).dropna(subset=["c"]).reset_index()
                    m1.rename(columns={"index": "dt"}, inplace=True)
                    if "dt" not in m1.columns:
                        m1 = m1.rename(columns={m1.columns[0]: "dt"})
                    json_data[pair] = m1.rename(columns={"index": "dt"}).to_dict("records")
                    print(f"  {pair}: {len(json_data[pair])} M1 bars")
            except Exception as e:
                print(f"  Skipping {pair}: {e}")

        print(f"  Loaded {len(json_data)} pairs into M1 bars")
        print("  Resampling to all 10 timeframes...")
        tf_data = preprocessor.resample_from_json(json_data)
        features = preprocessor.extract_features(tf_data)
        laplacians = preprocessor.compute_laplacians(tf_data)
        print(f"  Timeframes: {list(features.keys())}")
        for tf, arr in features.items():
            print(f"    {tf}: {arr.shape}")
        print(f"  [step 2 done in {_elapsed(t_step)}]")

        # Build targets from M1 returns
        t_step = _time.time()
        print("[3/5] Computing targets and labels...")
        m1_arr = features.get("M1")
        if m1_arr is None:
            print("ERROR: No M1 data after preprocessing.")
            sys.exit(1)

        # Cross-sectional mean return as regression target
        n_bars = m1_arr.shape[0]
        # Use close price return of first pair as proxy target
        close_idx = 3  # feature index for close-derived return
        targets = m1_arr[1:, 0, close_idx] - m1_arr[:-1, 0, close_idx]
        targets = np.concatenate([[0.0], targets]).astype(np.float32)

        buy_thresh = np.percentile(targets, 67)
        sell_thresh = np.percentile(targets, 33)
        labels = np.ones(n_bars, dtype=np.int64)
        labels[targets > buy_thresh] = 2
        labels[targets < sell_thresh] = 0
        sell_n = int((labels == 0).sum())
        hold_n = int((labels == 1).sum())
        buy_n = int((labels == 2).sum())
        print(f"  Label distribution: sell={sell_n}, hold={hold_n}, buy={buy_n}")
        print(f"  [step 3 done in {_elapsed(t_step)}]")

        # Train
        t_step = _time.time()
        print("[4/5] Training STGNN...")
        trainer = STGNNTrainer(
            n_pairs=m1_arr.shape[1],
            n_features=N_FEATURES,
            temporal_hidden=256,
            spatial_out=32,
            seq_len=30,
            n_folds=args.n_folds,
            epochs=args.epochs,
            patience=15,
            batch_size=args.batch_size,
        )
        results = trainer.train_full(features, laplacians, targets, labels, verbose=True)
        print(f"  [step 4 training done in {_elapsed(t_step)}]")

        # Save models
        t_step = _time.time()
        print("[5/5] Saving models...")
        stgnn_path = model_dir / "stgnn_weights.pt"
        cb_path = model_dir / "catboost_exec.cbm"
        trainer.save(str(stgnn_path), str(cb_path))
        print(f"  STGNN saved  : {stgnn_path}")
        print(f"  CatBoost saved: {cb_path}")

        print(f"  [step 5 done in {_elapsed(t_step)}]")

        # Print final CV metrics
        print("\n--- Cross-Validation Results ---")
        for fold in results.get("fold_results", []):
            print(f"  Fold {fold['fold']}: MSE={fold['mse']:.6f}, R2={fold['r2']:.4f}, Acc={fold['accuracy']*100:.1f}%")

        total_time = _elapsed(t_total)
        print("=" * 60)
        print(f"Training complete.  Total wall-clock time: {total_time}")
        print("=" * 60)
        sys.exit(0)

    # --- Smoke test mode ---
    print("STGNN Hierarchical Model -- Smoke Test")
    print("=" * 50)

    rng = np.random.default_rng(42)
    n_bars = 200
    n_pairs = 10  # Reduced for speed
    pairs = sorted(PAIRS_ALL)[:n_pairs]

    # 1. Generate synthetic multi-timeframe data
    print("\n[1] Generating synthetic data...")
    features = {}
    laplacians = {}
    tf_bars = {
        "M1": n_bars, "M5": n_bars // 5, "M15": n_bars // 15,
        "M30": n_bars // 30, "H1": n_bars // 60,
    }

    for tf_name, tb in tf_bars.items():
        tb = max(30, tb)
        features[tf_name] = rng.normal(0, 1, (tb, n_pairs, N_FEATURES)).astype(np.float32)
        laplacians[tf_name] = [np.eye(n_pairs, dtype=np.float32) + rng.normal(0, 0.01, (n_pairs, n_pairs)).astype(np.float32) for _ in range(tb)]
        print(f"  {tf_name}: {features[tf_name].shape}")

    targets = rng.normal(0, 0.001, n_bars).astype(np.float32)
    labels = np.ones(n_bars, dtype=np.int64)
    labels[targets > 0.0001] = 2
    labels[targets < -0.0001] = 0

    # 2. Test ChebGraphConv
    print("\n[2] Testing ChebGraphConv...")
    cheb = ChebGraphConv(N_FEATURES, 16, K=3)
    x = torch.tensor(features["M1"][:5], dtype=torch.float32)
    L = torch.tensor(laplacians["M1"][0], dtype=torch.float32)
    out = cheb(x, L)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (5, n_pairs, 16)

    # 3. Test SpatialBlock
    print("\n[3] Testing SpatialBlock...")
    sb = SpatialBlock(N_FEATURES, 32, 16, K=3)
    out2 = sb(x, L)
    print(f"  SpatialBlock output: {out2.shape}")

    # 4. Test TemporalEncoder
    print("\n[4] Testing TemporalEncoder...")
    te = TemporalEncoder(n_pairs * 16, 64, n_layers=2)
    seq = out2.reshape(5, -1).unsqueeze(0)  # (1, 5, N*F)
    te_out, te_hn = te(seq)
    print(f"  GRU output: {te_out.shape}, hidden: {te_hn.shape}")

    # 5. Test HierarchicalExchange
    print("\n[5] Testing HierarchicalExchange...")
    n_tf = len(tf_bars)
    he = HierarchicalExchange(64, n_tf)
    hs = [torch.randn(2, 64) for _ in range(n_tf)]
    enriched = he(hs)
    print(f"  {n_tf} hidden states enriched, shapes: {[e.shape for e in enriched]}")

    # 6. Test full STGNN
    print("\n[6] Testing HierarchicalSTGNN forward pass...")
    model = HierarchicalSTGNN(
        n_pairs=n_pairs, n_features=N_FEATURES,
        n_timeframes=n_tf, spatial_out=16, temporal_hidden=64,
    )

    # Build batch: (1, T, N, F) per timeframe
    batch_feat = {}
    batch_lap = {}
    for tf_name in tf_bars:
        t_len = min(10, len(features[tf_name]))
        batch_feat[tf_name] = torch.tensor(
            features[tf_name][:t_len], dtype=torch.float32
        ).unsqueeze(0)
        batch_lap[tf_name] = torch.tensor(
            laplacians[tf_name][0], dtype=torch.float32
        )

    result = model(batch_feat, batch_lap)
    print(f"  Prediction: {result['prediction'].shape}")
    print(f"  CatBoost features: {result['features_for_catboost'].shape}")
    print(f"  Hidden states: {len(result['hidden_states'])}")

    # 7. Test CatBoost head
    if HAS_CATBOOST:
        print("\n[7] Testing CatBoostExecutionHead...")
        cb_feat = rng.normal(0, 1, (100, 64 * n_tf)).astype(np.float32)
        cb_labels = rng.choice([0, 1, 2], 100)
        cb = CatBoostExecutionHead(iterations=10, depth=3, verbose=False)
        cb.train(cb_feat[:80], cb_labels[:80], cb_feat[80:], cb_labels[80:])
        pred = cb.predict(cb_feat[80:])
        print(f"  Classes: {pred['classes'][:5]}")
        print(f"  Probabilities shape: {pred['probabilities'].shape}")
    else:
        print("\n[7] CatBoost not installed, skipping...")

    # 8. Test trainer (mini run)
    print("\n[8] Testing STGNNTrainer (1 fold, 5 epochs)...")
    trainer = STGNNTrainer(
        n_pairs=n_pairs, n_features=N_FEATURES,
        temporal_hidden=64, spatial_out=16,
        seq_len=10, n_folds=1, epochs=5,
        patience=3, batch_size=4,
    )
    train_result = trainer.train_full(features, laplacians, targets, labels, verbose=True)
    print(f"  Fold results: {train_result['fold_results']}")

    print("\n" + "=" * 50)
    print("STGNN Smoke Test PASSED")
    print("=" * 50)
