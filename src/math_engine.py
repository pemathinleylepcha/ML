"""
Algo C2 v2 — Graph Laplacian + TDA Math Engine
Implements the normalized graph Laplacian, local residuals, spectral gap,
persistent homology proxies, and regime classification.

v2: Supports dynamic n_pairs (35 for v1, 43 for v2 dual-subnet architecture).
TDA regime thresholds scale proportionally with n_pairs.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh

# ── Universe import (optional — falls back to v1 defaults if universe.py absent) ──

try:
    from universe import N_NODES as _DEFAULT_N, SPECTRAL_GAP_WARN as _SPECTRAL_GAP_WARN
except ImportError:
    _DEFAULT_N = 35
    _SPECTRAL_GAP_WARN = 0.005

# ── Constants ───────────────────────────────────────────────────────────────

EPS = 1e-10
ROLLING_WINDOW = 60
MIN_WINDOW = 10
TDA_EMA_ALPHA = 0.3
RESIDUAL_DEAD_ZONE = 1e-04  # calibrated to FX log-return residual scale (p50≈6e-5, p75≈1.5e-4)
SPECTRAL_GAP_WARN = _SPECTRAL_GAP_WARN   # 0.004 for 43-node, 0.005 for 35-node
SPECTRAL_RECOMPUTE_INTERVAL = 5
STREAK_BONUS_THRESHOLD = 3

# Reference node count for threshold scaling (v1 baseline)
_TDA_BASELINE_N = 35


def _make_math_state(n: int) -> "MathState":
    """Create a MathState with correctly-sized arrays for n nodes."""
    s = MathState.__new__(MathState)
    s.bar_index = 0
    s.residuals = np.zeros(n)
    s.spectral_gap = 0.0
    s.beta_0 = 1
    s.beta_1 = 0
    s.h1_lifespan = 0.0
    s.regime = "NORMAL"
    s.sigma = 0.0
    s.correlation_matrix = np.eye(n)
    s.distance_matrix = np.zeros((n, n))
    s.adjacency_matrix = np.zeros((n, n))
    s.laplacian_matrix = np.eye(n)
    s.eigenvalues = np.zeros(n)
    s.streaks = np.zeros(n)
    s.valid = False
    return s


@dataclass
class MathState:
    """Output of one math engine update step."""
    bar_index: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(_DEFAULT_N))
    spectral_gap: float = 0.0
    beta_0: int = 1
    beta_1: int = 0
    h1_lifespan: float = 0.0
    regime: str = "NORMAL"
    sigma: float = 0.0
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.eye(_DEFAULT_N))
    distance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((_DEFAULT_N, _DEFAULT_N)))
    adjacency_matrix: np.ndarray = field(default_factory=lambda: np.zeros((_DEFAULT_N, _DEFAULT_N)))
    laplacian_matrix: np.ndarray = field(default_factory=lambda: np.eye(_DEFAULT_N))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(_DEFAULT_N))
    streaks: np.ndarray = field(default_factory=lambda: np.zeros(_DEFAULT_N))
    valid: bool = False


class MathEngine:
    """
    Core math engine for Algo C2.
    Maintains rolling state and computes graph-theoretic + topological features.
    """

    def __init__(self, n_pairs: int = _DEFAULT_N, rolling_window: int = ROLLING_WINDOW,
                 corr_shrinkage: float = 0.15):
        self.n = n_pairs
        self.window = rolling_window
        self.corr_shrinkage = corr_shrinkage   # toward-identity shrinkage (critics.md item 5)

        # Fix #14: Return ring buffer
        self.return_buffer: deque[np.ndarray] = deque(maxlen=rolling_window)

        # Cached state
        self._cached_spectral_gap = 0.0
        self._cached_eigenvalues = np.zeros(n_pairs)
        self._last_spectral_bar = -SPECTRAL_RECOMPUTE_INTERVAL  # force first computation

        # TDA EMA state (Fix #5)
        self._h1_ema = 0.0

        # Fix #9: Residual streak tracking per pair
        self._streaks = np.zeros(n_pairs)
        self._last_residual_sign = np.zeros(n_pairs)

        # Bar counter
        self._bar_count = 0

    def update(self, returns: np.ndarray) -> MathState:
        """
        Process one bar of returns for all pairs.

        Args:
            returns: 1D array of shape (n_pairs,) — log returns for the current bar.

        Returns:
            MathState with all computed features.

        Fix #7: Returns are pushed to buffer AFTER Laplacian computation
        to prevent cross-asset leakage.
        """
        state = _make_math_state(self.n)
        state.bar_index = self._bar_count
        self._bar_count += 1

        # Need at least MIN_WINDOW bars of history before computing
        if len(self.return_buffer) < MIN_WINDOW:
            self.return_buffer.append(returns.copy())
            return state

        # ── Step 1: Rolling correlation (Fix #13: 60-bar rolling) ──
        # Use buffer BEFORE pushing current returns (Fix #7: leakage prevention)
        hist = np.array(self.return_buffer)  # shape: (T, 35)
        corr = self._rolling_correlation(hist)
        state.correlation_matrix = corr

        # ── Step 2: Mantegna distance (Fix #2: ε-regularized) ──
        dist = self._mantegna_distance(corr)
        state.distance_matrix = dist  # Fix #16: shared distance matrix

        # ── Step 3: Gaussian kernel adjacency (Fix #1: adaptive σ) ──
        adj, sigma = self._gaussian_adjacency(dist)
        state.adjacency_matrix = adj
        state.sigma = sigma

        # ── Step 4: Normalized Laplacian (Fix #3: zero-degree guards) ──
        lap, degrees = self._normalized_laplacian(adj)
        state.laplacian_matrix = lap

        # ── Step 5: Spectral gap λ₂ (Fix #4: throttled, Fix #6: computed) ──
        should_recompute = (self._bar_count - self._last_spectral_bar) >= SPECTRAL_RECOMPUTE_INTERVAL
        if should_recompute:
            eigenvalues = self._eigendecomposition(lap)
            self._cached_eigenvalues = eigenvalues
            self._cached_spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
            self._last_spectral_bar = self._bar_count

        state.eigenvalues = self._cached_eigenvalues
        state.spectral_gap = self._cached_spectral_gap

        # ── Step 6: Local residuals (Fix #8: dead zone 0.02) ──
        residuals = self._local_residuals(adj, degrees, returns)
        state.residuals = residuals

        # ── Step 7: Streak tracking (Fix #9) ──
        self._update_streaks(residuals)
        state.streaks = self._streaks.copy()

        # ── Step 8: TDA features (Fix #5: β₀, β₁, H₁ with EMA) ──
        beta_0, beta_1, h1_raw = self._compute_tda(state.eigenvalues)
        self._h1_ema = TDA_EMA_ALPHA * h1_raw + (1 - TDA_EMA_ALPHA) * self._h1_ema
        state.beta_0 = beta_0
        state.beta_1 = beta_1
        state.h1_lifespan = self._h1_ema

        # ── Step 9: Regime classification ──
        state.regime = self._classify_regime(beta_0, beta_1, self._h1_ema, state.spectral_gap)

        # ── NOW push returns to buffer (Fix #7) ──
        self.return_buffer.append(returns.copy())

        state.valid = True
        return state

    # ── Internal methods ────────────────────────────────────────────────────

    def _rolling_correlation(self, hist: np.ndarray) -> np.ndarray:
        """Pearson correlation matrix from rolling return history."""
        # hist shape: (T, n_pairs)
        if hist.shape[0] < 2:
            return np.eye(self.n)

        # Center each column
        centered = hist - hist.mean(axis=0)
        stds = hist.std(axis=0, ddof=1)
        stds[stds < EPS] = EPS  # avoid division by zero

        # Correlation
        corr = (centered.T @ centered) / (hist.shape[0] - 1)
        outer_std = np.outer(stds, stds)
        corr = corr / outer_std

        # Clip to valid range
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)

        # Shrinkage toward identity (critics.md item 5): 0.85*corr + 0.15*I
        # Pulls off-diagonal correlations toward 0, stabilising the Laplacian
        # when the rolling window is short or assets are temporarily decorrelated.
        if self.corr_shrinkage > 0.0:
            corr = (1.0 - self.corr_shrinkage) * corr + self.corr_shrinkage * np.eye(self.n)

        return corr

    def _mantegna_distance(self, corr: np.ndarray) -> np.ndarray:
        """
        Mantegna distance: d_ij = sqrt(2(1 - corr_ij))
        Fix #2: ε-regularization to prevent NaN when corr ≈ 1.0
        """
        inner = 2.0 * (1.0 - corr)
        np.maximum(inner, EPS, out=inner)  # Fix #2
        dist = np.sqrt(inner)
        np.fill_diagonal(dist, 0.0)
        return dist

    def _gaussian_adjacency(self, dist: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Gaussian kernel adjacency matrix: A_ij = exp(-d²/2σ²)
        Fix #1: adaptive σ = median(D), fallback to mean if degenerate
        """
        upper_idx = np.triu_indices(self.n, k=1)
        upper_dists = dist[upper_idx]

        # Adaptive sigma (Fix #1)
        sigma = float(np.median(upper_dists))
        if sigma < 1e-6:
            sigma = float(np.mean(upper_dists))
        if sigma < EPS:
            sigma = 1.0  # ultimate fallback

        adj = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
        np.fill_diagonal(adj, 0.0)

        return adj, sigma

    def _normalized_laplacian(self, adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        Fix #3: zero-degree node guard
        """
        degrees = adj.sum(axis=1)

        # Fix #3: guard zero-degree nodes
        safe_degrees = degrees.copy()
        safe_degrees[safe_degrees < EPS] = EPS

        d_inv_sqrt = 1.0 / np.sqrt(safe_degrees)
        D_inv_sqrt = np.diag(d_inv_sqrt)

        lap = np.eye(self.n) - D_inv_sqrt @ adj @ D_inv_sqrt

        return lap, degrees

    def _eigendecomposition(self, lap: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of the Laplacian (sorted ascending)."""
        eigenvalues = eigh(lap, eigvals_only=True)
        eigenvalues.sort()
        # Clamp small negatives from numerical noise
        eigenvalues[eigenvalues < 0] = 0.0
        return eigenvalues

    def _local_residuals(self, adj: np.ndarray, degrees: np.ndarray,
                         returns: np.ndarray) -> np.ndarray:
        """
        Local residual per pair: ε_i = r_i - (A_i · r) / degree_i
        Fix #3: disconnected nodes get residual = 0
        Fix #8: dead zone of 0.02
        """
        residuals = np.zeros(self.n)
        for i in range(self.n):
            if degrees[i] < EPS:
                residuals[i] = 0.0  # Fix #3: disconnected
            else:
                neighbor_avg = np.dot(adj[i], returns) / degrees[i]
                residuals[i] = returns[i] - neighbor_avg

        return residuals

    def _update_streaks(self, residuals: np.ndarray):
        """
        Fix #9: Track consecutive bars where residual has the same sign.
        After STREAK_BONUS_THRESHOLD bars, signal gets log2(streak) bonus.
        """
        for i in range(self.n):
            if abs(residuals[i]) < RESIDUAL_DEAD_ZONE:
                sign = 0.0
            else:
                sign = 1.0 if residuals[i] > 0 else -1.0

            if sign == 0.0:
                self._streaks[i] = 0
                self._last_residual_sign[i] = 0
            elif sign == self._last_residual_sign[i]:
                self._streaks[i] += 1
            else:
                self._streaks[i] = 1
                self._last_residual_sign[i] = sign

    def get_streak_bonus(self, pair_idx: int) -> float:
        """Get the log2 streak bonus for a pair (0 if below threshold)."""
        streak = self._streaks[pair_idx]
        if streak >= STREAK_BONUS_THRESHOLD:
            return math.log2(streak)
        return 0.0

    def _compute_tda(self, eigenvalues: np.ndarray) -> tuple[int, int, float]:
        """
        Compute TDA features from eigenspectrum proxy.
        Fix #5: β₀, β₁, and H₁ max lifespan.

        β₀: connected components ≈ count of eigenvalues near zero
        β₁: cycle count ≈ count of eigenvalues in mid-range
        H₁ lifespan: max persistence of H₁ features
        """
        if len(eigenvalues) == 0:
            return 1, 0, 0.0

        # β₀: number of near-zero eigenvalues (connected components)
        beta_0 = int(np.sum(eigenvalues < 0.01))
        beta_0 = max(beta_0, 1)  # at least 1 component

        # β₁: eigenvalues in mid-range [0.3, 0.7] of max eigenvalue
        max_eig = eigenvalues[-1] if eigenvalues[-1] > EPS else 1.0
        mid_low = 0.3 * max_eig
        mid_high = 0.7 * max_eig
        mid_range = eigenvalues[(eigenvalues >= mid_low) & (eigenvalues <= mid_high)]
        beta_1 = len(mid_range)

        # H₁ max lifespan: spread of mid-range eigenvalues
        if len(mid_range) >= 2:
            h1_lifespan = float(mid_range[-1] - mid_range[0])
        else:
            h1_lifespan = 0.0

        return beta_0, beta_1, h1_lifespan

    def _classify_regime(self, beta_0: int, beta_1: int,
                         h1_lifespan: float, spectral_gap: float) -> str:
        """
        Regime classification based on TDA features.
        Order matters — most severe first.
        Thresholds scale proportionally with n_pairs relative to v1 baseline (35 nodes):
          FRAGMENTED:   b0 > round(20 * n/35)  →  35: >20,  43: >25
          TRANSITIONAL: b1 > round(4  * n/35)  →  35: >4,   43: >5
        H1_lifespan thresholds are topological and do not scale with N.
        """
        scale = self.n / _TDA_BASELINE_N
        frag_thr = round(20 * scale)
        trans_thr = round(4 * scale)

        if beta_0 > frag_thr:
            return "FRAGMENTED"
        if h1_lifespan > 0.8:
            return "HIGH_STRESS"
        if beta_1 > trans_thr:
            return "TRANSITIONAL"
        if beta_1 <= 1 and h1_lifespan < 0.2:
            return "LOW_VOL"
        return "NORMAL"


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Math engine smoke test...")
    engine = MathEngine(n_pairs=35)
    rng = np.random.default_rng(42)

    for i in range(80):
        returns = rng.normal(0, 0.001, 35)
        state = engine.update(returns)

    print(f"Bar {state.bar_index}, valid={state.valid}")
    print(f"Spectral gap L2 = {state.spectral_gap:.4f}")
    print(f"sigma = {state.sigma:.4f}")
    print(f"B0 = {state.beta_0}, B1 = {state.beta_1}")
    print(f"H1 lifespan (EMA) = {state.h1_lifespan:.4f}")
    print(f"Regime: {state.regime}")
    print(f"Top residuals: {np.argsort(np.abs(state.residuals))[-5:][::-1]}")
    print(f"Max |eps| = {np.max(np.abs(state.residuals)):.6f}")
    print("OK")
