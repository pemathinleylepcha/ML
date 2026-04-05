from __future__ import annotations

import numpy as np

from universe import EXOTIC_FX, FX_PAIRS, INDICES, METALS, ENERGY


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    degree = adj.sum(axis=-1, keepdims=True)
    degree[degree < 1e-8] = 1.0
    return adj / degree


def rolling_correlation_adjacency(close_window: np.ndarray, shrinkage: float = 0.15) -> np.ndarray:
    if close_window.ndim != 2:
        raise ValueError("close_window must have shape [steps, nodes]")
    n_nodes = close_window.shape[1]
    if n_nodes <= 1:
        return np.eye(n_nodes, dtype=np.float32)
    if close_window.shape[0] < 3:
        return np.eye(n_nodes, dtype=np.float32)
    safe_close = np.maximum(close_window.astype(np.float64, copy=False), 1e-10)
    returns = np.zeros_like(safe_close)
    returns[1:] = np.diff(np.log(safe_close), axis=0)
    active_returns = returns[1:]
    if active_returns.shape[0] < 2:
        return np.eye(n_nodes, dtype=np.float32)
    corr = np.eye(n_nodes, dtype=np.float64)
    stddev = np.std(active_returns, axis=0)
    variable_mask = stddev > 1e-12
    if int(np.count_nonzero(variable_mask)) > 1:
        variable_returns = active_returns[:, variable_mask]
        centered = variable_returns - np.mean(variable_returns, axis=0, keepdims=True)
        sample_count = max(centered.shape[0] - 1, 1)
        covariance = (centered.T @ centered) / float(sample_count)
        variable_std = np.std(variable_returns, axis=0, ddof=1)
        denom = np.outer(variable_std, variable_std)
        variable_corr = np.divide(
            covariance,
            denom,
            out=np.zeros_like(covariance),
            where=denom > 1e-12,
        )
        variable_corr = np.nan_to_num(variable_corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(variable_corr, 1.0)
        corr[np.ix_(variable_mask, variable_mask)] = variable_corr
    corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0], dtype=np.float64)
    inner = np.maximum(2.0 * (1.0 - corr), 1e-8)
    dist = np.sqrt(inner)
    sigma = float(np.median(dist[np.triu_indices(dist.shape[0], k=1)])) if dist.shape[0] > 1 else 1.0
    sigma = max(sigma, 1e-4)
    adj = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(adj, 0.0)
    return _normalize_adjacency(adj).astype(np.float32)


def fundamental_adjacency(node_names: tuple[str, ...]) -> np.ndarray:
    n = len(node_names)
    adj = np.zeros((n, n), dtype=np.float32)
    for i, left in enumerate(node_names):
        left_group = _group(left)
        for j, right in enumerate(node_names):
            if i == j:
                continue
            right_group = _group(right)
            if left_group == right_group:
                adj[i, j] = 1.0
            elif _shares_currency(left, right):
                adj[i, j] = 0.8
    return _normalize_adjacency(adj)


def session_adjacency(node_names: tuple[str, ...], market_open: np.ndarray) -> np.ndarray:
    n = len(node_names)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not market_open[i]:
            continue
        for j in range(n):
            if i != j and market_open[j]:
                adj[i, j] = 1.0
    return _normalize_adjacency(adj)


def build_edge_matrices(
    close_window: np.ndarray,
    node_names: tuple[str, ...],
    market_open: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "rolling_corr": rolling_correlation_adjacency(close_window),
        "fundamental": fundamental_adjacency(node_names),
        "session": session_adjacency(node_names, market_open),
    }


def _shares_currency(left: str, right: str) -> bool:
    if len(left) < 6 or len(right) < 6:
        return False
    left_base, left_quote = left[:3], left[3:6]
    right_base, right_quote = right[:3], right[3:6]
    return len({left_base, left_quote, right_base, right_quote}) < 4


def _group(symbol: str) -> str:
    if symbol == "BTCUSD":
        return "btc"
    if symbol in FX_PAIRS:
        return "fx"
    if symbol in INDICES:
        return "indices"
    if symbol in METALS:
        return "metals"
    if symbol in ENERGY:
        return "energy"
    if symbol in EXOTIC_FX:
        return "exotic"
    return "other"
