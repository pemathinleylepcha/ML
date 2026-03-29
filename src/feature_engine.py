"""
Algo C2 v2 — Feature Engine
v1: 16 per-pair indicators × 35 pairs + 8 graph = 568 features
v2: 16 per-pair × 29 tradeable + 7 signal-only × 14 + 8 graph = 570 features
    + cross-learning bridge (14) = 584 total per bar
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from math_engine import MathEngine, MathState

# ── Universe import (v2 constants; falls back to v1 if universe.py absent) ──

try:
    from universe import (
        ALL_INSTRUMENTS, TRADEABLE, SIGNAL_ONLY,
        PIP_SIZES as _UNIVERSE_PIP_SIZES,
        REGIONAL_CORRELATION_MAP,
    )
    _HAS_UNIVERSE = True
except ImportError:
    _HAS_UNIVERSE = False

# ── Constants ───────────────────────────────────────────────────────────────

EPS = 1e-10

# v1 pair lists — kept for backward compatibility
PAIRS_FX = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]
PAIRS_NON_FX = ["BTCUSD", "US30", "USDMXN", "USDZAR", "XAGUSD", "XAUUSD", "XBRUSD"]
PAIRS_ALL = sorted(PAIRS_FX + PAIRS_NON_FX)   # 35 instruments (v1)

_JPY_CROSSES = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP_SIZES: dict[str, float] = {}
for p in PAIRS_FX:
    PIP_SIZES[p] = 0.01 if p in _JPY_CROSSES else 0.0001
PIP_SIZES.update({
    "BTCUSD": 1.0, "US30": 1.0, "XAUUSD": 0.1,
    "XAGUSD": 0.01, "XBRUSD": 0.01, "USDMXN": 0.0001, "USDZAR": 0.0001,
})
# Extend with v2 signal-only instruments
if _HAS_UNIVERSE:
    for inst in SIGNAL_ONLY:
        if inst not in PIP_SIZES:
            PIP_SIZES[inst] = _UNIVERSE_PIP_SIZES.get(inst, 1.0)

# v2 pair list (43 instruments) — used by v2 feature matrix builder
PAIRS_ALL_V2: list[str] = ALL_INSTRUMENTS if _HAS_UNIVERSE else PAIRS_ALL


# ── Technical Indicators ───────────────────────────────────────────────────

def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """RSI(14). Returns 0-100."""
    if len(close) < period + 1:
        return np.nan
    deltas = np.diff(close[-(period + 1):])
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss < EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and histogram. Returns (macd_line, macd_hist) in price units."""
    if len(close) < slow + signal:
        return np.nan, np.nan
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_from_series(macd_line[-(signal + 10):], signal)
    hist = macd_line[-1] - signal_line
    return float(macd_line[-1]), float(hist)


def compute_bollinger(close: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands %B and bandwidth."""
    if len(close) < period:
        return np.nan, np.nan
    window = close[-period:]
    sma = window.mean()
    std = window.std(ddof=1)
    if std < EPS:
        return 0.5, 0.0
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (close[-1] - lower) / (upper - lower + EPS)
    bandwidth = (upper - lower) / (sma + EPS)
    return float(pct_b), float(bandwidth)


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> float:
    """ATR(14) in price units."""
    if len(close) < period + 1:
        return np.nan
    tr_vals = []
    for i in range(-period, 0):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        tr_vals.append(tr)
    return float(np.mean(tr_vals))


def compute_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       k_period: int = 14, d_period: int = 3, smooth: int = 3):
    """Stochastic %K and %D."""
    if len(close) < k_period + d_period + smooth:
        return np.nan, np.nan
    # Compute raw %K for enough bars to smooth
    raw_k = []
    for i in range(-(d_period + smooth), 0):
        start = i - k_period
        h = high[start:i + 1] if i + 1 != 0 else high[start:]
        l = low[start:i + 1] if i + 1 != 0 else low[start:]
        hh = h.max()
        ll = l.min()
        if hh - ll < EPS:
            raw_k.append(50.0)
        else:
            raw_k.append(100.0 * (close[i] - ll) / (hh - ll))
    raw_k = np.array(raw_k)
    # Smooth %K
    k_smoothed = np.convolve(raw_k, np.ones(smooth) / smooth, mode='valid')
    # %D = SMA of smoothed %K
    pct_k = k_smoothed[-1]
    pct_d = k_smoothed[-d_period:].mean()
    return float(pct_k), float(pct_d)


def compute_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 20) -> float:
    """CCI(20)."""
    if len(close) < period:
        return np.nan
    tp = (high[-period:] + low[-period:] + close[-period:]) / 3.0
    sma = tp.mean()
    mad = np.mean(np.abs(tp - sma))
    if mad < EPS:
        return 0.0
    return float((tp[-1] - sma) / (0.015 * mad))


def compute_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       period: int = 14) -> float:
    """Williams %R(14). Range: -100 to 0."""
    if len(close) < period:
        return np.nan
    hh = high[-period:].max()
    ll = low[-period:].min()
    if hh - ll < EPS:
        return -50.0
    return float(-100.0 * (hh - close[-1]) / (hh - ll))


# ── Helper functions ────────────────────────────────────────────────────────

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average over full array."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def _ema_from_series(data: np.ndarray, period: int) -> float:
    """EMA of a series, return last value."""
    if len(data) == 0:
        return 0.0
    result = _ema(data, period)
    return float(result[-1])


# ── Feature Matrix Builder ──────────────────────────────────────────────────

def compute_pair_features(ohlc: dict, pair: str, bar_idx: int,
                          lookback: int = 120) -> dict:
    """
    Compute 16 features for a single pair at a given bar index.

    Args:
        ohlc: dict with keys 'o', 'h', 'l', 'c', 'sp', 'tk' as arrays
        pair: instrument name
        bar_idx: current bar index (inclusive)
        lookback: how many bars to use for indicator computation

    Returns:
        dict of 16 feature values
    """
    pip = PIP_SIZES.get(pair, 0.0001)
    start = max(0, bar_idx - lookback + 1)
    end = bar_idx + 1

    o = ohlc["o"][start:end]
    h = ohlc["h"][start:end]
    l = ohlc["l"][start:end]
    c = ohlc["c"][start:end]
    sp = ohlc["sp"][start:end]
    tk = ohlc["tk"][start:end]

    if len(c) < 2:
        return {f"{pair}_{f}": np.nan for f in [
            "rsi", "macd", "macd_hist", "bb_pct_b", "bb_bandwidth",
            "atr_pips", "stoch_k", "stoch_d", "cci", "willr",
            "log_ret", "tick_vel", "spread_pip", "range_pips", "body_ratio", "mom5"
        ]}

    rsi = compute_rsi(c)
    macd_line, macd_hist = compute_macd(c)
    bb_pct_b, bb_bw = compute_bollinger(c)
    atr = compute_atr(h, l, c)
    atr_pips = atr / pip if atr is not np.nan else np.nan
    stoch_k, stoch_d = compute_stochastic(h, l, c)
    cci = compute_cci(h, l, c)
    willr = compute_williams_r(h, l, c)

    # Direct features
    log_ret = float(np.log(c[-1] / c[-2])) if c[-2] > 0 else 0.0
    tick_vel = float(tk[-1]) if len(tk) > 0 else 0.0
    spread_pip_val = float(sp[-1]) if len(sp) > 0 else 0.0
    range_pips = float((h[-1] - l[-1]) / pip)
    body = abs(c[-1] - o[-1])
    rng = h[-1] - l[-1]
    body_ratio = float(body / rng) if rng > EPS else 0.0

    # Mom5: 5-bar momentum z-score
    if len(c) >= 6:
        mom_raw = c[-1] - c[-6]
        mom_std = np.std(np.diff(c[-6:])) + EPS
        mom5 = float(mom_raw / mom_std)
    else:
        mom5 = 0.0

    # Convert MACD to pips
    if macd_line is not np.nan:
        macd_line = macd_line / pip
    if macd_hist is not np.nan:
        macd_hist = macd_hist / pip

    return {
        f"{pair}_rsi": rsi,
        f"{pair}_macd": macd_line,
        f"{pair}_macd_hist": macd_hist,
        f"{pair}_bb_pct_b": bb_pct_b,
        f"{pair}_bb_bandwidth": bb_bw,
        f"{pair}_atr_pips": atr_pips,
        f"{pair}_stoch_k": stoch_k,
        f"{pair}_stoch_d": stoch_d,
        f"{pair}_cci": cci,
        f"{pair}_willr": willr,
        f"{pair}_log_ret": log_ret,
        f"{pair}_tick_vel": tick_vel,
        f"{pair}_spread_pip": spread_pip_val,
        f"{pair}_range_pips": range_pips,
        f"{pair}_body_ratio": body_ratio,
        f"{pair}_mom5": mom5,
    }


def extract_graph_features(state: MathState) -> dict:
    """Extract 8 graph-level features from a MathState."""
    res = state.residuals
    return {
        "graph_residual_mean": float(np.mean(res)),
        "graph_residual_std": float(np.std(res)),
        "graph_residual_max": float(np.max(np.abs(res))),
        "graph_spectral_gap": float(state.spectral_gap),
        "graph_betti_h0": float(state.beta_0),
        "graph_betti_h1": float(state.beta_1),
        "graph_avg_correlation": float(np.mean(np.abs(
            state.correlation_matrix[np.triu_indices(state.correlation_matrix.shape[0], k=1)]
        ))),
        "graph_laplacian_trace": float(np.trace(state.laplacian_matrix)),
    }


def build_feature_matrix(data: dict, pairs: list[str] = None,
                         window: int = 120, step: int = 15,
                         target_pair: str = "EURUSD") -> pd.DataFrame:
    """
    Build the full 568-feature matrix with walk-forward windows.

    Args:
        data: dict of pair -> list of bar dicts [{dt, o, h, l, c, sp, tk}, ...]
        pairs: list of pair names to use (default: PAIRS_ALL)
        window: lookback window for indicators
        step: step size between samples
        target_pair: pair to use for label (next bar direction)

    Returns:
        DataFrame with 568 feature columns + 'label' + 'timestamp'
    """
    if pairs is None:
        pairs = PAIRS_ALL

    # Convert data to arrays
    ohlc_arrays = {}
    for pair in pairs:
        if pair not in data:
            continue
        bars = data[pair]
        ohlc_arrays[pair] = {
            "o": np.array([b["o"] for b in bars], dtype=float),
            "h": np.array([b["h"] for b in bars], dtype=float),
            "l": np.array([b["l"] for b in bars], dtype=float),
            "c": np.array([b["c"] for b in bars], dtype=float),
            "sp": np.array([b["sp"] for b in bars], dtype=float),
            "tk": np.array([b["tk"] for b in bars], dtype=float),
            "dt": [b["dt"] for b in bars],
        }

    available_pairs = [p for p in pairs if p in ohlc_arrays]
    n_pairs = len(available_pairs)
    pair_to_idx = {p: i for i, p in enumerate(available_pairs)}

    # Find minimum bar count across all pairs
    min_bars = min(len(ohlc_arrays[p]["c"]) for p in available_pairs)

    # Initialize math engine
    engine = MathEngine(n_pairs=n_pairs)

    # First pass: feed math engine all bars to build state
    math_states = []
    for t in range(min_bars):
        returns = np.zeros(n_pairs)
        for i, pair in enumerate(available_pairs):
            c = ohlc_arrays[pair]["c"]
            if t > 0 and c[t - 1] > 0:
                returns[i] = np.log(c[t] / c[t - 1])
        state = engine.update(returns)
        math_states.append(state)

    # Second pass: build feature rows at each step
    rows = []
    target_idx = pair_to_idx.get(target_pair)

    for t in range(window, min_bars - 1, step):
        row = {}
        row["timestamp"] = ohlc_arrays[available_pairs[0]]["dt"][t]

        # Per-pair features (16 x n_pairs)
        for pair in available_pairs:
            features = compute_pair_features(ohlc_arrays[pair], pair, t, lookback=window)
            row.update(features)

        # Graph features (8)
        if t < len(math_states) and math_states[t].valid:
            graph_feats = extract_graph_features(math_states[t])
            row.update(graph_feats)
        else:
            row.update({k: np.nan for k in [
                "graph_residual_mean", "graph_residual_std", "graph_residual_max",
                "graph_spectral_gap", "graph_betti_h0", "graph_betti_h1",
                "graph_avg_correlation", "graph_laplacian_trace",
            ]})

        # Label: next bar direction for target pair
        if target_idx is not None:
            c = ohlc_arrays[target_pair]["c"]
            if t + 1 < len(c) and c[t] > 0:
                row["label"] = 1 if c[t + 1] > c[t] else 0
            else:
                row["label"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Feature matrix: {len(df)} samples x {len(df.columns)} columns "
          f"({len(available_pairs)} pairs x 16 + 8 graph + label + timestamp)")
    return df


# ── v2: Signal-only feature extraction ──────────────────────────────────────

def compute_signal_only_features(ohlc: dict, instrument: str, bar_idx: int,
                                  lookback: int = 120,
                                  laplacian_residual: float = 0.0) -> dict:
    """
    Compute 7 cross-asset features for a signal-only node.

    Signal-only nodes (indices, metals, energy, exotic FX) contribute these
    features to the FX CatBoost input but never generate trade signals.

    Args:
        ohlc: dict with 'o','h','l','c','sp','tk' arrays
        instrument: instrument name (e.g. 'XAUUSD', 'AUS200')
        bar_idx: current bar index (inclusive)
        lookback: bars to use for indicator computation
        laplacian_residual: residual from the 43-node graph for this instrument

    Returns:
        dict of 7 features prefixed with instrument name
    """
    pip = PIP_SIZES.get(instrument, 1.0)
    start = max(0, bar_idx - lookback + 1)
    end = bar_idx + 1

    c = ohlc["c"][start:end]
    h = ohlc["h"][start:end]
    l = ohlc["l"][start:end]

    nan_result = {
        f"{instrument}_log_ret": np.nan,
        f"{instrument}_rsi_14": np.nan,
        f"{instrument}_macd_hist": np.nan,
        f"{instrument}_bb_bandwidth": np.nan,
        f"{instrument}_atr_14": np.nan,
        f"{instrument}_cci_20": np.nan,
        f"{instrument}_laplacian_residual": laplacian_residual,
    }

    if len(c) < 2:
        return nan_result

    log_ret = float(np.log(c[-1] / c[-2])) if c[-2] > EPS else 0.0
    rsi_14 = compute_rsi(c, period=14)
    _, macd_hist = compute_macd(c)
    _, bb_bw = compute_bollinger(c)
    atr_14 = compute_atr(h, l, c, period=14)
    cci_20 = compute_cci(h, l, c, period=20)

    # Normalise MACD histogram to pips (consistent with compute_pair_features)
    if macd_hist is not np.nan and not np.isnan(macd_hist):
        macd_hist = macd_hist / pip

    return {
        f"{instrument}_log_ret": log_ret,
        f"{instrument}_rsi_14": rsi_14,
        f"{instrument}_macd_hist": float(macd_hist) if macd_hist is not np.nan else np.nan,
        f"{instrument}_bb_bandwidth": float(bb_bw) if bb_bw is not np.nan else np.nan,
        f"{instrument}_atr_14": float(atr_14 / pip) if atr_14 is not np.nan else np.nan,
        f"{instrument}_cci_20": float(cci_20) if cci_20 is not np.nan else np.nan,
        f"{instrument}_laplacian_residual": laplacian_residual,
    }


# ── v2: Full 43-node feature matrix builder ──────────────────────────────────

def build_feature_matrix_v2(data: dict,
                             tradeable: list[str] = None,
                             signal_only: list[str] = None,
                             window: int = 120,
                             step: int = 15,
                             target_pair: str = "EURUSD") -> pd.DataFrame:
    """
    Build the v2 feature matrix for all 43 nodes.

    Features per row:
      - 16 per-pair indicators × len(tradeable) = 464 for 29 pairs
      - 7 cross-asset features × len(signal_only) = 98 for 14 nodes
      - 8 graph-level features from the 43-node Laplacian
      Total: 570 features + label + timestamp

    Args:
        data: dict of instrument -> list of bar dicts [{dt, o, h, l, c, sp, tk}, ...]
        tradeable: list of tradeable instruments (default: TRADEABLE from universe)
        signal_only: list of signal-only instruments (default: SIGNAL_ONLY from universe)
        window: lookback window for indicators
        step: step size between samples
        target_pair: pair used to generate the direction label

    Returns:
        DataFrame with feature columns + 'label' + 'timestamp'
    """
    if tradeable is None:
        tradeable = TRADEABLE if _HAS_UNIVERSE else PAIRS_ALL
    if signal_only is None:
        signal_only = SIGNAL_ONLY if _HAS_UNIVERSE else []

    all_instruments = tradeable + [s for s in signal_only if s not in tradeable]

    # Convert data to arrays
    ohlc_arrays: dict[str, dict] = {}
    for inst in all_instruments:
        if inst not in data:
            continue
        bars = data[inst]
        ohlc_arrays[inst] = {
            "o":  np.array([b["o"] for b in bars], dtype=float),
            "h":  np.array([b["h"] for b in bars], dtype=float),
            "l":  np.array([b["l"] for b in bars], dtype=float),
            "c":  np.array([b["c"] for b in bars], dtype=float),
            "sp": np.array([b["sp"] for b in bars], dtype=float),
            "tk": np.array([b["tk"] for b in bars], dtype=float),
            "dt": [b["dt"] for b in bars],
        }

    avail_tradeable = [p for p in tradeable if p in ohlc_arrays]
    avail_signal    = [p for p in signal_only if p in ohlc_arrays]
    avail_all       = avail_tradeable + avail_signal
    n_nodes = len(avail_all)

    if not avail_tradeable:
        raise ValueError("No tradeable instruments found in data")

    # Canonical index for Laplacian residuals
    node_idx_map = {inst: i for i, inst in enumerate(avail_all)}

    min_bars = min(len(ohlc_arrays[p]["c"]) for p in avail_all)

    # Build math engine at full 43-node size
    engine = MathEngine(n_pairs=n_nodes)

    # First pass: feed returns to build Laplacian state
    math_states: list[MathState] = []
    for t in range(min_bars):
        returns = np.zeros(n_nodes)
        for i, inst in enumerate(avail_all):
            c = ohlc_arrays[inst]["c"]
            if t > 0 and c[t - 1] > EPS:
                returns[i] = np.log(c[t] / c[t - 1])
        state = engine.update(returns)
        math_states.append(state)

    # Second pass: build feature rows
    rows = []
    for t in range(window, min_bars - 1, step):
        row: dict = {}
        row["timestamp"] = ohlc_arrays[avail_all[0]]["dt"][t]

        # ── Tradeable pair features (16 each) ──
        for pair in avail_tradeable:
            feats = compute_pair_features(ohlc_arrays[pair], pair, t, lookback=window)
            row.update(feats)

        # ── Signal-only cross-asset features (7 each) ──
        state_t = math_states[t] if t < len(math_states) else None
        for inst in avail_signal:
            lap_res = 0.0
            if state_t is not None and state_t.valid:
                idx = node_idx_map.get(inst, -1)
                if 0 <= idx < len(state_t.residuals):
                    lap_res = float(state_t.residuals[idx])
            feats = compute_signal_only_features(
                ohlc_arrays[inst], inst, t, lookback=window,
                laplacian_residual=lap_res,
            )
            row.update(feats)

        # ── Graph features (8) ──
        if state_t is not None and state_t.valid:
            row.update(extract_graph_features(state_t))
        else:
            row.update({k: np.nan for k in [
                "graph_residual_mean", "graph_residual_std", "graph_residual_max",
                "graph_spectral_gap", "graph_betti_h0", "graph_betti_h1",
                "graph_avg_correlation", "graph_laplacian_trace",
            ]})

        # ── Label: next bar direction for target pair ──
        if target_pair in ohlc_arrays:
            c = ohlc_arrays[target_pair]["c"]
            if t + 1 < len(c) and c[t] > EPS:
                row["label"] = 1 if c[t + 1] > c[t] else 0
            else:
                row["label"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    n_tradeable_feats = len(avail_tradeable) * 16
    n_signal_feats = len(avail_signal) * 7
    print(
        f"v2 feature matrix: {len(df)} samples × {len(df.columns)} columns "
        f"({len(avail_tradeable)} tradeable×16={n_tradeable_feats} + "
        f"{len(avail_signal)} signal-only×7={n_signal_feats} + 8 graph + 2 meta)"
    )
    return df


# ── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Feature engine smoke test with synthetic data...")
    rng = np.random.default_rng(42)
    n_bars = 300
    pairs = PAIRS_ALL[:5]  # use 5 pairs for speed

    # Generate synthetic OHLC data
    data = {}
    for pair in pairs:
        base = 1.0 + rng.random() * 0.5
        closes = base + np.cumsum(rng.normal(0, 0.001, n_bars))
        data[pair] = []
        for i in range(n_bars):
            c = closes[i]
            h = c + abs(rng.normal(0, 0.0005))
            l = c - abs(rng.normal(0, 0.0005))
            o = c + rng.normal(0, 0.0003)
            data[pair].append({
                "dt": f"2026-03-02 {i // 60:02d}:{i % 60:02d}",
                "o": round(o, 5), "h": round(h, 5),
                "l": round(l, 5), "c": round(c, 5),
                "sp": round(rng.uniform(0.5, 3.0), 2),
                "tk": int(rng.integers(5, 100)),
            })

    df = build_feature_matrix(data, pairs=pairs, window=60, step=10,
                              target_pair=pairs[0])
    print(f"Shape: {df.shape}")
    print(f"Non-NaN features in first row: {df.iloc[0].notna().sum()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print("OK")
