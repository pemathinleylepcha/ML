"""
Algo C2 v2 — Signal Pipeline
7-indicator composite voting, 6-gate filter, order placement logic,
position sizing, session whitelists, and v2 execution guard.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from math_engine import MathState, RESIDUAL_DEAD_ZONE, SPECTRAL_GAP_WARN

try:
    from market_neutral_model import predict_weights, weights_to_confidence, HAS_TORCH
except ImportError:
    HAS_TORCH = False

# v2: import tradeable set for execution guard
try:
    from universe import TRADEABLE as _TRADEABLE_SET, SIGNAL_ONLY as _SIGNAL_ONLY_SET
    _TRADEABLE_FROZENSET = frozenset(_TRADEABLE_SET)
    _SIGNAL_ONLY_FROZENSET = frozenset(_SIGNAL_ONLY_SET)
except ImportError:
    _TRADEABLE_FROZENSET = frozenset()
    _SIGNAL_ONLY_FROZENSET = frozenset()

# -- Constants ---------------------------------------------------------------

EPS = 1e-10

PAIRS_FX = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]
PAIRS_NON_FX = ["BTCUSD", "US30", "USDMXN", "USDZAR", "XAGUSD", "XAUUSD", "XBRUSD"]
PAIRS_ALL = sorted(PAIRS_FX + PAIRS_NON_FX)
FX_SET = set(PAIRS_FX)

_JPY_CROSSES = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP_SIZES = {}
for p in PAIRS_FX:
    PIP_SIZES[p] = 0.01 if p in _JPY_CROSSES else 0.0001
PIP_SIZES.update({
    "BTCUSD": 1.0, "US30": 1.0, "XAUUSD": 0.1,
    "XAGUSD": 0.01, "XBRUSD": 0.01, "USDMXN": 0.0001, "USDZAR": 0.0001,
})

# Pip value per standard lot (USD)
PIP_VALUES = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY": 6.70, "EURJPY": 6.70, "GBPJPY": 6.70, "AUDJPY": 6.70,
    "CADJPY": 6.70, "CHFJPY": 6.70, "NZDJPY": 6.70,
    "USDCHF": 11.10, "EURCHF": 11.10, "GBPCHF": 11.10, "AUDCHF": 11.10,
    "CADCHF": 11.10, "NZDCHF": 11.10,
    "EURGBP": 12.70,
    "USDCAD": 7.40, "AUDCAD": 7.40, "EURCAD": 7.40, "GBPCAD": 7.40,
    "NZDCAD": 7.30, "CADCHF": 7.50,
    "AUDNZD": 10.0, "EURNZD": 10.0, "GBPNZD": 10.0,
    "EURAUD": 10.0, "GBPAUD": 10.0,
}

# Default config
INIT_BALANCE = 50.0
LEVERAGE = 500
RISK_PCT = 0.02
CB_THR = 0.55  # proxy threshold (heuristic CB proxy; restore to 0.65 when real CatBoost model is loaded)
NET_SCORE_THR = 0.15
AGREE_R_THR = 0.50
ATR_LO = 0.3
ATR_HI = 2.8
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5
MIN_SL_SPREAD_MULT = 3.0   # Fix #11
MIN_TP_SPREAD_MULT = 5.0   # Fix #11
TIMEOUT_BARS = 30                # hold up to 30 bars before forced exit
MARGIN_LIMIT = 0.85
LOT_MIN = 0.001
LOT_MAX = 10.0
MAX_POSITIONS_PER_SIGNAL = 1    # one position at a time on $50 account
MIN_SIGNAL_GAP = 20             # minimum bars between new entries (cooldown)

# Session whitelists (UTC hours)
SESSION_WHITELISTS = {
    "Sydney": {"hours": (0, 3), "pairs": {"AUDUSD", "AUDNZD", "AUDCAD", "NZDUSD", "AUDCHF"}},
    "Tokyo":  {"hours": (3, 8), "pairs": {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY"}},
    "London": {"hours": (8, 17), "pairs": {"EURUSD", "GBPUSD", "EURGBP", "GBPCHF", "EURCHF"}},
    "NY":     {"hours": (17, 24), "pairs": {"EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"}},
}


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class GateResult:
    """Result of the 6-gate filter check."""
    passed: bool = False
    g1_net_score: bool = False
    g2_cb_floor: bool = False
    g3_ema_trend: bool = False
    g4_atr_window: bool = False
    g5_agreement: bool = False
    g6_regime: bool = False
    net_score: float = 0.0
    direction: Direction = Direction.FLAT


@dataclass
class Order:
    """A pending or open order."""
    pair: str = ""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    lot: float = 0.0
    margin: float = 0.0
    entry_bar: int = 0
    timeout_bar: int = 0
    # Fix #12: entry metadata
    entry_residual: float = 0.0
    entry_cb: float = 0.0
    entry_regime: str = "NORMAL"
    entry_spectral_gap: float = 0.0
    entry_streak: float = 0.0
    spread_cost_entry: float = 0.0


@dataclass
class ClosedTrade:
    """A completed trade with PnL."""
    pair: str = ""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    exit_price: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    lot: float = 0.0
    pnl_usd: float = 0.0
    pnl_pips: float = 0.0
    spread_cost: float = 0.0
    entry_bar: int = 0
    exit_bar: int = 0
    exit_reason: str = ""  # "TP", "SL", "TIMEOUT"
    entry_regime: str = "NORMAL"
    entry_residual: float = 0.0
    entry_cb: float = 0.0


# -- Indicator Voting --------------------------------------------------------

def compute_votes(rsi: float, macd: float, bb_pct_b: float,
                  stoch_k: float, cci: float, willr: float,
                  mom5: float) -> int:
    """
    7-indicator composite vote.
    Returns integer vote count: positive = bullish, negative = bearish.
    """
    votes = 0
    if not np.isnan(rsi):
        votes += 1 if rsi > 55 else (-1 if rsi < 45 else 0)
    if not np.isnan(macd):
        votes += 1 if macd > 0 else (-1 if macd < 0 else 0)
    if not np.isnan(bb_pct_b):
        votes += 1 if bb_pct_b > 0.6 else (-1 if bb_pct_b < 0.4 else 0)
    if not np.isnan(stoch_k):
        votes += 1 if stoch_k > 60 else (-1 if stoch_k < 40 else 0)
    if not np.isnan(cci):
        votes += 1 if cci > 50 else (-1 if cci < -50 else 0)
    if not np.isnan(willr):
        votes += 1 if willr > -40 else (-1 if willr < -60 else 0)
    if not np.isnan(mom5):
        votes += 1 if mom5 > 0.3 else (-1 if mom5 < -0.3 else 0)
    return votes


def vote_to_direction(votes: int) -> Direction:
    """Convert vote count to signal direction."""
    if votes >= 3:
        return Direction.LONG
    elif votes <= -3:
        return Direction.SHORT
    return Direction.FLAT


# -- Net Score ---------------------------------------------------------------

def compute_net_score(residuals: np.ndarray) -> tuple[float, Direction]:
    """
    Net score from all 35 residuals.
    Dead-band: |residual| < 0.05 counts as neutral.
    """
    long_count = np.sum(residuals > 1e-04)   # calibrated to FX log-return residual scale
    short_count = np.sum(residuals < -1e-04)
    net = (long_count - short_count) / len(residuals)
    if net > 0:
        return net, Direction.LONG
    elif net < 0:
        return net, Direction.SHORT
    return 0.0, Direction.FLAT


# -- CB Proxy ----------------------------------------------------------------

def compute_cb_proxy(votes: int, residual: float, streak: float) -> float:
    """
    Momentum-based CatBoost confidence proxy.
    Will be replaced by actual CatBoost model predictions.
    Returns value in [0, 1].
    """
    # Vote strength component (0-1)
    vote_strength = min(abs(votes) / 7.0, 1.0)
    # Residual magnitude component (0-1), capped at 0.005
    res_strength = min(abs(residual) / 5e-04, 1.0)  # cap at p95 of residual distribution
    # Streak bonus
    streak_bonus = min(streak / 5.0, 0.3) if streak >= 3 else 0.0

    cb = 0.4 * vote_strength + 0.35 * res_strength + 0.25 * streak_bonus
    return max(0.0, min(1.0, cb))


# -- Model-Based CB (replaces proxy when alpha model is available) ----------

def compute_cb_from_model(features: "np.ndarray", model, pairs: list[str],
                          scaler_mean: "np.ndarray" = None,
                          scaler_std: "np.ndarray" = None,
                          scale: float = 5.0) -> dict:
    """
    Compute confidence scores and directions from the trained AlphaNet.

    Args:
        features: (n_features,) feature snapshot for current bar
        model: trained AlphaNet instance
        pairs: ordered list of pair names matching model output
        scaler_mean: training feature mean
        scaler_std: training feature std
        scale: sigmoid scaling for confidence mapping

    Returns:
        dict with:
            'cb_scores': {pair: confidence} — replaces cb_scores from proxy
            'directions': {pair: Direction} — model-predicted direction
            'weights': {pair: raw_weight} — raw dollar-neutral weights
    """
    if not HAS_TORCH:
        return None

    weights = predict_weights(model, features, scaler_mean, scaler_std)
    conf = weights_to_confidence(weights, scale=scale)

    cb_scores = {}
    directions = {}
    raw_weights = {}

    for i, pair in enumerate(pairs):
        cb_scores[pair] = float(conf["confidence"][i])
        d = conf["direction"][i]
        if d > 0:
            directions[pair] = Direction.LONG
        elif d < 0:
            directions[pair] = Direction.SHORT
        else:
            directions[pair] = Direction.FLAT
        raw_weights[pair] = float(weights[i])

    return {
        "cb_scores": cb_scores,
        "directions": directions,
        "weights": raw_weights,
    }


# -- 6-Gate Filter -----------------------------------------------------------

def check_gates(net_score: float, direction: Direction, avg_cb: float,
                ema5_slope: float, atr_ratio: float, agree_r: float,
                regime: str, spectral_gap: float,
                cb_thr: float = CB_THR) -> GateResult:
    """
    Check all 6 gates. All must pass for order placement.
    """
    result = GateResult()
    result.net_score = net_score
    result.direction = direction

    # G1: Minimum directional consensus
    result.g1_net_score = abs(net_score) >= NET_SCORE_THR

    # G2: CatBoost confidence floor
    result.g2_cb_floor = avg_cb >= cb_thr

    # G3: EMA5 slope matches direction
    if direction == Direction.LONG:
        result.g3_ema_trend = ema5_slope > 0
    elif direction == Direction.SHORT:
        result.g3_ema_trend = ema5_slope < 0
    else:
        result.g3_ema_trend = False

    # G4: ATR ratio within safe window
    result.g4_atr_window = ATR_LO <= atr_ratio <= ATR_HI

    # G5: High-CB node agreement
    result.g5_agreement = agree_r >= AGREE_R_THR

    # G6: Regime guard (Fix #10)
    result.g6_regime = (regime != "FRAGMENTED") and (spectral_gap > SPECTRAL_GAP_WARN)

    result.passed = all([
        result.g1_net_score, result.g2_cb_floor, result.g3_ema_trend,
        result.g4_atr_window, result.g5_agreement, result.g6_regime
    ])

    return result


# -- EMA5 -------------------------------------------------------------------

def update_ema5(ema5: float, close: float) -> float:
    """EMA5 update: decay = 2/(5+1) = 0.333."""
    return ema5 * 0.667 + close * 0.333


# -- Position Sizing ---------------------------------------------------------

def compute_position_size(balance: float, sl_pips: float, pair: str,
                          price: float) -> dict:
    """
    Compute lot size, margin, and validate bounds.
    Returns dict with 'lot', 'margin', 'valid'.
    """
    pip_value = PIP_VALUES.get(pair, 10.0)
    risk_usd = balance * RISK_PCT

    if sl_pips < EPS:
        return {"lot": 0.0, "margin": 0.0, "valid": False}

    lot = risk_usd / (sl_pips * pip_value)
    lot = max(LOT_MIN, min(LOT_MAX, lot))

    margin = (lot * 100_000 * price) / LEVERAGE

    return {"lot": lot, "margin": margin, "valid": True}


# -- Order Construction ------------------------------------------------------

def create_order(pair: str, direction: Direction, mid_price: float,
                 atr_pips: float, spread_pips: float, balance: float,
                 bar_idx: int, residual: float = 0.0, cb: float = 0.0,
                 regime: str = "NORMAL", spectral_gap: float = 0.0,
                 streak: float = 0.0) -> Order | None:
    """
    Construct an order with TP/SL, position sizing, and metadata.
    Returns None if position sizing fails.
    """
    pip = PIP_SIZES.get(pair, 0.0001)

    # TP/SL in pips
    tp_pips = atr_pips * TP_ATR_MULT
    sl_pips = atr_pips * SL_ATR_MULT

    # Fix #11: enforce minimum spread multiples
    tp_pips = max(tp_pips, spread_pips * MIN_TP_SPREAD_MULT)
    sl_pips = max(sl_pips, spread_pips * MIN_SL_SPREAD_MULT)

    # Entry price with spread adjustment
    half_spread = (spread_pips / 2.0) * pip
    if direction == Direction.LONG:
        entry = mid_price + half_spread
        tp = entry + tp_pips * pip
        sl = entry - sl_pips * pip
    else:
        entry = mid_price - half_spread
        tp = entry - tp_pips * pip
        sl = entry + sl_pips * pip

    # Position sizing
    sizing = compute_position_size(balance, sl_pips, pair, mid_price)
    if not sizing["valid"]:
        return None

    order = Order(
        pair=pair,
        direction=direction,
        entry_price=entry,
        tp=tp,
        sl=sl,
        lot=sizing["lot"],
        margin=sizing["margin"],
        entry_bar=bar_idx,
        timeout_bar=bar_idx + TIMEOUT_BARS,
        entry_residual=residual,
        entry_cb=cb,
        entry_regime=regime,
        entry_spectral_gap=spectral_gap,
        entry_streak=streak,
        spread_cost_entry=spread_pips * pip * sizing["lot"] * 100_000 / 2.0,
    )
    return order


# -- Candidate Selection -----------------------------------------------------

def select_candidates(pairs: list[str], cb_scores: dict[str, float],
                      residuals: dict[str, float], directions: dict[str, Direction],
                      atr_ratios: dict[str, float], net_direction: Direction,
                      cb_thr: float = CB_THR,
                      max_candidates: int = MAX_POSITIONS_PER_SIGNAL) -> list[str]:
    """
    Select top trade candidates from FX pairs.
    Filter: FX-only, cb > threshold, residual aligned, ATR > 0.3x avg.
    Rank by cb * |residual|.
    """
    candidates = []
    for pair in pairs:
        if pair not in FX_SET:
            continue
        cb = cb_scores.get(pair, 0.0)
        res = residuals.get(pair, 0.0)
        d = directions.get(pair, Direction.FLAT)
        atr_r = atr_ratios.get(pair, 0.0)

        if cb < cb_thr:
            continue
        if atr_r < ATR_LO:
            continue
        # Residual must align with net direction
        if net_direction == Direction.LONG and res < RESIDUAL_DEAD_ZONE:
            continue
        if net_direction == Direction.SHORT and res > -RESIDUAL_DEAD_ZONE:
            continue
        if d == Direction.FLAT:
            continue

        score = cb * abs(res)
        candidates.append((pair, score))

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:max_candidates]]


# -- Session Whitelists ------------------------------------------------------

def get_active_sessions(hour_utc: int) -> list[str]:
    """Get active trading sessions for a given UTC hour."""
    active = []
    for session, cfg in SESSION_WHITELISTS.items():
        lo, hi = cfg["hours"]
        if lo <= hour_utc < hi:
            active.append(session)
    return active


def get_session_pairs(hour_utc: int) -> set[str]:
    """Get allowed pairs for current sessions."""
    pairs = set()
    for session, cfg in SESSION_WHITELISTS.items():
        lo, hi = cfg["hours"]
        if lo <= hour_utc < hi:
            pairs.update(cfg["pairs"])
    return pairs


# -- v2 Execution guard ------------------------------------------------------

def should_execute_trade(pair: str, signal: str = "LONG") -> bool:
    """
    Final execution guard for v2 dual-subnet architecture.

    Hard rule: signal-only nodes (indices, metals, energy, exotic FX)
    NEVER generate trade orders, regardless of their computed signal.
    This guard is checked at the execution layer, not just the gate layer,
    to provide defence-in-depth against accidental order placement.

    Args:
        pair: instrument name
        signal: "LONG", "SHORT", or "FLAT"

    Returns:
        True only if pair is in the tradeable set AND signal is not FLAT
    """
    if signal == "FLAT":
        return False

    # If universe.py is available, use its authoritative tradeable set
    if _TRADEABLE_FROZENSET:
        return pair in _TRADEABLE_FROZENSET

    # Fallback: reject known signal-only instruments by name pattern
    # (Indices, metals, energy — never tradeable in v2)
    _KNOWN_SIGNAL_ONLY = {
        "AUS200", "US30", "GER40", "UK100", "NAS100", "EUSTX50", "JPN225", "SPX500",
        "XTIUSD", "XBRUSD", "XAUUSD", "XAGUSD",
    }
    return pair not in _KNOWN_SIGNAL_ONLY


def is_signal_only(pair: str) -> bool:
    """Returns True if pair is a signal-only node (feature contributor, never trades)."""
    if _SIGNAL_ONLY_FROZENSET:
        return pair in _SIGNAL_ONLY_FROZENSET
    return False


# -- v2 CatBoost model integration ------------------------------------------

def load_catboost_v2_models(btc_model_path: str, fx_model_path: str,
                             feature_names_path: str = None):
    """
    Load trained CatBoost-BTC and CatBoost-FX models from disk.

    Args:
        btc_model_path: path to catboost_btc_v2.cbm
        fx_model_path:  path to catboost_fx_v2.cbm
        feature_names_path: optional path to catboost_v2_feature_names.json

    Returns:
        (btc_model, fx_model, meta_dict)
        meta_dict has keys: btc_feature_names, fx_feature_names, fx_pairs, horizon, lookback
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("catboost not installed: pip install catboost")

    btc_model = CatBoostClassifier()
    btc_model.load_model(btc_model_path)

    fx_model = CatBoostClassifier()
    fx_model.load_model(fx_model_path)

    meta = {}
    if feature_names_path:
        import json as _json
        with open(feature_names_path) as f:
            meta = _json.load(f)

    return btc_model, fx_model, meta


def compute_cb_from_catboost_v2(btc_features,
                                 fx_features_per_pair: dict,
                                 btc_model,
                                 fx_model,
                                 btc_feature_names: list,
                                 fx_feature_names: list,
                                 fx_pairs_list: list = None) -> dict:
    """
    Compute CatBoost v2 confidence scores and directions.

    Replaces compute_cb_proxy() and compute_cb_from_model() when
    CatBoost-BTC and CatBoost-FX models are loaded.

    Args:
        btc_features:       BTCFeatures object (from subnet_btc.BTCSubnet.compute())
        fx_features_per_pair: {pair: FXBarFeatures} (from subnet_fx.FXSubnet.compute_all())
        btc_model:          loaded CatBoostClassifier (BTC)
        fx_model:           loaded CatBoostClassifier (FX)
        btc_feature_names:  ordered feature name list for BTC model
        fx_feature_names:   ordered feature name list for FX model (last = pair_id)
        fx_pairs_list:      canonical FX pair order for pair_id encoding

    Returns:
        dict with:
            'cb_scores':  {pair: confidence float in [0,1]}
            'directions': {pair: Direction}
            'proba':      {pair: [P_sell, P_hold, P_buy]}
    """
    if fx_pairs_list is None:
        try:
            from universe import FX_PAIRS
            fx_pairs_list = list(FX_PAIRS)
        except ImportError:
            fx_pairs_list = []

    cb_scores  = {}
    directions = {}
    proba_out  = {}

    # BTC prediction
    if btc_features is not None and btc_features.valid and btc_model is not None:
        x_btc = btc_features.to_array(btc_feature_names).reshape(1, -1)
        p = btc_model.predict_proba(x_btc)[0]   # [P_sell, P_hold, P_buy]
        confidence = float(max(p[0], p[2]))       # max directional confidence
        cb_scores["BTCUSD"]  = confidence
        proba_out["BTCUSD"]  = p.tolist()
        if p[2] > p[0]:
            directions["BTCUSD"] = Direction.LONG
        elif p[0] > p[2]:
            directions["BTCUSD"] = Direction.SHORT
        else:
            directions["BTCUSD"] = Direction.FLAT

    # FX predictions (one call per pair, pair_id as last feature)
    if fx_model is not None:
        fx_feat_no_pid = fx_feature_names[:-1]   # strip pair_id
        for pid, pair in enumerate(fx_pairs_list):
            ff = fx_features_per_pair.get(pair)
            if ff is None or not ff.valid:
                continue
            x_fx = ff.to_array(fx_feat_no_pid)
            x_fx = np.append(x_fx, float(pid)).reshape(1, -1)
            p = fx_model.predict_proba(x_fx)[0]
            confidence = float(max(p[0], p[2]))
            cb_scores[pair]  = confidence
            proba_out[pair]  = p.tolist()
            if p[2] > p[0]:
                directions[pair] = Direction.LONG
            elif p[0] > p[2]:
                directions[pair] = Direction.SHORT
            else:
                directions[pair] = Direction.FLAT

    return {"cb_scores": cb_scores, "directions": directions, "proba": proba_out}


# -- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    print("Signal pipeline smoke test...")

    # Test voting
    votes = compute_votes(rsi=62, macd=0.5, bb_pct_b=0.7, stoch_k=65,
                          cci=80, willr=-30, mom5=0.5)
    d = vote_to_direction(votes)
    print(f"Votes: {votes}, Direction: {d.name}")
    assert votes == 7 and d == Direction.LONG

    votes2 = compute_votes(rsi=35, macd=-0.3, bb_pct_b=0.3, stoch_k=30,
                           cci=-70, willr=-70, mom5=-0.5)
    d2 = vote_to_direction(votes2)
    print(f"Votes: {votes2}, Direction: {d2.name}")
    assert votes2 == -7 and d2 == Direction.SHORT

    # Test net score
    res = np.array([0.06, -0.06, 0.03, 0.07, -0.02] * 7)
    ns, nd = compute_net_score(res)
    print(f"Net score: {ns:.3f}, Direction: {nd.name}")

    # Test CB proxy
    cb = compute_cb_proxy(votes=5, residual=0.003, streak=4)
    print(f"CB proxy: {cb:.3f}")

    # Test gates
    gate = check_gates(
        net_score=0.20, direction=Direction.LONG, avg_cb=0.70,
        ema5_slope=0.001, atr_ratio=1.2, agree_r=0.60,
        regime="NORMAL", spectral_gap=0.5
    )
    print(f"Gates passed: {gate.passed}")
    print(f"  G1={gate.g1_net_score} G2={gate.g2_cb_floor} G3={gate.g3_ema_trend} "
          f"G4={gate.g4_atr_window} G5={gate.g5_agreement} G6={gate.g6_regime}")
    assert gate.passed

    # Test position sizing
    sz = compute_position_size(balance=50.0, sl_pips=5.0, pair="EURUSD", price=1.08)
    print(f"Lot: {sz['lot']:.4f}, Margin: ${sz['margin']:.2f}")

    # Test order creation
    order = create_order("EURUSD", Direction.LONG, mid_price=1.08000,
                         atr_pips=7.0, spread_pips=1.0, balance=50.0,
                         bar_idx=100, residual=0.003, cb=0.72)
    if order:
        print(f"Order: {order.pair} {order.direction.name} "
              f"entry={order.entry_price:.5f} TP={order.tp:.5f} SL={order.sl:.5f} "
              f"lot={order.lot:.4f}")

    # Test sessions
    print(f"Sessions at 10 UTC: {get_active_sessions(10)}")
    print(f"Pairs at 10 UTC: {get_session_pairs(10)}")

    print("OK")
