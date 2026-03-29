"""
train_catboost_v2.py -- Dual-subnet CatBoost training for Algo C2 v2

Trains two classifiers:
  - CatBoost-BTC: 37-feature BTC signal model (24x7)
  - CatBoost-FX:  138-feature FX signal model, all 28 pairs pooled with pair_id
                  as a categorical feature (24x5)

Walk-forward cross-validation (TimeSeriesSplit), PBO analysis, model save.

Usage:
    # Single JSON file (output of process_fx_csv_43.py)
    python src/train_catboost_v2.py --data ./data/algo_c2_43.json

    # Directory of per-instrument JSON files
    python src/train_catboost_v2.py --data-dir D:/dataset-ml/DataExtractor

    # Versioned training (recommended — preserves previous generation)
    python src/train_catboost_v2.py --data ./data/algo_c2_43.json --model-version v1_5_c1

    # Full tuning options
    python src/train_catboost_v2.py --data ./data/algo_c2_43.json \\
        --model-version v1_5_c1 --n-folds 5 --horizon 5 --lookback 120 \\
        --iterations 500 --depth 6

Outputs (with --model-version v1_5_c1):
    models/v1_5_c1/catboost_btc_v1_5_c1.cbm
    models/v1_5_c1/catboost_fx_v1_5_c1.cbm
    models/v1_5_c1/catboost_feature_names_v1_5_c1.json

Outputs (legacy --model-dir only):
    <model-dir>/catboost_btc_v2.cbm
    <model-dir>/catboost_fx_v2.cbm
    <model-dir>/catboost_v2_feature_names.json

Generation registry: DOC/GENERATIONS.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Local imports
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from universe import ALL_INSTRUMENTS, FX_PAIRS, NODE_IDX
from math_engine import MathEngine
from bridge import BridgeComputer
from subnet_btc import BTCSubnet
from subnet_fx import FXSubnet
from pbo_analysis import compute_pbo, print_pbo_report
from signal_filter import strategy_sharpe_sized
from calibration import calibration_report, print_calibration_summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_BARS = 200       # bars to skip for indicator warmup
FX_LABEL_THRESHOLD = 5e-4   # ~0.5 pip for major FX (log-return units)
BTC_LABEL_THRESHOLD = 3e-3  # 0.3% for BTC
HOLD_CLASS = 1
BUY_CLASS  = 2
SELL_CLASS = 0

FX_PAIRS_LIST: list[str] = list(FX_PAIRS)   # canonical ordered list, index = pair_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _records_to_arrays(records: list[dict]) -> dict:
    """Convert list-of-dicts JSON records to numpy array dict."""
    return {
        "dt": np.array([r["dt"] for r in records]),
        "o":  np.array([r["o"]  for r in records], dtype=np.float32),
        "h":  np.array([r["h"]  for r in records], dtype=np.float32),
        "l":  np.array([r["l"]  for r in records], dtype=np.float32),
        "c":  np.array([r["c"]  for r in records], dtype=np.float32),
        "sp": np.array([r.get("sp", 0.0) for r in records], dtype=np.float32),
        "tk": np.array([r.get("tk", 1)   for r in records], dtype=np.float32),
    }


def load_single_json(json_path: str) -> dict[str, dict]:
    """Load a single all-instruments JSON file."""
    with open(json_path) as f:
        raw: dict = json.load(f)
    return {inst: _records_to_arrays(records) for inst, records in raw.items()}


def load_data_dir(data_dir: str) -> dict[str, dict]:
    """
    Load per-instrument JSON files from a directory.
    Expects files named {INSTRUMENT}.json or {INSTRUMENT}_*.json.
    """
    dpath = Path(data_dir)
    result: dict[str, dict] = {}
    for inst in ALL_INSTRUMENTS:
        # Try exact match first, then prefix match
        candidates = list(dpath.glob(f"{inst}.json")) + list(dpath.glob(f"{inst}_*.json"))
        if not candidates:
            continue
        fpath = sorted(candidates)[-1]   # latest if multiple
        with open(fpath) as f:
            raw = json.load(f)
        # File may be {inst: [...]} or directly [...]
        if isinstance(raw, dict) and inst in raw:
            result[inst] = _records_to_arrays(raw[inst])
        elif isinstance(raw, list):
            result[inst] = _records_to_arrays(raw)
    return result


def load_quarterly_csv_dir(data_dir: str, timeframe: str = "M5") -> dict[str, dict]:
    """
    Load data from DataExtractor/{year}/{Q1-Q4}/{instrument}/candles_{TF}.csv

    Concatenates all years and quarters chronologically, aligns all instruments
    to a common timestamp index (forward-fill gaps), returns ohlc array dict.
    """
    dpath = Path(data_dir)
    tf_file = f"candles_{timeframe}.csv"

    years    = sorted(d.name for d in dpath.iterdir() if d.is_dir() and d.name.isdigit())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    inst_frames: dict[str, list[pd.DataFrame]] = {}

    for year in years:
        for quarter in quarters:
            qpath = dpath / year / quarter
            if not qpath.exists():
                continue
            for inst_dir in sorted(qpath.iterdir()):
                if not inst_dir.is_dir():
                    continue
                csv_path = inst_dir / tf_file
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path, dtype=str)
                    inst_frames.setdefault(inst_dir.name, []).append(df)
                except Exception:
                    continue

    if not inst_frames:
        raise ValueError(f"No {tf_file} files found under {data_dir}")

    # Concatenate, deduplicate, sort each instrument
    raw: dict[str, pd.DataFrame] = {}
    for inst, frames in inst_frames.items():
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["bar_time"]).sort_values("bar_time").reset_index(drop=True)
        df["bar_time"] = pd.to_datetime(df["bar_time"], format="%Y.%m.%d %H:%M:%S")
        df = df.set_index("bar_time")
        for col in ["open", "high", "low", "close", "spread", "tick_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        raw[inst] = df

    print(f"  Loaded {len(raw)} instruments from {len(years)} years at {timeframe}")

    # Build master timeline (union of all timestamps)
    all_ts: set = set()
    for df in raw.values():
        all_ts.update(df.index.tolist())
    master = sorted(all_ts)
    print(f"  Master timeline: {len(master)} bars  ({master[0]} -> {master[-1]})")

    # Reindex + forward-fill each instrument onto master timeline
    result: dict[str, dict] = {}
    for inst, df in raw.items():
        reindexed = df.reindex(master)
        # Capture which bars are real (had actual data) before forward-filling.
        # Forward-filled bars (gaps/weekends/holidays) will have is_real=False.
        is_real = reindexed["close"].notna().values  # bool, length = len(master)
        df = reindexed.ffill().dropna(subset=["close"])
        if len(df) < WARMUP_BARS + 10:
            continue
        # Align is_real to rows surviving dropna (leading NaN rows are dropped)
        is_real = is_real[len(is_real) - len(df):]
        dt_arr = np.array([str(ts)[:16] for ts in df.index])
        result[inst] = {
            "dt":   dt_arr,
            "o":    df["open"].values.astype(np.float32),
            "h":    df["high"].values.astype(np.float32),
            "l":    df["low"].values.astype(np.float32),
            "c":    df["close"].values.astype(np.float32),
            "sp":   df["spread"].values.astype(np.float32),
            "tk":   df["tick_volume"].values.astype(np.float32),
            "real": is_real.astype(np.bool_),  # validity mask
        }

    print(f"  After alignment: {len(result)} instruments, {len(master)} bars each")
    return result


def _is_quarterly_dir(data_dir: str) -> bool:
    """Returns True if data_dir contains year-named subdirectories (quarterly structure)."""
    dpath = Path(data_dir)
    return any(d.is_dir() and d.name.isdigit() and len(d.name) == 4
               for d in dpath.iterdir())


def align_to_common_length(ohlc_all: dict[str, dict]) -> dict[str, dict]:
    """
    Trim all instruments to the shortest common length.
    Assumes data is already aligned (same timestamp per index).
    """
    if not ohlc_all:
        return ohlc_all
    min_len = min(len(v["c"]) for v in ohlc_all.values())
    return {k: {arr: v[arr][:min_len] for arr in v} for k, v in ohlc_all.items()}


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def make_labels(log_rets: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """
    Forward-return labels over [horizon] bars.

    label[t] = BUY  (2) if sum(log_ret[t+1 .. t+horizon]) >  threshold
             = SELL (0) if sum(log_ret[t+1 .. t+horizon]) < -threshold
             = HOLD (1) otherwise
    Last [horizon] bars get HOLD (no future data).
    """
    n = len(log_rets)
    labels = np.full(n, HOLD_CLASS, dtype=np.int32)
    cs = np.cumsum(log_rets)
    for t in range(n - horizon):
        fwd = cs[t + horizon] - cs[t]
        if fwd > threshold:
            labels[t] = BUY_CLASS
        elif fwd < -threshold:
            labels[t] = SELL_CLASS
    return labels


def make_labels_quantile(
    log_rets: np.ndarray,
    horizon: int,
    buy_pct: float = 0.30,
    sell_pct: float = 0.30,
    real_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rank-based labels: exactly floor(n*buy_pct) BUY, floor(n*sell_pct) SELL, rest HOLD.

    Uses argsort instead of quantile thresholds so ties (e.g. zero-return bars
    during non-trading hours on M5 FX) never collapse HOLD to zero.

    real_mask: optional bool array (same length as log_rets). When provided,
      ranking is performed only among real bars so that stale forward-filled bars
      (weekends/gaps) do not consume SELL/BUY rank slots and distort the label
      distribution among actual traded bars. Stale bars default to HOLD.
    """
    n = len(log_rets)
    cs = np.cumsum(log_rets)
    fwd = np.zeros(n, dtype=np.float64)
    fwd[:n - horizon] = cs[horizon:] - cs[:n - horizon]

    labels = np.full(n, HOLD_CLASS, dtype=np.int32)

    # Determine which labelable bars (bars with a full forward window) are real
    if real_mask is not None:
        labelable_real = real_mask[:n - horizon]
        real_idx = np.where(labelable_real)[0]   # positions of real bars within [0, n-horizon)
        labeled  = fwd[real_idx]
    else:
        real_idx = np.arange(n - horizon)
        labeled  = fwd[:n - horizon]

    n_lab  = len(labeled)
    n_sell = int(n_lab * sell_pct)
    n_buy  = int(n_lab * buy_pct)

    sorted_idx = np.argsort(labeled, kind="stable")   # indices into real_idx

    # Map back to global bar positions
    sell_positions = real_idx[sorted_idx[:n_sell]]
    buy_positions  = real_idx[sorted_idx[n_lab - n_buy:]]

    labels[sell_positions] = SELL_CLASS
    labels[buy_positions]  = BUY_CLASS
    return labels


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrices(
    ohlc_all: dict[str, dict],
    horizon: int = 5,
    lookback: int = 120,
    btc_threshold: float = BTC_LABEL_THRESHOLD,
    fx_threshold:  float = FX_LABEL_THRESHOLD,
    label_mode: str = "quantile",
    buy_pct: float = 0.30,
    sell_pct: float = 0.30,
    verbose: bool = True,
) -> dict:
    """
    Run bar-by-bar feature extraction using BTCSubnet and FXSubnet.

    Returns
    -------
    dict with keys:
      btc_X  : (n_btc, 37) float32
      btc_y  : (n_btc,)    int32   -- 0/1/2 labels
      btc_bar: (n_btc,)    int32   -- source bar index
      fx_X   : (n_fx, 139) float32 -- 138 features + pair_id
      fx_y   : (n_fx,)     int32
      fx_bar : (n_fx,)     int32
      fx_pid : (n_fx,)     int32   -- pair index 0-27
      feature_names_btc : list[str]
      feature_names_fx  : list[str]  (length 139, last = "pair_id")
    """
    if not ohlc_all:
        raise ValueError("Empty ohlc_all dict")

    # Reference instrument for length / timestamps
    ref_inst = "BTCUSD" if "BTCUSD" in ohlc_all else next(iter(ohlc_all))
    ref = ohlc_all[ref_inst]
    n_total = len(ref["c"])

    if verbose:
        print(f"  n_total={n_total} bars across {len(ohlc_all)} instruments")
        print(f"  Warmup={WARMUP_BARS}, horizon={horizon}, lookback={lookback}")

    # Pre-compute log returns for every instrument (needed for MathEngine + labels)
    log_rets: dict[str, np.ndarray] = {}
    for inst, ohlc in ohlc_all.items():
        c = ohlc["c"].astype(np.float64)
        lr = np.zeros(len(c), dtype=np.float64)
        safe = np.maximum(c, 1e-10)
        lr[1:] = np.diff(np.log(safe))
        log_rets[inst] = lr

    # Cost-adjusted log returns for labeling: subtract spread (one-way cost).
    # Labels should reflect net P&L, not gross return (critics.md item 3).
    # spread is stored in pip units; convert to log-return units: spread / close.
    cost_log_rets: dict[str, np.ndarray] = {}
    for inst, ohlc in ohlc_all.items():
        lr = log_rets[inst].copy()
        if "sp" in ohlc:
            c   = np.maximum(ohlc["c"].astype(np.float64), 1e-10)
            sp  = ohlc["sp"].astype(np.float64)
            cost = sp / c          # spread as fraction of price
            lr   = lr - cost       # deduct one-way cost from each bar's return
        cost_log_rets[inst] = lr

    # Returns matrix in ALL_INSTRUMENTS node order (for MathEngine)
    inst_ordered = [i for i in ALL_INSTRUMENTS if i in ohlc_all]
    n_inst = len(inst_ordered)
    node_rets = np.zeros((n_total, n_inst), dtype=np.float64)
    for col, inst in enumerate(inst_ordered):
        node_rets[:, col] = log_rets[inst]

    # Labels
    _label_fn = (
        (lambda lr, h: make_labels_quantile(lr, h, buy_pct, sell_pct))
        if label_mode == "quantile"
        else (lambda lr, h: make_labels(lr, h, btc_threshold))
    )
    _label_fn_fx = (
        (lambda lr, h: make_labels_quantile(lr, h, buy_pct, sell_pct))
        if label_mode == "quantile"
        else (lambda lr, h: make_labels(lr, h, fx_threshold))
    )

    # Pass real_mask so rank labeling is computed only among real (non-stale) bars.
    # Without this, stale weekend bars (with cost_log_ret ≈ -spread/close, small
    # negative) consume the bottom 30% SELL rank slots; after validity filtering
    # removes them, almost no SELL labels remain in the training set.
    btc_real = ohlc_all.get("BTCUSD", {}).get("real")
    btc_labels = _label_fn(
        cost_log_rets.get("BTCUSD", np.zeros(n_total)), horizon
    ) if btc_real is None else make_labels_quantile(
        cost_log_rets.get("BTCUSD", np.zeros(n_total)), horizon,
        buy_pct, sell_pct, real_mask=btc_real
    )

    fx_labels: dict[str, np.ndarray] = {}
    for pair in FX_PAIRS_LIST:
        if pair in cost_log_rets:
            pair_real = ohlc_all.get(pair, {}).get("real")
            if label_mode == "quantile" and pair_real is not None:
                fx_labels[pair] = make_labels_quantile(
                    cost_log_rets[pair], horizon, buy_pct, sell_pct, real_mask=pair_real
                )
            else:
                fx_labels[pair] = _label_fn_fx(cost_log_rets[pair], horizon)

    if verbose:
        btc_dist = np.bincount(btc_labels, minlength=3)
        n_lab = n_total - horizon
        print(f"  Label mode: {label_mode}  (buy_pct={buy_pct}, sell_pct={sell_pct})")
        print(f"  BTC label dist: SELL={btc_dist[0]}({btc_dist[0]/n_lab*100:.1f}%)  "
              f"HOLD={btc_dist[1]}({btc_dist[1]/n_lab*100:.1f}%)  "
              f"BUY={btc_dist[2]}({btc_dist[2]/n_lab*100:.1f}%)")
        if fx_labels:
            sample_pair = FX_PAIRS_LIST[0]
            if sample_pair in fx_labels:
                fd = np.bincount(fx_labels[sample_pair], minlength=3)
                print(f"  FX  label dist ({sample_pair}): "
                      f"SELL={fd[0]}({fd[0]/n_lab*100:.1f}%)  "
                      f"HOLD={fd[1]}({fd[1]/n_lab*100:.1f}%)  "
                      f"BUY={fd[2]}({fd[2]/n_lab*100:.1f}%)")

    # Initialize pipeline components
    engine = MathEngine(n_pairs=n_inst)
    bridge = BridgeComputer()
    btc_subnet = BTCSubnet()
    fx_subnet  = FXSubnet()

    # Storage
    btc_X_rows: list[np.ndarray] = []
    btc_y_vals: list[int]        = []
    btc_bar_ids: list[int]       = []
    fx_X_rows: list[np.ndarray]  = []
    fx_y_vals: list[int]         = []
    fx_bar_ids: list[int]        = []
    fx_pid_vals: list[int]       = []

    btc_feat_names: Optional[list[str]] = None
    fx_feat_names:  Optional[list[str]] = None

    has_btc = "BTCUSD" in ohlc_all
    btc_ohlc = ohlc_all.get("BTCUSD")

    report_every = max(1, (n_total - WARMUP_BARS) // 20)

    for t in range(WARMUP_BARS, n_total - horizon):
        if verbose and (t - WARMUP_BARS) % report_every == 0:
            pct = 100 * (t - WARMUP_BARS) / (n_total - WARMUP_BARS - horizon)
            print(f"  [{pct:5.1f}%] bar {t}/{n_total}", end="\r", flush=True)

        # Update math engine
        math_state = engine.update(node_rets[t])

        # BTC data for bridge
        btc_close = float(btc_ohlc["c"][t]) if has_btc else 50000.0
        btc_hist  = btc_ohlc["c"][max(0, t - 240): t + 1] if has_btc else np.array([btc_close])
        btc_tk    = float(btc_ohlc["tk"][t]) if has_btc else 10.0
        btc_sp    = float(btc_ohlc["sp"][t]) if has_btc else 5.0
        # Approximate ATR as 14-bar rolling std of log returns × close
        btc_lr_window = log_rets.get("BTCUSD", np.zeros(n_total))[max(0, t - 14): t + 1]
        btc_atr = float(np.std(btc_lr_window) * btc_close * 1.414) if len(btc_lr_window) > 1 else btc_close * 0.002

        # FX data for bridge
        fx_closes_now  = {p: float(ohlc_all[p]["c"][t])  for p in FX_PAIRS_LIST if p in ohlc_all}
        fx_logrets_now = {p: float(log_rets[p][t])        for p in FX_PAIRS_LIST if p in log_rets}
        fx_spreads_now = {p: float(ohlc_all[p]["sp"][t])  for p in FX_PAIRS_LIST if p in ohlc_all}

        bridge_state = bridge.update(
            dt_str=str(ref["dt"][t]),
            btc_close=btc_close,
            btc_tick_vel=btc_tk,
            btc_spread=btc_sp,
            btc_atr=btc_atr,
            btc_h1_lifespan=math_state.h1_lifespan if math_state.valid else 0.0,
            btc_close_history=btc_hist,
            math_state=math_state,
            fx_closes=fx_closes_now,
            fx_log_rets=fx_logrets_now,
            fx_spreads=fx_spreads_now,
        )

        # BTC features
        if has_btc:
            # Skip forward-filled (stale) bars — no real trading occurred
            if "real" in btc_ohlc and not btc_ohlc["real"][t]:
                pass
            else:
                bf = btc_subnet.compute(btc_ohlc, t, math_state, bridge_state, lookback=lookback)
                if bf.valid:
                    if btc_feat_names is None:
                        btc_feat_names = btc_subnet.feature_names
                    btc_X_rows.append(bf.to_array(btc_feat_names))
                    btc_y_vals.append(int(btc_labels[t]))
                    btc_bar_ids.append(t)

        # FX features
        fx_results = fx_subnet.compute_all(ohlc_all, t, math_state, bridge_state, lookback=lookback)
        if fx_feat_names is None and fx_subnet._feature_names:
            fx_feat_names = fx_subnet.feature_names

        if fx_feat_names is not None:
            for pid, pair in enumerate(FX_PAIRS_LIST):
                ff = fx_results.get(pair)
                if ff and ff.valid:
                    # Skip forward-filled (stale) bars for this pair
                    pair_ohlc = ohlc_all.get(pair, {})
                    if "real" in pair_ohlc and not pair_ohlc["real"][t]:
                        continue
                    row = np.append(ff.to_array(fx_feat_names), float(pid))
                    fx_X_rows.append(row)
                    lbl = int(fx_labels[pair][t]) if pair in fx_labels else HOLD_CLASS
                    fx_y_vals.append(lbl)
                    fx_bar_ids.append(t)
                    fx_pid_vals.append(pid)

    if verbose:
        print()  # newline after \r
        # Validity mask stats
        if "real" in (ohlc_all.get("BTCUSD") or {}):
            btc_real = ohlc_all["BTCUSD"]["real"]
            n_real = int(btc_real[WARMUP_BARS:n_total - horizon].sum())
            n_total_lab = n_total - WARMUP_BARS - horizon
            print(f"  Validity mask: BTC {n_real}/{n_total_lab} real bars "
                  f"({n_real/max(n_total_lab,1)*100:.1f}%)")
        sample_fx = FX_PAIRS_LIST[0] if FX_PAIRS_LIST else None
        if sample_fx and "real" in (ohlc_all.get(sample_fx) or {}):
            fx_real = ohlc_all[sample_fx]["real"]
            n_real_fx = int(fx_real[WARMUP_BARS:n_total - horizon].sum())
            print(f"  Validity mask: FX  {n_real_fx}/{n_total_lab} real bars "
                  f"({n_real_fx/max(n_total_lab,1)*100:.1f}%) [sample: {sample_fx}]")

    # Assemble arrays
    btc_X = np.array(btc_X_rows, dtype=np.float32) if btc_X_rows else np.zeros((0, 37), dtype=np.float32)
    btc_y = np.array(btc_y_vals, dtype=np.int32)
    btc_bar = np.array(btc_bar_ids, dtype=np.int32)

    n_fx_features = len(fx_feat_names) + 1 if fx_feat_names else 139
    fx_X = np.array(fx_X_rows, dtype=np.float32) if fx_X_rows else np.zeros((0, n_fx_features), dtype=np.float32)
    fx_y = np.array(fx_y_vals, dtype=np.int32)
    fx_bar = np.array(fx_bar_ids, dtype=np.int32)
    fx_pid = np.array(fx_pid_vals, dtype=np.int32)

    return {
        "btc_X": btc_X,  "btc_y": btc_y,  "btc_bar": btc_bar,
        "fx_X":  fx_X,   "fx_y":  fx_y,   "fx_bar":  fx_bar, "fx_pid": fx_pid,
        "feature_names_btc": btc_feat_names or [],
        "feature_names_fx":  (fx_feat_names or []) + ["pair_id"],
    }


# ---------------------------------------------------------------------------
# CatBoost helpers
# ---------------------------------------------------------------------------

def _make_catboost(iterations: int, depth: int, learning_rate: float,
                   cat_features: list[int] | None = None,
                   auto_class_weights: str | None = "SqrtBalanced") -> "CatBoostClassifier":
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        classes_count=3,
        random_seed=42,
        verbose=0,
        cat_features=cat_features or [],
        early_stopping_rounds=50,
        auto_class_weights=auto_class_weights,
    )


def _fx_df(X: np.ndarray, cat_col: int) -> "pd.DataFrame":
    """Convert FX feature array to DataFrame with pair_id as string.

    CatBoost requires string (or object) dtype for categorical columns —
    a float32 numpy array with cat_features raises CatBoostError at both
    Pool construction and predict_proba time.
    Column-label assignment replaces the column dtype entirely (unlike iloc
    which tries in-place coercion and raises LossySetitemError).
    """
    df = pd.DataFrame(X.astype(np.float32))
    df[cat_col] = df[cat_col].astype(int).astype(str)
    return df


def _fx_pool(X: np.ndarray, y: np.ndarray, cat_col: int) -> "Pool":
    """CatBoost Pool for FX training/eval data."""
    return Pool(_fx_df(X, cat_col), y, cat_features=[cat_col])


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    n_keep: int,
    cat_col: int | None = None,
    iterations: int = 100,
    verbose: bool = True,
    label: str = "",
) -> np.ndarray:
    """
    Train a quick CatBoost selector, rank features by PredictionValuesChange
    importance, return column indices of top-n_keep features.

    cat_col (pair_id for FX) is always kept and placed last in the returned
    indices so downstream code can continue assuming pair_id = X[:, -1].
    """
    n_sample = min(len(X), 50_000)
    idx = np.random.default_rng(42).choice(len(X), n_sample, replace=False)
    Xs, ys = X[idx], y[idx]

    if cat_col is not None:
        pool = Pool(_fx_df(Xs, cat_col), ys, cat_features=[cat_col])
        sel = _make_catboost(iterations, depth=6, learning_rate=0.1,
                             cat_features=[cat_col], auto_class_weights=None)
    else:
        pool = Pool(Xs, ys)
        sel = _make_catboost(iterations, depth=6, learning_rate=0.1,
                             auto_class_weights=None)

    sel.fit(pool, verbose=False)
    imp = sel.get_feature_importance()   # shape (n_cols,)

    if cat_col is not None:
        non_cat = [i for i in range(X.shape[1]) if i != cat_col]
        ranked  = sorted(non_cat, key=lambda i: imp[i], reverse=True)
        top     = sorted(ranked[: n_keep - 1])   # keep cat_col slot
        selected = np.array(top + [cat_col], dtype=np.int32)
    else:
        ranked   = sorted(range(X.shape[1]), key=lambda i: imp[i], reverse=True)
        selected = np.array(sorted(ranked[:n_keep]), dtype=np.int32)

    if verbose:
        top5_names = ranked[:5]
        top5_imp   = [f"col{i}={imp[i]:.2f}" for i in top5_names]
        print(f"  Feature select [{label}]: {X.shape[1]} -> {len(selected)} cols  "
              f"top5: {top5_imp}")

    return selected


def _strategy_sharpe(proba: np.ndarray, log_rets: np.ndarray) -> float:
    """
    Compute strategy Sharpe from probability matrix and forward log-returns.

    signal = P(BUY) - P(SELL) in [-1, 1]
    strategy_ret = signal × actual_log_ret
    sharpe = mean/std × sqrt(n)  (unnormalized)
    """
    if len(proba) == 0:
        return 0.0
    signal = proba[:, BUY_CLASS] - proba[:, SELL_CLASS]
    strat = signal * log_rets[:len(signal)]
    std = float(np.std(strat)) + 1e-8
    return float(np.mean(strat) / std * math.sqrt(max(len(strat), 1)))


def _log_loss(proba: np.ndarray, y_true: np.ndarray, eps: float = 1e-7) -> float:
    """Cross-entropy loss."""
    n = len(y_true)
    if n == 0:
        return 1.0
    p = np.clip(proba[np.arange(n), y_true], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def _manual_tssplit(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Manual TimeSeriesSplit fallback (no sklearn required)."""
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end   = fold_size * (i + 1)
        if val_end > n:
            val_end = n
        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(train_end, val_end)
        if len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def walk_forward_cv(
    btc_X: np.ndarray, btc_y: np.ndarray, btc_bar: np.ndarray,
    fx_X:  np.ndarray, fx_y:  np.ndarray, fx_bar:  np.ndarray, fx_pid: np.ndarray,
    log_rets_btc: np.ndarray, log_rets_fx: dict[str, np.ndarray],
    n_splits: int = 5,
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.05,
    auto_class_weights: str | None = "SqrtBalanced",
    purge_bars: int = 0,
    entry_threshold: float = 0.45,
    exit_threshold: float = 0.35,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Walk-forward CV for both BTC and FX models.

    Returns (btc_fold_results, fx_fold_results) lists for PBO analysis.
    Each element: {fold, stage1_val_loss, strategy_sharpe, n_bars, accuracy}
    """
    if HAS_SKLEARN:
        tss = TimeSeriesSplit(n_splits=n_splits)
        btc_splits = list(tss.split(btc_X))
        # FX splits based on unique bar indices to keep train/val aligned
        unique_bars = np.unique(fx_bar)
        bar_splits  = list(tss.split(unique_bars))
        # Map back to fx_X row indices
        fx_splits = []
        for tr_b, va_b in bar_splits:
            tr_bars_set = set(unique_bars[tr_b].tolist())
            va_bars_set = set(unique_bars[va_b].tolist())
            tr_idx = np.where(np.isin(fx_bar, list(tr_bars_set)))[0]
            va_idx = np.where(np.isin(fx_bar, list(va_bars_set)))[0]
            fx_splits.append((tr_idx, va_idx))
    else:
        btc_splits = _manual_tssplit(len(btc_X), n_splits)
        unique_bars = np.unique(fx_bar)
        bar_splits  = _manual_tssplit(len(unique_bars), n_splits)
        fx_splits = []
        for tr_b, va_b in bar_splits:
            tr_bars_set = set(unique_bars[tr_b].tolist())
            va_bars_set = set(unique_bars[va_b].tolist())
            tr_idx = np.where(np.isin(fx_bar, list(tr_bars_set)))[0]
            va_idx = np.where(np.isin(fx_bar, list(va_bars_set)))[0]
            fx_splits.append((tr_idx, va_idx))

    btc_fold_results: list[dict] = []
    fx_fold_results:  list[dict] = []

    n_folds = min(len(btc_splits), len(fx_splits))

    for fold_idx in range(n_folds):
        if verbose:
            print(f"\n  -- Fold {fold_idx + 1}/{n_folds} --")

        # ── BTC fold ──────────────────────────────────────────────────────
        if len(btc_X) > 0:
            tr_idx, va_idx = btc_splits[fold_idx]
            # Purge: drop training samples whose bar is within purge_bars of
            # the validation window start (critics.md item 7 — CV leakage).
            if purge_bars > 0 and len(va_idx) > 0:
                va_bar_min = int(btc_bar[va_idx].min())
                tr_idx = tr_idx[btc_bar[tr_idx] < va_bar_min - purge_bars]
            Xtr, ytr = btc_X[tr_idx], btc_y[tr_idx]
            Xva, yva = btc_X[va_idx], btc_y[va_idx]
            va_bars  = btc_bar[va_idx]

            if len(np.unique(ytr)) < 3:
                if verbose:
                    print(f"    BTC: skipped (only {len(np.unique(ytr))} classes in train fold)")
                btc_fold_results.append({"fold": fold_idx, "stage1_val_loss": 1.0,
                                         "strategy_sharpe": 0.0, "n_bars": len(va_idx), "accuracy": 0.0})
            else:
                btc_model = _make_catboost(iterations, depth, learning_rate,
                                           auto_class_weights=auto_class_weights)
                btc_model.fit(
                    Pool(Xtr, ytr),
                    eval_set=Pool(Xva, yva),
                    verbose=False,
                )

                proba_tr = btc_model.predict_proba(Xtr)
                proba_va = btc_model.predict_proba(Xva)
                acc_va   = float((np.argmax(proba_va, axis=1) == yva).mean())
                train_ll = _log_loss(proba_tr, ytr)

                # Strategy Sharpe: hysteresis-filtered signals on BTC log returns
                lr_va = np.array([log_rets_btc[b] if b < len(log_rets_btc) else 0.0 for b in va_bars])
                sr_va = strategy_sharpe_sized(proba_va, lr_va, entry_threshold, exit_threshold)

                cal = calibration_report(proba_va, yva, label="BTC", verbose=verbose)
                btc_fold_results.append({
                    "fold": fold_idx,
                    "stage1_val_loss": round(train_ll, 6),
                    "strategy_sharpe": round(sr_va, 4),
                    "n_bars":          len(va_idx),
                    "accuracy":        round(acc_va, 4),
                    **{f"cal_{k}": v for k, v in cal.items() if k != "curves"},
                })
                if verbose:
                    print(f"    BTC: acc={acc_va:.4f}  sharpe={sr_va:.4f}  train_ll={train_ll:.4f}")

        # ── FX fold ───────────────────────────────────────────────────────
        if len(fx_X) > 0 and fold_idx < len(fx_splits):
            tr_idx, va_idx = fx_splits[fold_idx]
            Xtr, ytr = fx_X[tr_idx], fx_y[tr_idx]
            Xva, yva = fx_X[va_idx], fx_y[va_idx]
            va_bars_fx = fx_bar[va_idx]
            va_pids_fx = fx_pid[va_idx]

            # Purge FX fold boundary (critics.md item 7)
            if purge_bars > 0 and len(va_idx) > 0:
                va_bar_min = int(fx_bar[va_idx].min())
                tr_idx = tr_idx[fx_bar[tr_idx] < va_bar_min - purge_bars]
                Xtr, ytr = fx_X[tr_idx], fx_y[tr_idx]

            # pair_id is the last column — categorical feature index
            cat_col_idx = Xtr.shape[1] - 1

            if len(np.unique(ytr)) < 3:
                if verbose:
                    print(f"    FX:  skipped (only {len(np.unique(ytr))} classes in train fold)")
                fx_fold_results.append({"fold": fold_idx, "stage1_val_loss": 1.0,
                                        "strategy_sharpe": 0.0, "n_bars": len(va_idx), "accuracy": 0.0})
            else:
                fx_model = _make_catboost(iterations, depth, learning_rate,
                                          cat_features=[cat_col_idx],
                                          auto_class_weights=auto_class_weights)
                fx_model.fit(
                    _fx_pool(Xtr, ytr, cat_col_idx),
                    eval_set=_fx_pool(Xva, yva, cat_col_idx),
                    verbose=False,
                )

                proba_tr = fx_model.predict_proba(_fx_df(Xtr, cat_col_idx))
                proba_va = fx_model.predict_proba(_fx_df(Xva, cat_col_idx))
                acc_va   = float((np.argmax(proba_va, axis=1) == yva).mean())
                train_ll = _log_loss(proba_tr, ytr)

                # Strategy Sharpe: computed per-pair then averaged.
                # A single hysteresis filter across all 28 pairs concatenated is
                # wrong — state from EURUSD bleeds into GBPJPY.  Per-pair Sharpe
                # gives each pair an independent filter and is averaged (equal weight).
                pair_sharpes = []
                for pid in range(len(FX_PAIRS_LIST)):
                    pair = FX_PAIRS_LIST[pid]
                    pmask = va_pids_fx == pid
                    if pmask.sum() < 10:
                        continue
                    p_proba = proba_va[pmask]
                    p_bars  = va_bars_fx[pmask]
                    p_lr = np.array([
                        log_rets_fx[pair][b] if pair in log_rets_fx and b < len(log_rets_fx[pair]) else 0.0
                        for b in p_bars
                    ])
                    pair_sharpes.append(strategy_sharpe_sized(p_proba, p_lr, entry_threshold, exit_threshold))
                sr_va = float(np.mean(pair_sharpes)) if pair_sharpes else 0.0

                cal = calibration_report(proba_va, yva, label="FX", verbose=verbose)
                fx_fold_results.append({
                    "fold": fold_idx,
                    "stage1_val_loss": round(train_ll, 6),
                    "strategy_sharpe": round(sr_va, 4),
                    "n_bars":          len(va_idx),
                    "accuracy":        round(acc_va, 4),
                    **{f"cal_{k}": v for k, v in cal.items() if k != "curves"},
                })
                if verbose:
                    print(f"    FX:  acc={acc_va:.4f}  sharpe={sr_va:.4f}  train_ll={train_ll:.4f}")

    return btc_fold_results, fx_fold_results


# ---------------------------------------------------------------------------
# Final training on full data
# ---------------------------------------------------------------------------

def train_final(
    btc_X: np.ndarray, btc_y: np.ndarray,
    fx_X:  np.ndarray, fx_y:  np.ndarray,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    auto_class_weights: str | None = "SqrtBalanced",
    verbose: bool = True,
) -> tuple[Optional["CatBoostClassifier"], Optional["CatBoostClassifier"]]:
    """Train final models on full dataset."""
    btc_model = None
    fx_model  = None

    if len(btc_X) > 0:
        if verbose:
            print(f"\n  Training final CatBoost-BTC on {len(btc_X)} samples...")
        btc_model = _make_catboost(iterations, depth, learning_rate,
                                   auto_class_weights=auto_class_weights)
        btc_model.fit(Pool(btc_X, btc_y), verbose=50 if verbose else 0)

    if len(fx_X) > 0:
        if verbose:
            print(f"\n  Training final CatBoost-FX on {len(fx_X)} samples ({len(fx_X)//len(FX_PAIRS_LIST)} bars x {len(FX_PAIRS_LIST)} pairs)...")
        cat_col = fx_X.shape[1] - 1
        fx_model = _make_catboost(iterations, depth, learning_rate,
                                  cat_features=[cat_col],
                                  auto_class_weights=auto_class_weights)
        fx_model.fit(_fx_pool(fx_X, fx_y, cat_col), verbose=50 if verbose else 0)

    return btc_model, fx_model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args) -> None:
    print("\n=== Algo C2 v2 — CatBoost Dual-Subnet Training ===\n")

    if not HAS_CATBOOST:
        print("ERROR: catboost not installed. Run: pip install catboost")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.data:
        print(f"Loading {args.data} ...")
        ohlc_all = load_single_json(args.data)
        ohlc_all = align_to_common_length(ohlc_all)
    elif args.data_dir:
        if _is_quarterly_dir(args.data_dir):
            print(f"Detected quarterly CSV structure in {args.data_dir} ...")
            ohlc_all = load_quarterly_csv_dir(args.data_dir, timeframe=args.timeframe)
            ohlc_all = align_to_common_length(ohlc_all)
        else:
            print(f"Loading per-instrument JSON files from {args.data_dir} ...")
            ohlc_all = load_data_dir(args.data_dir)
            ohlc_all = align_to_common_length(ohlc_all)
    else:
        print("ERROR: provide --data or --data-dir")
        sys.exit(1)
    print(f"Loaded {len(ohlc_all)} instruments, {len(next(iter(ohlc_all.values()))['c'])} bars each")

    # ── Build feature matrices ────────────────────────────────────────────
    print("\n[1/4] Building feature matrices...")
    matrices = build_feature_matrices(
        ohlc_all,
        horizon=args.horizon,
        lookback=args.lookback,
        label_mode=args.label_mode,
        buy_pct=args.buy_pct,
        sell_pct=args.sell_pct,
        verbose=True,
    )

    btc_X, btc_y, btc_bar = matrices["btc_X"], matrices["btc_y"], matrices["btc_bar"]
    fx_X,  fx_y,  fx_bar  = matrices["fx_X"],  matrices["fx_y"],  matrices["fx_bar"]
    fx_pid = matrices["fx_pid"]

    print(f"  BTC: {btc_X.shape}, class dist: {np.bincount(btc_y)}")
    print(f"  FX:  {fx_X.shape}, class dist: {np.bincount(fx_y)}")

    # ── Feature selection (critics.md item 2) ────────────────────────────
    if not args.no_feature_select and len(btc_X) > 0 and len(fx_X) > 0:
        print(f"\n[1b/4] Feature selection (BTC top-{args.btc_top_k}, FX top-{args.fx_top_k})...")
        # BTC
        btc_sel = select_top_features(btc_X, btc_y, n_keep=args.btc_top_k,
                                      label="BTC", verbose=True)
        btc_X = btc_X[:, btc_sel]
        btc_feat_names = matrices["feature_names_btc"]
        matrices["feature_names_btc"] = [btc_feat_names[i] for i in btc_sel
                                          if i < len(btc_feat_names)]
        # FX — pair_id is last column, always kept, cat_col index shifts to new last
        fx_cat_orig = fx_X.shape[1] - 1
        fx_sel = select_top_features(fx_X, fx_y, n_keep=args.fx_top_k,
                                     cat_col=fx_cat_orig, label="FX", verbose=True)
        fx_X = fx_X[:, fx_sel]
        fx_feat_names = matrices["feature_names_fx"]
        matrices["feature_names_fx"] = [fx_feat_names[i] for i in fx_sel
                                         if i < len(fx_feat_names)]
        print(f"  After selection — BTC: {btc_X.shape}  FX: {fx_X.shape}")

    # Pre-extract FORWARD log returns for Sharpe calculation.
    # fwd[t] = sum(log_ret[t+1 .. t+horizon]) — matches the label horizon exactly.
    # Using current bar's return (log_ret[t]) was wrong: it's uncorrelated with
    # future direction, producing misleadingly large negative Sharpe values.
    ref = ohlc_all.get("BTCUSD") or next(iter(ohlc_all.values()))
    n_total = len(ref["c"])
    horizon = args.horizon

    def _fwd_rets(c_arr: np.ndarray, h: int) -> np.ndarray:
        """Vectorised forward return: fwd[t] = sum log_ret[t+1..t+h]."""
        safe = np.maximum(c_arr.astype(np.float64), 1e-10)
        lr   = np.zeros(len(safe))
        lr[1:] = np.diff(np.log(safe))
        cs   = np.cumsum(lr)
        fwd  = np.zeros(len(safe))
        fwd[:len(safe) - h] = cs[h:] - cs[:len(safe) - h]
        return fwd

    btc_lr = np.zeros(n_total)
    if "BTCUSD" in ohlc_all:
        btc_lr = _fwd_rets(ohlc_all["BTCUSD"]["c"], horizon)

    fx_lr: dict[str, np.ndarray] = {}
    for pair in FX_PAIRS_LIST:
        if pair in ohlc_all:
            fx_lr[pair] = _fwd_rets(ohlc_all[pair]["c"], horizon)

    # ── Walk-forward CV ───────────────────────────────────────────────────
    print(f"\n[2/4] Walk-forward CV ({args.n_folds} folds)...")
    acw = None if args.no_class_weights else "SqrtBalanced"
    btc_fold_results, fx_fold_results = walk_forward_cv(
        btc_X, btc_y, btc_bar,
        fx_X,  fx_y,  fx_bar, fx_pid,
        btc_lr, fx_lr,
        n_splits=args.n_folds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        auto_class_weights=acw,
        purge_bars=args.purge_bars,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        verbose=True,
    )

    # ── PBO analysis ──────────────────────────────────────────────────────
    print("\n[3/4] PBO Analysis...")
    pbo_btc = compute_pbo(btc_fold_results)
    print_pbo_report(pbo_btc, phase_name="CatBoost-BTC")

    pbo_fx = compute_pbo(fx_fold_results)
    print_pbo_report(pbo_fx, phase_name="CatBoost-FX")

    # ── Calibration summary ───────────────────────────────────────────────
    print("\n  --- Calibration Summary ---")
    btc_cal_folds = [f for f in btc_fold_results if "cal_log_loss" in f]
    fx_cal_folds  = [f for f in fx_fold_results  if "cal_log_loss" in f]
    if btc_cal_folds:
        print_calibration_summary(
            [{k[4:]: v for k, v in f.items() if k.startswith("cal_")} for f in btc_cal_folds],
            label="BTC"
        )
    if fx_cal_folds:
        print_calibration_summary(
            [{k[4:]: v for k, v in f.items() if k.startswith("cal_")} for f in fx_cal_folds],
            label="FX"
        )

    # ── Final training + save ─────────────────────────────────────────────
    print("\n[4/4] Final training on full dataset...")
    btc_model, fx_model = train_final(
        btc_X, btc_y, fx_X, fx_y,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        auto_class_weights=acw,
        verbose=True,
    )

    # ── Resolve output paths (versioned vs legacy) ────────────────────────
    ver = getattr(args, "model_version", None)
    if ver:
        model_dir = Path("./models") / ver
        btc_filename  = f"catboost_btc_{ver}.cbm"
        fx_filename   = f"catboost_fx_{ver}.cbm"
        feat_filename = f"catboost_feature_names_{ver}.json"
    else:
        model_dir = Path(args.model_dir)
        btc_filename  = "catboost_btc_v2.cbm"
        fx_filename   = "catboost_fx_v2.cbm"
        feat_filename = "catboost_v2_feature_names.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    if btc_model is not None:
        btc_path = model_dir / btc_filename
        btc_model.save_model(str(btc_path))
        print(f"\n  Saved: {btc_path}")

    if fx_model is not None:
        fx_path = model_dir / fx_filename
        fx_model.save_model(str(fx_path))
        print(f"  Saved: {fx_path}")

    # Save feature names for inference — includes version + training metadata
    feat_names_path = model_dir / feat_filename
    with open(feat_names_path, "w") as f:
        json.dump({
            "version":   ver or "v2_legacy",
            "btc": matrices["feature_names_btc"],
            "fx":  matrices["feature_names_fx"],
            "fx_pairs": FX_PAIRS_LIST,
            "horizon":  args.horizon,
            "lookback": args.lookback,
            "trained_on": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }, f, indent=2)
    print(f"  Saved: {feat_names_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train CatBoost-BTC and CatBoost-FX (Algo C2 v2)")
    data_grp = p.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--data",     type=str, help="Single all-instruments JSON file")
    data_grp.add_argument("--data-dir", type=str, help="Directory of per-instrument JSON files")

    p.add_argument("--timeframe",  type=str, default="M5",
                   choices=["M1","M5","M15","M30","H1","H4","H12","D1"],
                   help="Timeframe to load from quarterly CSVs (default: M5)")
    p.add_argument("--model-dir",  type=str, default="./models",
                   help="Directory to save trained models (default: ./models)")
    p.add_argument("--n-folds",    type=int, default=5,
                   help="Number of walk-forward CV folds (default: 5)")
    p.add_argument("--horizon",    type=int, default=5,
                   help="Label horizon in bars (default: 5)")
    p.add_argument("--lookback",   type=int, default=120,
                   help="Indicator lookback window in bars (default: 120)")
    p.add_argument("--iterations", type=int, default=300,
                   help="CatBoost iterations per fold (default: 300)")
    p.add_argument("--depth",      type=int, default=6,
                   help="CatBoost tree depth (default: 6)")
    p.add_argument("--lr",         type=float, default=0.05,
                   help="CatBoost learning rate (default: 0.05)")
    p.add_argument("--label-mode", type=str, default="quantile",
                   choices=["quantile", "threshold"],
                   help="Label strategy: quantile (balanced) or threshold (fixed, default: quantile)")
    p.add_argument("--buy-pct",    type=float, default=0.30,
                   help="Fraction of bars labeled BUY in quantile mode (default: 0.30)")
    p.add_argument("--sell-pct",   type=float, default=0.30,
                   help="Fraction of bars labeled SELL in quantile mode (default: 0.30)")
    p.add_argument("--entry-threshold", type=float, default=0.45,
                   help="Hysteresis entry threshold: min P(direction) to open (default: 0.45)")
    p.add_argument("--exit-threshold",  type=float, default=0.35,
                   help="Hysteresis exit threshold: min P(direction) to hold  (default: 0.35)")
    p.add_argument("--no-feature-select", action="store_true",
                   help="Disable feature selection (keep all features)")
    p.add_argument("--btc-top-k", type=int, default=20,
                   help="Top-K BTC features to keep after selection (default: 20)")
    p.add_argument("--fx-top-k",  type=int, default=40,
                   help="Top-K FX features to keep after selection (default: 40)")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable SqrtBalanced class weights in CatBoost")
    p.add_argument("--purge-bars",  type=int, default=0,
                   help="Bars to purge at train/val boundary to prevent label leakage (default: 0)")
    p.add_argument("--model-version", type=str, default=None,
                   help="Generation tag e.g. v1_5_c1 — auto-creates models/<version>/ and "
                        "names files catboost_btc_<version>.cbm. Overrides --model-dir. "
                        "Register new versions in DOC/GENERATIONS.md")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Smoke test (--smoke flag or __main__ with no args)
# ---------------------------------------------------------------------------

def smoke_test():
    """Runs a fast end-to-end test with synthetic data."""
    import io, contextlib

    print("Smoke test: synthetic data through full pipeline...")
    rng = np.random.default_rng(42)
    n = 600

    # Minimal instrument set: BTC + 3 FX pairs
    insts = ["BTCUSD", "EURUSD", "GBPUSD", "USDJPY"]
    ohlc_all: dict[str, dict] = {}
    for inst in insts:
        base = 50000.0 if inst == "BTCUSD" else (1.1 if "USD" in inst else 130.0)
        c = base + np.cumsum(rng.normal(0, base * 0.001, n))
        ohlc_all[inst] = {
            "dt": np.array([f"2026-01-{1 + t//1440:02d} {(t%1440)//60:02d}:{t%60:02d}" for t in range(n)]),
            "o":  (c + rng.normal(0, base * 0.0003, n)).astype(np.float32),
            "h":  (c + np.abs(rng.normal(0, base * 0.0005, n))).astype(np.float32),
            "l":  (c - np.abs(rng.normal(0, base * 0.0005, n))).astype(np.float32),
            "c":  c.astype(np.float32),
            "sp": np.abs(rng.normal(0.0002, 0.0001, n)).astype(np.float32),
            "tk": rng.integers(5, 50, n).astype(np.float32),
        }

    matrices = build_feature_matrices(ohlc_all, horizon=3, lookback=60, verbose=False)

    btc_X, btc_y = matrices["btc_X"], matrices["btc_y"]
    fx_X,  fx_y  = matrices["fx_X"],  matrices["fx_y"]

    print(f"  BTC features: {btc_X.shape}, classes: {np.bincount(btc_y)}")
    print(f"  FX  features: {fx_X.shape},  classes: {np.bincount(fx_y)}")
    assert btc_X.shape[1] == len(matrices["feature_names_btc"]), "BTC feature name mismatch"
    assert fx_X.shape[1] == len(matrices["feature_names_fx"]),   "FX feature name mismatch"

    # Label sanity: all three classes should appear with enough data
    assert 0 in btc_y or 2 in btc_y, "BTC labels degenerate"

    print("  Feature matrix OK")
    print("  feature_names_btc[:3]:", matrices["feature_names_btc"][:3])
    print("  feature_names_fx[:3]:", matrices["feature_names_fx"][:3])
    print("  feature_names_fx[-1]:", matrices["feature_names_fx"][-1])

    if not HAS_CATBOOST:
        print("  catboost not installed — skipping model train (pip install catboost)")
    else:
        # Mini CatBoost fit
        btc_m = _make_catboost(50, 4, 0.1)
        btc_m.fit(Pool(btc_X, btc_y), verbose=False)
        acc = float((btc_m.predict(btc_X).flatten().astype(int) == btc_y).mean())
        print(f"  CatBoost-BTC train acc (overfit check): {acc:.3f}")

    print("Smoke test PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--smoke"):
        smoke_test()
    else:
        run(parse_args())
