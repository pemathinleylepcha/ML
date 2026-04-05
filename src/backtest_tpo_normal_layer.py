from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from math_engine import MathEngine
from research_backtester import BinaryHysteresisFilter
from research_dataset import SESSION_NAMES, _load_base_frames, encode_session_codes
from research_inference import CompactFXInference
from tpo_normal_layer import build_tpo_normal_decision
from universe import FX_PAIRS, PIP_SIZES, PIP_VALUES


DEFAULT_DATA_ROOT = Path("data/DataExtractor")
DEFAULT_MODEL_DIR = Path("models/live_compact_no_bridge_optk10")
DEFAULT_REPORT_PATH = Path("data/remote_clean_2025_runs/clean_2025_no_bridge_optk10_report.json")
DEFAULT_OUTPUT = Path("data/tpo_normal_backtest_2025.json")

ATR_MULT_TP = 2.0
ATR_MULT_SL = 1.5
MIN_TP_SPREAD = 5.0
MIN_SL_SPREAD = 3.0
LOT_MIN = 0.01
LOT_MAX = 10.0
BASE_BALANCE = 50.0
RISK_PCT = 0.01
WARMUP_BARS = 90


@dataclass(slots=True)
class PairArrays:
    dt: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    sp_raw: np.ndarray
    tk: np.ndarray
    spread_cost: np.ndarray
    atr_price: np.ndarray
    atr_norm: np.ndarray
    feature_matrix: np.ndarray
    p_buy: np.ndarray
    p_sell: np.ndarray
    session_code: np.ndarray
    lap_residual: np.ndarray


@dataclass(slots=True)
class StrategySignalSeries:
    direction: np.ndarray
    confidence: np.ndarray
    sl_distance: np.ndarray
    tp_distance: np.ndarray
    lot: np.ndarray


@dataclass(slots=True)
class TradeSummary:
    trade_count: int
    win_count: int
    trade_returns: list[float]
    trade_pnls: list[float]
    vol_bucket_counts: dict[str, int]
    vol_bucket_wins: dict[str, int]
    vol_bucket_pnl: dict[str, float]
    pair_stats: dict[str, dict[str, float]]
    exit_reason_counts: dict[str, int]


def _estimate_spread_cost(pair: str, spread_raw: float) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    if spread_raw <= 0.0:
        return 0.0
    if spread_raw < pip * 50.0:
        return float(spread_raw)
    return float(spread_raw) * (pip / 10.0)


def _compute_lot(balance: float, sl_distance: float, pair: str) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    pip_value = float(PIP_VALUES.get(pair, 10.0))
    if sl_distance <= 0.0:
        return 0.0
    sl_pips = sl_distance / max(pip, 1e-8)
    if sl_pips <= 0.0:
        return 0.0
    risk_usd = float(balance) * RISK_PCT
    lot = risk_usd / max(sl_pips * pip_value, 1e-8)
    return float(np.clip(lot, LOT_MIN, LOT_MAX))


def _lagged_logret_array(close: np.ndarray, lag: int) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return out
    safe = np.maximum(close.astype(np.float64, copy=False), 1e-10)
    idx = np.arange(n, dtype=np.int64)
    prev_idx = np.maximum(0, idx - lag)
    valid = idx > 0
    out[valid] = np.log(safe[valid] / safe[prev_idx[valid]]).astype(np.float32)
    return out


def _rolling_zscore_from_past(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    vals = values.astype(np.float64, copy=False)
    csum = np.concatenate(([0.0], np.cumsum(vals)))
    csum_sq = np.concatenate(([0.0], np.cumsum(vals * vals)))
    idx = np.arange(len(vals), dtype=np.int64)
    start = np.maximum(0, idx - window)
    count = idx - start

    sums = csum[idx] - csum[start]
    sums_sq = csum_sq[idx] - csum_sq[start]
    means = np.divide(sums, count, out=np.zeros(len(vals), dtype=np.float64), where=count > 0)
    variances = np.divide(sums_sq, count, out=np.zeros(len(vals), dtype=np.float64), where=count > 0) - means * means
    variances = np.maximum(variances, 0.0)

    z = np.zeros(len(vals), dtype=np.float64)
    mask = count > 0
    z[mask] = (vals[mask] - means[mask]) / np.sqrt(variances[mask] + 1e-8)
    return z.astype(np.float32)


def _atr_proxy_series(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int = 24) -> np.ndarray:
    n = len(close)
    out = np.maximum(high - low, 0.0).astype(np.float32)
    if n <= 1:
        return out
    tr = np.zeros(n, dtype=np.float64)
    tr[1:] = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ]
    )
    csum = np.concatenate(([0.0], np.cumsum(tr)))
    for idx in range(1, n):
        start = max(1, idx - window + 1)
        count = idx - start + 1
        out[idx] = float((csum[idx + 1] - csum[start]) / max(count, 1))
    return out


def _load_common_frames(data_root: Path, start: str | None, end: str | None) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    base_frames = _load_base_frames(data_root, symbols=FX_PAIRS, start=start, end=end)
    missing = [symbol for symbol in FX_PAIRS if symbol not in base_frames]
    if missing:
        raise ValueError(f"Missing M5 history for FX pairs: {missing}")

    common_index: pd.DatetimeIndex | None = None
    for symbol in FX_PAIRS:
        frame = base_frames[symbol]
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) < WARMUP_BARS + 10:
        raise ValueError("Insufficient common M5 history across all FX pairs")
    common_index = common_index.sort_values()
    aligned = {symbol: base_frames[symbol].loc[common_index].copy() for symbol in FX_PAIRS}
    return common_index, aligned


def _compute_lap_residual_matrix(aligned: dict[str, pd.DataFrame]) -> np.ndarray:
    close_matrix = np.column_stack([aligned[symbol]["c"].to_numpy(dtype=np.float64) for symbol in FX_PAIRS])
    returns_matrix = np.zeros_like(close_matrix, dtype=np.float64)
    if close_matrix.shape[0] > 1:
        returns_matrix[1:] = np.diff(np.log(np.maximum(close_matrix, 1e-10)), axis=0)

    engine = MathEngine(n_pairs=len(FX_PAIRS))
    residuals = np.zeros_like(returns_matrix, dtype=np.float32)
    for idx in range(len(returns_matrix)):
        state = engine.update(returns_matrix[idx])
        if state.valid:
            residuals[idx] = state.residuals.astype(np.float32, copy=False)
    return residuals


def _feature_block_for_symbol(
    symbol: str,
    frame: pd.DataFrame,
    session_codes: np.ndarray,
    lap_residual: np.ndarray,
    feature_names: list[str],
) -> PairArrays:
    o = frame["o"].to_numpy(dtype=np.float64)
    h = frame["h"].to_numpy(dtype=np.float64)
    l = frame["l"].to_numpy(dtype=np.float64)
    c = frame["c"].to_numpy(dtype=np.float64)
    sp_raw = frame["sp"].to_numpy(dtype=np.float64)
    tk = frame["tk"].to_numpy(dtype=np.float64)
    safe_close = np.maximum(np.abs(c), 1e-10)
    spread_cost = np.asarray([_estimate_spread_cost(symbol, float(value)) for value in sp_raw], dtype=np.float32)
    atr_price = _atr_proxy_series(c, h, l, window=24)
    range_norm = ((h - l) / safe_close).astype(np.float32)
    liquidity_stress = (spread_cost / safe_close + 1.0 / np.sqrt(tk + 1.0)).astype(np.float32)

    all_features = {
        "local_ret_1": _lagged_logret_array(c, lag=1),
        "local_ret_3": _lagged_logret_array(c, lag=3),
        "local_ret_6": _lagged_logret_array(c, lag=6),
        "local_atr_norm": (atr_price / safe_close).astype(np.float32),
        "local_range_norm": range_norm,
        "local_spread_z": _rolling_zscore_from_past(spread_cost, window=64),
        "local_tick_z": _rolling_zscore_from_past(tk.astype(np.float32), window=64),
        "local_liquidity_stress": liquidity_stress,
        "local_lap_residual": lap_residual.astype(np.float32, copy=False),
        "local_session_code": session_codes.astype(np.float32),
    }

    missing = [name for name in feature_names if name not in all_features]
    if missing:
        raise ValueError(f"Feature builder is missing expected names: {missing}")

    feature_matrix = np.column_stack([all_features[name] for name in feature_names]).astype(np.float32, copy=False)
    return PairArrays(
        dt=frame.index.to_numpy(dtype="datetime64[ns]"),
        o=o.astype(np.float32),
        h=h.astype(np.float32),
        l=l.astype(np.float32),
        c=c.astype(np.float32),
        sp_raw=sp_raw.astype(np.float32),
        tk=tk.astype(np.float32),
        spread_cost=spread_cost.astype(np.float32),
        atr_price=atr_price.astype(np.float32),
        atr_norm=all_features["local_atr_norm"].astype(np.float32, copy=False),
        feature_matrix=feature_matrix,
        p_buy=np.zeros(len(frame), dtype=np.float32),
        p_sell=np.zeros(len(frame), dtype=np.float32),
        session_code=session_codes.astype(np.int8),
        lap_residual=lap_residual.astype(np.float32, copy=False),
    )


def _legacy_signal_series(
    p_buy: np.ndarray,
    p_sell: np.ndarray,
    atr_price: np.ndarray,
    spread_cost: np.ndarray,
    pair: str,
    entry_threshold: float,
    exit_threshold: float,
    confidence_threshold: float,
) -> StrategySignalSeries:
    filt = BinaryHysteresisFilter(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        confidence_threshold=confidence_threshold,
    )
    direction = np.zeros(len(p_buy), dtype=np.int8)
    confidence = np.zeros(len(p_buy), dtype=np.float32)
    sl_distance = np.maximum(ATR_MULT_SL * atr_price, spread_cost * MIN_SL_SPREAD).astype(np.float32)
    tp_distance = np.maximum(ATR_MULT_TP * atr_price, spread_cost * MIN_TP_SPREAD).astype(np.float32)
    lot = np.zeros(len(p_buy), dtype=np.float32)

    for idx in range(len(p_buy)):
        direction[idx] = filt.step(float(p_sell[idx]), float(p_buy[idx]))
        if direction[idx] != 0:
            confidence[idx] = abs(float(p_buy[idx]) - float(p_sell[idx]))
            lot[idx] = _compute_lot(BASE_BALANCE, float(sl_distance[idx]), pair)
    return StrategySignalSeries(direction=direction, confidence=confidence, sl_distance=sl_distance, tp_distance=tp_distance, lot=lot)


def _tpo_signal_series(
    pair: str,
    arrays: PairArrays,
    entry_threshold: float,
    exit_threshold: float,
    confidence_threshold: float,
) -> tuple[StrategySignalSeries, dict[str, object]]:
    legacy_filter = BinaryHysteresisFilter(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        confidence_threshold=confidence_threshold,
    )
    normal_filter = BinaryHysteresisFilter(
        entry_threshold=max(0.53, entry_threshold - 0.04),
        exit_threshold=max(0.49, exit_threshold - 0.02),
        confidence_threshold=max(0.05, confidence_threshold - 0.04),
    )

    direction = np.zeros(len(arrays.c), dtype=np.int8)
    confidence = np.zeros(len(arrays.c), dtype=np.float32)
    sl_distance = np.zeros(len(arrays.c), dtype=np.float32)
    tp_distance = np.zeros(len(arrays.c), dtype=np.float32)
    lot = np.zeros(len(arrays.c), dtype=np.float32)

    protector_reason_counts: dict[str, int] = {}
    raw_signal_count = 0
    blocked_signal_count = 0

    for idx in range(len(arrays.c)):
        legacy_direction = legacy_filter.step(float(arrays.p_sell[idx]), float(arrays.p_buy[idx]))
        legacy_confidence = abs(float(arrays.p_buy[idx]) - float(arrays.p_sell[idx]))
        if idx < 8:
            continue

        decision = build_tpo_normal_decision(
            close=arrays.c[: idx + 1].astype(np.float64, copy=False),
            high=arrays.h[: idx + 1].astype(np.float64, copy=False),
            low=arrays.l[: idx + 1].astype(np.float64, copy=False),
            atr_price=float(arrays.atr_price[idx]),
            spread_price=float(arrays.spread_cost[idx]),
            legacy_direction=legacy_direction,
            legacy_confidence=legacy_confidence,
            legacy_p_buy=float(arrays.p_buy[idx]),
            legacy_p_sell=float(arrays.p_sell[idx]),
        )

        if decision.direction != 0:
            raw_signal_count += 1
            if decision.protector_blocked:
                blocked_signal_count += 1
                protector_reason_counts[decision.protector_reason] = protector_reason_counts.get(decision.protector_reason, 0) + 1

        tpo_buy = 0.5
        tpo_sell = 0.5
        if decision.direction > 0:
            tpo_buy = 0.5 + 0.5 * decision.confidence
            tpo_sell = 1.0 - tpo_buy
        elif decision.direction < 0:
            tpo_sell = 0.5 + 0.5 * decision.confidence
            tpo_buy = 1.0 - tpo_sell

        direction[idx] = normal_filter.step(float(tpo_sell), float(tpo_buy))
        if direction[idx] != 0:
            confidence[idx] = float(decision.confidence)
            sl_distance[idx] = max(float(decision.sl_distance), float(arrays.spread_cost[idx]) * MIN_SL_SPREAD)
            tp_distance[idx] = max(float(decision.tp_distance), float(arrays.spread_cost[idx]) * MIN_TP_SPREAD)
            lot[idx] = _compute_lot(BASE_BALANCE, float(sl_distance[idx]), pair) * max(float(decision.lot_scale), 0.0)

    meta = {
        "raw_signal_count": int(raw_signal_count),
        "blocked_signal_count": int(blocked_signal_count),
        "blocked_signal_rate": float(blocked_signal_count / raw_signal_count) if raw_signal_count > 0 else 0.0,
        "protector_reason_counts": protector_reason_counts,
    }
    return StrategySignalSeries(direction=direction, confidence=confidence, sl_distance=sl_distance, tp_distance=tp_distance, lot=lot), meta


def _vol_bucket_label(value: float, low_thr: float, high_thr: float) -> str:
    if value <= low_thr:
        return "low"
    if value <= high_thr:
        return "normal"
    return "high"


def _trade_pnl_usd(pair: str, direction: int, entry_price: float, exit_price: float, lot: float) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    pip_value = float(PIP_VALUES.get(pair, 10.0))
    move = (exit_price - entry_price) * float(direction)
    pips = move / max(pip, 1e-8)
    return float(pips * pip_value * lot)


def _simulate_pair_strategy(
    pair: str,
    arrays: PairArrays,
    signals: StrategySignalSeries,
    start_idx: int,
    vol_low_thr: float,
    vol_high_thr: float,
) -> TradeSummary:
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    vol_bucket_counts = {"low": 0, "normal": 0, "high": 0}
    vol_bucket_wins = {"low": 0, "normal": 0, "high": 0}
    vol_bucket_pnl = {"low": 0.0, "normal": 0.0, "high": 0.0}
    exit_reason_counts: dict[str, int] = {}
    pair_stats = {
        pair: {
            "trade_count": 0.0,
            "win_count": 0.0,
            "net_pnl_usd": 0.0,
        }
    }

    current_direction = 0
    entry_price = 0.0
    entry_idx = -1
    entry_lot = 0.0
    current_sl_distance = 0.0
    current_tp_distance = 0.0
    current_bucket = "normal"

    def close_trade(exit_price: float, reason: str) -> None:
        nonlocal current_direction, entry_price, entry_idx, entry_lot, current_sl_distance, current_tp_distance, current_bucket
        if current_direction == 0:
            return
        pnl_usd = _trade_pnl_usd(pair, current_direction, entry_price, exit_price, entry_lot)
        trade_return = pnl_usd / BASE_BALANCE
        trade_returns.append(float(trade_return))
        trade_pnls.append(float(pnl_usd))
        vol_bucket_counts[current_bucket] += 1
        vol_bucket_pnl[current_bucket] += float(pnl_usd)
        pair_stats[pair]["trade_count"] += 1.0
        pair_stats[pair]["net_pnl_usd"] += float(pnl_usd)
        if pnl_usd > 0.0:
            vol_bucket_wins[current_bucket] += 1
            pair_stats[pair]["win_count"] += 1.0
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        current_direction = 0
        entry_price = 0.0
        entry_idx = -1
        entry_lot = 0.0
        current_sl_distance = 0.0
        current_tp_distance = 0.0
        current_bucket = "normal"

    for idx in range(start_idx, len(arrays.c) - 1):
        bar = idx + 1
        desired_direction = int(signals.direction[idx])
        open_price = float(arrays.o[bar])
        high_price = float(arrays.h[bar])
        low_price = float(arrays.l[bar])

        if current_direction != 0 and desired_direction != current_direction:
            reason = "signal_exit" if desired_direction == 0 else "signal_flip"
            close_trade(open_price, reason)

        if current_direction == 0 and desired_direction != 0 and signals.lot[idx] > 0.0:
            current_direction = desired_direction
            entry_price = open_price
            entry_idx = idx
            entry_lot = float(signals.lot[idx])
            current_sl_distance = float(signals.sl_distance[idx])
            current_tp_distance = float(signals.tp_distance[idx])
            current_bucket = _vol_bucket_label(float(arrays.atr_norm[idx]), vol_low_thr, vol_high_thr)

        if current_direction == 0:
            continue

        if current_direction > 0:
            stop_price = entry_price - current_sl_distance
            take_price = entry_price + current_tp_distance
            hit_sl = low_price <= stop_price
            hit_tp = high_price >= take_price
            if hit_sl and hit_tp:
                close_trade(stop_price, "sl_tp_same_bar")
            elif hit_sl:
                close_trade(stop_price, "stop_loss")
            elif hit_tp:
                close_trade(take_price, "take_profit")
        else:
            stop_price = entry_price + current_sl_distance
            take_price = entry_price - current_tp_distance
            hit_sl = high_price >= stop_price
            hit_tp = low_price <= take_price
            if hit_sl and hit_tp:
                close_trade(stop_price, "sl_tp_same_bar")
            elif hit_sl:
                close_trade(stop_price, "stop_loss")
            elif hit_tp:
                close_trade(take_price, "take_profit")

    if current_direction != 0:
        close_trade(float(arrays.c[-1]), "end_of_test")

    win_count = int(sum(1 for value in trade_pnls if value > 0.0))
    return TradeSummary(
        trade_count=len(trade_returns),
        win_count=win_count,
        trade_returns=trade_returns,
        trade_pnls=trade_pnls,
        vol_bucket_counts=vol_bucket_counts,
        vol_bucket_wins=vol_bucket_wins,
        vol_bucket_pnl=vol_bucket_pnl,
        pair_stats=pair_stats,
        exit_reason_counts=exit_reason_counts,
    )


def _merge_trade_summaries(summaries: list[TradeSummary]) -> TradeSummary:
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    vol_bucket_counts = {"low": 0, "normal": 0, "high": 0}
    vol_bucket_wins = {"low": 0, "normal": 0, "high": 0}
    vol_bucket_pnl = {"low": 0.0, "normal": 0.0, "high": 0.0}
    pair_stats: dict[str, dict[str, float]] = {}
    exit_reason_counts: dict[str, int] = {}

    for summary in summaries:
        trade_returns.extend(summary.trade_returns)
        trade_pnls.extend(summary.trade_pnls)
        for bucket in vol_bucket_counts:
            vol_bucket_counts[bucket] += summary.vol_bucket_counts.get(bucket, 0)
            vol_bucket_wins[bucket] += summary.vol_bucket_wins.get(bucket, 0)
            vol_bucket_pnl[bucket] += summary.vol_bucket_pnl.get(bucket, 0.0)
        for pair, stats in summary.pair_stats.items():
            merged = pair_stats.setdefault(pair, {"trade_count": 0.0, "win_count": 0.0, "net_pnl_usd": 0.0})
            merged["trade_count"] += stats.get("trade_count", 0.0)
            merged["win_count"] += stats.get("win_count", 0.0)
            merged["net_pnl_usd"] += stats.get("net_pnl_usd", 0.0)
        for reason, count in summary.exit_reason_counts.items():
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + count

    win_count = int(sum(1 for value in trade_pnls if value > 0.0))
    return TradeSummary(
        trade_count=len(trade_returns),
        win_count=win_count,
        trade_returns=trade_returns,
        trade_pnls=trade_pnls,
        vol_bucket_counts=vol_bucket_counts,
        vol_bucket_wins=vol_bucket_wins,
        vol_bucket_pnl=vol_bucket_pnl,
        pair_stats=pair_stats,
        exit_reason_counts=exit_reason_counts,
    )


def _summary_payload(summary: TradeSummary) -> dict[str, object]:
    returns = np.asarray(summary.trade_returns, dtype=np.float64)
    pnls = np.asarray(summary.trade_pnls, dtype=np.float64)
    sharpe = 0.0
    if len(returns) > 1:
        sharpe = float(returns.mean() / (returns.std(ddof=0) + 1e-8) * np.sqrt(len(returns)))
    pair_table = []
    for pair, stats in sorted(summary.pair_stats.items(), key=lambda item: item[1]["net_pnl_usd"], reverse=True):
        trade_count = int(stats["trade_count"])
        win_rate = float(stats["win_count"] / trade_count) if trade_count > 0 else 0.0
        pair_table.append(
            {
                "symbol": pair,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "net_pnl_usd": float(stats["net_pnl_usd"]),
            }
        )

    vol_payload: dict[str, object] = {}
    for bucket, count in summary.vol_bucket_counts.items():
        wins = summary.vol_bucket_wins[bucket]
        vol_payload[bucket] = {
            "trade_count": int(count),
            "win_rate": float(wins / count) if count > 0 else 0.0,
            "net_pnl_usd": float(summary.vol_bucket_pnl[bucket]),
        }

    return {
        "trade_count": int(summary.trade_count),
        "win_rate": float(summary.win_count / summary.trade_count) if summary.trade_count > 0 else 0.0,
        "trade_sharpe": sharpe,
        "avg_trade_return_pct": float(100.0 * returns.mean()) if len(returns) > 0 else 0.0,
        "net_return_pct_on_50usd": float(100.0 * returns.sum()),
        "net_pnl_usd_on_50usd": float(pnls.sum()) if len(pnls) > 0 else 0.0,
        "avg_trade_pnl_usd": float(pnls.mean()) if len(pnls) > 0 else 0.0,
        "exit_reason_counts": summary.exit_reason_counts,
        "volatility_buckets": vol_payload,
        "top_pairs_by_pnl": pair_table[:10],
        "top_pairs_by_trade_count": sorted(pair_table, key=lambda item: item["trade_count"], reverse=True)[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical backtest for TPO normal-trading layer vs compact legacy logic")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-bars", type=int, default=0, help="Optional tail truncation for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = CompactFXInference(args.model_dir)

    with open(args.report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    backtest_cfg = report["config"]["backtest"]
    entry_threshold = float(backtest_cfg["entry_threshold"])
    exit_threshold = float(backtest_cfg["exit_threshold"])
    confidence_threshold = float(backtest_cfg["confidence_threshold"])

    common_index, aligned = _load_common_frames(Path(args.data_root), start=args.start, end=args.end)
    if args.max_bars and len(common_index) > args.max_bars:
        common_index = common_index[-args.max_bars:]
        aligned = {symbol: frame.loc[common_index].copy() for symbol, frame in aligned.items()}

    session_codes = encode_session_codes(common_index.to_numpy(dtype="datetime64[ns]"))
    residuals = _compute_lap_residual_matrix(aligned)

    pair_arrays: dict[str, PairArrays] = {}
    all_atr_norm: list[np.ndarray] = []
    for col, symbol in enumerate(FX_PAIRS):
        arrays = _feature_block_for_symbol(
            symbol=symbol,
            frame=aligned[symbol],
            session_codes=session_codes,
            lap_residual=residuals[:, col],
            feature_names=model.feature_names,
        )
        proba = model.predict_proba(arrays.feature_matrix)
        arrays.p_sell = proba[:, 0].astype(np.float32)
        arrays.p_buy = proba[:, 1].astype(np.float32)
        pair_arrays[symbol] = arrays
        all_atr_norm.append(arrays.atr_norm[WARMUP_BARS:-1])

    flat_atr = np.concatenate(all_atr_norm) if all_atr_norm else np.zeros(0, dtype=np.float32)
    if len(flat_atr) == 0:
        raise ValueError("No ATR-normalized values available for volatility bucket computation")
    vol_low_thr, vol_high_thr = np.quantile(flat_atr, [1.0 / 3.0, 2.0 / 3.0]).astype(float)

    legacy_summaries: list[TradeSummary] = []
    tpo_summaries: list[TradeSummary] = []
    protector_meta: dict[str, dict[str, object]] = {}

    for symbol in FX_PAIRS:
        arrays = pair_arrays[symbol]
        legacy_signals = _legacy_signal_series(
            p_buy=arrays.p_buy,
            p_sell=arrays.p_sell,
            atr_price=arrays.atr_price,
            spread_cost=arrays.spread_cost,
            pair=symbol,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            confidence_threshold=confidence_threshold,
        )
        tpo_signals, tpo_meta = _tpo_signal_series(
            pair=symbol,
            arrays=arrays,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            confidence_threshold=confidence_threshold,
        )
        legacy_summaries.append(
            _simulate_pair_strategy(
                pair=symbol,
                arrays=arrays,
                signals=legacy_signals,
                start_idx=WARMUP_BARS - 1,
                vol_low_thr=vol_low_thr,
                vol_high_thr=vol_high_thr,
            )
        )
        tpo_summaries.append(
            _simulate_pair_strategy(
                pair=symbol,
                arrays=arrays,
                signals=tpo_signals,
                start_idx=WARMUP_BARS - 1,
                vol_low_thr=vol_low_thr,
                vol_high_thr=vol_high_thr,
            )
        )
        protector_meta[symbol] = tpo_meta

    legacy_summary = _merge_trade_summaries(legacy_summaries)
    tpo_summary = _merge_trade_summaries(tpo_summaries)

    raw_signal_count = int(sum(int(meta["raw_signal_count"]) for meta in protector_meta.values()))
    blocked_signal_count = int(sum(int(meta["blocked_signal_count"]) for meta in protector_meta.values()))
    protector_reason_counts: dict[str, int] = {}
    for meta in protector_meta.values():
        for reason, count in meta["protector_reason_counts"].items():
            protector_reason_counts[reason] = protector_reason_counts.get(reason, 0) + int(count)

    output = {
        "config": {
            "data_root": str(Path(args.data_root)),
            "model_dir": str(Path(args.model_dir)),
            "report_path": str(Path(args.report_path)),
            "start": args.start,
            "end": args.end,
            "n_common_bars": int(len(common_index)),
            "n_pairs": len(FX_PAIRS),
            "fixed_balance_usd": BASE_BALANCE,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "confidence_threshold": confidence_threshold,
            "assumptions": {
                "signal_timing": "Signals are computed on closed bar t and executed at next bar open t+1.",
                "tp_sl_intrabar_rule": "If TP and SL are both touched in the same bar, the stop-loss is taken first (conservative).",
                "lot_sizing": "Lots are computed with a fixed 50 USD balance proxy per trade for cross-strategy comparability.",
            },
        },
        "volatility_thresholds": {
            "low_to_normal_atr_norm": float(vol_low_thr),
            "normal_to_high_atr_norm": float(vol_high_thr),
        },
        "legacy_compact_only": _summary_payload(legacy_summary),
        "tpo_normal_with_legacy_protector": _summary_payload(tpo_summary),
        "tpo_protector": {
            "raw_signal_count": raw_signal_count,
            "blocked_signal_count": blocked_signal_count,
            "blocked_signal_rate": float(blocked_signal_count / raw_signal_count) if raw_signal_count > 0 else 0.0,
            "reason_counts": protector_reason_counts,
        },
        "comparison": {
            "trade_count_delta": int(tpo_summary.trade_count - legacy_summary.trade_count),
            "win_rate_delta": float(
                (tpo_summary.win_count / tpo_summary.trade_count) if tpo_summary.trade_count > 0 else 0.0
            ) - float(
                (legacy_summary.win_count / legacy_summary.trade_count) if legacy_summary.trade_count > 0 else 0.0
            ),
            "trade_sharpe_delta": float(_summary_payload(tpo_summary)["trade_sharpe"]) - float(_summary_payload(legacy_summary)["trade_sharpe"]),
            "net_pnl_usd_delta": float(np.sum(tpo_summary.trade_pnls) - np.sum(legacy_summary.trade_pnls)),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(json.dumps(output["comparison"], indent=2))


if __name__ == "__main__":
    main()
