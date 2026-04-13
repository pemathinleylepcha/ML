from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from research_dataset import compute_atr_series
from staged_v5.config import BacktestConfig
from staged_v5.evaluation.contracts import BarState, EntryContext, OpenPosition, RejectionCounters
from staged_v5.evaluation.entry_strategies import ENTRY_REGISTRY
from staged_v5.evaluation.exit_strategies import EXIT_REGISTRY, evaluate_weekend_close
from staged_v5.utils.critical_fixes import enforce_correlation_exposure


def adjusted_backtest_config(cfg: BacktestConfig, val_ece: float | None = None) -> BacktestConfig:
    adjusted = deepcopy(cfg)
    if adjusted.ece_gate_threshold > 0.0 and val_ece is not None and val_ece > adjusted.ece_gate_threshold:
        adjusted.position.max_positions = max(2, adjusted.position.max_positions // 2)
    return adjusted


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252.0 * 24.0 * 12.0))


def _compute_atr_matrix(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    atr = np.zeros_like(close, dtype=np.float64)
    for node_idx in range(close.shape[1]):
        atr[:, node_idx] = compute_atr_series(
            high[:, node_idx].astype(np.float32),
            low[:, node_idx].astype(np.float32),
            close[:, node_idx].astype(np.float32),
            period=period,
        )
    return atr


def _realized_return(direction: int, entry_price: float, exit_price: float) -> float:
    if direction == 1:
        return float((exit_price - entry_price) / max(abs(entry_price), 1e-8))
    return float((entry_price - exit_price) / max(abs(entry_price), 1e-8))


def _build_threshold_diagnostics(confidences: list[float], returns: list[float]) -> list[dict[str, float | int]]:
    if not confidences:
        return []
    confidence_arr = np.asarray(confidences, dtype=np.float64)
    returns_arr = np.asarray(returns, dtype=np.float64)
    buckets = ((0.50, 0.57), (0.57, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01))
    diagnostics: list[dict[str, float | int]] = []
    for lo, hi in buckets:
        mask = (confidence_arr >= lo) & (confidence_arr < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            diagnostics.append({"min_confidence": lo, "max_confidence": hi, "count": 0, "win_rate": 0.0, "avg_return": 0.0})
            continue
        bucket_returns = returns_arr[mask]
        diagnostics.append(
            {
                "min_confidence": lo,
                "max_confidence": hi,
                "count": count,
                "win_rate": float(np.mean(bucket_returns > 0.0)),
                "avg_return": float(np.mean(bucket_returns)),
            }
        )
    return diagnostics


def _mean_by_action(counts: dict[str, int], sums: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    keys = set(counts) | set(sums)
    for key in keys:
        count = int(counts.get(key, 0))
        metrics[key] = float(sums.get(key, 0.0) / count) if count > 0 else 0.0
    return metrics


def _rate_by_action(action_counts: dict[str, int], metric_counts: dict[str, int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    keys = set(action_counts) | set(metric_counts)
    for key in keys:
        total = int(action_counts.get(key, 0))
        metrics[key] = float(metric_counts.get(key, 0) / total) if total > 0 else 0.0
    return metrics


def backtest_probabilities(
    prob_buy: np.ndarray,
    prob_entry: np.ndarray | None,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volatility: np.ndarray,
    session_codes: np.ndarray,
    pair_names: tuple[str, ...],
    cfg: BacktestConfig,
    *,
    timestamps: np.ndarray | None = None,
    anchor_node_features: np.ndarray | None = None,
    anchor_tpo_features: np.ndarray | None = None,
    neural_gate_runtime=None,
) -> dict:
    n_steps, n_nodes = prob_buy.shape
    atr = _compute_atr_matrix(high, low, close)
    entry_cfg = cfg.entry
    exit_cfg = cfg.exit
    pos_cfg = cfg.position

    entry_fn = ENTRY_REGISTRY[entry_cfg.entry_type]
    exit_fn = EXIT_REGISTRY[exit_cfg.exit_type]

    is_weekend_close = np.zeros(n_steps, dtype=bool)
    if exit_cfg.close_before_weekend and timestamps is not None:
        import pandas as pd

        ts_index = pd.DatetimeIndex(timestamps)
        is_weekend_close = np.asarray((ts_index.dayofweek == 4) & (ts_index.hour >= exit_cfg.weekend_close_hour_utc), dtype=bool)

    active_cooldown = np.zeros(n_nodes, dtype=np.int32)
    open_positions: list[OpenPosition] = []
    trade_returns: list[float] = []
    trade_confidences: list[float] = []
    bar_returns = np.zeros(n_steps, dtype=np.float64)
    trade_count = 0
    wins = 0
    blocked_by_exposure = 0
    rejection_counters = RejectionCounters()
    exit_reason_counts: dict[str, int] = {}
    max_open_positions = 0
    hold_bars: list[int] = []
    gate_action_counts = getattr(neural_gate_runtime, "action_counts", {}) if neural_gate_runtime is not None else {}
    gate_fill_counts = getattr(neural_gate_runtime, "fill_counts", {}) if neural_gate_runtime is not None else {}
    gate_no_fill_counts = getattr(neural_gate_runtime, "no_fill_counts", {}) if neural_gate_runtime is not None else {}
    gate_near_miss_counts = getattr(neural_gate_runtime, "near_miss_counts", {}) if neural_gate_runtime is not None else {}
    gate_reward_sums = getattr(neural_gate_runtime, "reward_sums", {}) if neural_gate_runtime is not None else {}

    def _close_position(position: OpenPosition, current_bar: int, exit_price: float, reason: str) -> None:
        nonlocal trade_count, wins
        realized = _realized_return(position.direction, position.entry_price, exit_price)
        trade_returns.append(realized)
        trade_confidences.append(position.confidence)
        bar_returns[current_bar] += realized
        trade_count += 1
        if realized > 0.0:
            wins += 1
        else:
            active_cooldown[position.node_idx] = max(active_cooldown[position.node_idx], pos_cfg.cooldown_bars)
        hold_bars.append(max(current_bar - position.entry_bar, 0))
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    start_bar = max(1, entry_cfg.latency_bars)
    if entry_cfg.entry_type == "neural_gate":
        if neural_gate_runtime is None:
            raise ValueError("entry_type='neural_gate' requires neural_gate_runtime")
        if timestamps is None or anchor_node_features is None or anchor_tpo_features is None:
            raise ValueError("entry_type='neural_gate' requires timestamps, anchor_node_features, and anchor_tpo_features")
    for t in range(start_bar, n_steps):
        rejection_counters.bars_seen += 1
        active_cooldown = np.maximum(active_cooldown - 1, 0)

        still_open: list[OpenPosition] = []
        for position in open_positions:
            if t <= position.entry_bar:
                still_open.append(position)
                continue

            bar = BarState(
                bar_index=t,
                node_idx=position.node_idx,
                prob_buy=float(prob_buy[t, position.node_idx]),
                prob_entry=float(prob_entry[t, position.node_idx]) if prob_entry is not None else None,
                high=float(high[t, position.node_idx]),
                low=float(low[t, position.node_idx]),
                close=float(close[t, position.node_idx]),
                atr=float(atr[t, position.node_idx]),
                volatility=float(volatility[t, position.node_idx]),
                session_code=int(session_codes[t]),
                pair_name=position.pair_name,
            )

            weekend_decision = evaluate_weekend_close(
                position,
                bar,
                exit_cfg,
                is_weekend_bar=bool(is_weekend_close[t]),
            )
            if weekend_decision is not None:
                _close_position(position, t, weekend_decision.exit_price, weekend_decision.reason)
                continue

            decision = exit_fn(position, bar, exit_cfg)
            if decision is not None:
                _close_position(position, t, decision.exit_price, decision.reason)
                continue

            still_open.append(position)

        open_positions = still_open
        max_open_positions = max(max_open_positions, len(open_positions))

        if t + entry_cfg.latency_bars >= n_steps:
            rejection_counters.latency_skip += 1
            continue
        if session_codes[t] == 0:
            rejection_counters.session_closed += 1
            continue
        if len(open_positions) >= pos_cfg.max_positions:
            rejection_counters.max_positions_blocked += 1
            continue

        open_nodes = {p.node_idx for p in open_positions}
        active_pairs = [p.pair_name for p in open_positions]
        ranked = np.argsort(-np.abs(prob_buy[t] - 0.5))

        for node_idx in ranked:
            if len(open_positions) >= pos_cfg.max_positions:
                break
            if node_idx in open_nodes:
                rejection_counters.exposure_cooldown_blocked += 1
                rejection_counters.already_open_blocked += 1
                continue
            if active_cooldown[node_idx] > 0:
                rejection_counters.exposure_cooldown_blocked += 1
                rejection_counters.cooldown_blocked += 1
                continue

            proposed = active_pairs + [pair_names[node_idx]]
            capped = enforce_correlation_exposure(proposed, active_pairs, pos_cfg.max_group_exposure)
            if len(capped) < len(proposed):
                blocked_by_exposure += 1
                rejection_counters.exposure_cooldown_blocked += 1
                rejection_counters.correlation_exposure_blocked += 1
                continue

            entry_bar = t + entry_cfg.latency_bars
            bar = BarState(
                bar_index=t,
                node_idx=int(node_idx),
                prob_buy=float(prob_buy[t, node_idx]),
                prob_entry=float(prob_entry[t, node_idx]) if prob_entry is not None else None,
                high=float(high[t, node_idx]),
                low=float(low[t, node_idx]),
                close=float(close[t, node_idx]),
                atr=float(atr[t, node_idx]),
                volatility=float(volatility[t, node_idx]),
                session_code=int(session_codes[t]),
                pair_name=pair_names[node_idx],
            )

            rejection_counters.total_evaluated += 1
            context = EntryContext(
                timestamp=timestamps[t] if timestamps is not None else None,
                entry_timestamp=timestamps[entry_bar] if timestamps is not None else None,
                anchor_node_features=anchor_node_features[t, node_idx] if anchor_node_features is not None else None,
                anchor_tpo_features=anchor_tpo_features[t, node_idx] if anchor_tpo_features is not None else None,
                neural_gate_runtime=neural_gate_runtime,
            )
            order = entry_fn(
                bar,
                entry_cfg,
                exit_cfg,
                counters=rejection_counters,
                context=context,
                entry_bar_high=float(high[entry_bar, node_idx]),
                entry_bar_low=float(low[entry_bar, node_idx]),
                entry_bar_close=float(close[entry_bar, node_idx]),
                entry_bar_atr=float(max(atr[entry_bar, node_idx], 1e-8)),
            )
            if order is None:
                continue

            rejection_counters.entries_created += 1
            open_positions.append(
                OpenPosition(
                    node_idx=order.node_idx,
                    pair_name=order.pair_name,
                    direction=order.direction,
                    signal_bar=order.signal_bar,
                    entry_bar=order.entry_bar,
                    entry_price=order.entry_price,
                    tp_price=order.tp_price,
                    sl_price=order.sl_price,
                    confidence=order.confidence,
                    entry_atr=order.entry_atr,
                )
            )
            active_pairs.append(pair_names[node_idx])
            open_nodes.add(node_idx)
            max_open_positions = max(max_open_positions, len(open_positions))

    if open_positions:
        final_bar = n_steps - 1
        for position in open_positions:
            _close_position(position, final_bar, float(close[final_bar, position.node_idx]), "end_of_test")

    return {
        "trade_count": int(trade_count),
        "win_rate": float(wins / trade_count) if trade_count else 0.0,
        "strategy_sharpe": _sharpe(bar_returns),
        "net_return": float(bar_returns.sum()),
        "confidence_hit_rate": float(np.mean(trade_confidences)) if trade_confidences else 0.0,
        "blocked_by_exposure": int(blocked_by_exposure),
        "exit_reason_counts": exit_reason_counts,
        "max_open_positions": int(max_open_positions),
        "avg_hold_bars": float(np.mean(hold_bars)) if hold_bars else 0.0,
        "entry_rejection_counters": asdict(rejection_counters),
        "threshold_diagnostics": _build_threshold_diagnostics(trade_confidences, trade_returns),
        "gate_action_counts": gate_action_counts,
        "gate_reject_count": int(getattr(neural_gate_runtime, "reject_count", 0)) if neural_gate_runtime is not None else 0,
        "gate_no_fill_count": int(getattr(neural_gate_runtime, "no_fill_count", 0)) if neural_gate_runtime is not None else 0,
        "gate_fill_counts": gate_fill_counts,
        "gate_no_fill_counts": gate_no_fill_counts,
        "gate_near_miss_counts": gate_near_miss_counts,
        "gate_mean_reward_by_action": _mean_by_action(gate_action_counts, gate_reward_sums) if neural_gate_runtime is not None else {},
        "gate_fill_rate_by_action": _rate_by_action(gate_action_counts, gate_fill_counts) if neural_gate_runtime is not None else {},
        "gate_no_fill_rate_by_action": _rate_by_action(gate_action_counts, gate_no_fill_counts) if neural_gate_runtime is not None else {},
        "gate_near_miss_rate_by_action": _rate_by_action(gate_action_counts, gate_near_miss_counts) if neural_gate_runtime is not None else {},
        "bar_returns": bar_returns,
    }
