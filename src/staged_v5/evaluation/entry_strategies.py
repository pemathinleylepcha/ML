from __future__ import annotations

from typing import Callable

from staged_v5.config import EntryConfig, ExitConfig
from staged_v5.evaluation.contracts import BarState, EntryContext, Order, RejectionCounters
from staged_v5.utils.critical_fixes import adaptive_entry_threshold


def _bump_counter(counters: RejectionCounters | None, field_name: str) -> None:
    if counters is None:
        return
    setattr(counters, field_name, getattr(counters, field_name) + 1)


def _compute_direction(bar: BarState, entry_cfg: EntryConfig, counters: RejectionCounters | None = None) -> int:
    threshold = adaptive_entry_threshold(
        entry_cfg.base_entry_threshold,
        float(bar.volatility),
        entry_cfg.threshold_volatility_coeff,
    )
    threshold = float(threshold)
    p = bar.prob_buy
    directional_confidence = max(p, 1.0 - p)
    if directional_confidence > entry_cfg.max_confidence_threshold:
        _bump_counter(counters, "direction_threshold_failed")
        return 0
    spread_ok = bool(abs(p - (1.0 - p)) >= entry_cfg.probability_spread_threshold)
    if spread_ok and p >= threshold:
        return 1
    if spread_ok and p <= 1.0 - threshold:
        return -1
    _bump_counter(counters, "direction_threshold_failed")
    return 0


def _check_entry_gate(
    bar: BarState,
    direction: int,
    entry_cfg: EntryConfig,
    counters: RejectionCounters | None = None,
) -> bool:
    if bar.prob_entry is None:
        return True
    gate_conf = bar.prob_entry if direction == 1 else (1.0 - bar.prob_entry)
    passed = gate_conf >= entry_cfg.entry_gate_threshold
    if not passed:
        _bump_counter(counters, "entry_head_failed")
    return passed


def _trade_confidence(prob_buy: float, direction: int, prob_entry: float | None) -> float:
    directional_conf = prob_buy if direction == 1 else (1.0 - prob_buy)
    if prob_entry is None:
        return float(directional_conf)
    entry_conf = prob_entry if direction == 1 else (1.0 - prob_entry)
    return float(min(directional_conf, entry_conf))


def _sanitize_entry_atr(raw_atr: float, reference_price: float, entry_cfg: EntryConfig) -> float:
    capped = entry_cfg.max_entry_atr_pct * max(abs(reference_price), 1e-8)
    return float(max(min(raw_atr, capped), 1e-8))


def _apply_entry_slippage(price: float, direction: int, atr: float, entry_cfg: EntryConfig) -> float:
    slip = entry_cfg.slippage_atr * atr
    if direction == 1:
        return float(price + slip)
    return float(price - slip)


def _compute_tp_sl(entry_price: float, direction: int, entry_atr: float, exit_cfg: ExitConfig) -> tuple[float, float]:
    atr_stop_dist = exit_cfg.stop_loss_atr * entry_atr
    pct_stop_dist = exit_cfg.max_loss_pct_per_trade * max(abs(entry_price), 1e-8)
    stop_dist = min(atr_stop_dist, pct_stop_dist)
    if direction == 1:
        tp = entry_price + exit_cfg.take_profit_atr * entry_atr
        sl = entry_price - stop_dist
    else:
        tp = entry_price - exit_cfg.take_profit_atr * entry_atr
        sl = entry_price + stop_dist
    return float(tp), float(sl)


def evaluate_limit_entry(
    bar: BarState,
    entry_cfg: EntryConfig,
    exit_cfg: ExitConfig,
    counters: RejectionCounters | None = None,
    *,
    context: EntryContext | None = None,
    entry_bar_high: float,
    entry_bar_low: float,
    entry_bar_close: float,
    entry_bar_atr: float,
    **_extras,
) -> Order | None:
    direction = _compute_direction(bar, entry_cfg, counters)
    if direction == 0:
        return None
    if not _check_entry_gate(bar, direction, entry_cfg, counters):
        return None
    entry_atr = _sanitize_entry_atr(entry_bar_atr, entry_bar_close, entry_cfg)
    limit_price = bar.close - direction * entry_cfg.limit_offset_atr * entry_atr
    if direction == 1:
        if entry_bar_low > limit_price:
            _bump_counter(counters, "limit_no_fill")
            return None
    else:
        if entry_bar_high < limit_price:
            _bump_counter(counters, "limit_no_fill")
            return None
    entry_price = _apply_entry_slippage(limit_price, direction, entry_atr, entry_cfg)
    tp, sl = _compute_tp_sl(entry_price, direction, entry_atr, exit_cfg)
    return Order(
        node_idx=bar.node_idx,
        pair_name=bar.pair_name,
        direction=direction,
        entry_price=entry_price,
        tp_price=tp,
        sl_price=sl,
        confidence=_trade_confidence(bar.prob_buy, direction, bar.prob_entry),
        entry_atr=entry_atr,
        signal_bar=bar.bar_index,
        entry_bar=bar.bar_index + entry_cfg.latency_bars,
    )


def evaluate_market_entry(
    bar: BarState,
    entry_cfg: EntryConfig,
    exit_cfg: ExitConfig,
    counters: RejectionCounters | None = None,
    *,
    context: EntryContext | None = None,
    entry_bar_high: float,
    entry_bar_low: float,
    entry_bar_close: float,
    entry_bar_atr: float,
    **_extras,
) -> Order | None:
    direction = _compute_direction(bar, entry_cfg, counters)
    if direction == 0:
        return None
    if not _check_entry_gate(bar, direction, entry_cfg, counters):
        return None
    entry_atr = _sanitize_entry_atr(entry_bar_atr, entry_bar_close, entry_cfg)
    entry_price = _apply_entry_slippage(entry_bar_close, direction, entry_atr, entry_cfg)
    tp, sl = _compute_tp_sl(entry_price, direction, entry_atr, exit_cfg)
    return Order(
        node_idx=bar.node_idx,
        pair_name=bar.pair_name,
        direction=direction,
        entry_price=entry_price,
        tp_price=tp,
        sl_price=sl,
        confidence=_trade_confidence(bar.prob_buy, direction, bar.prob_entry),
        entry_atr=entry_atr,
        signal_bar=bar.bar_index,
        entry_bar=bar.bar_index + entry_cfg.latency_bars,
    )


def evaluate_neural_gate_entry(
    bar: BarState,
    entry_cfg: EntryConfig,
    exit_cfg: ExitConfig,
    counters: RejectionCounters | None = None,
    *,
    context: EntryContext | None = None,
    entry_bar_high: float,
    entry_bar_low: float,
    entry_bar_close: float,
    entry_bar_atr: float,
    **_extras,
) -> Order | None:
    if context is None or context.neural_gate_runtime is None:
        raise ValueError("neural_gate entry requires EntryContext.neural_gate_runtime")
    if context.timestamp is None:
        raise ValueError("neural_gate entry requires the current anchor timestamp")
    if context.anchor_node_features is None or context.anchor_tpo_features is None:
        raise ValueError("neural_gate entry requires anchor node/tpo features")

    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction, direction_from_prob
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import build_gate_state_vector
    import torch

    runtime = context.neural_gate_runtime
    effective_prob_buy = float(bar.prob_buy)
    direction = direction_from_prob(effective_prob_buy)
    event = GateEvent(
        timestamp=context.timestamp,
        pair_name=bar.pair_name,
        direction=direction,
        prob_buy=effective_prob_buy,
        prob_entry=float(bar.prob_entry) if bar.prob_entry is not None else None,
        close=float(bar.close),
        atr=float(max(entry_bar_atr, 1e-8)),
        volatility=float(bar.volatility),
        session_code=int(bar.session_code),
        spread=float(context.anchor_node_features[4]),
        tick_count=float(context.anchor_node_features[5]),
        tpo_features=context.anchor_tpo_features,
    )
    state = build_gate_state_vector(
        prob_buy=event.prob_buy,
        prob_entry=event.prob_entry,
        atr=event.atr,
        volatility=event.volatility,
        spread=event.spread,
        tick_count=event.tick_count,
        session_code=event.session_code,
        tpo_features=event.tpo_features,
        pair_name=event.pair_name,
        anchor_timestamp=event.timestamp,
        tick_store=runtime.tick_store,
    )
    runtime.model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=runtime.device).unsqueeze(0)
        logits = runtime.model(state_tensor)
        action = NeuralGateAction(int(torch.argmax(logits, dim=-1).item()))
    runtime.action_counts[action.name.lower()] = runtime.action_counts.get(action.name.lower(), 0) + 1
    if action == NeuralGateAction.REJECT_WAIT:
        runtime.reject_count += 1
        _bump_counter(counters, "neural_gate_reject")
        return None

    result = simulate_action(
        event,
        action,
        runtime.tick_store,
        reference_price_mode=runtime.config.reference_price_mode,
        reject_wait_penalty=runtime.config.reject_wait_penalty,
        entry_slippage_atr=entry_cfg.slippage_atr,
        market_order_slippage_ticks=runtime.config.market_order_slippage_ticks,
        limit_at_poc_near_miss_reward=runtime.config.limit_at_poc_near_miss_reward,
        limit_at_poc_no_fill_penalty=runtime.config.limit_at_poc_no_fill_penalty,
        passive_limit_no_fill_penalty=runtime.config.passive_limit_no_fill_penalty,
        horizon_bars=max(runtime.config.mfe_horizon_bars, runtime.config.mae_horizon_bars),
    )
    runtime.reward_sums[result.action] = runtime.reward_sums.get(result.action, 0.0) + float(result.reward)
    if result.filled:
        runtime.fill_counts[result.action] = runtime.fill_counts.get(result.action, 0) + 1
    if result.no_fill:
        runtime.no_fill_counts[result.action] = runtime.no_fill_counts.get(result.action, 0) + 1
    if result.near_miss:
        runtime.near_miss_counts[result.action] = runtime.near_miss_counts.get(result.action, 0) + 1
    if not result.filled or result.fill_price is None:
        runtime.no_fill_count += 1
        _bump_counter(counters, "neural_gate_no_fill")
        _bump_counter(counters, "limit_no_fill")
        return None

    entry_atr = _sanitize_entry_atr(entry_bar_atr, entry_bar_close, entry_cfg)
    entry_price = float(result.fill_price)
    tp, sl = _compute_tp_sl(entry_price, direction, entry_atr, exit_cfg)
    return Order(
        node_idx=bar.node_idx,
        pair_name=bar.pair_name,
        direction=direction,
        entry_price=entry_price,
        tp_price=tp,
        sl_price=sl,
        confidence=_trade_confidence(bar.prob_buy, direction, bar.prob_entry),
        entry_atr=entry_atr,
        signal_bar=bar.bar_index,
        entry_bar=bar.bar_index + entry_cfg.latency_bars,
    )


ENTRY_REGISTRY: dict[str, Callable[..., Order | None]] = {
    "limit": evaluate_limit_entry,
    "market": evaluate_market_entry,
    "neural_gate": evaluate_neural_gate_entry,
}
