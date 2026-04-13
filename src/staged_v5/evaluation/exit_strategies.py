from __future__ import annotations

from typing import Callable

from staged_v5.config import ExitConfig
from staged_v5.evaluation.contracts import BarState, ExitDecision, OpenPosition


def _apply_stop_slippage(stop_price: float, direction: int, entry_atr: float, cfg: ExitConfig) -> float:
    slip = cfg.slippage_atr * entry_atr
    if direction == 1:
        return float(stop_price - slip)
    return float(stop_price + slip)


def _is_breakeven_stop(position: OpenPosition) -> bool:
    return abs(position.sl_price - position.entry_price) <= max(1e-8, abs(position.entry_price) * 1e-8)


def _check_tp_sl(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> ExitDecision | None:
    if position.direction == 1:
        hit_tp = bar.high >= position.tp_price
        hit_sl = bar.low <= position.sl_price
    else:
        hit_tp = bar.low <= position.tp_price
        hit_sl = bar.high >= position.sl_price

    if hit_tp and hit_sl:
        favorable = bar.close >= position.entry_price if position.direction == 1 else bar.close <= position.entry_price
        if favorable:
            return ExitDecision(exit_price=position.tp_price, reason="tp_sl_same_bar_tp")
        exit_price = _apply_stop_slippage(position.sl_price, position.direction, position.entry_atr, cfg)
        reason = "trailing_breakeven" if _is_breakeven_stop(position) else "tp_sl_same_bar_sl"
        return ExitDecision(exit_price=exit_price, reason=reason)
    if hit_tp:
        return ExitDecision(exit_price=position.tp_price, reason="take_profit")
    if hit_sl:
        exit_price = _apply_stop_slippage(position.sl_price, position.direction, position.entry_atr, cfg)
        reason = "trailing_breakeven" if _is_breakeven_stop(position) else "stop_loss"
        return ExitDecision(exit_price=exit_price, reason=reason)
    return None


def _apply_trailing(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> None:
    if cfg.trailing_activate_atr <= 0.0:
        return
    if position.direction == 1:
        unrealized_atr = (bar.high - position.entry_price) / max(position.entry_atr, 1e-8)
        if unrealized_atr >= cfg.trailing_activate_atr:
            position.sl_price = max(position.sl_price, position.entry_price)
    else:
        unrealized_atr = (position.entry_price - bar.low) / max(position.entry_atr, 1e-8)
        if unrealized_atr >= cfg.trailing_activate_atr:
            position.sl_price = min(position.sl_price, position.entry_price)


def _check_signal_exit(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> bool:
    if position.direction == 1:
        return bar.prob_buy < cfg.exit_threshold
    return bar.prob_buy > (1.0 - cfg.exit_threshold)


def evaluate_trailing_atr_exit(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> ExitDecision | None:
    tp_sl = _check_tp_sl(position, bar, cfg)
    if tp_sl is not None:
        return tp_sl
    _apply_trailing(position, bar, cfg)
    if (bar.bar_index - position.entry_bar) >= cfg.max_hold_bars:
        return ExitDecision(exit_price=bar.close, reason="horizon_exit")
    if _check_signal_exit(position, bar, cfg):
        return ExitDecision(exit_price=bar.close, reason="signal_exit")
    return None


def evaluate_time_only_exit(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> ExitDecision | None:
    tp_sl = _check_tp_sl(position, bar, cfg)
    if tp_sl is not None:
        return tp_sl
    if (bar.bar_index - position.entry_bar) >= cfg.max_hold_bars:
        return ExitDecision(exit_price=bar.close, reason="horizon_exit")
    return None


def evaluate_signal_only_exit(position: OpenPosition, bar: BarState, cfg: ExitConfig) -> ExitDecision | None:
    tp_sl = _check_tp_sl(position, bar, cfg)
    if tp_sl is not None:
        return tp_sl
    if _check_signal_exit(position, bar, cfg):
        return ExitDecision(exit_price=bar.close, reason="signal_exit")
    if (bar.bar_index - position.entry_bar) >= cfg.max_hold_bars:
        return ExitDecision(exit_price=bar.close, reason="horizon_exit")
    return None


def evaluate_weekend_close(
    position: OpenPosition,
    bar: BarState,
    cfg: ExitConfig,
    *,
    is_weekend_bar: bool,
) -> ExitDecision | None:
    if not cfg.close_before_weekend or not is_weekend_bar:
        return None
    return ExitDecision(exit_price=bar.close, reason="weekend_close")


EXIT_REGISTRY: dict[str, Callable[..., ExitDecision | None]] = {
    "trailing_atr": evaluate_trailing_atr_exit,
    "time_only": evaluate_time_only_exit,
    "signal_only": evaluate_signal_only_exit,
}
