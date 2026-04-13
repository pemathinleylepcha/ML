from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def test_entry_config_defaults():
    from staged_v5.config import EntryConfig

    cfg = EntryConfig()
    assert cfg.entry_type == "limit"
    assert cfg.base_entry_threshold == 0.65
    assert cfg.latency_bars == 1
    assert cfg.slippage_atr == 0.01
    print("PASS test_entry_config_defaults")


def test_exit_config_defaults():
    from staged_v5.config import ExitConfig

    cfg = ExitConfig()
    assert cfg.exit_type == "trailing_atr"
    assert cfg.stop_loss_atr == 0.70
    assert cfg.max_hold_bars == 6
    assert cfg.close_before_weekend is True
    assert cfg.enable_tp_shift is False
    print("PASS test_exit_config_defaults")


def test_position_config_defaults():
    from staged_v5.config import PositionConfig

    cfg = PositionConfig()
    assert cfg.max_positions == 6
    assert cfg.cooldown_bars == 3
    assert cfg.max_group_exposure == 2
    print("PASS test_position_config_defaults")


def test_backtest_config_nested_defaults():
    from staged_v5.config import BacktestConfig, EntryConfig, ExitConfig, PositionConfig

    cfg = BacktestConfig()
    assert isinstance(cfg.entry, EntryConfig)
    assert isinstance(cfg.exit, ExitConfig)
    assert isinstance(cfg.position, PositionConfig)
    assert cfg.entry.base_entry_threshold == 0.65
    assert cfg.exit.stop_loss_atr == 0.70
    assert cfg.position.max_positions == 6
    print("PASS test_backtest_config_nested_defaults")


def test_backtest_config_from_flat():
    from staged_v5.config import BacktestConfig

    flat = {
        "base_entry_threshold": 0.78,
        "threshold_volatility_coeff": 12.0,
        "exit_threshold": 0.52,
        "probability_spread_threshold": 0.15,
        "latency_bars": 1,
        "cooldown_bars": 3,
        "max_positions": 6,
        "max_hold_bars": 21,
        "entry_gate_threshold": 0.55,
        "max_confidence_threshold": 1.01,
        "max_group_exposure": 2,
        "take_profit_atr": 1.0,
        "stop_loss_atr": 0.7,
        "max_loss_pct_per_trade": 0.01,
        "max_entry_atr_pct": 0.01,
        "slippage_atr": 0.02,
        "ece_gate_threshold": 0.0,
        "trailing_activate_atr": 0.5,
        "use_limit_entries": True,
        "limit_offset_atr": 0.1,
    }
    cfg = BacktestConfig.from_flat(flat)
    assert cfg.entry.base_entry_threshold == 0.78
    assert cfg.entry.entry_type == "limit"
    assert cfg.entry.slippage_atr == 0.02
    assert cfg.exit.max_hold_bars == 21
    assert cfg.exit.stop_loss_atr == 0.7
    assert cfg.exit.slippage_atr == 0.02
    assert cfg.exit.max_loss_pct_per_trade == 0.01
    assert cfg.position.max_positions == 6
    assert cfg.position.cooldown_bars == 3
    assert cfg.ece_gate_threshold == 0.0
    print("PASS test_backtest_config_from_flat")


def test_backtest_config_from_flat_market_entry():
    from staged_v5.config import BacktestConfig

    flat = {"use_limit_entries": False}
    cfg = BacktestConfig.from_flat(flat)
    assert cfg.entry.entry_type == "market"
    print("PASS test_backtest_config_from_flat_market_entry")


def test_backtest_config_to_flat():
    from staged_v5.config import BacktestConfig

    cfg = BacktestConfig()
    flat = cfg.to_flat()
    assert flat["base_entry_threshold"] == 0.65
    assert flat["stop_loss_atr"] == 0.70
    assert flat["max_positions"] == 6
    assert flat["use_limit_entries"] is True
    assert flat["entry_type"] == "limit"
    print("PASS test_backtest_config_to_flat")


def test_bar_state_creation():
    from staged_v5.evaluation.contracts import BarState

    bar = BarState(
        bar_index=10,
        node_idx=3,
        prob_buy=0.82,
        prob_entry=0.75,
        high=100.5,
        low=99.5,
        close=100.0,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    assert bar.bar_index == 10
    assert bar.pair_name == "EURUSD"
    print("PASS test_bar_state_creation")


def test_rejection_counters_creation():
    from staged_v5.evaluation.contracts import RejectionCounters

    counters = RejectionCounters()
    assert counters.bars_seen == 0
    assert counters.total_evaluated == 0
    assert counters.direction_threshold_failed == 0
    assert counters.entry_head_failed == 0
    assert counters.limit_no_fill == 0
    print("PASS test_rejection_counters_creation")


def test_order_creation():
    from staged_v5.evaluation.contracts import Order

    order = Order(
        node_idx=3,
        pair_name="EURUSD",
        direction=1,
        entry_price=100.01,
        tp_price=100.51,
        sl_price=99.65,
        confidence=0.82,
        entry_atr=0.5,
        signal_bar=9,
        entry_bar=10,
    )
    assert order.direction == 1
    assert order.tp_price == 100.51
    print("PASS test_order_creation")


def test_exit_decision_creation():
    from staged_v5.evaluation.contracts import ExitDecision

    decision = ExitDecision(exit_price=100.5, reason="take_profit")
    assert decision.reason == "take_profit"
    print("PASS test_exit_decision_creation")


def test_open_position_tp_extensions_default():
    from staged_v5.evaluation.contracts import OpenPosition

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=100.5,
        sl_price=99.5,
        confidence=0.8,
        entry_atr=0.5,
    )
    assert pos.tp_extensions == 0
    pos.sl_price = 100.0
    assert pos.sl_price == 100.0
    print("PASS test_open_position_tp_extensions_default")


def test_exit_trailing_atr_take_profit():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_trailing_atr_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=100.5,
        sl_price=99.65,
        confidence=0.8,
        entry_atr=0.5,
    )
    bar = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.6,
        low=99.9,
        close=100.4,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70)
    decision = evaluate_trailing_atr_exit(pos, bar, cfg)
    assert decision is not None
    assert decision.reason == "take_profit"
    assert decision.exit_price == 100.5
    print("PASS test_exit_trailing_atr_take_profit")


def test_exit_trailing_atr_stop_loss():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_trailing_atr_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=100.5,
        sl_price=99.65,
        confidence=0.8,
        entry_atr=0.5,
    )
    bar = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.1,
        low=99.6,
        close=99.7,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    cfg = ExitConfig(slippage_atr=0.0)
    decision = evaluate_trailing_atr_exit(pos, bar, cfg)
    assert decision is not None
    assert decision.reason == "stop_loss"
    print("PASS test_exit_trailing_atr_stop_loss")


def test_exit_trailing_atr_breakeven():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_trailing_atr_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(trailing_activate_atr=0.5, slippage_atr=0.0)
    bar1 = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.6,
        low=100.0,
        close=100.3,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_trailing_atr_exit(pos, bar1, cfg)
    assert decision is None
    assert pos.sl_price == 100.0
    print("PASS test_exit_trailing_atr_breakeven")


def test_exit_trailing_atr_horizon():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_trailing_atr_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(max_hold_bars=3)
    bar = BarState(
        bar_index=9,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.2,
        low=99.8,
        close=100.1,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_trailing_atr_exit(pos, bar, cfg)
    assert decision is not None
    assert decision.reason == "horizon_exit"
    assert decision.exit_price == 100.1
    print("PASS test_exit_trailing_atr_horizon")


def test_exit_trailing_atr_signal_exit():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_trailing_atr_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(exit_threshold=0.52)
    bar = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.48,
        prob_entry=0.75,
        high=100.2,
        low=99.8,
        close=100.0,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_trailing_atr_exit(pos, bar, cfg)
    assert decision is not None
    assert decision.reason == "signal_exit"
    print("PASS test_exit_trailing_atr_signal_exit")


def test_exit_time_only_no_trailing():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_time_only_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(max_hold_bars=2, slippage_atr=0.0)
    bar = BarState(
        bar_index=8,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.6,
        low=99.8,
        close=100.3,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_time_only_exit(pos, bar, cfg)
    assert decision is not None
    assert decision.reason == "horizon_exit"
    assert pos.sl_price == 99.3
    print("PASS test_exit_time_only_no_trailing")


def test_exit_signal_only_holds_when_signal_agrees():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_signal_only_exit

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(exit_threshold=0.52, max_hold_bars=100)
    bar = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.75,
        prob_entry=0.75,
        high=100.2,
        low=99.8,
        close=100.1,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_signal_only_exit(pos, bar, cfg)
    assert decision is None
    print("PASS test_exit_signal_only_holds_when_signal_agrees")


def test_weekend_close_guard():
    from staged_v5.config import ExitConfig
    from staged_v5.evaluation.contracts import BarState, OpenPosition
    from staged_v5.evaluation.exit_strategies import evaluate_weekend_close

    pos = OpenPosition(
        node_idx=0,
        pair_name="EURUSD",
        direction=1,
        signal_bar=5,
        entry_bar=6,
        entry_price=100.0,
        tp_price=101.0,
        sl_price=99.3,
        confidence=0.8,
        entry_atr=1.0,
    )
    cfg = ExitConfig(close_before_weekend=True)
    bar = BarState(
        bar_index=7,
        node_idx=0,
        prob_buy=0.80,
        prob_entry=0.75,
        high=100.2,
        low=99.8,
        close=100.1,
        atr=1.0,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    decision = evaluate_weekend_close(pos, bar, cfg, is_weekend_bar=True)
    assert decision is not None
    assert decision.reason == "weekend_close"
    assert decision.exit_price == 100.1
    decision2 = evaluate_weekend_close(pos, bar, cfg, is_weekend_bar=False)
    assert decision2 is None
    cfg_off = ExitConfig(close_before_weekend=False)
    decision3 = evaluate_weekend_close(pos, bar, cfg_off, is_weekend_bar=True)
    assert decision3 is None
    print("PASS test_weekend_close_guard")


def test_entry_limit_buy():
    from staged_v5.config import EntryConfig, ExitConfig
    from staged_v5.evaluation.contracts import BarState
    from staged_v5.evaluation.entry_strategies import evaluate_limit_entry

    bar = BarState(
        bar_index=1,
        node_idx=0,
        prob_buy=0.82,
        prob_entry=0.75,
        high=100.5,
        low=99.5,
        close=100.0,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    entry_cfg = EntryConfig(
        base_entry_threshold=0.60,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        slippage_atr=0.0,
        limit_offset_atr=0.10,
        max_entry_atr_pct=0.01,
    )
    exit_cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70, max_loss_pct_per_trade=1.0)
    order = evaluate_limit_entry(
        bar,
        entry_cfg,
        exit_cfg,
        entry_bar_high=100.5,
        entry_bar_low=99.5,
        entry_bar_close=100.0,
        entry_bar_atr=0.5,
    )
    assert order is not None
    assert order.direction == 1
    assert order.pair_name == "EURUSD"
    print("PASS test_entry_limit_buy")


def test_entry_limit_no_fill():
    from staged_v5.config import EntryConfig, ExitConfig
    from staged_v5.evaluation.contracts import BarState, RejectionCounters
    from staged_v5.evaluation.entry_strategies import evaluate_limit_entry

    bar = BarState(
        bar_index=1,
        node_idx=0,
        prob_buy=0.82,
        prob_entry=0.75,
        high=100.2,
        low=100.1,
        close=100.15,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    counters = RejectionCounters()
    entry_cfg = EntryConfig(
        base_entry_threshold=0.60,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        slippage_atr=0.0,
        limit_offset_atr=0.10,
    )
    exit_cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70, max_loss_pct_per_trade=1.0)
    order = evaluate_limit_entry(
        bar,
        entry_cfg,
        exit_cfg,
        counters=counters,
        entry_bar_high=100.2,
        entry_bar_low=100.11,
        entry_bar_close=100.15,
        entry_bar_atr=0.5,
    )
    assert order is None
    assert counters.limit_no_fill == 1
    print("PASS test_entry_limit_no_fill")


def test_entry_market_buy():
    from staged_v5.config import EntryConfig, ExitConfig
    from staged_v5.evaluation.contracts import BarState
    from staged_v5.evaluation.entry_strategies import evaluate_market_entry

    bar = BarState(
        bar_index=1,
        node_idx=0,
        prob_buy=0.82,
        prob_entry=0.75,
        high=100.5,
        low=99.5,
        close=100.0,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    entry_cfg = EntryConfig(
        entry_type="market",
        base_entry_threshold=0.60,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        slippage_atr=0.0,
    )
    exit_cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70, max_loss_pct_per_trade=1.0)
    order = evaluate_market_entry(
        bar,
        entry_cfg,
        exit_cfg,
        entry_bar_high=100.5,
        entry_bar_low=99.5,
        entry_bar_close=100.0,
        entry_bar_atr=0.5,
    )
    assert order is not None
    assert order.direction == 1
    assert order.entry_price == 100.0
    print("PASS test_entry_market_buy")


def test_entry_below_threshold_rejected():
    from staged_v5.config import EntryConfig, ExitConfig
    from staged_v5.evaluation.contracts import BarState, RejectionCounters
    from staged_v5.evaluation.entry_strategies import evaluate_limit_entry

    bar = BarState(
        bar_index=1,
        node_idx=0,
        prob_buy=0.55,
        prob_entry=0.75,
        high=100.5,
        low=99.5,
        close=100.0,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    counters = RejectionCounters()
    entry_cfg = EntryConfig(
        base_entry_threshold=0.70,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        slippage_atr=0.0,
    )
    exit_cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70, max_loss_pct_per_trade=1.0)
    order = evaluate_limit_entry(
        bar,
        entry_cfg,
        exit_cfg,
        counters=counters,
        entry_bar_high=100.5,
        entry_bar_low=99.5,
        entry_bar_close=100.0,
        entry_bar_atr=0.5,
    )
    assert order is None
    assert counters.direction_threshold_failed == 1
    print("PASS test_entry_below_threshold_rejected")


def test_entry_gate_blocks_low_entry_prob():
    from staged_v5.config import EntryConfig, ExitConfig
    from staged_v5.evaluation.contracts import BarState, RejectionCounters
    from staged_v5.evaluation.entry_strategies import evaluate_limit_entry

    bar = BarState(
        bar_index=1,
        node_idx=0,
        prob_buy=0.82,
        prob_entry=0.20,
        high=100.5,
        low=99.5,
        close=100.0,
        atr=0.5,
        volatility=0.001,
        session_code=1,
        pair_name="EURUSD",
    )
    counters = RejectionCounters()
    entry_cfg = EntryConfig(
        base_entry_threshold=0.60,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        slippage_atr=0.0,
    )
    exit_cfg = ExitConfig(take_profit_atr=1.0, stop_loss_atr=0.70, max_loss_pct_per_trade=1.0)
    order = evaluate_limit_entry(
        bar,
        entry_cfg,
        exit_cfg,
        counters=counters,
        entry_bar_high=100.5,
        entry_bar_low=99.5,
        entry_bar_close=100.0,
        entry_bar_atr=0.5,
    )
    assert order is None
    assert counters.entry_head_failed == 1
    print("PASS test_entry_gate_blocks_low_entry_prob")


def test_backtest_v5_default_matches_v4_signature():
    from staged_v5.config import BacktestConfig
    from staged_v5.evaluation.backtest import backtest_probabilities

    prob_buy = np.array(
        [[0.50, 0.50], [0.82, 0.80], [0.81, 0.79], [0.50, 0.50]],
        dtype=np.float32,
    )
    prob_entry = np.full_like(prob_buy, 0.80)
    close = np.array(
        [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
        dtype=np.float32,
    )
    high = np.array(
        [[100.0, 100.0], [100.0, 100.0], [101.2, 101.0], [100.5, 100.5]],
        dtype=np.float32,
    )
    low = np.array(
        [[100.0, 100.0], [100.0, 100.0], [99.9, 99.9], [99.8, 99.8]],
        dtype=np.float32,
    )
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig.from_flat(
        {
            "base_entry_threshold": 0.57,
            "threshold_volatility_coeff": 0.0,
            "probability_spread_threshold": 0.05,
            "latency_bars": 1,
            "max_positions": 2,
            "entry_gate_threshold": 0.50,
            "max_confidence_threshold": 0.95,
            "take_profit_atr": 1.0,
            "stop_loss_atr": 0.70,
            "max_loss_pct_per_trade": 1.0,
            "slippage_atr": 0.0,
            "trailing_activate_atr": 0.0,
            "use_limit_entries": True,
            "limit_offset_atr": 0.0,
            "max_hold_bars": 6,
            "cooldown_bars": 0,
            "max_group_exposure": 10,
        }
    )
    result = backtest_probabilities(
        prob_buy,
        prob_entry,
        close,
        high,
        low,
        volatility,
        session_codes,
        ("EURUSD", "GBPUSD"),
        cfg,
    )
    expected_keys = {
        "trade_count",
        "win_rate",
        "strategy_sharpe",
        "net_return",
        "confidence_hit_rate",
        "blocked_by_exposure",
        "exit_reason_counts",
        "max_open_positions",
        "avg_hold_bars",
        "entry_rejection_counters",
        "threshold_diagnostics",
        "gate_action_counts",
        "gate_reject_count",
        "gate_no_fill_count",
        "gate_fill_counts",
        "gate_no_fill_counts",
        "gate_near_miss_counts",
        "gate_mean_reward_by_action",
        "gate_fill_rate_by_action",
        "gate_no_fill_rate_by_action",
        "gate_near_miss_rate_by_action",
        "bar_returns",
    }
    assert expected_keys == set(result.keys())
    assert result["trade_count"] >= 0
    print("PASS test_backtest_v5_default_matches_v4_signature")


def test_backtest_v5_trailing_produces_trades():
    from staged_v5.config import BacktestConfig
    from staged_v5.evaluation.backtest import backtest_probabilities

    prob_buy = np.array([[0.50], [0.82], [0.81], [0.51]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.1]], dtype=np.float32)
    high = np.array([[100.5], [100.5], [100.5], [100.6]], dtype=np.float32)
    low = np.array([[99.5], [99.5], [99.5], [99.9]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig.from_flat(
        {
            "base_entry_threshold": 0.57,
            "threshold_volatility_coeff": 0.0,
            "probability_spread_threshold": 0.05,
            "latency_bars": 1,
            "max_positions": 1,
            "entry_gate_threshold": 0.50,
            "max_confidence_threshold": 0.95,
            "take_profit_atr": 1.0,
            "stop_loss_atr": 0.70,
            "max_loss_pct_per_trade": 1.0,
            "slippage_atr": 0.0,
            "trailing_activate_atr": 0.5,
            "use_limit_entries": False,
            "max_hold_bars": 6,
            "cooldown_bars": 0,
            "max_group_exposure": 10,
        }
    )
    result = backtest_probabilities(
        prob_buy,
        prob_entry,
        close,
        high,
        low,
        volatility,
        session_codes,
        ("EURUSD",),
        cfg,
    )
    assert result["trade_count"] >= 1
    print("PASS test_backtest_v5_trailing_produces_trades")


def test_continuous_ga_finds_better_score():
    from staged_v5.utils.ga_search import run_continuous_ga

    def scorer(genome):
        return sum(genome)

    result = run_continuous_ga(
        n_genes=4,
        scorer=scorer,
        population_size=10,
        generations=5,
        mutation_rate=0.2,
        crossover_rate=0.5,
    )
    assert result.best_score > 2.0
    assert len(result.best_genome) == 4
    assert all(0.0 <= gene <= 1.0 for gene in result.best_genome)
    print("PASS test_continuous_ga_finds_better_score")


def test_decode_ga_genome_round_trip():
    from staged_v5.config import BacktestConfig, GA_PARAM_SPACE, decode_ga_genome

    base = BacktestConfig()
    genome = [0.5] * len(GA_PARAM_SPACE)
    cfg = decode_ga_genome(genome, base)
    assert abs(cfg.entry.base_entry_threshold - 0.725) < 0.01
    assert cfg.exit.max_hold_bars in (16, 17)
    print("PASS test_decode_ga_genome_round_trip")


def test_v5_backtest_from_v4_flat_config_produces_results():
    from staged_v5.config import BacktestConfig
    from staged_v5.evaluation.backtest import backtest_probabilities

    v4_flat = {
        "base_entry_threshold": 0.65,
        "threshold_volatility_coeff": 12.0,
        "exit_threshold": 0.52,
        "probability_spread_threshold": 0.15,
        "latency_bars": 1,
        "cooldown_bars": 3,
        "max_positions": 6,
        "max_hold_bars": 6,
        "entry_gate_threshold": 0.55,
        "max_confidence_threshold": 1.01,
        "max_group_exposure": 2,
        "take_profit_atr": 1.0,
        "stop_loss_atr": 0.7,
        "max_loss_pct_per_trade": 0.005,
        "max_entry_atr_pct": 0.01,
        "slippage_atr": 0.01,
        "ece_gate_threshold": 0.0,
        "trailing_activate_atr": 0.5,
        "use_limit_entries": True,
        "limit_offset_atr": 0.1,
    }
    cfg = BacktestConfig.from_flat(v4_flat)
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.4], [100.2]], dtype=np.float32)
    high = np.array([[100.0], [100.0], [100.5], [100.3]], dtype=np.float32)
    low = np.array([[100.0], [100.0], [99.9], [100.0]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    result = backtest_probabilities(
        prob_buy,
        prob_entry,
        close,
        high,
        low,
        volatility,
        session_codes,
        ("EURUSD",),
        cfg,
    )
    assert "trade_count" in result
    assert "strategy_sharpe" in result
    assert "exit_reason_counts" in result
    assert "entry_rejection_counters" in result
    print("PASS test_v5_backtest_from_v4_flat_config_produces_results")


def test_backtest_tracks_entry_rejection_waterfall():
    from staged_v5.config import BacktestConfig
    from staged_v5.evaluation.backtest import backtest_probabilities

    prob_buy = np.array([[0.50], [0.60], [0.60], [0.60]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.80]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float32)
    high = np.array([[100.0], [100.0], [100.2], [100.2]], dtype=np.float32)
    low = np.array([[100.0], [100.0], [99.8], [99.8]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.0)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig.from_flat(
        {
            "base_entry_threshold": 0.80,
            "threshold_volatility_coeff": 0.0,
            "probability_spread_threshold": 0.05,
            "latency_bars": 1,
            "max_positions": 1,
            "entry_gate_threshold": 0.50,
            "max_confidence_threshold": 0.95,
            "take_profit_atr": 1.0,
            "stop_loss_atr": 0.70,
            "max_loss_pct_per_trade": 1.0,
            "slippage_atr": 0.0,
            "trailing_activate_atr": 0.5,
            "use_limit_entries": True,
            "limit_offset_atr": 0.0,
            "max_hold_bars": 6,
            "cooldown_bars": 0,
            "max_group_exposure": 10,
        }
    )
    result = backtest_probabilities(
        prob_buy,
        prob_entry,
        close,
        high,
        low,
        volatility,
        session_codes,
        ("EURUSD",),
        cfg,
    )
    counters = result["entry_rejection_counters"]
    assert result["trade_count"] == 0
    assert counters["bars_seen"] > 0
    assert counters["total_evaluated"] > 0
    assert counters["direction_threshold_failed"] > 0
    print("PASS test_backtest_tracks_entry_rejection_waterfall")


def test_adjusted_backtest_config_v5():
    from staged_v5.config import BacktestConfig, PositionConfig
    from staged_v5.evaluation.backtest import adjusted_backtest_config

    cfg = BacktestConfig(position=PositionConfig(max_positions=6), ece_gate_threshold=0.09)
    adjusted = adjusted_backtest_config(cfg, 0.105)
    assert adjusted.position.max_positions == 3
    unchanged = adjusted_backtest_config(cfg, 0.08)
    assert unchanged.position.max_positions == 6
    print("PASS test_adjusted_backtest_config_v5")


def test_neural_gate_config_round_trip():
    from staged_v5.config import BacktestConfig, EntryConfig, NeuralGateConfig

    cfg = BacktestConfig(
        entry=EntryConfig(entry_type="neural_gate"),
        neural_gate=NeuralGateConfig(
            enabled=True,
            action_temperature=0.9,
            reference_price_mode="poc",
            positive_fill_reward_boost=3.0,
            market_order_slippage_ticks=2.5,
            reject_wait_penalty=-6e-5,
            limit_at_poc_near_miss_reward=6e-5,
            limit_at_poc_no_fill_penalty=-3e-5,
            passive_limit_no_fill_penalty=-9e-5,
        ),
    )
    flat = cfg.to_flat()
    restored = BacktestConfig.from_flat(flat)
    assert restored.entry.entry_type == "neural_gate"
    assert restored.neural_gate.enabled is True
    assert restored.neural_gate.action_temperature == 0.9
    assert restored.neural_gate.reference_price_mode == "poc"
    assert restored.neural_gate.positive_fill_reward_boost == 3.0
    assert restored.neural_gate.market_order_slippage_ticks == 2.5
    assert restored.neural_gate.reject_wait_penalty == -6e-5
    assert restored.neural_gate.limit_at_poc_near_miss_reward == 6e-5
    assert restored.neural_gate.limit_at_poc_no_fill_penalty == -3e-5
    assert restored.neural_gate.passive_limit_no_fill_penalty == -9e-5
    assert restored.neural_gate.mfe_horizon_bars == 120
    assert restored.neural_gate.mae_horizon_bars == 120
    print("PASS test_neural_gate_config_round_trip")


def test_build_gate_events_preserves_oracle_direction():
    from staged_v5.execution_gate.dataset import build_gate_events

    events = build_gate_events(
        timestamps=np.array(["2026-01-01T01:00:00"], dtype="datetime64[s]"),
        prob_buy=np.array([[0.8]], dtype=np.float32),
        prob_entry=np.array([[0.7]], dtype=np.float32),
        close=np.array([[1.1000]], dtype=np.float32),
        atr=np.array([[0.0010]], dtype=np.float32),
        volatility=np.array([[0.01]], dtype=np.float32),
        session_codes=np.array([1], dtype=np.int64),
        pair_names=("EURUSD",),
        anchor_node_features=np.array([[[0.0, 0.0, 0.0, 0.0, 1.0, 10.0]]], dtype=np.float32),
        anchor_tpo_features=np.zeros((1, 1, 8), dtype=np.float32),
        valid_mask=np.array([[True]]),
    )
    assert len(events) == 1
    assert events[0].direction == 1
    assert abs(events[0].prob_buy - 0.8) < 1e-6
    print("PASS test_build_gate_events_preserves_oracle_direction")


def test_tick_proxy_feature_summary_no_future_leak():
    from staged_v5.execution_gate.features import GATE_STATE_FEATURE_NAMES, TickProxyStore, build_gate_state_vector

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 00:59:58", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 3.0, "bid_ask_imbalance": 0.1, "price_velocity": 4.0},
                {"dt": "2026-01-01 00:59:59", "o": 1.1000, "h": 1.1002, "l": 1.0998, "c": 1.1001, "sp": 1.2, "tk": 11, "tick_velocity": 3.0, "spread_z": 7.0, "bid_ask_imbalance": 0.2, "price_velocity": 11.0},
                {"dt": "2026-01-01 01:00:00", "o": 1.1001, "h": 1.1003, "l": 1.1000, "c": 1.1002, "sp": 1.4, "tk": 12, "tick_velocity": 99.0, "spread_z": 99.0, "bid_ask_imbalance": 0.9, "price_velocity": 101.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        state = build_gate_state_vector(
            prob_buy=0.8,
            prob_entry=0.6,
            atr=0.001,
            volatility=0.01,
            spread=1.0,
            tick_count=100.0,
            session_code=1,
            tpo_features=np.zeros(8, dtype=np.float32),
            pair_name="EURUSD",
            anchor_timestamp=np.datetime64("2026-01-01T01:00:00"),
            tick_store=store,
        )
        spread_idx = GATE_STATE_FEATURE_NAMES.index("spread_z_last")
        velocity_idx = GATE_STATE_FEATURE_NAMES.index("price_velocity_last")
        assert state[spread_idx] == 7.0
        assert state[velocity_idx] == 11.0
        print("PASS test_tick_proxy_feature_summary_no_future_leak")


def test_trade_through_limit_fill_requires_through_not_touch():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 00:59:59", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1002, "l": 1.0998, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:01", "o": 1.1000, "h": 1.1002, "l": 1.0997, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        touch_only = simulate_action(event, NeuralGateAction.LIMIT_2TICK, store, horizon_bars=1)
        assert touch_only.filled is False
        assert touch_only.near_miss is True
        assert abs(touch_only.reward - (-2.5e-5)) < 1e-9
        through_fill = simulate_action(event, NeuralGateAction.LIMIT_2TICK, store, horizon_bars=2)
        assert through_fill.filled is True
        print("PASS test_trade_through_limit_fill_requires_through_not_touch")


def test_passive_limit_no_fill_uses_penalty_even_on_near_miss():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 00:59:59", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1002, "l": 1.1000, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        result = simulate_action(event, NeuralGateAction.PASSIVE_LIMIT_1TICK, store, horizon_bars=1)
        assert result.filled is False
        assert result.near_miss is True
        assert abs(result.reward - (-2.5e-5)) < 1e-9
        print("PASS test_passive_limit_no_fill_uses_penalty_even_on_near_miss")


def test_intermediate_limit_actions_use_expected_fill_prices():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 00:59:59", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1002, "l": 1.0993, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        limit_2tick = simulate_action(event, NeuralGateAction.LIMIT_2TICK, store, horizon_bars=1)
        limit_3tick = simulate_action(event, NeuralGateAction.LIMIT_3TICK, store, horizon_bars=1)
        limit_half_atr = simulate_action(event, NeuralGateAction.LIMIT_HALF_ATR, store, horizon_bars=1)
        assert limit_2tick.filled is True
        assert abs(limit_2tick.fill_price - 1.0998) < 1e-9
        assert limit_3tick.filled is True
        assert abs(limit_3tick.fill_price - 1.0997) < 1e-9
        assert limit_half_atr.filled is True
        assert abs(limit_half_atr.fill_price - 1.0995) < 1e-9
        print("PASS test_intermediate_limit_actions_use_expected_fill_prices")


def test_reject_wait_uses_opportunity_cost_penalty():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        pd.DataFrame(
            [
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        ).to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        result = simulate_action(event, NeuralGateAction.REJECT_WAIT, store)
        assert result.filled is False
        assert abs(result.reward - (-5e-5)) < 1e-9
        print("PASS test_reject_wait_uses_opportunity_cost_penalty")


def test_simulate_action_accepts_singleton_list_action():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 00:59:59", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1002, "l": 1.0998, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        result = simulate_action(event, [NeuralGateAction.MARKET_NOW], store, horizon_bars=1)
        assert result.filled is True
        assert result.action == "market_now"
        print("PASS test_simulate_action_accepts_singleton_list_action")


def test_market_now_uses_tick_floor_for_slippage():
    from staged_v5.execution_gate.contracts import GateEvent, NeuralGateAction
    from staged_v5.execution_gate.environment import simulate_action
    from staged_v5.execution_gate.features import TickProxyStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        frame = pd.DataFrame(
            [
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
                {"dt": "2026-01-01 01:00:01", "o": 1.1000, "h": 1.1002, "l": 1.0998, "c": 1.1001, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 1.0},
            ]
        )
        frame.to_csv(path, index=False)
        store = TickProxyStore(tmp_dir)
        event = GateEvent(
            timestamp=np.datetime64("2026-01-01T01:00:00"),
            pair_name="EURUSD",
            direction=1,
            prob_buy=0.8,
            prob_entry=0.7,
            close=1.1000,
            atr=0.0010,
            volatility=0.01,
            session_code=1,
            spread=1.0,
            tick_count=10.0,
            tpo_features=np.zeros(8, dtype=np.float32),
        )
        result = simulate_action(
            event,
            NeuralGateAction.MARKET_NOW,
            store,
            entry_slippage_atr=0.01,
            market_order_slippage_ticks=1.0,
            horizon_bars=1,
        )
        assert result.filled is True
        assert abs(result.fill_price - 1.1001) < 1e-9
        print("PASS test_market_now_uses_tick_floor_for_slippage")


def test_grpo_zero_variance_skip():
    from staged_v5.execution_gate.grpo import grpo_update_step
    from staged_v5.execution_gate.model import MicrostructureGate

    model = MicrostructureGate(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    states = torch.zeros((4, 4), dtype=torch.float32)
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with torch.no_grad():
        logits = model(states)
        old_log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
    result = grpo_update_step(
        model,
        optimizer,
        states,
        actions,
        torch.zeros(4, dtype=torch.float32),
        old_log_probs,
    )
    assert result["skipped"] is True
    assert result["zero_variance"] is True
    print("PASS test_grpo_zero_variance_skip")


def test_scale_rewards_boosts_limits_not_market():
    from staged_v5.execution_gate.contracts import NeuralGateAction
    from staged_v5.execution_gate.grpo import _scale_rewards

    rewards = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    actions = torch.tensor(
        [
            int(NeuralGateAction.MARKET_NOW),
            int(NeuralGateAction.LIMIT_2TICK),
            int(NeuralGateAction.PASSIVE_LIMIT_1TICK),
        ],
        dtype=torch.long,
    )
    filled_mask = torch.tensor([True, True, True], dtype=torch.bool)
    scaled = _scale_rewards(rewards, actions, filled_mask, positive_fill_reward_boost=2.0)
    assert abs(float(scaled[0].item()) - 1.0) < 1e-9
    assert abs(float(scaled[1].item()) - 2.0) < 1e-9
    assert abs(float(scaled[2].item()) - 1.0) < 1e-9
    print("PASS test_scale_rewards_boosts_limits_not_market")


def test_grpo_update_increases_better_action_probability():
    from staged_v5.execution_gate.grpo import grpo_update_step
    from staged_v5.execution_gate.model import MicrostructureGate

    model = MicrostructureGate(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    states = torch.zeros((4, 4), dtype=torch.float32)
    with torch.no_grad():
        reference_logits = model(states).detach()
        before = torch.softmax(model(states[:1]), dim=-1)[0, 1].item()
        dist = torch.distributions.Categorical(logits=reference_logits)
        old_log_probs = dist.log_prob(torch.tensor([1, 1, 1, 0], dtype=torch.long))
    result = grpo_update_step(
        model,
        optimizer,
        states,
        torch.tensor([1, 1, 1, 0], dtype=torch.long),
        torch.tensor([2.0, 2.0, 1.5, -1.0], dtype=torch.float32),
        old_log_probs,
        reference_logits=reference_logits,
        filled_mask=torch.tensor([True, True, True, False]),
    )
    after = torch.softmax(model(states[:1]), dim=-1)[0, 1].item()
    assert result["skipped"] is False
    assert after > before
    print("PASS test_grpo_update_increases_better_action_probability")


def test_sample_group_actions_returns_flat_actions():
    from train_execution_gate_v5 import _sample_group_actions
    from staged_v5.execution_gate.contracts import NeuralGateAction

    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)
    actions, old_log_probs = _sample_group_actions(logits, group_size=8, temperature=1.0)
    assert tuple(actions.shape) == (8,)
    assert tuple(old_log_probs.shape) == (8,)
    assert all(isinstance(value, (int, np.integer)) for value in actions.detach().cpu().tolist())
    assert all(0 <= int(value) < len(NeuralGateAction) for value in actions.detach().cpu().tolist())
    print("PASS test_sample_group_actions_returns_flat_actions")


def test_neural_gate_backtest_smoke():
    from staged_v5.config import BacktestConfig, EntryConfig, ExitConfig, NeuralGateConfig, PositionConfig
    from staged_v5.execution_gate.contracts import NeuralGateRuntime
    from staged_v5.execution_gate.features import TickProxyStore
    from staged_v5.evaluation.backtest import backtest_probabilities

    class DummyGate(torch.nn.Module):
        def forward(self, state):
            logits = torch.tensor([[0.0, 5.0, -2.0, -2.0, -2.0, -2.0]], dtype=torch.float32, device=state.device)
            return logits.repeat(state.shape[0], 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "EURUSD_1000ms.csv"
        rows = []
        for second in range(60):
            rows.append(
                {
                    "dt": pd.Timestamp("2026-01-01 00:59:00") + pd.Timedelta(seconds=second),
                    "o": 1.1000,
                    "h": 1.1001,
                    "l": 1.0999,
                    "c": 1.1000,
                    "sp": 1.0,
                    "tk": 10,
                    "tick_velocity": 2.0,
                    "spread_z": 1.0,
                    "bid_ask_imbalance": 0.1,
                    "price_velocity": 0.5,
                }
            )
        rows.extend(
            [
                {"dt": "2026-01-01 01:00:00", "o": 1.1000, "h": 1.1001, "l": 1.0999, "c": 1.1000, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 0.5},
                {"dt": "2026-01-01 01:00:01", "o": 1.1002, "h": 1.1003, "l": 1.1001, "c": 1.1002, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 0.5},
                {"dt": "2026-01-01 01:00:02", "o": 1.1003, "h": 1.1016, "l": 1.1002, "c": 1.1014, "sp": 1.0, "tk": 10, "tick_velocity": 2.0, "spread_z": 1.0, "bid_ask_imbalance": 0.1, "price_velocity": 0.5},
            ]
        )
        pd.DataFrame(rows).to_csv(path, index=False)
        cfg = BacktestConfig(
            entry=EntryConfig(entry_type="neural_gate", latency_bars=1, slippage_atr=0.0),
            neural_gate=NeuralGateConfig(enabled=True),
            exit=ExitConfig(take_profit_atr=0.2, stop_loss_atr=0.7, max_loss_pct_per_trade=1.0, max_hold_bars=4, slippage_atr=0.0),
            position=PositionConfig(max_positions=1, cooldown_bars=0, max_group_exposure=10),
        )
        runtime = NeuralGateRuntime(
            model=DummyGate(),
            config=cfg.neural_gate,
            tick_store=TickProxyStore(tmp_dir),
            device=torch.device("cpu"),
        )
        prob_buy = np.array([[0.50], [0.85], [0.80], [0.75], [0.55]], dtype=np.float32)
        prob_entry = np.array([[0.50], [0.70], [0.70], [0.60], [0.50]], dtype=np.float32)
        close = np.array([[1.1000], [1.1000], [1.1004], [1.1012], [1.1013]], dtype=np.float32)
        high = np.array([[1.1000], [1.1001], [1.1006], [1.1017], [1.1015]], dtype=np.float32)
        low = np.array([[1.1000], [1.0999], [1.1001], [1.1009], [1.1010]], dtype=np.float32)
        volatility = np.full_like(prob_buy, 0.01)
        session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
        timestamps = np.array(
            [
                np.datetime64("2026-01-01T00:59:00"),
                np.datetime64("2026-01-01T01:00:00"),
                np.datetime64("2026-01-01T01:01:00"),
                np.datetime64("2026-01-01T01:02:00"),
                np.datetime64("2026-01-01T01:03:00"),
            ]
        )
        anchor_node_features = np.zeros((5, 1, 14), dtype=np.float32)
        anchor_node_features[:, 0, 4] = 1.0
        anchor_node_features[:, 0, 5] = 100.0
        anchor_tpo_features = np.zeros((5, 1, 8), dtype=np.float32)
        result = backtest_probabilities(
            prob_buy,
            prob_entry,
            close,
            high,
            low,
            volatility,
            session_codes,
            ("EURUSD",),
            cfg,
            timestamps=timestamps,
            anchor_node_features=anchor_node_features,
            anchor_tpo_features=anchor_tpo_features,
            neural_gate_runtime=runtime,
        )
        assert result["gate_action_counts"]["market_now"] > 0
        assert result["gate_fill_rate_by_action"]["market_now"] > 0.0
        assert "market_now" in result["gate_mean_reward_by_action"]
        assert result["trade_count"] >= 1
        print("PASS test_neural_gate_backtest_smoke")


_ALL_TESTS = [value for key, value in sorted(locals().items()) if key.startswith("test_")]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for test_fn in _ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
