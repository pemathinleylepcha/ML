from __future__ import annotations

import json
import logging
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import staged_v4.data.fx_features as fx_features_module
from tick_resampler import parse_tick_csv, resample_to_1000ms, stream_resample_tick_csv
from staged_v4.config import ALL_TIMEFRAMES, BacktestConfig, DEFAULT_SEQ_LENS, GAConfig, STGNNBlockConfig, SubnetConfig, TrainingConfig
from staged_v4.data import (
    StagedPanels,
    build_bridge_batches,
    build_btc_feature_batch,
    build_fx_feature_batch,
    build_btc_sequence_batch_from_panels,
    build_fx_sequence_batch_from_panels,
    compute_tpo_feature_panel,
    generate_synthetic_panels,
    load_feature_batches,
    prepare_staged_cache,
)
from staged_v4.data.fx_features import _build_symbol_aux_features, build_fx_timeframe_batch
from staged_v4.data.dataset import _presweep_tpo, _read_tick_frame, _resample_frame, _stream_resample_tick_source
from staged_v4.data.dataset import _load_symbol_timeframe
from staged_v4.data.jit_sequences import _to_device
from staged_v4.evaluation.backtest import adjusted_backtest_config, backtest_probabilities
from staged_v4.models import BTCSubnet, ConditionalBridge, STGNNBlock
from staged_v4.training.train_staged import (
    _build_edge_tensors,
    _build_subnet_sequence_batch,
    _compute_subnet_loss,
    _gpu_memory_state,
    _resolve_cached_splits,
    run_staged_experiment,
)
from staged_v4.utils.calibration_helpers import apply_platt_scaler, fit_platt_scaler
from staged_v4.utils.graph_helpers import rolling_correlation_adjacency
from staged_v4.utils import runtime_logging as runtime_logging_module


def test_synthetic_panels_cover_all_timeframes() -> None:
    btc_panels, fx_panels = generate_synthetic_panels(n_anchor=32, anchor_timeframe="M1")
    assert set(btc_panels.panels) == set(ALL_TIMEFRAMES)
    assert set(fx_panels.panels) == set(ALL_TIMEFRAMES)
    subset = ("tick", "M1", "M5", "H1", "D1", "MN1")
    btc_batch = build_btc_feature_batch(btc_panels, timeframes=subset)
    fx_batch = build_fx_feature_batch(fx_panels, timeframes=subset)
    assert set(btc_batch.timeframe_batches) == set(subset)
    assert set(fx_batch.timeframe_batches) == set(subset)


def test_bridge_zero_outside_overlap() -> None:
    bridge = ConditionalBridge(8, 8)
    context = torch.ones((3, 8), dtype=torch.float32)
    overlap = torch.tensor([1, 0, 1], dtype=torch.bool)
    out = bridge(context, overlap, n_nodes=4)
    assert out.shape == (3, 4, 8)
    assert torch.allclose(out[1], torch.zeros_like(out[1]))
    assert torch.count_nonzero(out[0]) > 0


def test_stgnn_block_shapes() -> None:
    block = STGNNBlock("M1", raw_input_dim=14, tpo_input_dim=8, cfg=STGNNBlockConfig(hidden_dim=16, output_dim=16, n_heads=4))
    node_features = torch.randn(2, 12, 5, 14)
    tpo_features = torch.randn(2, 12, 5, 8)
    volatility = torch.rand(2, 12, 5)
    valid_mask = torch.ones(2, 12, 5, dtype=torch.bool)
    session_codes = torch.zeros(2, 12, dtype=torch.long)
    edge_matrices = {name: torch.eye(5).unsqueeze(0).repeat(2, 1, 1) for name in ("rolling_corr", "fundamental", "session")}
    state = block(node_features, tpo_features, volatility, valid_mask, session_codes, edge_matrices)
    assert state.node_embeddings.shape == (2, 5, 16)
    assert state.directional_logits.shape == (2, 5)
    assert state.entry_logits is not None and state.entry_logits.shape == (2, 5)


def test_cooperative_rotation() -> None:
    btc_panels, _ = generate_synthetic_panels(n_anchor=40, anchor_timeframe="M1")
    btc_batch = build_btc_feature_batch(btc_panels, timeframes=("M1", "M5"))
    device = torch.device("cpu")
    subnet = BTCSubnet(SubnetConfig(timeframe_order=("M1", "M5"), exchange_every_k_batches=2), STGNNBlockConfig(hidden_dim=16, output_dim=16, n_heads=4))
    batch0 = _build_subnet_sequence_batch(btc_batch, np.arange(0, 8, dtype=np.int32), DEFAULT_SEQ_LENS, device)
    edges0 = _build_edge_tensors(batch0, device)
    out0 = subnet.cooperative_step(batch0.timeframe_batches, edges0, active_idx=0, batch_index=0)
    assert out0.state.active_timeframe == "M1"
    batch1 = _build_subnet_sequence_batch(btc_batch, np.arange(8, 16, dtype=np.int32), DEFAULT_SEQ_LENS, device)
    edges1 = _build_edge_tensors(batch1, device)
    out1 = subnet.cooperative_step(batch1.timeframe_batches, edges1, active_idx=out0.next_active_idx, batch_index=2)
    assert out1.next_active_idx == 1


def test_tpo_and_platt() -> None:
    close = np.linspace(1.0, 1.1, 128, dtype=np.float32)
    high = close + 0.01
    low = close - 0.01
    features, vol = compute_tpo_feature_panel(high, low, close)
    assert features.shape == (128, 8)
    assert vol.shape == (128,)
    logits = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    artifact = fit_platt_scaler(logits, labels)
    probs = apply_platt_scaler(artifact, logits)
    assert probs.shape == (4,)
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_jit_preswept_tpo_matches_fallback_sequences() -> None:
    _, fx_panels = generate_synthetic_panels(n_anchor=64, anchor_timeframe="M1")
    anchor_indices = np.array([12, 24], dtype=np.int32)
    device = torch.device("cpu")

    fallback_batch = build_fx_sequence_batch_from_panels(fx_panels, anchor_indices, DEFAULT_SEQ_LENS, device)
    preswept_panels = StagedPanels(
        subnet_name=fx_panels.subnet_name,
        symbols=fx_panels.symbols,
        anchor_timeframe=fx_panels.anchor_timeframe,
        panels=fx_panels.panels,
        anchor_timestamps=fx_panels.anchor_timestamps,
        anchor_lookup=fx_panels.anchor_lookup,
        walkforward_splits=fx_panels.walkforward_splits,
        split_frequency=fx_panels.split_frequency,
        tpo_panels=_presweep_tpo(fx_panels.panels, fx_panels.symbols, tuple(fx_panels.panels.keys())),
    )
    preswept_batch = build_fx_sequence_batch_from_panels(preswept_panels, anchor_indices, DEFAULT_SEQ_LENS, device)

    for timeframe in ("tick", "M1", "M5", "H1"):
        np.testing.assert_allclose(
            preswept_batch.timeframe_batches[timeframe].tpo_features.cpu().numpy(),
            fallback_batch.timeframe_batches[timeframe].tpo_features.cpu().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            preswept_batch.timeframe_batches[timeframe].volatility.cpu().numpy(),
            fallback_batch.timeframe_batches[timeframe].volatility.cpu().numpy(),
            atol=1e-6,
        )


def test_load_symbol_timeframe_returns_none_when_no_candle_or_lower_source() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        out = _load_symbol_timeframe(
            candle_root=root,
            tick_root=root,
            symbol="MISSING",
            timeframe="M1",
            start="2026-03-01",
            end="2026-03-02",
            lower_frame=None,
            lower_timeframe="tick",
            relevant_quarters=[],
            tick_file_map={},
        )
    assert out is None


def test_rolling_correlation_adjacency_ignores_constant_columns_without_warning() -> None:
    close_window = np.array(
        [
            [100.0, 100.0, 100.0],
            [100.1, 100.0, 100.2],
            [100.2, 100.0, 100.4],
            [100.3, 100.0, 100.6],
        ],
        dtype=np.float64,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        adj = rolling_correlation_adjacency(close_window)
    assert adj.shape == (3, 3)
    assert np.isfinite(adj).all()
    assert not any("invalid value encountered in divide" in str(item.message) for item in caught)


def test_memory_guard_reduces_workers_when_available_memory_is_low() -> None:
    with patch.object(runtime_logging_module, "runtime_snapshot", return_value={"system_mem_available_mb": 3000.0}):
        workers, guard = runtime_logging_module.guard_worker_budget(
            16,
            min_available_mb=4096.0,
            critical_available_mb=2048.0,
        )
    assert workers == 2
    assert guard["state"] == "low"


def test_gpu_memory_state_is_safe_on_cpu() -> None:
    guard = _gpu_memory_state(
        torch.device("cpu"),
        min_available_mb=4096.0,
        critical_available_mb=2048.0,
    )
    assert guard["state"] == "ok"
    assert guard["vram_free_mb"] is None
    assert guard["vram_total_mb"] is None


def test_gpu_memory_state_resolves_default_cuda_device_index() -> None:
    free_bytes = 8 * 1024 ** 3
    total_bytes = 24 * 1024 ** 3
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)) as mem_get_info,
        patch("torch.cuda.memory_allocated", return_value=512 * 1024 ** 2) as memory_allocated,
        patch("torch.cuda.memory_reserved", return_value=1024 * 1024 ** 2) as memory_reserved,
    ):
        guard = _gpu_memory_state(
            torch.device("cuda"),
            min_available_mb=4096.0,
            critical_available_mb=2048.0,
        )
    mem_get_info.assert_called_once_with(torch.device("cuda:0"))
    memory_allocated.assert_called_once_with(torch.device("cuda:0"))
    memory_reserved.assert_called_once_with(torch.device("cuda:0"))
    assert guard["state"] == "ok"
    assert guard["vram_free_mb"] == free_bytes / (1024 ** 2)
    assert guard["vram_total_mb"] == total_bytes / (1024 ** 2)


def test_memory_guard_raises_when_available_memory_is_critical() -> None:
    logger = logging.getLogger("memory_guard_test")
    with patch.object(runtime_logging_module, "runtime_snapshot", return_value={"system_mem_available_mb": 1024.0}):
        try:
            runtime_logging_module.enforce_memory_guard(
                logger,
                None,
                "unit_test",
                min_available_mb=4096.0,
                critical_available_mb=2048.0,
                raise_on_critical=True,
            )
        except MemoryError as exc:
            assert "critical_available_mb" in str(exc)
        else:
            raise AssertionError("Expected MemoryError from critical memory guard state")


def test_jit_to_device_preserves_values_on_cpu() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = _to_device(arr, torch.float32, torch.device("cpu"))
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    np.testing.assert_allclose(tensor.numpy(), arr)


def test_pushgateway_metrics_render_training_progress() -> None:
    payload = {
        "runtime": {
            "hostname": "thinley",
            "rss_mb": 512.0,
            "system_mem_available_mb": 8192.0,
        },
        "state": "running",
        "stage": "stage2_fx",
        "fold": 1,
        "fold_total": 2,
        "progress": {"current": 12, "total": 24, "ratio": 0.5},
        "details": {"epoch": 1},
    }
    rendered = runtime_logging_module._render_pushgateway_metrics(
        "/workspace/Algo-C2-Codex/data/remote_runs/demo_run/status.json",
        payload,
    ).decode("utf-8")
    assert 'algoc2_training_status{run="demo_run",stage="stage2_fx",state="running",host="thinley"} 1.0' in rendered
    assert 'algoc2_training_progress_ratio{run="demo_run",stage="stage2_fx",host="thinley"} 0.5' in rendered
    assert 'algoc2_training_fold_total{run="demo_run",host="thinley"} 2.0' in rendered
    assert 'algoc2_training_runtime_rss_mb{run="demo_run",host="thinley"} 512.0' in rendered


def test_pushgateway_metrics_render_training_throughput() -> None:
    payload = {
        "runtime": {
            "hostname": "thinley",
        },
        "state": "running",
        "stage": "stage2_fx",
        "details": {
            "batches": 128,
            "elapsed_sec": 32.0,
            "batches_per_sec": 4.0,
            "batch_size": 16,
        },
    }
    rendered = runtime_logging_module._render_pushgateway_metrics(
        "/workspace/Algo-C2-Codex/data/remote_runs/demo_run/status.json",
        payload,
    ).decode("utf-8")
    assert 'algoc2_training_stage_batches{run="demo_run",stage="stage2_fx",host="thinley"} 128.0' in rendered
    assert 'algoc2_training_stage_elapsed_sec{run="demo_run",stage="stage2_fx",host="thinley"} 32.0' in rendered
    assert 'algoc2_training_stage_batches_per_sec{run="demo_run",stage="stage2_fx",host="thinley"} 4.0' in rendered
    assert 'algoc2_training_stage_batch_size{run="demo_run",stage="stage2_fx",host="thinley"} 16.0' in rendered


def test_backtest_tp_exit_and_position_cap() -> None:
    prob_buy = np.array(
        [
            [0.50, 0.50],
            [0.82, 0.80],
            [0.81, 0.79],
            [0.50, 0.50],
        ],
        dtype=np.float32,
    )
    prob_entry = np.full_like(prob_buy, 0.80)
    close = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.2, 100.15],
        ],
        dtype=np.float32,
    )
    high = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.5, 100.35],
        ],
        dtype=np.float32,
    )
    low = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [99.95, 99.95],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        max_hold_bars=6,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        take_profit_atr=0.35,
        stop_loss_atr=0.35,
        max_loss_pct_per_trade=0.05,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 1
    assert result["max_open_positions"] == 1
    assert any(reason in result["exit_reason_counts"] for reason in ("take_profit", "tp_sl_same_bar_tp"))
    assert result["win_rate"] == 1.0


def test_backtest_entry_gate_blocks_low_entry_prob() -> None:
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.20], [0.20], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.4], [100.2]], dtype=np.float32)
    high = np.array([[100.0], [100.0], [100.5], [100.3]], dtype=np.float32)
    low = np.array([[100.0], [100.0], [99.9], [100.0]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        max_loss_pct_per_trade=0.05,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 0


def test_backtest_trailing_stop_moves_to_breakeven() -> None:
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.51]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.1]], dtype=np.float32)
    high = np.array([[100.5], [100.5], [100.5], [100.6]], dtype=np.float32)
    low = np.array([[99.5], [99.5], [99.5], [99.9]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        take_profit_atr=2.0,
        stop_loss_atr=0.7,
        max_loss_pct_per_trade=0.05,
        trailing_activate_atr=0.5,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 1
    assert result["exit_reason_counts"]["trailing_breakeven"] == 1
    assert abs(result["net_return"]) < 1e-9


def test_backtest_caps_inverted_high_confidence_trades() -> None:
    prob_buy = np.array([[0.50], [0.92], [0.91], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [101.0], [100.8]], dtype=np.float32)
    high = np.array([[100.0], [100.0], [101.2], [100.9]], dtype=np.float32)
    low = np.array([[100.0], [100.0], [99.9], [100.7]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.60,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.70,
        max_loss_pct_per_trade=0.05,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 0


def test_compute_subnet_loss_applies_label_smoothing() -> None:
    logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    entry_logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    direction_labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    entry_labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, True]], dtype=torch.bool)
    subnet_state = SimpleNamespace(
        timeframe_states={
            "M1": SimpleNamespace(
                directional_logits=logits,
                entry_logits=entry_logits,
            )
        }
    )
    sequence_batch = SimpleNamespace(
        timeframe_batches={
            "M1": SimpleNamespace(
                direction_labels=direction_labels,
                entry_labels=entry_labels,
                label_valid_mask=valid_mask,
                tradable_indices=(0, 1),
            )
        }
    )
    smooth = 0.10
    loss = _compute_subnet_loss(
        subnet_state,
        sequence_batch,
        active_timeframe="M1",
        active_boost=1.0,
        label_smooth=smooth,
    )
    smoothed_targets = direction_labels * (1.0 - smooth) + 0.5 * smooth
    smoothed_entry_targets = entry_labels * (1.0 - smooth) + 0.5 * smooth
    expected = nn.BCEWithLogitsLoss()(logits[valid_mask], smoothed_targets[valid_mask]) + 0.5 * nn.BCEWithLogitsLoss()(
        entry_logits[valid_mask],
        smoothed_entry_targets[valid_mask],
    )
    assert torch.allclose(loss, expected)


def test_backtest_hard_loss_cap_limits_single_trade_drawdown() -> None:
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float32)
    high = np.array([[100.0], [105.0], [105.0], [100.0]], dtype=np.float32)
    low = np.array([[100.0], [95.0], [95.0], [99.0]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        take_profit_atr=1.0,
        stop_loss_atr=0.70,
        max_loss_pct_per_trade=0.005,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 1
    assert result["exit_reason_counts"]["stop_loss"] == 1
    assert result["net_return"] >= -0.00501
    assert result["net_return"] <= -0.00499


def test_backtest_sanitizes_entry_atr_for_take_profit_geometry() -> None:
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float32)
    high = np.array([[100.0], [105.0], [100.0], [101.1]], dtype=np.float32)
    low = np.array([[100.0], [95.0], [99.9], [99.9]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        take_profit_atr=1.0,
        stop_loss_atr=0.70,
        max_loss_pct_per_trade=0.05,
        max_entry_atr_pct=0.01,
        slippage_atr=0.0,
        use_limit_entries=False,
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
    assert result["trade_count"] == 1
    assert any(reason in result["exit_reason_counts"] for reason in ("take_profit", "tp_sl_same_bar_tp"))
    assert result["net_return"] > 0.0099


def test_backtest_applies_slippage_on_entry_and_stop_loss() -> None:
    prob_buy = np.array([[0.50], [0.82], [0.81], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float32)
    high = np.array([[100.0], [101.0], [101.0], [100.0]], dtype=np.float32)
    low = np.array([[100.0], [99.0], [99.0], [99.0]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.95,
        take_profit_atr=2.0,
        stop_loss_atr=0.50,
        max_loss_pct_per_trade=0.05,
        max_entry_atr_pct=0.01,
        slippage_atr=0.05,
        use_limit_entries=False,
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
    expected_entry = 100.0 + 0.05 * 1.0
    expected_exit = (expected_entry - 0.50 * 1.0) - 0.05 * 1.0
    expected_return = (expected_exit - expected_entry) / expected_entry
    assert result["trade_count"] == 1
    assert result["exit_reason_counts"]["stop_loss"] == 1
    assert abs(result["net_return"] - expected_return) < 1e-6


def test_adjusted_backtest_config_halves_positions_on_high_ece() -> None:
    base_cfg = BacktestConfig(max_positions=6, ece_gate_threshold=0.09)
    adjusted = adjusted_backtest_config(base_cfg, 0.105)
    assert adjusted.max_positions == 3
    unchanged = adjusted_backtest_config(base_cfg, 0.08)
    assert unchanged.max_positions == 6
    disabled = adjusted_backtest_config(BacktestConfig(max_positions=6, ece_gate_threshold=0.0), 0.50)
    assert disabled.max_positions == 6


def test_backtest_threshold_diagnostics_include_subthreshold_bucket() -> None:
    prob_buy = np.array([[0.50], [0.58], [0.58], [0.50]], dtype=np.float32)
    prob_entry = np.array([[0.50], [0.80], [0.80], [0.50]], dtype=np.float32)
    close = np.array([[100.0], [100.0], [100.4], [100.2]], dtype=np.float32)
    high = np.array([[100.0], [100.0], [100.6], [100.3]], dtype=np.float32)
    low = np.array([[100.0], [100.0], [99.9], [100.1]], dtype=np.float32)
    volatility = np.full_like(prob_buy, 0.001)
    session_codes = np.ones(prob_buy.shape[0], dtype=np.int64)
    cfg = BacktestConfig(
        base_entry_threshold=0.57,
        threshold_volatility_coeff=0.0,
        probability_spread_threshold=0.05,
        latency_bars=1,
        max_positions=1,
        entry_gate_threshold=0.50,
        max_confidence_threshold=0.70,
        use_limit_entries=False,
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
    assert result["threshold_diagnostics"][0]["min_confidence"] == 0.50
    assert result["threshold_diagnostics"][0]["max_confidence"] == 0.57


def test_large_tick_tpo_guard_skips_without_source() -> None:
    n_rows = 500_001
    timestamps = pd.date_range("2026-03-01", periods=n_rows, freq="1s")
    close = np.linspace(1.0, 1.01, n_rows, dtype=np.float32)
    high = close + 0.001
    low = close - 0.001
    spread = np.full(n_rows, 0.0001, dtype=np.float32)
    valid = np.ones(n_rows, dtype=np.bool_)
    _, node_tpo, node_vol, direction, entry, label_valid, diagnostics = _build_symbol_aux_features(
        symbol="XAUUSD",
        timeframe="tick",
        timestamps=pd.Index(timestamps),
        high_col=high,
        low_col=low,
        close_col=close,
        spread_col=spread,
        tradable=False,
        valid_col=valid,
    )
    assert diagnostics["tpo_skipped_large"] is True
    assert np.count_nonzero(node_tpo) == 0
    assert np.count_nonzero(node_vol) == 0
    assert np.all(direction == -1)
    assert np.count_nonzero(entry) == 0
    assert np.count_nonzero(label_valid) == 0


def test_tpo_source_floor_uses_m5_for_tick_and_m1() -> None:
    btc_panels, fx_panels = generate_synthetic_panels(n_anchor=32, anchor_timeframe="M1")
    subset = ("tick", "M1", "M5")
    btc_batch = build_btc_feature_batch(btc_panels, timeframes=subset)
    fx_batch = build_fx_feature_batch(fx_panels, timeframes=subset)

    for batch in (btc_batch, fx_batch):
        m5_batch = batch.timeframe_batches["M5"]
        m5_index = pd.Index(m5_batch.timestamps)
        for timeframe in ("tick", "M1"):
            tf_batch = batch.timeframe_batches[timeframe]
            tf_index = pd.Index(tf_batch.timestamps)
            lookup = np.maximum(m5_index.searchsorted(tf_index, side="right") - 1, 0)
            np.testing.assert_allclose(tf_batch.tpo_features, m5_batch.tpo_features[lookup], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(tf_batch.volatility, m5_batch.volatility[lookup], rtol=1e-5, atol=1e-5)


def test_jit_sequence_batches_match_prebuilt_batches_for_m1_m5() -> None:
    btc_panels_full, fx_panels_full = generate_synthetic_panels(n_anchor=24, anchor_timeframe="M1")
    subset = ("M1", "M5")
    btc_panels = StagedPanels(
        subnet_name=btc_panels_full.subnet_name,
        symbols=btc_panels_full.symbols,
        anchor_timeframe=btc_panels_full.anchor_timeframe,
        panels={tf: btc_panels_full.panels[tf] for tf in subset},
        anchor_timestamps=btc_panels_full.anchor_timestamps,
        anchor_lookup={tf: btc_panels_full.anchor_lookup[tf] for tf in subset},
        walkforward_splits=btc_panels_full.walkforward_splits,
        split_frequency=btc_panels_full.split_frequency,
        tpo_panels={tf: btc_panels_full.tpo_panels[tf] for tf in btc_panels_full.tpo_panels if tf in subset},
    )
    fx_panels = StagedPanels(
        subnet_name=fx_panels_full.subnet_name,
        symbols=fx_panels_full.symbols,
        anchor_timeframe=fx_panels_full.anchor_timeframe,
        panels={tf: fx_panels_full.panels[tf] for tf in subset},
        anchor_timestamps=fx_panels_full.anchor_timestamps,
        anchor_lookup={tf: fx_panels_full.anchor_lookup[tf] for tf in subset},
        walkforward_splits=fx_panels_full.walkforward_splits,
        split_frequency=fx_panels_full.split_frequency,
        tpo_panels={tf: fx_panels_full.tpo_panels[tf] for tf in fx_panels_full.tpo_panels if tf in subset},
    )
    btc_batch = build_btc_feature_batch(btc_panels, timeframes=subset)
    fx_batch = build_fx_feature_batch(fx_panels, timeframes=subset)
    anchor_indices = np.array([5, 11, 17], dtype=np.int32)
    seq_lens = {"M1": DEFAULT_SEQ_LENS["M1"], "M5": DEFAULT_SEQ_LENS["M5"]}
    device = torch.device("cpu")

    btc_prebuilt = _build_subnet_sequence_batch(btc_batch, anchor_indices, seq_lens, device)
    btc_jit = build_btc_sequence_batch_from_panels(btc_panels, anchor_indices, seq_lens, device)
    fx_prebuilt = _build_subnet_sequence_batch(fx_batch, anchor_indices, seq_lens, device)
    fx_jit = build_fx_sequence_batch_from_panels(fx_panels, anchor_indices, seq_lens, device)

    for timeframe in subset:
        np.testing.assert_allclose(
            btc_jit.timeframe_batches[timeframe].node_features.detach().cpu().numpy(),
            btc_prebuilt.timeframe_batches[timeframe].node_features.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            btc_jit.timeframe_batches[timeframe].tpo_features.detach().cpu().numpy(),
            btc_prebuilt.timeframe_batches[timeframe].tpo_features.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            fx_jit.timeframe_batches[timeframe].node_features.detach().cpu().numpy(),
            fx_prebuilt.timeframe_batches[timeframe].node_features.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            fx_jit.timeframe_batches[timeframe].tpo_features.detach().cpu().numpy(),
            fx_prebuilt.timeframe_batches[timeframe].tpo_features.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            fx_jit.timeframe_batches[timeframe].volatility.detach().cpu().numpy(),
            fx_prebuilt.timeframe_batches[timeframe].volatility.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            fx_jit.timeframe_batches[timeframe].label_valid_mask.detach().cpu().numpy(),
            fx_prebuilt.timeframe_batches[timeframe].label_valid_mask.detach().cpu().numpy(),
        )


def test_cache_roundtrip() -> None:
    cache_root = Path("data/staged_v4_cache_test")
    manifest = prepare_staged_cache(
        output_root=cache_root,
        mode="synthetic",
        anchor_timeframe="M1",
        timeframes=("tick", "M1", "M5"),
        synthetic_n_anchor=32,
    )
    assert manifest["fx_timeframes"] == ["tick", "M1", "M5"]
    btc_batch, fx_batch, walkforward_splits, split_meta = load_feature_batches(cache_root)
    assert set(btc_batch.timeframe_batches) == {"tick", "M1", "M5"}
    assert set(fx_batch.timeframe_batches) == {"tick", "M1", "M5"}
    assert split_meta["split_frequency"] == "week"
    assert split_meta["outer_holdout_blocks"] == 1
    assert len(walkforward_splits) >= 0


def test_cached_split_override_regenerates_more_folds() -> None:
    anchor_timestamps = np.array(
        pd.date_range("2026-03-02", periods=28, freq="1D").to_numpy(dtype="datetime64[ns]")
    )
    cached_splits = [
        {
            "fold": 0,
            "split_frequency": "week",
            "train_blocks": ["2026-03-02/2026-03-08", "2026-03-09/2026-03-15"],
            "val_block": "2026-03-16/2026-03-22",
            "train_idx": np.arange(0, 14, dtype=np.int32),
            "val_idx": np.arange(14, 21, dtype=np.int32),
        }
    ]
    split_meta = {"split_frequency": "week", "outer_holdout_blocks": 1}
    training_cfg = TrainingConfig(
        anchor_timeframe="M1",
        split_frequency="week",
        outer_holdout_blocks=0,
        min_train_blocks=2,
        purge_bars=0,
    )
    splits, split_frequency = _resolve_cached_splits(
        anchor_timestamps,
        cached_splits,
        split_meta,
        training_cfg,
        logger=logging.getLogger("test"),
        status_file=None,
    )
    assert split_frequency == "week"
    assert len(splits) == 2
    assert splits[0]["val_block"] == "2026-03-16/2026-03-22"
    assert splits[1]["val_block"] == "2026-03-23/2026-03-29"


def test_fx_shard_resume_roundtrip() -> None:
    _, fx_panels = generate_synthetic_panels(n_anchor=32, anchor_timeframe="M1")
    shard_root = Path("data/staged_v4_fx_shards_test")
    if shard_root.exists():
        for path in shard_root.glob("*.npz"):
            path.unlink()
    else:
        shard_root.mkdir(parents=True, exist_ok=True)
    first = build_fx_feature_batch(
        fx_panels,
        timeframes=("M5",),
        include_signal_only_tpo=False,
        max_workers=2,
        status_file=None,
        logger=None,
    )
    batch1 = first.timeframe_batches["M5"]
    batch2 = build_fx_timeframe_batch(
        fx_panels,
        "M5",
        include_signal_only_tpo=False,
        max_workers=2,
        logger=None,
        status_file=None,
        shard_root=shard_root,
    )
    batch3 = build_fx_timeframe_batch(
        fx_panels,
        "M5",
        include_signal_only_tpo=False,
        max_workers=2,
        logger=None,
        status_file=None,
        shard_root=shard_root,
    )
    assert len(list(shard_root.glob("*.npz"))) == len(fx_panels.symbols)
    np.testing.assert_allclose(batch1.node_features, batch2.node_features, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch2.tpo_features, batch3.tpo_features, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch2.volatility, batch3.volatility, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(batch2.direction_labels, batch3.direction_labels)
    np.testing.assert_array_equal(batch2.entry_labels, batch3.entry_labels)
    np.testing.assert_array_equal(batch2.label_valid_mask, batch3.label_valid_mask)


def test_fx_timeframe_batch_continues_after_symbol_exception() -> None:
    _, fx_panels = generate_synthetic_panels(n_anchor=24, anchor_timeframe="M1")
    shard_root = Path("data/staged_v4_fx_failed_symbol_test")
    if shard_root.exists():
        for path in shard_root.glob("*.npz"):
            path.unlink()
    else:
        shard_root.mkdir(parents=True, exist_ok=True)

    original = fx_features_module._build_symbol_aux_features
    first_symbol = fx_panels.symbols[0]

    def _patched(*args, **kwargs):
        symbol = args[0]
        if symbol == first_symbol:
            raise RuntimeError("synthetic failure")
        return original(*args, **kwargs)

    fx_features_module._build_symbol_aux_features = _patched
    try:
        batch = build_fx_timeframe_batch(
            fx_panels,
            "M5",
            include_signal_only_tpo=False,
            max_workers=2,
            logger=None,
            status_file=None,
            shard_root=shard_root,
            symbol_timeout_sec=1.0,
            batch_deadline_sec=30.0,
        )
    finally:
        fx_features_module._build_symbol_aux_features = original

    failed_idx = fx_panels.symbols.index(first_symbol)
    assert np.count_nonzero(batch.tpo_features[:, failed_idx, :]) == 0
    assert np.count_nonzero(batch.volatility[:, failed_idx]) == 0
    assert np.all(batch.direction_labels[:, failed_idx] == -1)
    assert np.count_nonzero(batch.entry_labels[:, failed_idx]) == 0
    assert np.count_nonzero(batch.label_valid_mask[:, failed_idx]) == 0
    assert (shard_root / f"{first_symbol}.npz").exists()


def test_stream_resample_tick_checkpoint_matches_in_memory() -> None:
    tick_root = Path("data/staged_v4_tick_stream_test")
    tick_root.mkdir(parents=True, exist_ok=True)
    tick_path = tick_root / "BTCUSD_1000ms.csv"
    dt_index = pd.date_range("2026-03-01 00:00:01", periods=180, freq="1s")
    frame = pd.DataFrame(
        {
            "dt": dt_index,
            "o": np.linspace(100.0, 101.0, len(dt_index)),
            "h": np.linspace(100.1, 101.1, len(dt_index)),
            "l": np.linspace(99.9, 100.9, len(dt_index)),
            "c": np.linspace(100.0, 101.0, len(dt_index)),
            "sp": np.full(len(dt_index), 0.01),
            "tk": np.ones(len(dt_index), dtype=np.int32),
        }
    )
    frame.to_csv(tick_path, index=False)
    loaded = _read_tick_frame(tick_path)
    expected = _resample_frame(loaded, "M1")
    actual = _stream_resample_tick_source(tick_path, "M1")
    assert actual is not None and expected is not None
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
    checkpoint = tick_root / "_derived" / "M1" / "BTCUSD_M1.csv"
    assert checkpoint.exists()


def test_stream_resample_tick_csv_matches_in_memory() -> None:
    raw_root = Path("data/tick_resampler_stream_test")
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / "BTCUSD_raw.csv"
    raw_path.write_text(
        "\n".join(
            [
                "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>",
                "2026.03.01\t00:00:00.100\t100.0\t100.2\t\t\t6",
                "2026.03.01\t00:00:00.500\t100.1\t100.3\t\t\t6",
                "2026.03.01\t00:00:01.100\t100.2\t100.4\t\t\t6",
                "2026.03.01\t00:00:01.900\t100.3\t100.5\t\t\t6",
                "2026.03.01\t00:00:02.050\t100.4\t100.6\t\t\t6",
            ]
        ),
        encoding="utf-8",
    )
    ticks = parse_tick_csv(raw_path)
    expected = resample_to_1000ms(ticks, "BTCUSD")
    actual = stream_resample_tick_csv(raw_path, "BTCUSD", chunksize=2)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_freq=False)


def test_smoke_train() -> None:
    output = Path("data/staged_v4_smoke.json")
    report = run_staged_experiment(
        mode="synthetic",
        output=str(output),
        anchor_timeframe="M1",
        max_folds=1,
        training_cfg=TrainingConfig(anchor_timeframe="M1", batch_size=8, epochs_stage1=1, epochs_stage2=1, epochs_stage3=0),
        subnet_cfg=SubnetConfig(timeframe_order=("M1", "M5"), exchange_every_k_batches=2),
        ga_cfg=GAConfig(population_size=4, generations=0),
        synthetic_n_anchor=24,
        include_signal_only_tpo=False,
    )
    assert output.exists()
    assert report["summary"]["folds"] >= 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "summary" in payload


def main() -> None:
    tests = [
        test_synthetic_panels_cover_all_timeframes,
        test_bridge_zero_outside_overlap,
        test_stgnn_block_shapes,
        test_cooperative_rotation,
        test_tpo_and_platt,
        test_compute_subnet_loss_applies_label_smoothing,
        test_rolling_correlation_adjacency_ignores_constant_columns_without_warning,
        test_memory_guard_reduces_workers_when_available_memory_is_low,
        test_gpu_memory_state_is_safe_on_cpu,
        test_memory_guard_raises_when_available_memory_is_critical,
        test_jit_to_device_preserves_values_on_cpu,
        test_pushgateway_metrics_render_training_progress,
        test_pushgateway_metrics_render_training_throughput,
        test_large_tick_tpo_guard_skips_without_source,
        test_tpo_source_floor_uses_m5_for_tick_and_m1,
        test_jit_sequence_batches_match_prebuilt_batches_for_m1_m5,
        test_cache_roundtrip,
        test_cached_split_override_regenerates_more_folds,
        test_fx_shard_resume_roundtrip,
        test_fx_timeframe_batch_continues_after_symbol_exception,
        test_backtest_trailing_stop_moves_to_breakeven,
        test_stream_resample_tick_checkpoint_matches_in_memory,
        test_stream_resample_tick_csv_matches_in_memory,
        test_backtest_tp_exit_and_position_cap,
        test_backtest_entry_gate_blocks_low_entry_prob,
        test_backtest_caps_inverted_high_confidence_trades,
        test_backtest_hard_loss_cap_limits_single_trade_drawdown,
        test_backtest_sanitizes_entry_atr_for_take_profit_geometry,
        test_backtest_applies_slippage_on_entry_and_stop_loss,
        test_adjusted_backtest_config_halves_positions_on_high_ece,
        test_backtest_threshold_diagnostics_include_subthreshold_bucket,
        test_smoke_train,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"passed {len(tests)} staged_v4 tests")


if __name__ == "__main__":
    main()
