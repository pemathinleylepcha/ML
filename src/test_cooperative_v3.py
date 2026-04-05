from __future__ import annotations

import unittest

import numpy as np
import torch

from cooperative_v3.config import DualSubnetSystemConfig
from cooperative_v3.contracts import TimeframeState
from cooperative_v3.execution import KellyConfig, fractional_kelly_fraction, normalize_kelly_allocations, run_fractional_kelly_backtest
from cooperative_v3.layers import CausalTemporalSelfAttention
from cooperative_v3.meta import CatBoostMetaClassifier, MetaFeatureBuilder, MetaFeatureMatrix
from cooperative_v3.real_data import _build_scalp_entry_labels
from cooperative_v3.subnet import AdjacentTimeframeExchangeController
from cooperative_v3.synthetic import build_synthetic_dual_subnet_batches
from cooperative_v3.system import DualSubnetCooperativeSystem
from research_dataset import SESSION_CODES


class CooperativeV3Tests(unittest.TestCase):
    def test_system_smoke_builds_all_timeframes_and_meta_features(self):
        config = DualSubnetSystemConfig()
        system = DualSubnetCooperativeSystem(config)
        btc_batch, fx_batch = build_synthetic_dual_subnet_batches(config=config)
        with torch.no_grad():
            output = system(btc_batch, fx_batch, batch_index=8, bridge_enabled=True)
        self.assertIn("tick", output.btc_state.timeframe_states)
        self.assertIn("M1", output.fx_state.timeframe_states)
        self.assertGreater(output.meta_features.X.shape[0], 0)
        self.assertGreater(output.meta_features.X.shape[1], 0)

    def test_bridge_is_zero_when_overlap_closed(self):
        config = DualSubnetSystemConfig()
        system = DualSubnetCooperativeSystem(config)
        btc_batch, fx_batch = build_synthetic_dual_subnet_batches(config=config)
        for tf_batch in fx_batch.timeframe_batches.values():
            tf_batch.overlap_mask[:] = False
            tf_batch.market_open_mask[:] = False
        with torch.no_grad():
            btc_state = system.forward_btc_only(btc_batch, batch_index=8)
            contexts = system.bridge.build_fx_contexts(btc_state, fx_batch)
        self.assertTrue(contexts)
        for value in contexts.values():
            self.assertTrue(torch.allclose(value, torch.zeros_like(value)))

    def test_fractional_kelly_is_bounded_and_normalized(self):
        cfg = KellyConfig(fractional_scale=0.25, max_fraction_per_trade=0.05, portfolio_cap=0.10)
        fractions = [
            fractional_kelly_fraction(0.57, 1.5, cfg),
            fractional_kelly_fraction(0.61, 1.2, cfg),
            fractional_kelly_fraction(0.53, 2.0, cfg),
        ]
        normalized = normalize_kelly_allocations(fractions, cfg)
        self.assertLessEqual(float(normalized.sum()), cfg.portfolio_cap + 1e-6)
        self.assertTrue((normalized >= 0.0).all())

    def test_meta_builder_uses_macro_and_micro_outputs(self):
        config = DualSubnetSystemConfig()
        system = DualSubnetCooperativeSystem(config)
        btc_batch, fx_batch = build_synthetic_dual_subnet_batches(config=config)
        with torch.no_grad():
            output = system(btc_batch, fx_batch, batch_index=8, bridge_enabled=True)
        builder = MetaFeatureBuilder(macro_timeframe="M5", micro_timeframe="M5")
        matrix = builder.build(output, fx_batch)
        self.assertIn("macro_prob", matrix.feature_names)
        self.assertIn("micro_entry_prob", matrix.feature_names)
        self.assertIn("open_ready", matrix.feature_names)
        self.assertIn("momentum_score", matrix.feature_names)
        self.assertIn("volatility_ratio", matrix.feature_names)
        self.assertIn("lap_laggard_score", matrix.feature_names)
        self.assertIn("lap_neighbor_momentum", matrix.feature_names)
        self.assertEqual(matrix.X.shape[0], len(matrix.references))

    def test_exchange_memory_persists_until_reset(self):
        config = DualSubnetSystemConfig()
        system = DualSubnetCooperativeSystem(config)
        system.eval()
        btc_batch, _ = build_synthetic_dual_subnet_batches(config=config)
        with torch.no_grad():
            no_exchange_state = system.forward_btc_only(btc_batch, batch_index=7)
            system.reset_cooperative_state()
            first_state = system.forward_btc_only(btc_batch, batch_index=8)
            second_state = system.forward_btc_only(btc_batch, batch_index=9)
            system.reset_cooperative_state()
            reset_state = system.forward_btc_only(btc_batch, batch_index=9)
        self.assertTrue(first_state.next_exchange_contexts)
        self.assertFalse(
            torch.allclose(
                no_exchange_state.timeframe_states["M1"].pooled_context,
                first_state.timeframe_states["M1"].pooled_context,
            )
        )
        self.assertTrue(system.btc_subnet.exchange_memory == {})
        self.assertFalse(
            torch.allclose(
                second_state.timeframe_states["M1"].pooled_context,
                reset_state.timeframe_states["M1"].pooled_context,
            )
        )
        self.assertNotIn("MN1", first_state.next_exchange_contexts)

    def test_lower_timeframes_use_wider_slower_neighborhood(self):
        controller = AdjacentTimeframeExchangeController(
            timeframe_order=("M5", "M15", "M30", "H1", "H4"),
            exchange_every_k_batches=8,
        )
        states = {
            "M15": TimeframeState("M15", None, torch.full((1, 3), 10.0), torch.zeros(1, 1)),
            "M30": TimeframeState("M30", None, torch.full((1, 3), 20.0), torch.zeros(1, 1)),
            "H1": TimeframeState("H1", None, torch.full((1, 3), 30.0), torch.zeros(1, 1)),
            "H4": TimeframeState("H4", None, torch.full((1, 3), 40.0), torch.zeros(1, 1)),
        }
        contexts = controller.build_next_contexts(states)
        expected_m5 = (0.7 * states["M15"].pooled_context) + (0.3 * states["M30"].pooled_context)
        expected_m15 = (0.7 * states["M30"].pooled_context) + (0.3 * states["H1"].pooled_context)
        self.assertTrue(torch.allclose(contexts["M5"], expected_m5))
        self.assertTrue(torch.allclose(contexts["M15"], expected_m15))
        self.assertTrue(torch.allclose(contexts["H1"], states["H4"].pooled_context))
        self.assertNotIn("H4", contexts)

    def test_temporal_attention_responds_to_session_context(self):
        layer = CausalTemporalSelfAttention(hidden_dim=8, n_heads=2, ff_multiplier=2, dropout=0.0)
        sequence = torch.randn(1, 4, 2, 8)
        valid_mask = torch.ones(1, 4, 2, dtype=torch.bool)
        session_a = torch.zeros(1, 4, dtype=torch.long)
        session_b = torch.full((1, 4), 4, dtype=torch.long)
        regime = torch.zeros(1, 4, 2, dtype=torch.float32)
        with torch.no_grad():
            out_a = layer(sequence, valid_mask, session_codes=session_a, regime_signal=regime)
            out_b = layer(sequence, valid_mask, session_codes=session_b, regime_signal=regime)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_temporal_attention_handles_all_invalid_sequences(self):
        layer = CausalTemporalSelfAttention(hidden_dim=8, n_heads=2, ff_multiplier=2, dropout=0.0)
        sequence = torch.randn(2, 5, 3, 8)
        valid_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
        with torch.no_grad():
            output = layer(sequence, valid_mask)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_scalp_gate_blocks_unready_open(self):
        probs = torch.tensor([0.65, 0.68, 0.32], dtype=torch.float32).numpy()
        labels = torch.tensor([1, 1, 0], dtype=torch.int32).numpy()
        returns = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32).numpy()
        baseline = run_fractional_kelly_backtest(probs, labels, returns, threshold=0.08)
        gated = run_fractional_kelly_backtest(
            probs,
            labels,
            returns,
            threshold=0.08,
            trade_session_mask=[1, 1, 1],
            open_ready=[0, 1, 1],
            momentum_score=[0.8, 0.8, 0.8],
            momentum_floor=0.35,
            volatility_ratio=[1.0, 1.0, 1.0],
            volatility_ratio_floor=0.75,
            volatility_ratio_cap=1.8,
            breakout_signed=[0.20, 0.20, -0.20],
            breakout_floor=0.05,
        )
        self.assertEqual(baseline.trade_count, 3)
        self.assertEqual(gated.trade_count, 2)

    def test_scalp_soft_gate_rescales_without_zeroing_trades(self):
        probs = torch.tensor([0.68, 0.68, 0.32], dtype=torch.float32).numpy()
        labels = torch.tensor([1, 1, 0], dtype=torch.int32).numpy()
        returns = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32).numpy()
        baseline = run_fractional_kelly_backtest(probs, labels, returns, threshold=0.08)
        softened = run_fractional_kelly_backtest(
            probs,
            labels,
            returns,
            threshold=0.08,
            trade_session_mask=[1, 1, 1],
            open_ready=[1, 1, 1],
            momentum_score=[0.12, 0.8, 0.8],
            momentum_floor=0.35,
            volatility_ratio=[0.4, 1.0, 1.0],
            volatility_ratio_floor=0.75,
            volatility_ratio_cap=1.8,
            breakout_signed=[0.03, 0.20, -0.20],
            breakout_floor=0.05,
        )
        self.assertEqual(softened.trade_count, baseline.trade_count)
        self.assertLess(softened.avg_fraction, baseline.avg_fraction)

    def test_scalp_hard_gate_can_still_block_weak_setup(self):
        probs = torch.tensor([0.68, 0.68, 0.32], dtype=torch.float32).numpy()
        labels = torch.tensor([1, 1, 0], dtype=torch.int32).numpy()
        returns = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32).numpy()
        gated = run_fractional_kelly_backtest(
            probs,
            labels,
            returns,
            threshold=0.08,
            trade_session_mask=[1, 1, 1],
            open_ready=[1, 1, 1],
            momentum_score=[0.12, 0.8, 0.8],
            momentum_floor=0.35,
            volatility_ratio=[0.4, 1.0, 1.0],
            volatility_ratio_floor=0.75,
            volatility_ratio_cap=1.8,
            breakout_signed=[0.03, 0.20, -0.20],
            breakout_floor=0.05,
            scalp_hard_gate=True,
        )
        self.assertEqual(gated.trade_count, 2)

    def test_laplacian_laggard_gate_requires_aligned_neighbor_pressure(self):
        probs = torch.tensor([0.68, 0.68, 0.32], dtype=torch.float32).numpy()
        labels = torch.tensor([1, 1, 0], dtype=torch.int32).numpy()
        returns = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32).numpy()
        gated = run_fractional_kelly_backtest(
            probs,
            labels,
            returns,
            threshold=0.08,
            lap_neighbor_momentum=[0.3, -0.2, -0.25],
            lap_laggard_score=[0.2, 0.2, 0.2],
            lap_laggard_floor=0.02,
            lap_allocation_scale=0.25,
            lap_hard_gate=True,
        )
        self.assertEqual(gated.trade_count, 2)

    def test_laplacian_soft_gate_keeps_trades_but_rescales(self):
        probs = torch.tensor([0.68, 0.68, 0.32], dtype=torch.float32).numpy()
        labels = torch.tensor([1, 1, 0], dtype=torch.int32).numpy()
        returns = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32).numpy()
        baseline = run_fractional_kelly_backtest(probs, labels, returns, threshold=0.08)
        softened = run_fractional_kelly_backtest(
            probs,
            labels,
            returns,
            threshold=0.08,
            lap_neighbor_momentum=[0.3, -0.2, -0.25],
            lap_laggard_score=[0.2, 0.1, 0.2],
            lap_laggard_floor=0.02,
            lap_allocation_scale=0.25,
            lap_alignment_scale=0.35,
            lap_hard_gate=False,
        )
        self.assertEqual(softened.trade_count, baseline.trade_count)
        self.assertLess(softened.avg_fraction, baseline.avg_fraction)

    def test_meta_classifier_caps_stage2_features(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(256, 8)).astype(np.float32)
        y = ((1.6 * x[:, 0]) + (1.2 * x[:, 1]) - (0.8 * x[:, 2]) > 0.0).astype(np.float32)
        matrix = MetaFeatureMatrix(
            X=x,
            feature_names=[
                "macro_prob",
                "micro_prob",
                "noise_0",
                "noise_1",
                "noise_2",
                "noise_3",
                "macro_emb_0",
                "micro_emb_0",
            ],
            references=[{"idx": idx} for idx in range(len(y))],
            y=y,
        )
        model = CatBoostMetaClassifier(iterations=16, depth=2, learning_rate=0.1, max_features=4)
        model.fit(matrix)
        self.assertEqual(model.fit_mode, "catboost")
        self.assertIsNotNone(model.selected_feature_indices)
        self.assertLessEqual(len(model.selected_feature_indices), 4)
        selected_names = {matrix.feature_names[int(idx)] for idx in model.selected_feature_indices.tolist()}
        self.assertIn("macro_prob", selected_names)
        self.assertIn("micro_prob", selected_names)

    def test_scalp_entry_labels_require_open_and_alignment(self):
        close = np.asarray([1.0000, 1.0010, 1.0020, 1.0040, 1.0055, 1.0065], dtype=np.float32)
        high = close + 0.0006
        low = close - 0.0004
        spread = np.full(len(close), 0.00008, dtype=np.float32)
        tick = np.asarray([100, 105, 110, 150, 155, 160], dtype=np.float32)
        real = np.ones(len(close), dtype=bool)
        sessions = np.full(len(close), SESSION_CODES["overlap"], dtype=np.int8)
        direction = np.ones(len(close), dtype=np.float32)
        valid = np.ones(len(close), dtype=bool)
        entry = _build_scalp_entry_labels(
            "EURUSD",
            close,
            high,
            low,
            spread,
            tick,
            real,
            sessions,
            direction,
            valid,
            opening_range_bars=3,
        )
        self.assertEqual(float(entry[0]), 0.0)
        self.assertEqual(float(entry[1]), 0.0)
        self.assertGreater(float(entry[3:].sum()), 0.0)

    def test_scalp_entry_labels_reject_bad_spread(self):
        close = np.asarray([1.0000, 1.0010, 1.0020, 1.0040, 1.0055, 1.0065], dtype=np.float32)
        high = close + 0.0006
        low = close - 0.0004
        spread = np.full(len(close), 1000.0, dtype=np.float32)
        tick = np.asarray([100, 105, 110, 150, 155, 160], dtype=np.float32)
        real = np.ones(len(close), dtype=bool)
        sessions = np.full(len(close), SESSION_CODES["overlap"], dtype=np.int8)
        direction = np.ones(len(close), dtype=np.float32)
        valid = np.ones(len(close), dtype=bool)
        entry = _build_scalp_entry_labels(
            "EURUSD",
            close,
            high,
            low,
            spread,
            tick,
            real,
            sessions,
            direction,
            valid,
            opening_range_bars=3,
        )
        self.assertEqual(float(entry.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
