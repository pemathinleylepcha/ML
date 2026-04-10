from __future__ import annotations

import unittest

import numpy as np

from tpo_normal_layer import build_tpo_normal_decision, compute_tpo_memory_state, compute_tpo_profile


class TPONormalLayerTests(unittest.TestCase):
    def test_compute_tpo_profile_returns_reasonable_bounds(self) -> None:
        close = np.linspace(1.0950, 1.1050, 60, dtype=np.float64)
        high = close + 0.0008
        low = close - 0.0008
        profile = compute_tpo_profile(high=high, low=low, close=close, atr_price=0.0015, lookback=48, n_bins=16)

        self.assertGreaterEqual(profile.value_area_high, profile.value_area_low)
        self.assertGreater(profile.profile_high, profile.profile_low)
        self.assertGreaterEqual(profile.poc, profile.profile_low)
        self.assertLessEqual(profile.poc, profile.profile_high)

    def test_tpo_normal_buy_signal_outside_value_area(self) -> None:
        base = np.linspace(1.1000, 1.1015, 50, dtype=np.float64)
        tail = np.array([1.0987, 1.0985, 1.0986, 1.0989, 1.0992, 1.0997], dtype=np.float64)
        close = np.concatenate([base, tail])
        high = close + 0.0007
        low = close - 0.0007

        decision = build_tpo_normal_decision(
            close=close,
            high=high,
            low=low,
            atr_price=0.0012,
            spread_price=0.00008,
            legacy_direction=0,
            legacy_confidence=0.05,
            legacy_p_buy=0.52,
            legacy_p_sell=0.48,
        )

        self.assertEqual(decision.direction, 1)
        self.assertGreater(decision.confidence, 0.0)
        self.assertFalse(decision.protector_blocked)
        self.assertGreaterEqual(len(decision.memory.profiles), 2)
        self.assertGreater(decision.memory.support_score, 0.0)

    def test_legacy_protector_blocks_when_conflict_and_high_confidence(self) -> None:
        close = np.concatenate(
            [
                np.linspace(1.1000, 1.1010, 50, dtype=np.float64),
                np.array([1.1018, 1.1022, 1.1025, 1.1021, 1.1018, 1.1015], dtype=np.float64),
            ]
        )
        high = close + 0.0006
        low = close - 0.0006

        decision = build_tpo_normal_decision(
            close=close,
            high=high,
            low=low,
            atr_price=0.0011,
            spread_price=0.00008,
            legacy_direction=1,
            legacy_confidence=0.20,
            legacy_p_buy=0.60,
            legacy_p_sell=0.40,
        )

        self.assertTrue(decision.protector_blocked)
        self.assertEqual(decision.direction, 0)

    def test_memory_state_builds_multiple_profile_levels(self) -> None:
        close = np.concatenate(
            [
                np.linspace(1.1000, 1.1030, 120, dtype=np.float64),
                np.linspace(1.1030, 1.1012, 40, dtype=np.float64),
            ]
        )
        high = close + 0.0006
        low = close - 0.0006

        memory = compute_tpo_memory_state(
            high=high,
            low=low,
            close=close,
            atr_price=0.0012,
            lookbacks=(24, 48, 96, 144),
            n_bins=24,
        )

        self.assertGreaterEqual(len(memory.profiles), 3)
        self.assertGreaterEqual(memory.composite_profile.value_area_high, memory.composite_profile.value_area_low)
        self.assertGreaterEqual(memory.value_area_overlap, 0.0)
        self.assertLessEqual(memory.value_area_overlap, 1.0)

    def test_memory_state_marks_zero_atr_as_degenerate(self) -> None:
        close = np.linspace(1.1000, 1.1010, 64, dtype=np.float64)
        high = close + 0.0004
        low = close - 0.0004

        memory = compute_tpo_memory_state(high=high, low=low, close=close, atr_price=0.0)

        self.assertTrue(memory.degenerate)
        self.assertTrue(memory.composite_profile.degenerate)
        self.assertEqual(memory.composite_profile.distance_to_poc_atr, 0.0)
        self.assertEqual(memory.composite_profile.value_area_width_atr, 0.0)
        self.assertEqual(memory.support_score, 0.0)
        self.assertEqual(memory.resistance_score, 0.0)
        self.assertEqual(memory.rejection_score, 0.0)
        self.assertEqual(memory.poc_drift_atr, 0.0)
        self.assertEqual(memory.value_area_overlap, 0.0)

    def test_memory_state_marks_tiny_atr_as_degenerate(self) -> None:
        close = np.linspace(1.1000, 1.1010, 64, dtype=np.float64)
        high = close + 0.0004
        low = close - 0.0004

        memory = compute_tpo_memory_state(high=high, low=low, close=close, atr_price=1e-10)

        self.assertTrue(memory.degenerate)
        self.assertTrue(memory.composite_profile.degenerate)
        self.assertEqual(memory.composite_profile.distance_to_poc_atr, 0.0)
        self.assertEqual(memory.composite_profile.value_area_width_atr, 0.0)


if __name__ == "__main__":
    unittest.main()
