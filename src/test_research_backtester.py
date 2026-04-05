from __future__ import annotations

import unittest

import numpy as np

from research_backtester import run_probability_backtest


class ResearchBacktesterTests(unittest.TestCase):
    def test_backtester_reports_trade_win_rate(self):
        metrics = run_probability_backtest(
            p_buy=np.array([0.75, 0.75, 0.50, 0.75, 0.50], dtype=np.float32),
            p_sell=np.array([0.25, 0.25, 0.50, 0.25, 0.50], dtype=np.float32),
            forward_returns=np.array([0.01, 0.01, 0.0, -0.02, 0.0], dtype=np.float32),
            regime_codes=np.ones(5, dtype=np.float32),
            session_codes=np.ones(5, dtype=np.int8),
            confidence_threshold=0.12,
            persistence_threshold=0.95,
            point_lookback=1,
        )

        self.assertEqual(metrics.trade_count, 2)
        self.assertAlmostEqual(metrics.win_rate, 0.5, places=6)

    def test_persistent_signal_trails_tp_and_sl_forward(self):
        common_kwargs = {
            "p_buy": np.array([0.75, 0.75, 0.75, 0.40], dtype=np.float32),
            "p_sell": np.array([0.25, 0.25, 0.25, 0.60], dtype=np.float32),
            "forward_returns": np.array([0.01, 0.01, 0.01, 0.0], dtype=np.float32),
            "regime_codes": np.ones(4, dtype=np.float32),
            "session_codes": np.ones(4, dtype=np.int8),
            "confidence_threshold": 0.12,
            "point_lookback": 1,
        }
        no_trail = run_probability_backtest(
            persistence_threshold=0.95,
            **common_kwargs,
        )
        with_trail = run_probability_backtest(
            persistence_threshold=0.64,
            **common_kwargs,
        )

        self.assertEqual(no_trail.trade_count, 2)
        self.assertEqual(with_trail.trade_count, 1)
        self.assertGreaterEqual(with_trail.net_return, no_trail.net_return)
        self.assertAlmostEqual(with_trail.win_rate, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
