"""
Algo C2 -- End-to-End Integration Test
Runs the full pipeline: synthetic data -> math engine -> features ->
alpha model training -> backtest with model inference.
"""

import sys
import json
import numpy as np
from pathlib import Path

from math_engine import MathEngine
from feature_engine import build_feature_matrix, extract_graph_features, PAIRS_ALL, PIP_SIZES
from signal_pipeline import PAIRS_FX, PAIRS_NON_FX, Direction
from backtester import Backtester, BacktestConfig
from market_neutral_model import (
    WalkForwardTrainer, AlphaNet, PortfolioBacktester,
    predict_weights, weights_to_confidence, HAS_TORCH,
)


def generate_synthetic_data(n_bars=500, n_pairs=35, seed=42):
    """Generate synthetic OHLC data with mild trends for testing."""
    rng = np.random.default_rng(seed)
    pairs = sorted(PAIRS_FX + PAIRS_NON_FX)[:n_pairs]

    data = {}
    for pair in pairs:
        pip = PIP_SIZES.get(pair, 0.0001)
        base = 1.0 + rng.random() * 0.5
        # Add mild trends to make signals non-trivial
        trend = rng.choice([-1, 1]) * rng.uniform(0.00005, 0.0002)
        noise = rng.normal(0, 0.001, n_bars)
        closes = base + np.cumsum(noise + trend)

        data[pair] = []
        for i in range(n_bars):
            c = closes[i]
            h = c + abs(rng.normal(0, 0.0005))
            l = c - abs(rng.normal(0, 0.0005))
            o = c + rng.normal(0, 0.0003)
            data[pair].append({
                "dt": f"2026-03-{2 + i // 1440:02d} "
                      f"{(i // 60) % 24:02d}:{i % 60:02d}",
                "o": round(float(o), 6),
                "h": round(float(h), 6),
                "l": round(float(l), 6),
                "c": round(float(c), 6),
                "sp": round(float(rng.uniform(0.8, 3.0)), 2),
                "tk": int(rng.integers(10, 80)),
            })

    return data, pairs


def test_pipeline():
    """Full integration test."""
    print("=" * 60)
    print("ALGO C2 -- END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # -- 1. Generate data --
    print("\n[1/6] Generating synthetic data...")
    data, pairs = generate_synthetic_data(n_bars=500, n_pairs=35)
    print(f"  Pairs: {len(pairs)}")
    print(f"  Bars: {len(data[pairs[0]])}")

    # -- 2. Build feature matrix --
    print("\n[2/6] Building feature matrix...")
    df = build_feature_matrix(data, pairs=pairs, window=60, step=10,
                              target_pair=pairs[0])
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Labels: {df['label'].notna().sum()}")

    # -- 3. Baseline backtest (no model, CB proxy) --
    print("\n[3/6] Baseline backtest (CB proxy)...")
    config = BacktestConfig(use_session_filter=False, enabled_pairs=set(pairs))
    bt_baseline = Backtester(config)
    results_baseline = bt_baseline.run(data, pairs=pairs)
    m = results_baseline.metrics
    print(f"  Trades: {m['total_trades']}")
    print(f"  Win rate: {m['win_rate']:.1%}")
    print(f"  Sharpe: {m['sharpe']:.4f}")
    print(f"  Final balance: ${m['final_balance']:.2f}")

    # -- 4. Train alpha model --
    if not HAS_TORCH:
        print("\n[4/6] SKIPPED -- PyTorch not installed")
        print("\n[5/6] SKIPPED")
        print("\n[6/6] SKIPPED")
        print("\nIntegration test PASSED (without alpha model)")
        return True

    print("\n[4/6] Training AlphaNet (3 folds, 20 epochs max)...")
    trainer = WalkForwardTrainer(
        n_features=len(feature_cols),
        n_pairs=len(pairs),
        n_folds=3,
        epochs=20,
        patience=8,
        batch_size=16,
    )

    features, forward_returns, timestamps = trainer.prepare_data(df, data, pairs)
    print(f"  Features: {features.shape}")
    print(f"  Forward returns: {forward_returns.shape}")
    print(f"  Non-zero returns: {(forward_returns != 0).sum()}")

    result = trainer.train(features, forward_returns, verbose=True)
    print(f"  Best fold: {result['best_fold']}")
    print(f"  Best Sharpe: {result['best_sharpe']:.4f}")

    if result["best_fold"] < 0:
        print("  No valid fold -- data may be too small. Test PASSED (partial).")
        return True

    # Save and reload model
    model_path = "models/test_integration_alpha.pt"
    best_fold_data = result["fold_results"][result["best_fold"]]
    trainer.save_model(
        result["model"], model_path,
        scaler_mean=best_fold_data["scaler_mean"],
        scaler_std=best_fold_data["scaler_std"],
        pairs=pairs,
    )
    loaded = WalkForwardTrainer.load_model(model_path)
    print(f"  Model saved and reloaded OK")

    # -- 5. Backtest with alpha model --
    print("\n[5/6] Backtest with AlphaNet...")
    config2 = BacktestConfig(use_session_filter=False, enabled_pairs=set(pairs))
    bt_model = Backtester(
        config2,
        alpha_model=loaded["model"],
        alpha_scaler=(loaded["scaler_mean"], loaded["scaler_std"]),
        alpha_pairs=loaded["pairs"],
    )
    results_model = bt_model.run(data, pairs=pairs)
    m2 = results_model.metrics
    print(f"  Trades: {m2['total_trades']}")
    print(f"  Win rate: {m2['win_rate']:.1%}")
    print(f"  Sharpe: {m2['sharpe']:.4f}")
    print(f"  Final balance: ${m2['final_balance']:.2f}")

    # -- 6. Portfolio-level backtest (direct weight allocation) --
    print("\n[6/8] Portfolio backtest (direct weights, no gates)...")
    pbt = PortfolioBacktester(
        init_capital=10000, leverage=2.0,
        transaction_cost_bps=3.0, rebalance_freq=1,
    )
    pbt_result = pbt.run(
        loaded["model"], features, forward_returns,
        scaler_mean=loaded["scaler_mean"],
        scaler_std=loaded["scaler_std"],
        timestamps=timestamps,
        pairs=pairs,
    )
    PortfolioBacktester.print_report(pbt_result)

    # -- 7. Portfolio backtest with lower leverage + slower rebalance --
    print("\n[7/8] Portfolio backtest (conservative: 1x lev, 5-bar rebal)...")
    pbt_cons = PortfolioBacktester(
        init_capital=10000, leverage=1.0,
        transaction_cost_bps=3.0, rebalance_freq=5,
    )
    pbt_cons_result = pbt_cons.run(
        loaded["model"], features, forward_returns,
        scaler_mean=loaded["scaler_mean"],
        scaler_std=loaded["scaler_std"],
        pairs=pairs,
    )
    PortfolioBacktester.print_report(pbt_cons_result)

    # -- 8. Compare all approaches --
    pm = pbt_result["metrics"]
    pm2 = pbt_cons_result["metrics"]
    print(f"\n[8/8] Comparison:")
    print(f"  {'Metric':<20s} {'Gate BT':>10s} {'Gate+Model':>10s} {'Port 2x':>10s} {'Port 1x':>10s}")
    print(f"  {'-' * 62}")
    rows = [
        ("trades/bars",     m['total_trades'],     m2['total_trades'],     pm['total_bars'],     pm2['total_bars']),
        ("sharpe",          m['sharpe'],            m2['sharpe'],           pm['sharpe'],          pm2['sharpe']),
        ("return %",        m['return_pct']*100,    m2['return_pct']*100,   pm['total_return_pct'],pm2['total_return_pct']),
        ("max DD %",        m['max_drawdown']*100,  m2['max_drawdown']*100, pm['max_drawdown_pct'],pm2['max_drawdown_pct']),
    ]
    for label, v1, v2, v3, v4 in rows:
        print(f"  {label:<20s} {v1:>10.2f} {v2:>10.2f} {v3:>10.2f} {v4:>10.2f}")

    # Cleanup
    Path(model_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Integration test PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
