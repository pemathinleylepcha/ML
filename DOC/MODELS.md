# Algo C2 — Models & Analysis

## Pipeline (3 steps — in order)

```
Step 1  process_fx_csv_35.py     CSV → JSON
Step 2  algo_c2_analysis.py      Report (indicators + Laplacian + ensembles)
Step 3  algo_c2_analysis.py      Outlier detection (runs after Step 2)
```

```bash
python process_fx_csv_35.py --input_dir ./csvs
python algo_c2_analysis.py --data algo_c2_5day_data.json --output results.json
python algo_c2_analysis.py --data algo_c2_5day_data.json --window 300 --step 5
```

---

## Decision tree ensemble suite (6 models)

| # | Model | Description | AUC (5-day) |
|---|-------|-------------|-------------|
| 1 | **HistGradientBoosting** | CatBoost-equivalent. Native NaN handling, ordered-like boosting, monotone constraints. **Production model.** | 0.476 |
| 2 | GradientBoosting | Classic GBDT. Interpretable via `staged_predict`. | 0.506 |
| 3 | RandomForest | Variance-reducing bagging, OOB score. | 0.495 |
| 4 | ExtraTrees | Extra randomised trees, fastest for high-dim. | 0.500 |
| 5 | AdaBoost | Adaptive boosting on shallow trees. Focuses on hard examples. | 0.496 |
| 6 | BaggingTrees | Confidence via predict_proba spread across estimators. OOB score. | 0.484 |

**Note on AUC ~0.50:** 1-minute FX return direction is not reliably predictable from a single bar. Near-random AUC is expected and correct — the value is in **feature importances** (which cross-pair signals matter) not in direct prediction.

### Model hyperparameters

```python
HistGradientBoostingClassifier(
    max_iter=200, max_depth=5, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.1,
    early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=15
)

GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=15
)

RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    max_features='sqrt', oob_score=True, n_jobs=-1
)
```

---

## Feature matrix (568 features)

Built from rolling windows (default 120 bars, step 15):

### Per-pair features (16 features × 35 pairs = 560)

| Feature | Description |
|---------|-------------|
| `rsi` | RSI(14) |
| `macd` | MACD line (pips) |
| `macd_hist` | MACD histogram |
| `bb_pct_b` | Bollinger %B position |
| `bb_bandwidth` | Bollinger bandwidth |
| `atr_pips` | ATR(14) in pips |
| `stoch_k` | Stochastic %K |
| `stoch_d` | Stochastic %D (smoothed) |
| `cci` | CCI(20) |
| `willr` | Williams %R(14) |
| `log_ret` | Log return |
| `tick_vel` | Tick count per bar |
| `spread_pip` | Mean spread in pips |
| `range_pips` | Intrabar range in pips |
| `body_ratio` | Candle body / range ratio |
| `mom5` | 5-bar momentum z-score |

### Graph features (8 features)

| Feature | Description |
|---------|-------------|
| `graph_residual_mean` | Mean Laplacian residual across all pairs |
| `graph_residual_std` | Residual spread |
| `graph_residual_max` | Largest absolute residual |
| `spectral_gap` | λ₂ — algebraic connectivity |
| `betti_h0` | β₀ proxy (near-zero eigenvalues) |
| `betti_h1` | β₁ proxy (mid-range eigenvalues) |
| `avg_correlation` | Mean |corr| across upper triangle |
| `laplacian_trace` | tr(L) |

---

## Bridge features (v1_5_c1 — 24 total)

Cross-learning features exchanged between BTC and FX subnets each bar.
See `src/bridge.py` and [CHANGELOG.md](CHANGELOG.md) for history.

### BTC → FX (9 features, gated)

| Feature | Description |
|---------|-------------|
| `btc_weekend_regime` | BTC regime at Sunday close (0=LOW_VOL..4=FRAGMENTED) |
| `btc_weekend_regime_shift` | Regime change Saturday→Sunday (−1/0/+1) |
| `btc_vol_percentile` | BTC ATR percentile over 30-day history |
| `btc_risk_sentiment` | 4h log return — risk-on/off proxy |
| `btc_spectral_gap_delta` | Δλ₂ over last 60 bars |
| `btc_tick_velocity_z` | Tick count z-score (60-bar rolling) |
| `btc_spread_z` | Spread z-score |
| `btc_h1_lifespan` | H1 topological feature lifespan |
| `btc_bridge_gate` | Conditional gate ∈ [0,1] |

### FX → BTC (6 features)

| Feature | Description |
|---------|-------------|
| `fx_session_phase` | Current FX session (0=pre_tokyo..5=closed) |
| `fx_regime_summary` | Mean log return across 7 major pairs |
| `fx_dxy_proxy` | Weighted USD composite (ICE basket approximation) |
| `fx_avg_spread_z` | Average spread z-score across all pairs |
| `fx_jpy_cross_mean_ret` | Mean return of JPY crosses (risk proxy) |
| `fx_spectral_gap` | λ₂ from FX-only subgraph |

### Shared (9 features — v1_5_c1, was 5 in v2_legacy)

| Feature | Added | Description |
|---------|-------|-------------|
| `graph_betti_0` | v2 | β₀ — connected components |
| `graph_betti_1` | v2 | β₁ — independent cycles |
| `graph_max_h1_life` | v2 | Max H1 persistence lifespan |
| `graph_spectral_gap` | v2 | λ₂ from full 43-node graph |
| `graph_regime` | v2 | Regime label (0=LOW_VOL..4=FRAGMENTED) |
| `prev_regime` | **v1_5_c1** | Regime at previous bar |
| `bars_in_regime` | **v1_5_c1** | `log1p(n)` bars in current regime |
| `session_transition` | **v1_5_c1** | 1.0 on first bar of new session |
| `regime_ema` | **v1_5_c1** | EWMA (α=0.1) smoothed regime |

---

## Top feature importances (5-day real data)

From ensemble aggregate (higher = more predictive of next EURUSD bar direction):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | AUDCAD_cci | 0.1759 |
| 2 | BTCUSD_bb_bandwidth | 0.0642 |
| 3 | XAUUSD_log_ret | 0.0589 |
| 4 | GBPCAD_log_ret | 0.0545 |
| 5 | US30_body_ratio | 0.0497 |
| 6 | XAUUSD_mom5 | 0.0436 |
| 7 | BTCUSD_spread_pip | 0.0367 |
| 8 | CHFJPY_tick_vel | 0.0363 |
| 9 | AUDNZD_rsi | 0.0363 |
| 10 | GBPCHF_body_ratio | 0.0355 |

**Interpretation:** AUDCAD CCI (17.6%), BTC volatility regime (6.4%), and gold log returns (5.9%) are the most discriminative cross-market signals. These are the features to prioritise in the production CatBoost model.

---

## Outlier detection (3 methods)

| Method | Type | What it catches |
|--------|------|-----------------|
| IsolationForest | Tree-based anomaly isolation | Global structural anomalies, regime breaks |
| LocalOutlierFactor | Density-based local anomaly | Localised spikes within dense regions |
| EllipticEnvelope (MCD) | Robust covariance on PCA | Multivariate distributional outliers |

**Consensus rule:** flagged by ≥2 of 3 detectors = consensus outlier  
**Contamination:** 2% per detector  
**Result on 5-day data:** 1 consensus outlier (0.3% of samples)

The consensus outlier timestamp is in `results.json → outliers.consensus.timestamps`. This is the most anomalous 1-minute bar structurally across the full 568-feature space — likely a macro release or liquidity event.

---

## Laplacian analysis results (Mar 2–6 2026)

```
Eigenspectrum (first 10):
λ1=0.000  λ2=0.899  λ3=0.968  λ4=0.977  λ5=0.994
λ6=0.999  λ7=1.019  λ8=1.022  λ9=1.026  λ10=1.032

Spectral gap (λ₂): 0.899   — strong algebraic connectivity
β₀ = 1                     — single connected component
β₁ = 34                    — dense cycle structure
Regime: TRENDING
```

**Top 10 Laplacian mispricings (last 60-bar window):**

| Rank | Pair | Residual ε | Signal |
|------|------|-----------|--------|
| 1 | NZDUSD | −0.001242 | SHORT |
| 2 | XBRUSD | −0.000546 | LONG |
| 3 | USDCAD | +0.000451 | FLAT |
| 4 | GBPUSD | +0.000406 | LONG |
| 5 | USDZAR | −0.000401 | FLAT |
| 6 | XAGUSD | +0.000362 | LONG |
| 7 | USDCHF | −0.000344 | SHORT |
| 8 | EURGBP | −0.000315 | SHORT |
| 9 | GBPCHF | +0.000285 | LONG |
| 10 | CHFJPY | −0.000273 | SHORT |

**Alpha interpretation:** NZDUSD ε=−0.00124 means NZD is significantly lagging what its correlated network peers (AUD, EUR, JPY crosses) predict. When a pair has a large residual AND an aligned indicator signal, this is the highest-conviction trade opportunity.
