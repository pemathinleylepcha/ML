# Algo C2 — Project Transfer Document
## Chat Session: Math Engine Refinement + Real Tick Dashboard

---

## 1. PROJECT OVERVIEW

**Algo C2** is a quantitative trading pipeline that ingests raw tick data from 35 financial instruments and transforms it into high-conviction trade signals using:

- **Normalized Graph Laplacian** — maps cross-asset correlation structure, generates local residual (alpha/mispricing signal)
- **Persistent Homology (TDA)** — measures the "shape" of market data via Betti numbers to detect regime changes
- **CatBoost (Ordered Boosting)** — final decision engine combining microstructure biases, Laplacian residuals, and topological regime features
- **6-Gate Signal Filter** — ensures all conditions are met before order placement

### Asset Universe (35 nodes)

**28 FX pairs (tradeable):**
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY

**7 Non-FX (graph anchors only, no orders placed):**
BTCUSD, US30, USDMXN, USDZAR, XAGUSD, XAUUSD, XBRUSD

### Data

- **Source:** Real tick data from broker feed
- **Period:** Mon 2 Mar – Fri 6 Mar 2026
- **Bars:** 7,175 (1-minute OHLC)
- **Fields per bar:** `dt` (timestamp), `o` (open), `h` (high), `l` (low), `c` (close), `sp` (spread in sub-pips), `tk` (tick count)
- **File:** `algo_c2_5day_data.json` (21.9 MB raw, 2.4 MB gzip compressed)
- **Alignment:** Forward-filled to master timeline; some pairs start later (US30 at 01:00, USDZAR at 04:00, XBRUSD at 03:00)

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Production | C++ / C# / Rx.NET | Sub-millisecond execution |
| Messaging | Kafka (partitioned by pair) | Tick ingestion |
| State | Redis (HSET by Pair:Timestamp) | Feature vector storage |
| Routing | RabbitMQ | Execution triggers |
| Research | Python / Polars / Ripser / CatBoost | Prototyping & training |
| Dashboard | Self-contained HTML + Chart.js | Monitoring & backtesting |

---

## 2. WHAT THIS CHAT ACCOMPLISHED

### 2.1 Comprehensive Audit

Identified **16 bugs, missing features, and architectural issues** across 4 subsystems:
- Mathematical engine (6 issues)
- CatBoost / feature engineering (5 issues)
- Backtesting framework (3 issues — was completely missing)
- Visualization / dashboard (2 issues)

### 2.2 Files Produced

| File | Description | Status |
|------|-------------|--------|
| `algo_c2_v2_real_tick.html` | **Primary deliverable.** Complete dashboard with real tick data embedded (3.2 MB), all 16 engine fixes, 6-gate filter, TDA regime detection, 6-tab UI | ✅ Complete |
| `algo_c2_v2_refined.html` | Synthetic-data version of the v2 engine (for testing without real data) | ✅ Complete |
| `algo_c2_math_engine_v2.js` | Standalone JS module with all math engine fixes, drop-in replaceable functions | ✅ Complete |
| `math_engine.py` | Python research implementation of the refined Laplacian + TDA engine | ✅ Complete |
| `feature_engine.py` | Python CatBoost feature engineering pipeline with all fixes | ✅ Complete |
| `backtester.py` | Walk-forward cross-validation framework (was completely missing) | ✅ Complete |
| `AUDIT_AND_REFINEMENTS.md` | Full audit document with all 16 issues, severity ratings, and fixes | ✅ Complete |
| `algo_c2_dashboard.jsx` | React version of the dashboard (alternative to HTML) | ✅ Complete |

### 2.3 The 16 Fixes Applied

#### Mathematical Engine

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **CRITICAL** | σ in Gaussian kernel was undefined/static — adjacency matrix weights were wrong | Adaptive σ via median heuristic, recomputed every bar from rolling distance matrix |
| 2 | **HIGH** | Mantegna distance produced NaN when corr=1.0 exactly (thin liquidity) | Epsilon regularization: `sqrt(max(1e-10, 2*(1-clip(corr,-1,1))))` |
| 3 | **HIGH** | Zero-degree nodes caused division-by-zero in residual computation | Guard clause: nodes with degree < ε get residual=0, flagged as disconnected |
| 4 | **MEDIUM** | Full eigendecomposition on hot path was O(N³) unnecessarily | Removed from hot path; only used for λ₂ computation every 5 bars |
| 5 | **MEDIUM** | TDA Betti numbers not computed at all in dashboard | Added β₀, β₁, H₁ lifespan via union-find on distance matrix + EMA smoothing (α=0.3) |
| 6 | **MEDIUM** | Spectral gap (λ₂) was missing — no regime transition warning | Added via Jacobi eigenvalue iteration; λ₂→0 means graph fragmentation imminent |

#### Signal Logic

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 7 | **CRITICAL** | Cross-asset leakage: neighbor returns at T used to compute signals for trading at T | Returns pushed into buffer AFTER Laplacian computation; correlation uses T-1 |
| 8 | **HIGH** | Residual dead zone too aggressive (0.05) — filtered 70%+ of signals | Lowered to 0.02 |
| 9 | **HIGH** | n_score had no temporal memory — noise spikes weighted same as persistent divergence | Residual streak tracking per pair; 3+ bars in same direction gets log₂ bonus |
| 10 | **MEDIUM** | No regime awareness — traded through flash crashes | Gate 6 blocks FRAGMENTED regime + spectral gap < 0.005 |

#### Order Placement

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 11 | **HIGH** | TP/SL ignored spread cost — could be unprofitable even on winning trades | Minimum SL = 3× spread, minimum TP = 5× spread |
| 12 | **MEDIUM** | No entry metadata for post-trade analysis | Each position tracks: entry residual, CB, regime, spectral gap, streak count |

#### Infrastructure

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 13 | **HIGH** | Correlation was static (from initial 5-day matrix) | Rolling 60-bar correlation matrix, recomputed per bar |
| 14 | **MEDIUM** | No return buffer | Ring buffer (Float64Array) with O(1) push per pair |
| 15 | **MEDIUM** | Adjacency/degree recomputed from scratch each time | Single-pass shared pipeline: corr → dist → σ → A → D → L → residuals + TDA |
| 16 | **LOW** | Distance matrix computed twice (for Laplacian and TDA) | Single computation shared between both |

---

## 3. ARCHITECTURE DETAILS

### 3.1 Signal Pipeline Flow (v2)

```
Tick Data (35 pairs)
    │
    ▼
Rolling Return Buffer (60-bar ring buffer per pair)
    │
    ▼  [FIX #7: returns pushed AFTER Laplacian]
Rolling Correlation Matrix (35×35, recomputed per bar)  [FIX #13]
    │
    ├──► Mantegna Distance [FIX #2: ε-regularized]
    │        │
    │        ├──► Adaptive σ [FIX #1: median heuristic]
    │        │
    │        ├──► Adjacency → Degree → Laplacian L [FIX #3: zero-degree guard]
    │        │        │
    │        │        ├──► Local Residuals (alpha signal)
    │        │        └──► Spectral Gap λ₂ [FIX #6: every 5 bars]
    │        │
    │        └──► Persistent Homology → β₀, β₁, H₁ [FIX #5: EMA smoothed]
    │                    │
    │                    └──► Regime Classification
    │
    ▼
Signal Features: ε + λ₂ + regime + streak [FIX #9]
    │
    ▼
6-Gate Filter [FIX #10: Gate 6 = regime safety]
  G1: Net vote ≥ 0.15
  G2: Avg CB ≥ threshold (adjustable 0.55–0.85)
  G3: EMA trend alignment
  G4: ATR within 0.4–2.4× average
  G5: High-CB consensus ≥ 50%
  G6: Regime ≠ FRAGMENTED AND λ₂ > 0.005  ← NEW
    │
    ▼
Order Placement (FX only, top 3 by n_score_v2)
  [FIX #11: spread-adjusted TP/SL]
  [FIX #12: entry metadata tracked]
```

### 3.2 Regime Classification (from TDA)

| Regime | Condition | Trading |
|--------|-----------|---------|
| LOW_VOL | β₁ ≤ 1 AND H₁ < 0.2 | ✅ Trade normally |
| NORMAL | Default | ✅ Trade normally |
| TRANSITIONAL | β₁ > 4 | ✅ Trade with caution |
| HIGH_STRESS | H₁ > 0.8 | ✅ Trade with caution |
| FRAGMENTED | β₀ > 20 | ❌ Gate 6 blocks all entries |

### 3.3 Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Signal** | Equity curve, net signal score (bar chart), cumulative PnL, spectral gap λ₂, key metrics (balance, PnL, trades, win rate, PF, max DD, spread cost, regime) |
| **Topology** | Laplacian residual heatmap (35 nodes), adaptive σ chart, disconnected nodes chart, TDA metrics (σ, λ₂, β₀, β₁, H₁ lifespan) |
| **Performance** | Win/loss doughnut, trade duration histogram, detailed metrics (expectancy, avg winner/loser, return %) |
| **Book** | Open positions table (with regime at entry), closed trades table (last 20) |
| **Optimisation** | 6-gate filter pass rates (individual gate performance), gate pass % over time, lot sizes |
| **Regime** | Color-coded regime timeline, β₀/β₁ charts over time, H₁ lifespan chart, regime-conditional performance table |

### 3.4 Data Format (Compressed JSON)

The real tick data is embedded in the HTML as gzip-compressed, base64-encoded JSON:

```json
{
  "dt": ["2026-03-02 00:05", "2026-03-02 00:06", ...],  // 7175 timestamps
  "pairs": ["AUDCAD", "AUDCHF", ...],                     // 35 pair names
  "d": {
    "AUDCAD": [[o,h,l,c,sp,tk], [o,h,l,c,sp,tk], ...],  // 7175 bars
    "AUDCHF": [[o,h,l,c,sp,tk], ...],
    ...
  }
}
```

Loading sequence: `atob()` → `DecompressionStream('gzip')` → `JSON.parse()` → sim starts.

---

## 4. KEY CONSTANTS & CONFIGURATION

```javascript
// Pip sizes
PIP_SZ = { EURUSD: 1e-4, USDJPY: 0.01, BTCUSD: 1, XAUUSD: 0.1, ... }

// Pip values in USD (per standard lot)
PIP_USD = { EURUSD: 10, USDJPY: 6.7, BTCUSD: 1, XAUUSD: 10, ... }

// Engine configuration
ROLLING_WINDOW = 60    // bars for correlation computation
MIN_WINDOW = 10        // minimum bars before Laplacian is valid
EPS = 1e-10            // numerical floor
TDA_EMA_ALPHA = 0.3    // Betti number smoothing
RESIDUAL_DEAD_ZONE = 0.02  // residuals below this are FLAT
SPECTRAL_GAP_WARN = 0.005  // λ₂ below this triggers Gate 6

// Trading configuration
INIT_BALANCE = 50      // USD
LEVERAGE = 500         // 1:500
CB_THR = 0.65          // CatBoost conviction threshold (adjustable)
RISK_PCT = 0.02        // 2% risk per trade
```

---

## 5. WHAT'S NEXT (Not yet done)

### 5.1 High Priority
- [ ] Wire real CatBoost model (currently using momentum-based CB proxy)
- [ ] Implement walk-forward backtesting on the real tick data (framework exists in `backtester.py`, needs integration)
- [ ] Add SHAP feature importance visualization
- [ ] Replace synthetic CB score with actual trained CatBoost predictions

### 5.2 Medium Priority
- [ ] Add rolling Sharpe ratio chart
- [ ] Add persistence diagram visualization (birth-death scatter)
- [ ] Implement 2-tier model architecture (universal + per-cluster)
- [ ] Add temporal decay features (rolling EMA z-scores across lookbacks)
- [ ] Session-aware pair whitelisting (e.g., JPY pairs active in Tokyo session)

### 5.3 Production Path
- [ ] Port math engine to C++ (Eigen for Laplacian, GUDHI for TDA)
- [ ] Implement Rx.NET tumbling window for real-time bar assembly
- [ ] Redis state management (HSET by Pair:Timestamp)
- [ ] CatBoost C++ API inference (ModelCalcerWrapper)
- [ ] FIX protocol order routing

---

## 6. REAL DATA STATISTICS (2–6 Mar 2026)

| Pair | Avg Spread (pip) | ATR14 (pip) | Return |
|------|-----------------|-------------|--------|
| EURUSD | 1.03 | 0.7 | -1.30% |
| GBPUSD | 1.70 | 0.9 | +0.04% |
| USDJPY | 1.99 | 1.1 | +1.03% |
| BTCUSD | 25.23 | 170.1 | +3.97% |
| XAUUSD | 4.05 | 509.2 | -2.94% |
| XAGUSD | 9.18 | 105.4 | -11.39% |
| XBRUSD | 11.18 | 34.0 | +16.60% |
| US30 | 6.15 | 136.5 | -2.06% |
| USDZAR | 207.93 | 180.1 | +3.36% |

Full 35-pair stats available in the audit.

---

## 7. FILE LOCATIONS

All output files were saved to `/mnt/user-data/outputs/`:

```
algo_c2_v2_real_tick.html    — 3.2 MB  (main dashboard with real data)
algo_c2_v2_refined.html      — 47 KB   (synthetic data version)
algo_c2_math_engine_v2.js    — 18 KB   (standalone math engine module)
algo_c2_dashboard.jsx        — 22 KB   (React version)
math_engine.py               — 9 KB    (Python research engine)
feature_engine.py             — 11 KB   (Python feature pipeline)
backtester.py                — 14 KB   (Walk-forward framework)
AUDIT_AND_REFINEMENTS.md     — 8 KB    (Full audit document)
```

---

*Generated from Claude chat session — Algo C2 v2 refinement, March 2026*
