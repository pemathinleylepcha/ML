# Algo C2 — Architecture Critique & Gap Analysis

External design review. 11 items — each tracked against current implementation.
Referenced in code as `critics.md item N`.
Item 11 added from GPT session-regime analysis (2026-03-27).

---

## Item 1 — Validity mask

**Issue:** Training should skip stale/forward-filled bars (weekends, gaps, holidays).
**Status:** ✅ Implemented — `real` bool mask in `build_feature_matrices()`. Skips bars where `is_stale_fill=True`. BTC 70.8% real bars, FX 50.9%.
**Location:** `src/train_catboost_v2.py` — `build_feature_matrices()`, validity mask section.

---

## Item 2 — Feature budget cap

**Issue:** Unconstrained features → noise, slower inference. Recommended BTC: 15–25, FX: 25–40.
**Status:** ✅ Implemented — `select_top_features()` with `--btc-top-k 20 --fx-top-k 40`. Uses PredictionValuesChange importance from a quick 100-iter selector model.
**Location:** `src/train_catboost_v2.py` — `select_top_features()` + `[1b/4]` step.
**Scores (v2_legacy):** BTC 37→20, FX 140→40. Top BTC feature: `lap_residual` (71.4%).

---

## Item 3 — Net edge labels (cost-adjusted)

**Issue:** Labels based on gross log return include spread + slippage, overstating edge.
**Status:** ✅ Implemented — `make_labels_quantile()` deducts spread from log return before ranking. Formula: `adj_ret = log_ret - spread_cost`.
**Location:** `src/train_catboost_v2.py` — `make_labels_quantile()`, "Labels should reflect net P&L" comment.

---

## Item 4 — Conditional BTC→FX bridge gate

**Issue:** BTC signals are noise during closed FX sessions and fragmented BTC regimes.
**Status:** ✅ Implemented — `_compute_bridge_gate()` returns 0–1 scalar. Zero on FX weekend/closed, 0.3 on FRAGMENTED regime, 0.7 on HIGH_STRESS.
**Location:** `src/bridge.py` — `BridgeComputer._compute_bridge_gate()`.

---

## Item 5 — Rolling correlation stabilization

**Issue:** Raw correlation matrices are rank-deficient and noisy with short windows.
**Status:** ✅ Implemented — Ledoit-Wolf style shrinkage: `0.85 * corr + 0.15 * I`.
**Location:** `src/math_engine.py:210` — `corr_shrinkage` parameter.

---

## Item 6 — Feature pruning (SHAP / MI)

**Issue:** PredictionValuesChange is a fast proxy but misses interaction effects. SHAP or mutual information would give better rankings.
**Status:** ⚠️ Partial — Using PredictionValuesChange (fast, built-in CatBoost). SHAP not implemented — adds significant compute time.
**Gap:** Could switch to `shap.TreeExplainer` on the selector model for better ranking. Low priority until FX WFE improves.

---

## Item 7 — Purged CV windows (leakage prevention)

**Issue:** Walk-forward folds with adjacent train/val boundaries leak label information when horizon > 1 bar.
**Status:** ✅ Implemented — `purge_bars` drops training rows within `N` bars of the validation window start.
**Location:** `src/train_catboost_v2.py` — `walk_forward_cv()`, purge section.
**Note:** v2_legacy used default `purge_bars=0` (gap). v1_5_c1 uses `--purge-bars 5` (= label horizon, correct).

---

## Item 8 — Hysteresis (entry > exit threshold)

**Issue:** Using the same threshold for entry and exit creates excessive churn — every bar re-evaluates.
**Status:** ✅ Implemented — `entry_threshold=0.45, exit_threshold=0.35` (now 0.40/0.30 in live). `strategy_sharpe_sized()` applies hysteresis filter during CV evaluation.
**Location:** `src/train_catboost_v2.py` — `strategy_sharpe_sized()`. Live: `src/live_mt5.py` — `ENTRY_THRESH / EXIT_THRESH`.

---

## Item 9 — Dynamic position sizing

**Issue:** Fixed lot size ignores signal confidence and regime — equal risk on high/low confidence trades.
**Status:** ❌ Not implemented — `live_mt5.py` uses fixed lot size. `sig.size` is computed but not scaled by regime.
**Gap:** Formula proposed: `lot = base_risk × confidence × regime_scale`. Requires defining `base_risk` per pair and `regime_scale` lookup.
**Priority:** Medium — affects P&L variance more than raw edge.

---

## Item 10 — Calibration metrics

**Issue:** CatBoost raw probabilities are not calibrated — 0.6 P(BUY) does not mean 60% win rate.
**Status:** ⚠️ Measured, not corrected — calibration report (ECE/Brier/log-loss) runs after each CV fold and prints curves. No Platt/Isotonic correction applied.
**Scores (v2_legacy):** BTC ECE=0.32 (poor), FX ECE=0.09 (acceptable).
**Gap:** Planned in `v1_5_c2` — fit `CalibratedClassifierCV(method='isotonic')` on held-out CV probabilities, wrap the CatBoost model.
**Location:** `src/calibration.py` — report only. `src/train_catboost_v2.py` — printed but not applied.

---

## Item 11 — Session-aware regime memory

**Source:** GPT analysis (2026-03-27) — *"Your model has regime detection, but no regime memory across sessions, so it resets behavior instead of adapting."*

**Problem identified:**
```
Tokyo  → LOW_VOL
London → HIGH_VOL
Model treats each bar independently
→ No continuity, weak session transitions, poor timing at session boundaries
Markets don't reset every bar — they evolve across sessions.
```

**GPT's 6 fixes:**

| # | Fix | Status |
|---|-----|--------|
| 1 | Session-aware regime state — carry forward last regime across sessions | ✅ `_prev_regime` in `BridgeComputer` |
| 2 | `prev_regime` feature — previous bar's regime as model input | ✅ Added to `shared_features()` |
| 3 | `bars_in_regime` — regime duration as persistence signal | ✅ `log1p(n)` in `shared_features()` |
| 4 | Session transition flag — 1.0 on first bar of new session | ✅ `session_transition` in `shared_features()` |
| 5 | Regime transition probability — EWMA instead of instant switch | ✅ `regime_ema` (α=0.1) in `shared_features()` |
| 6 | Weight signals by session-specific regime confidence | ❌ Not implemented — bridge_gate is session-weighted but not per-regime-within-session |

**Status:** ⚠️ 5/6 fixes implemented in `v1_5_c1` (bridge.py). Fix 6 (per-session regime confidence weighting) is a gap.
**Location:** `src/bridge.py` — `BridgeState.shared_features()`, `BridgeComputer.__init__()` + `update()`.
**Version:** v1_5_c1 — 4 new shared features, bridge feature count 20 → 24. Requires retraining to activate.
**Gap (fix 6):** Would need a `session_regime_confidence` lookup — e.g. London HIGH_STRESS gets higher gate weight than Tokyo HIGH_STRESS. Could fold into `_compute_bridge_gate()` as a 2D table `(session, regime) → weight`. Planned `v1_5_c2`.

---

## Summary table

| # | Item | Status | Version fixed / planned |
|---|------|--------|------------------------|
| 1 | Validity mask | ✅ Done | v2_legacy |
| 2 | Feature budget | ✅ Done | v2_legacy |
| 3 | Net edge labels | ✅ Done | v2_legacy |
| 4 | Bridge gate | ✅ Done | v2_legacy |
| 5 | Corr shrinkage | ✅ Done | v2_legacy |
| 6 | SHAP pruning | ⚠️ Partial | — (low priority) |
| 7 | Purged CV | ✅ Fixed | v2_legacy=gap(0), v1_5_c1=fixed(5) |
| 8 | Hysteresis | ✅ Done | v2_legacy |
| 9 | Dynamic sizing | ❌ Missing | planned v1_5_c2 |
| 10 | Calibration | ⚠️ Measured only | planned v1_5_c2 |
| 11 | Session regime memory | ⚠️ 5/6 done | v1_5_c1 (fix 6 gap → v1_5_c2) |
