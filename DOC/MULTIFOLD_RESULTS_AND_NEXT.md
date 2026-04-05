# Multi-Fold Validation Results & Next Steps

## exec_v13: Honest no-GA fixed-config run

Report: `data/remote_runs/staged_v4_weekly_exec_v13_cpu_fixed_noga_fg_report.json`

Config: `ga_generations=0`, fixed `threshold=0.57, vol_coeff=10.0, spread_threshold=0.10, max_group_exposure=2`

### Fold comparison

| Metric | Fold 0 (week 3) | Fold 1 (week 4) |
|--------|-----------------|-----------------|
| Train blocks | weeks 1-2 | weeks 1-3 |
| AUC | 0.6806 | **0.7287** |
| Dir accuracy | 0.6123 | **0.6477** |
| Sharpe | **-5.48** | **+11.13** |
| Trades | 3,592 | 1,066 |
| Win rate | 49.1% | 46.7% |
| Net return | **-20.91** | **+0.022** |
| avg_hold_bars | 1.86 | 1.90 |

### Threshold diagnostics

**Fold 0 (negative Sharpe):**
| Bucket | Count | Win rate | Avg return |
|--------|-------|----------|------------|
| 0.50-0.57 | 900 | 47.0% | **-0.00776** |
| 0.57-0.60 | 1,645 | 52.2% | **-0.00848** |
| 0.60-0.65 | 817 | 46.0% | +1.62e-5 |
| 0.65-0.70 | 230 | 46.5% | +1.79e-5 |

**Fold 1 (positive Sharpe):**
| Bucket | Count | Win rate | Avg return |
|--------|-------|----------|------------|
| 0.50-0.57 | 273 | 48.4% | +3.92e-5 |
| 0.57-0.60 | 342 | 46.2% | +2.05e-5 |
| 0.60-0.65 | 449 | 45.9% | +8.70e-6 |
| 0.65-0.70 | 2 | 100% | +2.17e-4 |

### Previous GA-tuned run (exec_v12) for comparison

| Metric | v12 (GA) fold 0 | v13 (no GA) fold 0 | v12 (GA) fold 1 | v13 (no GA) fold 1 |
|--------|-----------------|---------------------|-----------------|---------------------|
| Sharpe | +12.80 | **-5.48** | +17.31 | **+11.13** |
| Trades | 687 | 3,592 | 1,897 | 1,066 |
| max_group_exposure | **1** (GA) | 2 (fixed) | 2 (GA) | 2 (fixed) |
| threshold | 0.60 (GA) | 0.57 (fixed) | **0.55** (GA) | 0.57 (fixed) |

GA flipped fold 0 from -5.48 to +12.80 primarily by setting `max_group_exposure=1` (cutting trades from 3592 to 687). That is pure in-sample fitting.

---

## Diagnosis

### Why fold 0 fails

1. **Overtrading.** 3,592 trades on 7,199 bars = 1 trade every 2 bars. The 0.57 threshold lets in too many marginal signals. The 0.50-0.60 buckets account for 2,545 trades with deeply negative avg returns (-0.008). These trades are slightly better than coin-flip on direction but the asymmetric SL (0.7 ATR) makes losers bigger than winners.

2. **Undertrained model.** Only 2 weeks of training data. AUC 0.68 is weaker than fold 1's 0.73. The model's probability estimates in the 0.57-0.60 band don't carry enough precision to overcome transaction costs.

3. **Return magnitude mismatch.** Fold 0's 0.50-0.60 buckets show avg returns around -0.008 (large). Fold 1's same buckets show avg returns around +3e-5 (tiny positive). The asymmetric TP/SL interacts differently depending on signal quality — when the directional signal is weak, the 0.7 ATR stop eats more than the 1.0 ATR take-profit captures.

### Why fold 1 works

Fold 1 trains on 3 weeks and evaluates on 1 week (3338 bars). More training data → better AUC (0.73) → tighter probability distribution → fewer trades pass the same 0.57 threshold (1066 vs 3592). The model also has better calibration: fold 1's avg returns are small but consistently positive across all buckets.

### The honest state of the signal

- **AUC is real and improving with data:** 0.68 (2-week train) → 0.73 (3-week train)
- **The execution layer is fragile:** same fixed config produces Sharpe -5.48 and +11.13 depending on model quality
- **The 0.57 threshold is too loose for a 2-week model** but works for a 3-week model
- **GA was masking this fragility** by per-fold-tuning exposure limits

---

## What to optimize next

### Direction 1: Raise the threshold to be safe for the weakest model

The fold 0 bleed comes from the 0.50-0.60 band (2,545 trades, all negative). The 0.60+ band is breakeven-to-positive in both folds.

**Concrete change:** raise `base_entry_threshold` from 0.57 to 0.60.

Expected impact on fold 0:
- Eliminates the 0.50-0.57 bucket (900 trades, -0.008 avg return)
- Eliminates the 0.57-0.60 bucket (1,645 trades, -0.008 avg return)
- Keeps 0.60+ bucket (1,047 trades, ~breakeven to slightly positive)
- Should flip fold 0 from deep negative to roughly flat

Expected impact on fold 1:
- Eliminates 615 trades from the 0.50-0.60 range (already slightly positive)
- Keeps 451 trades in the 0.60+ range
- May reduce Sharpe slightly (fewer trades, less compounding)

**This is the single highest-leverage change.** It's not adding complexity — it's removing the worst trades.

### Direction 2: More training data

The fundamental issue is that 2 weeks isn't enough to train the STGNN. Fold 1 with 3 weeks is materially better. Building a Feb-Mar cache (8+ weeks) would give every fold at least 2-3 weeks of training while providing enough folds for real PBO.

But this is a data/infra step, not a code change. Park it until Direction 1 is validated.

### Direction 3: Adaptive threshold based on training data size

Instead of a fixed threshold, scale the entry threshold inversely with model confidence quality. Practically: if the calibration fold's ECE is above some cutoff, use a higher threshold. This makes the execution layer self-adjusting for model quality.

This is more complex and should come after Directions 1 and 2 prove the base signal.

### Direction 4: Tighten SL to reduce loss asymmetry

Fold 0's 0.57-0.60 bucket has 52.2% win rate but -0.008 avg return — winning more often than losing but losing bigger. The SL at 0.7 ATR is wider than the typical winning trade's capture. Options:
- Tighten SL from 0.7 to 0.5 ATR (but risk more premature stops)
- Keep SL at 0.7 but add a trailing stop after 1 ATR profit
- Widen TP from 1.0 to 1.2 ATR to compensate

Lower priority than Direction 1 — fixing the threshold eliminates these trades entirely.

---

## Recommended next run

```bash
python -B src/staged_v4/training/train_staged.py \
    --mode real \
    --cache-root "..." \
    --output ".../staged_v4_weekly_exec_v14/report.json" \
    --outer-holdout-blocks 0 \
    --min-train-blocks 2 \
    --max-folds 2 \
    --ga-generations 0 \
    --tradeable-tpo-only \
    ...
```

With `base_entry_threshold=0.60` in config.py (the only change from v13).

Expected: fold 0 improves from -5.48 to roughly flat or slightly positive, fold 1 stays positive but with fewer trades.

## Files to modify

| Step | File | Change |
|------|------|--------|
| Threshold raise | `src/staged_v4/config.py` | `base_entry_threshold: 0.57 -> 0.60` |

One line. No other changes.

---

## exec_v14: Threshold 0.60 no-GA result

Report: `data/remote_runs/staged_v4_weekly_exec_v14_cpu_fixed_noga_t060_report.json`

### Fold comparison (v13 vs v14)

| Metric | v13 fold 0 (t=0.57) | v14 fold 0 (t=0.60) | v13 fold 1 | v14 fold 1 |
|--------|---------------------|---------------------|------------|------------|
| Sharpe | -5.48 | **-4.47** | +11.13 | **+20.15** |
| Trades | 3,592 | 2,736 | 1,066 | 1,010 |
| Win rate | 49.1% | 48.4% | 46.7% | 48.9% |
| Net return | -20.91 | **-13.94** | +0.022 | **+0.048** |

### Fold 0 threshold diagnostics (v14)

| Bucket | Count | Win rate | Avg return |
|--------|-------|----------|------------|
| 0.50-0.57 | 666 | 46.7% | **-0.01050** |
| 0.57-0.60 | 542 | 48.7% | +2.16e-5 |
| 0.60-0.65 | 1,085 | 50.9% | **-0.00642** |
| 0.65-0.70 | 443 | 44.7% | +1.30e-5 |

### Diagnosis

The threshold raise helped marginally (Sharpe -5.48→-4.47) but fold 0 is still negative.

**Key insight:** The 0.50-0.57 bucket still has 666 trades despite `base_entry_threshold=0.60`. This is because the threshold diagnostics track `_trade_confidence = min(directional_conf, entry_conf)`, not raw directional probability. These trades passed the 0.60 directional threshold but the entry head's lower confidence pulls the combined metric below 0.57.

**The real problem is the 0.60-0.65 bucket:** 1,085 trades that passed the 0.60 threshold with -0.00642 avg return. These are legitimate threshold-passing trades that still lose money. No threshold raise will fix this — the 2-week model's directional signal at 0.60+ confidence doesn't carry enough precision to overcome transaction costs on fold 0's validation week.

### Conclusion

**This is a data problem, not a threshold problem.** The 2-week model (fold 0) produces AUC 0.68 — just not enough signal. The 3-week model (fold 1) produces AUC 0.73 and is consistently profitable at every confidence level. The fix is more training data.

### Next step: Feb-Mar cache build

Build a cache covering Feb 1 - Mar 31 2026 (8-9 weekly blocks). This gives:
- Every fold trains on 3+ weeks minimum (eliminating the 2-week undertraining problem)
- 6-7 validation folds with `min_train_blocks=3, outer_holdout_blocks=0`
- Meaningful PBO with 20+ CSCV combinations
- Honest assessment of whether the fold 1 signal generalizes

**Requirements:**
- Feb 2026 tick data on the remote box
- Feb 2026 candle data in DataExtractor format
- Cache build command with `--start 2026-02-01 --end 2026-03-31`

**No code changes needed** — the pipeline, backtest, and config are all ready. This is purely a data/infra step.
