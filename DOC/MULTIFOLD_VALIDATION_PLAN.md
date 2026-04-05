# Multi-Fold Validation Plan

## Goal

Validate the staged_v4 Sharpe +4.56 result across 2+ folds to get PBO, DSR, and fold dispersion.

## Current state

- Cache: `staged_v4_remote_cache_20260301_20260331_weekly` (March 2026, 4 weekly blocks)
- Current exec_v3 result: Sharpe +4.56 on fold 0 (train weeks 1-2, val week 3)
- Splits are baked into cache metadata with `outer_holdout_blocks=1` → only 1 fold
- Code now supports regenerating splits at training time via `_resolve_cached_splits()`

## Root cause of the runtime inconsistency

The code changes are correct but the remote run likely failed for one of these reasons:

### Most likely: stale `.pyc` on the remote box

The `[:-0]` fix in `dataset.py:370` changes `build_walkforward_splits` behavior. If the remote box has a cached `__pycache__/dataset.cpython-*.pyc` from the old code, Python will use the old bytecode where `unique_blocks[:-0]` returns `[]`, causing `build_walkforward_splits(outer_holdout_blocks=0)` to return an empty list. The fallback at `train_staged.py:318` then returns the cached 1-fold splits.

**Fix:** Always run with `python -B` (already done for v3) AND explicitly clear `__pycache__`:
```bash
find src/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
python -B src/staged_v4/training/train_staged.py ...
```

### Also possible: manifest missing `outer_holdout_blocks`

If the remote cache was built by older code (specifically `save_feature_batches` which doesn't write `outer_holdout_blocks` to manifest), then `split_meta["outer_holdout_blocks"]` is `None`. This triggers `needs_regen = True` (correct), but if the old `dataset.py` is in the pycache, regeneration produces `[]` and falls back to cached splits.

**Fix:** Same as above — clear pycache. The manifest issue resolves itself because `cached_outer_holdout is None` triggers regeneration regardless.

## Implementation: Diagnostic logging + clean rerun

### Change 1: Add split resolution logging

**File:** `src/staged_v4/training/train_staged.py`

Add explicit logging right after line 362 (after `_resolve_cached_splits` returns) and right after line 425 (after `max_folds` clipping):

```python
# After line 362:
logger.info(
    "state=splits_resolved n_splits=%d split_frequency=%s source=%s",
    len(splits), split_frequency,
    "regenerated" if splits is not cached_splits else "cached",
)

# After line 425 (after max_folds clipping):
logger.info(
    "state=splits_final n_splits=%d max_folds=%s",
    len(splits), max_folds,
)
```

This makes it unambiguous in the log whether regeneration happened and how many splits survived clipping.

### Change 2: Assert split count before fold loop

**File:** `src/staged_v4/training/train_staged.py`

Add after line 425:
```python
if len(splits) == 0:
    logger.error("state=no_splits_available outer_holdout_blocks=%d min_train_blocks=%d",
                 training_cfg.outer_holdout_blocks, training_cfg.min_train_blocks)
    raise ValueError("No walk-forward splits available with current settings")
```

### No changes to backtest.py, config.py, cache.py, or dataset.py

The split regeneration and `[:-0]` fix are already correct. The issue is purely runtime (stale bytecode).

## Rerun command

```bash
# On remote box:
cd /path/to/Algo-C2-Codex

# Step 1: Clear ALL pycache to eliminate stale bytecode
find src/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Step 2: Run with -B to prevent new pycache creation
python -B src/staged_v4/training/train_staged.py \
    --mode real \
    --cache-root "D:\work\Algo-C2-Codex\data\staged_v4_remote_cache_20260301_20260331_weekly" \
    --output "D:\work\Algo-C2-Codex\data\remote_runs\staged_v4_weekly_exec_v4\report.json" \
    --outer-holdout-blocks 0 \
    --min-train-blocks 2 \
    --max-folds 2 \
    --ga-generations 3 \
    --tradeable-tpo-only \
    --log-file "D:\work\Algo-C2-Codex\data\remote_runs\staged_v4_weekly_exec_v4\train.log" \
    --status-file "D:\work\Algo-C2-Codex\data\remote_runs\staged_v4_weekly_exec_v4\status.json"
```

## Expected output

```
state=splits_resolved n_splits=2 split_frequency=week source=regenerated
state=splits_final n_splits=2 max_folds=2
fold=1/2 state=start split={"fold": 0, "train_blocks": ["...week1", "...week2"], "val_block": "...week3"}
...
fold=2/2 state=start split={"fold": 1, "train_blocks": ["...week1", "...week2", "...week3"], "val_block": "...week4"}
...
```

Report should have:
- 2 entries in `fold_results`
- PBO computed (currently says "Insufficient folds for PBO (need >= 2)")
- DSR computed
- `fold_dispersion` showing Sharpe variance across folds

## What to look for in results

| Scenario | Meaning | Next step |
|----------|---------|-----------|
| Both folds Sharpe > 0 | Signal generalizes across weeks | Extend to multi-month cache for production validation |
| Fold 0 positive, Fold 1 negative | Week-specific edge or regime sensitivity | Check if week 4 had unusual macro events; consider longer training |
| Both folds Sharpe < 0 | Fold 0 v3 result was lucky | Re-examine execution parameters; possible overfit to week 3 |
| PBO < 0.40 | Low probability of backtest overfitting | Good sign for live deployment |
| DSR > 0 | Sharpe survives multiple testing deflation | Strong validation signal |
