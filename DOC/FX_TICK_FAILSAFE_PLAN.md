# Plan: Fail-Safe FX Tick Feature Generation

## Context

The `staged_v4` remote cache pipeline stalls at FX tick feature generation. 31/42 symbols complete, but ~11 signal-only symbols (XAGUSD, XTIUSD, UK100, EUSTX50, XBRUSD, AUS200, JPN225, XAUUSD, etc.) hang indefinitely. The pipeline never reaches bridge/manifest/training stages.

**Three compounding root causes identified:**

1. **Missing M5 fallback triggers tick-level TPO** -- Signal-only symbols that lack M5 panel data fall into the slow path at `_build_symbol_aux_features` line 137, which calls `compute_tpo_feature_panel` on millions of raw tick bars. Each bar calls the C extension `compute_tpo_memory_state` sequentially -- computationally infeasible.

2. **No timeout/cancellation in wait loop** -- `fx_features.py` line 315 `while pending:` polls with 30s timeout but never breaks, never cancels stuck futures, never gives up. One stuck symbol blocks the entire pipeline forever.

3. **ThreadPoolExecutor for CPU-bound work** -- The GIL serializes all threads when doing Python/C work. Even if the C extension releases the GIL, the thread model provides no isolation for truly stuck operations.

---

## Changes (in priority order)

### Change 1: Hard safeguard in `_build_symbol_aux_features` (fixes the hang)
**File:** `src/staged_v4/data/fx_features.py` lines 127-138

Add a `MAX_TPO_BARS = 500_000` guard. When `tpo_source_frame is None` AND `len(close_col) > MAX_TPO_BARS`, skip TPO computation entirely (leave zeros). This prevents the pathological tick-level TPO loop.

```python
MAX_TPO_BARS = 500_000

if valid_col.any():
    tpo_started = time.perf_counter()
    if tpo_source_frame is not None and tpo_lookup is not None:
        # fast path: M5-sourced TPO (existing code unchanged)
        ...
    elif len(close_col) <= MAX_TPO_BARS:
        node_tpo, node_vol = compute_tpo_feature_panel(high_col, low_col, close_col)
    # else: zero-fill (already initialized)
    tpo_duration = time.perf_counter() - tpo_started
```

Add `"tpo_skipped_large": bool` to diagnostics dict.

### Change 2: Per-symbol timeout + graceful failure in wait loop
**File:** `src/staged_v4/data/fx_features.py` lines 315-408

- Add `symbol_timeout_sec` parameter (default 300s for tick, 120s otherwise)
- In the wait loop, check elapsed time per pending future; if exceeded, cancel + zero-fill
- Change `except Exception: raise` (line 379) to `except Exception: continue` with logging
- Track `failed_symbols: list[str]` for summary logging

### Change 3: Global batch deadline
**File:** `src/staged_v4/data/fx_features.py` same wait loop

Add `batch_deadline = time.perf_counter() + 3600.0` (configurable). If exceeded, cancel all remaining futures, append to failed list, and `break`.

### Change 4: Defense-in-depth guard in TPO
**File:** `src/staged_v4/data/tpo_features.py` line 44

After `if n_bars == 0:`, add `if n_bars > 500_000: return features, volatility`. Belt-and-suspenders in case the fx_features safeguard is bypassed.

### Change 5: Summary logging for partial completion
**File:** `src/staged_v4/data/fx_features.py` end of `build_fx_timeframe_batch`

Log failed symbol count and names before returning. No changes to `TimeframeFeatureBatch` shape -- zeros are the natural default for skipped symbols.

### Change 6 (Optional): ProcessPoolExecutor for tick
**File:** `src/staged_v4/data/fx_features.py` line 248

Add `use_process_pool: bool = False` parameter. When True and timeframe is tick, use `ProcessPoolExecutor`. Requires `.copy()` on numpy array slices before submission. **Lower priority** since Change 1 eliminates the CPU-bound bottleneck entirely.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/staged_v4/data/fx_features.py` | Changes 1-3, 5, optionally 6 |
| `src/staged_v4/data/tpo_features.py` | Change 4 |
| `src/staged_v4/data/cache.py` | No changes needed (shard cleanup is already correct) |

## What Does NOT Change

- Output format (`TimeframeFeatureBatch` shape and fields)
- Shard resume capability (existing 31 shards still load)
- `_save_symbol_aux_shard` / `_load_symbol_aux_shard` format
- M5/M15/H1+ timeframe behavior (they use the fast path already)
- `cache.py` call site (line 303-311)

## Verification

1. Run existing tests: `python -m pytest src/test_staged_v4.py -v`
2. Quick smoke test: confirm `_build_symbol_aux_features` returns instantly when `close_col` has 1M+ rows and no M5 source
3. Integration: re-run the cache build for tick timeframe -- all 42 symbols should complete (31 from shards, ~11 with zero-filled TPO), producing `fx/tick.npz`
4. Verify M5 timeframe still computes real TPO features (regression check)

---

## Outcome (2026-04-03)

**Status: RESOLVED** — All changes implemented, cache completed, first full training run succeeded.

### What was applied
- Changes 1-5 all implemented in `fx_features.py`
- `_MAX_TPO_BARS = 500_000` guard + `tpo_skipped_large` diagnostic
- `symbol_timeout_sec` (default 300s for tick) + `batch_deadline_sec` (default 3600s)
- `_mark_failed_symbol()` helper for graceful zero-fill on timeout/error
- Partial completion summary logging

### First training result (March 2026 weekly fold, tick/M1/M5, 1 fold)

| Metric | staged_v4 | compact benchmark |
|--------|-----------|-------------------|
| AUC | **0.6675** | 0.5215 |
| Log loss | 0.5872 | — |
| ECE | 0.0436 | — |
| Dir. accuracy | 0.5993 | — |
| Sharpe | -8.23 | **7.92** |
| Win rate | 36.82% | **52.70%** |
| Trades | 13,179 | — |

### Diagnosis
The model has strong directional discrimination (AUC 0.6675 >> 0.5215) but the trade execution layer inverts it into losses. 13K trades with 36% win rate suggests:
- Entry threshold too loose (no probability calibration gate)
- Spread/cost not factored into the entry decision
- Signal is present but the execution layer needs calibration

### Next focus
Execution/calibration layer, not pipeline robustness. The cache infrastructure is now stable and fail-safe.
