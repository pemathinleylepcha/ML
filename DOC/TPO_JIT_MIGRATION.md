# TPO JIT Migration: Pre-Sweep at Panel Load

Date: 2026-04-05

## Problem

TPO feature computation is the dominant cost in both the cache and JIT training paths.

**Cache path** (`fx_features.py:107-146`):
- Computes TPO once per symbol per timeframe, stores in `.npz` shards
- Upfront cost: hours for full symbol set, hangs on tick (>500K bars), requires `_MAX_TPO_BARS` guard
- Once cached, training is fast — but the cache build itself is the blocker

**JIT path** (`jit_sequences.py:321-331`):
- Recomputes TPO per minibatch × per timeframe × per symbol
- Each FX batch: 42 symbols × ~300 bars × 4 lookbacks (24, 48, 96, 192)
- With batch_size=8, ~1000 anchors, 11 timeframes: **~6.3M profile computations per timeframe per epoch**
- This makes JIT training unacceptably slow

**Root cause**: `compute_tpo_feature_panel` (`tpo_features.py:71-101`) loops every bar sequentially, calling `compute_tpo_memory_state` which builds 4 histograms per bar. The per-bar cost is O(max_lookback). TPO depends only on price history — not on the model, training split, or batch — so recomputing it per batch is pure waste.

---

## Approach A: Columnar Pre-Sweep at Panel Load (Primary — Implement This)

### Concept

Compute TPO features **once per symbol per timeframe** immediately after panel loading, before any training loop runs. Store the results alongside the panel data. At batch construction time, replace the TPO computation with a simple array slice.

This mirrors how every other feature (OHLC, returns, ATR, session codes) is already handled.

### Detailed changes

#### Step 1: Add `tpo_panels` field to `StagedPanels`

**File:** `src/staged_v4/data/dataset.py` line 22-31

Add a new field to the `StagedPanels` dataclass:

```python
@dataclass(slots=True)
class StagedPanels:
    subnet_name: str
    symbols: tuple[str, ...]
    anchor_timeframe: str
    panels: dict[str, dict[str, pd.DataFrame]]
    anchor_timestamps: np.ndarray
    anchor_lookup: dict[str, np.ndarray]
    walkforward_splits: list[dict[str, object]]
    split_frequency: str
    tpo_panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    # tpo_panels[timeframe][symbol] = (tpo_features: ndarray[n_bars, 8], volatility: ndarray[n_bars])
```

The field holds pre-computed TPO features and volatility arrays, keyed by `[source_timeframe][symbol]`.

#### Step 2: Add `_presweep_tpo` helper function

**File:** `src/staged_v4/data/dataset.py` — new function, add near end of file before `load_staged_panels`

```python
def _presweep_tpo(
    panels: dict[str, dict[str, pd.DataFrame]],
    symbols: tuple[str, ...],
    requested_timeframes: tuple[str, ...],
    logger=None,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Compute TPO features once per source timeframe per symbol.

    Uses TPO_SOURCE_TIMEFRAME to determine which timeframe provides TPO
    data for each requested timeframe. Deduplicates so each source is
    only computed once even if multiple target timeframes map to it.

    Returns:
        tpo_panels[source_timeframe][symbol] = (features[n_bars, 8], volatility[n_bars])
    """
    from staged_v4.config import TPO_SOURCE_TIMEFRAME
    from staged_v4.data.tpo_features import compute_tpo_feature_panel

    # Collect unique source timeframes needed
    source_timeframes: set[str] = set()
    for tf in requested_timeframes:
        source_tf = TPO_SOURCE_TIMEFRAME.get(tf, tf)
        if source_tf in panels:
            source_timeframes.add(source_tf)

    tpo_panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for source_tf in sorted(source_timeframes):
        symbol_frames = panels[source_tf]
        tpo_panels[source_tf] = {}
        for symbol in symbols:
            if symbol not in symbol_frames:
                continue
            frame = symbol_frames[symbol]
            high = frame["h"].fillna(0.0).to_numpy(dtype=np.float32)
            low = frame["l"].fillna(0.0).to_numpy(dtype=np.float32)
            close = frame["c"].fillna(0.0).to_numpy(dtype=np.float32)
            features, volatility = compute_tpo_feature_panel(high, low, close)
            tpo_panels[source_tf][symbol] = (features, volatility)
        if logger is not None:
            logger.info(
                "state=tpo_presweep source_tf=%s symbols=%d",
                source_tf,
                len(tpo_panels[source_tf]),
            )
    return tpo_panels
```

**Cost estimate for 4-week M5 data:**
- ~8,000 bars per symbol × 42 FX symbols = ~336K `compute_tpo_memory_state` calls
- Each call: 4 lookbacks × histogram build = lightweight
- Expected wall time: 30–60 seconds total for all FX symbols on M5
- BTC (1 symbol): <1 second

#### Step 3: Call `_presweep_tpo` inside `load_staged_panels`

**File:** `src/staged_v4/data/dataset.py` line 561-578 (inside `_make` closure or after it)

After building the panels and before returning, run the pre-sweep and attach to `StagedPanels`:

```python
def _make(subnet_name, symbols, panels):
    lookup = {}
    for timeframe, symbol_map in panels.items():
        if symbol_map:
            tf_index = next(iter(symbol_map.values())).index
            lookup[timeframe] = _build_anchor_lookup(anchor_timestamps, tf_index)
        else:
            lookup[timeframe] = np.zeros(len(anchor_timestamps), dtype=np.int32)
    tpo = _presweep_tpo(panels, symbols, requested_timeframes, logger)
    return StagedPanels(
        subnet_name=subnet_name,
        symbols=symbols,
        anchor_timeframe=anchor_timeframe,
        panels=panels,
        anchor_timestamps=anchor_timestamps,
        anchor_lookup=lookup,
        walkforward_splits=walkforward_splits,
        split_frequency=split_frequency,
        tpo_panels=tpo,
    )
```

Also update `generate_synthetic_panels` to pass `tpo_panels={}` (synthetic data doesn't need real TPO).

#### Step 4: Replace JIT TPO computation with slice from `tpo_panels`

**File:** `src/staged_v4/data/jit_sequences.py`

##### 4a. BTC batch builder (line 167-174)

Current code:
```python
if source_timeframe == timeframe:
    source_tpo, source_vol = compute_tpo_feature_panel(ext_high, ext_low, ext_close)
    window_tpo = source_tpo[local_start:local_end]
    window_vol = source_vol[local_start:local_end]
else:
    window_tpo, window_vol = _tpo_for_target_window(...)
```

Replace with:
```python
source_tf = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
symbol_tpo = panels.tpo_panels.get(source_tf, {}).get(symbol)
if symbol_tpo is not None:
    all_tpo, all_vol = symbol_tpo
    if source_tf == timeframe:
        window_tpo = all_tpo[ext_start:ext_end][local_start:local_end]
        window_vol = all_vol[ext_start:ext_end][local_start:local_end]
    else:
        # Use anchor_lookup to map from target timeframe to source timeframe
        source_lookup = panels.anchor_lookup[source_tf]
        target_positions = source_lookup[anchor_indices[batch_idx]]  # single position
        # For sequence window, need per-step lookup
        source_index = next(iter(panels.panels[source_tf].values())).index
        target_ts_idx = source_index.searchsorted(pd.Index(window_ts), side="right") - 1
        target_ts_idx = np.clip(target_ts_idx, 0, len(all_tpo) - 1)
        window_tpo = all_tpo[target_ts_idx][:, None, :]
        window_vol = all_vol[target_ts_idx][:, None]
else:
    # Fallback: compute on the fly (should not happen if presweep ran)
    source_tpo, source_vol = compute_tpo_feature_panel(ext_high, ext_low, ext_close)
    window_tpo = source_tpo[local_start:local_end]
    window_vol = source_vol[local_start:local_end]
```

##### 4b. FX batch builder (lines 321-331)

Current code loops per symbol and calls `compute_tpo_feature_panel`. Replace with:

```python
source_tf = TPO_SOURCE_TIMEFRAME.get(timeframe, timeframe)
window_tpo = np.zeros((len(target_ts), n_nodes, 8), dtype=np.float32)
window_vol = np.zeros((len(target_ts), n_nodes), dtype=np.float32)

for col_idx, symbol in enumerate(symbols):
    if not include_signal_only_tpo and not tradable_mask[col_idx]:
        continue
    symbol_tpo = panels.tpo_panels.get(source_tf, {}).get(symbol)
    if symbol_tpo is not None:
        all_tpo, all_vol = symbol_tpo
        if source_tf == timeframe:
            window_tpo[:, col_idx, :] = all_tpo[ext_start:ext_end][local_start:local_end]
            window_vol[:, col_idx] = all_vol[ext_start:ext_end][local_start:local_end]
        else:
            source_index = next(iter(panels.panels[source_tf].values())).index
            ts_idx = source_index.searchsorted(pd.Index(target_ts), side="right") - 1
            ts_idx = np.clip(ts_idx, 0, len(all_tpo) - 1)
            window_tpo[:, col_idx, :] = all_tpo[ts_idx]
            window_vol[:, col_idx] = all_vol[ts_idx]
    else:
        # Fallback
        features, vol = compute_tpo_feature_panel(
            ext_high[:, col_idx], ext_low[:, col_idx], ext_close[:, col_idx]
        )
        window_tpo[:, col_idx, :] = features[local_start:local_end]
        window_vol[:, col_idx] = vol[local_start:local_end]
```

##### 4c. Remove `_tpo_for_target_window` function (lines 71-102)

This function becomes dead code after the above changes. It can be removed entirely, or kept behind a `# legacy` comment if you want a rollback path.

#### Step 5: Update `__init__.py` exports

**File:** `src/staged_v4/data/__init__.py`

No new public exports needed — `tpo_panels` is an internal field on `StagedPanels`.

### Files to modify

| Step | File | Change |
|------|------|--------|
| 1 | `src/staged_v4/data/dataset.py:22-31` | Add `tpo_panels` field to `StagedPanels` |
| 2 | `src/staged_v4/data/dataset.py` (new fn) | Add `_presweep_tpo()` helper |
| 3 | `src/staged_v4/data/dataset.py:561-578` | Call `_presweep_tpo` in `_make`, pass to constructor |
| 3b | `src/staged_v4/data/dataset.py:583+` | Pass `tpo_panels={}` in `generate_synthetic_panels` |
| 4a | `src/staged_v4/data/jit_sequences.py:167-174` | BTC: slice from `tpo_panels` instead of computing |
| 4b | `src/staged_v4/data/jit_sequences.py:321-331` | FX: slice from `tpo_panels` instead of computing |
| 4c | `src/staged_v4/data/jit_sequences.py:71-102` | Remove `_tpo_for_target_window` (dead code) |

### What does NOT change

- `tpo_features.py` — `compute_tpo_feature_panel` and `compute_rolling_volatility` stay as-is; they're still called, just once at load time instead of per batch
- `tpo_normal_layer.py` — the core TPO engine is untouched
- `fx_features.py` — the cache builder path is unaffected (it still pre-bakes for cached mode)
- `train_staged.py` — no changes; it already passes `StagedPanels` through to the batch builders
- `config.py` — `TPO_SOURCE_TIMEFRAME` mapping stays the same
- Cache path (`load_feature_batches`) — still works for pre-built caches; this only changes JIT
- Output shapes — `TimeframeSequenceBatch.tpo_features` shape is identical
- Model — `STGNNBlock` receives the same tensor shapes

### Verification

1. **Unit test**: run `python -m pytest src/test_staged_v4.py -v` — must pass unchanged
2. **Synthetic smoke**: run a synthetic training (`--mode synthetic`). Synthetic panels will have `tpo_panels={}`, so the fallback path fires. Verify output report matches pre-change synthetic run.
3. **JIT smoke**: run the JIT real smoke (`src/run_staged_v4_real_jit_smoke.py`) with a small date range. Compare:
   - TPO features for a sample of (symbol, timeframe, anchor_index) should be numerically identical to the old per-batch computation
   - Wall time should drop significantly (expect 5-10x on FX batch construction)
4. **Regression**: compare a short training run's fold metrics (AUC, Sharpe) pre- and post-change. Must be identical — this is a pure performance refactor with no behavioral change.

### Expected performance

| Scenario | Before (per-batch TPO) | After (pre-sweep + slice) |
|----------|----------------------|--------------------------|
| FX M5, 4 weeks, 42 symbols | ~6.3M profile calls/epoch | ~336K calls total (one-time) |
| Per-batch FX cost | ~50K profile calls | 0 (array slice only) |
| JIT startup overhead | ~0 | +30-60s for pre-sweep |
| Epoch wall time (FX batches) | Minutes per epoch | Seconds per epoch |

---

## Approach B: Vectorized Rolling TPO (Complementary Optimization)

### Concept

Replace the per-bar Python `for idx in range(n_bars)` loop in `compute_tpo_feature_panel` with a sliding-window histogram approach. Instead of rebuilding the histogram from scratch at each bar, maintain a running bin-count array and incrementally add/remove bars.

### How it works

For a fixed lookback L:
1. Initialize histogram from bars 0..L-1
2. At bar i (where i >= L): add bar i's bins, remove bar (i-L)'s bins
3. Extract POC, value area, balance score from the running histogram
4. Repeat for each of the 4 lookbacks (24, 48, 96, 192)

This turns per-bar cost from O(lookback) to O(n_bins) amortized.

### Changes

| File | Change |
|------|--------|
| `src/staged_v4/data/tpo_features.py:71-101` | Replace `for idx in range(n_bars)` loop with sliding-window approach |
| `src/tpo_normal_layer.py` | Add `compute_tpo_profile_incremental(counts, edges, close_price, atr)` that accepts pre-built histogram counts |

### Estimated speedup

10-20x on the inner loop. For 8K M5 bars × 42 symbols, this would reduce the pre-sweep in Approach A from ~30-60s to ~3-6s.

### When to implement

After Approach A is working and validated. This is a pure optimization of the pre-sweep step — not needed for correctness or even for reasonable performance on 4-8 week date ranges. Consider it if/when training extends to months of data.

---

## Approach C: Lazy TPO with LRU Cache (Alternative — Lower Priority)

### Concept

Cache TPO results keyed by `(symbol, source_timeframe, bar_index)`. Since training iterates anchors roughly in order, adjacent batches share overlapping TPO windows.

### Why it's lower priority

- Cache key space is large: 42 symbols × 8K bars × unique source timeframes
- TPO state depends on the full lookback window, not just the bar index — overlap savings require careful window management
- Bookkeeping complexity is high relative to the benefit
- Approach A eliminates the need entirely by computing upfront

### When to consider

Only if memory constraints prevent holding the full pre-swept TPO arrays in RAM (e.g., if training extends to a year of tick data with many timeframes). For 4-8 weeks of M5 data, the pre-swept arrays are ~42 symbols × 8K bars × 8 features × 4 bytes = ~10.5 MB — negligible.

---

## Decision summary

| Approach | Complexity | Speedup | Implement when |
|----------|-----------|---------|----------------|
| **A: Pre-sweep at load** | Low (5 files, ~80 lines changed) | ~100x on batch TPO | **Now** |
| B: Vectorized rolling | Medium (2 files, algorithmic rewrite) | ~10-20x on pre-sweep | After A, if pre-sweep is slow |
| C: LRU cache | High (new caching layer) | Variable | Only if memory-constrained |
