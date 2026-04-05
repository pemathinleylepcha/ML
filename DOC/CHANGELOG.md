# Algo C2 — Changelog

All architectural and feature changes that affect model training or inference.
Ordered newest first. Each entry maps to a generation tag in GENERATIONS.md.

Active blockers and operational issues are tracked separately in:
- [PROJECT_ISSUES_LOG.md](/d:/Algo-C2-Codex/DOC/PROJECT_ISSUES_LOG.md)

Documentation rule:
- Use the changelog for model and architecture changes.
- Use the issue log for active failures, runtime problems, data gaps, and resolved operational incidents.

---

## v1_5_c1 — 2026-03-27

**Correction 1: Session-Aware Regime Memory**

### Problem

The bridge had session encoding (`fx_session_phase`) and regime encoding
(`graph_regime`), both computed fresh every bar independently. The model had
no way to know:
- What regime the previous bar was in
- How long we have been in the current regime
- Whether a session boundary just happened
- Whether a regime flip is a real transition or noise

Each bar looked like a fresh start. Markets don't reset every bar — they
evolve across sessions. A Tokyo LOW_VOL regime that transitions to London
HIGH_STRESS carries context the model was blind to.

### Changes

**`src/bridge.py`**

`BridgeState.shared_features()` expanded from 5 → 9 features:

| Feature | Type | Description |
|---------|------|-------------|
| `prev_regime` | new | Regime encoding at previous bar (0=LOW_VOL..4=FRAGMENTED) |
| `bars_in_regime` | new | `log1p(n)` bars elapsed in current regime — persistence signal |
| `session_transition` | new | 1.0 on first bar of a new session (Tokyo→London etc), else 0 |
| `regime_ema` | new | EWMA (α=0.1) of regime — soft memory, resists instant flips |

`BridgeComputer` internal state additions:
- `_prev_regime: float` — carries last bar's regime
- `_prev_session: float` — carries last bar's session phase
- `_bars_in_regime: int` — counter, resets on regime change
- `_regime_ema: float` — EWMA accumulator

### Why each feature matters

**`prev_regime`** — Gives the model the raw delta: "we were LOW_VOL, now we're HIGH_STRESS". Without this, the model cannot see transitions at all.

**`bars_in_regime`** — Log-scaled duration. A regime that just started (bars=0) is very different from one that has held for 4 hours (bars=240). Regime persistence is a known predictor of continuation vs reversal.

**`session_transition`** — Tokyo/London/NY boundaries are the highest-variance moments in FX. A flag on the exact first bar of each session lets the model learn session-opening behaviour explicitly.

**`regime_ema`** — Raw regime flips bar-to-bar on noisy data. EWMA α=0.1 provides a smooth "regime temperature" — the model gets the current hard label AND the soft memory of where regime was trending.

### Impact on existing model

**Not backward compatible.** `v2_legacy` was trained with 5 shared features.
`v1_5_c1` uses 9. Cannot load v2_legacy model and run new bridge features
against it — feature count mismatch.

**Do not overwrite `models/test_v2/`.** v2_legacy stays as the reference
baseline. Retrain with `--model-version v1_5_c1` to produce the new generation.

### Verification

```python
# bridge.py smoke test shows correct behaviour:
# 2026-03-27 02:00 (Tokyo, LOW_VOL starts):
#   regime=0, prev=2, bars_in=0.00, transition=1.0, ema=1.800
# 2026-03-27 03:00 (same session, same regime):
#   regime=0, prev=0, bars_in=0.69, transition=0.0, ema=1.620
# 2026-03-27 09:00 (London opens, regime flips HIGH_STRESS):
#   regime=3, prev=0, bars_in=0.00, transition=1.0, ema=1.758
```

---

## v2_legacy — 2026-03-27 (original)

Initial production model trained on 16 months of MT5 M5 data (43 instruments).

**Architecture:**
- CatBoost-BTC: 37 features, BTC 24x7 signal
- CatBoost-FX: 138 features, 28 pairs pooled with `pair_id` categorical
- Bridge: 20 features (9 BTC→FX gated, 6 FX→BTC, 5 shared)
- Walk-forward CV: 5 folds, TimeSeriesSplit
- Labels: quantile (30% BUY / 30% SELL / 40% HOLD), cost-adjusted

**Scores:**
- BTC PBO: 0.60
- FX WFE: poor (negative)
- ECE: 0.20 BTC / 0.23 FX
- Paper trading live: 2026-03-27 onwards

**Known gaps at training time (see critics.md):**
- No regime memory across sessions (fixed in v1_5_c1)
- No short-term bar features
- No probability calibration
- No correlation cluster cap
