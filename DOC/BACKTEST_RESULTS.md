# Algo C2 — Backtest Results & Graph Interpretation

## Simulation parameters

| Parameter | Value |
|-----------|-------|
| Data | Real tick (Mon 2 – Fri 6 Mar 2026) |
| Balance | $50 |
| Leverage | 1:500 |
| Risk per trade | 2% |
| TP | ATR × 2.0 |
| SL | ATR × 1.5 |
| R:R | 1.33 |
| Break-even WR | ~43% |
| Max open positions | 3 per bar-close signal |
| Margin limit | 85% of balance |
| Timeout | 10 bars |
| Spread | Real tick spread deducted on entry + exit |

---

## Deliverable files

| File | Size | Purpose |
|------|------|---------|
| `algo_c2_backtest_v2.html` | ~2.6 MB | 5-day backtest with pair selector + session whitelists |
| `algo_c2_tick_sim.html` | ~4.2 MB | Real tick sim — real H/L for TP/SL, real spread |
| `analysis_results_35.json` | ~118 KB | Full model + outlier + Laplacian results (JSON) |

Both HTML files are self-contained — open directly in Chrome or Firefox. No server required. The tick sim decompresses all 35-pair data in-browser on load (~10s).

---

## Backtest tabs

| Tab | Content |
|-----|---------|
| Execution | Equity curve, EURUSD price + TP/SL overlay, CB scores, open positions |
| Pair selector | Enable/disable pairs globally + per session. Click = global toggle, double-click = session toggle |
| Daily P&L | Mon–Fri day cards, daily PnL chart, session breakdown |
| Order book | Per-trade PnL bars, cumulative PnL, closed order table with spread cost |
| Performance | Win rate, avg win/loss, EV, profit factor, max DD, return, spread drag |
| Optimisation | 5-gate pass rates, lot size history |

---

## Controls

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| CB threshold | 0.55 – 0.85 | 0.65 | Gate 2: min CatBoost confidence to count as high-conviction |
| Risk % | 1 – 5% | 2% | Lot sizing — position size scales with balance |
| Speed | Turbo/Fast/Normal/Slow | Fast | Simulation playback speed |
| Pair selector | All 35 pairs | FX enabled | Enable/disable per pair and per session |

---

## Graph interpretation guide

### The 35-node graph

```
FX pairs     → circles (28 nodes, tradeable)
Non-FX pairs → diamonds (7 nodes, graph anchors only)
```

**Edges:**
- **Teal** = positive correlation (pairs move together)
- **Coral** = negative correlation (pairs move inversely)
- Edge thickness ∝ correlation strength

**View modes:**
| Mode | What it shows |
|------|---------------|
| Correlation strength | Raw correlation. Teal = positive, coral = negative |
| Laplacian residuals | Blue node = positive ε (lagging peers), red = negative ε (leading) |
| Node degree | How well-connected each node is to the rest of the network |
| Currency clusters | 8 currency families colour-coded |

### Reading a residual

**Positive ε (blue):** The pair's last-bar return was LOWER than its correlated neighbours predicted. It underperformed. Mean-reversion thesis: LONG.

**Negative ε (red):** The pair's last-bar return was HIGHER than predicted. It overperformed. Mean-reversion thesis: SHORT.

Largest residuals = strongest structural alpha candidates. Best when residual direction agrees with the 7-indicator signal.

### Spectral gap (λ₂)

| λ₂ value | Meaning |
|----------|---------|
| > 0.5 | Strongly connected — trends propagate quickly across all pairs |
| 0.1 – 0.5 | Moderate connectivity |
| < 0.1 | Fragmented — clusters are semi-isolated |

Current: **λ₂ = 0.899** → strongly connected, TRENDING regime.

### Betti numbers

| Feature | Value | Meaning |
|---------|-------|---------|
| β₀ = 1 | Single component | All 35 pairs are connected into one market cluster |
| β₁ = 34 | Dense cycles | High loop count = many arbitrage triangles / multi-pair dependencies |

---

## Signal summary (end of 5-day window)

**LONG signals (17 pairs):**
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADJPY, EURCAD, EURCHF, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, XBRUSD

**SHORT signals (11 pairs):**
CADCHF, CHFJPY, EURAUD, EURGBP, NZDCHF, NZDUSD, USDCHF, USDJPY, USDMXN, USDZAR, XAGUSD

**FLAT (6 pairs):**
EURJPY, NZDCAD, NZDJPY, USDCAD, US30, XAUUSD

**Market theme:** GBP dominant strength (all GBP crosses LONG, +7 vote). USD weakness (USDCHF, USDJPY both −7). NZD weakness (NZDUSD −7). BTC/Brent bid.

---

## Production integration notes

When deploying to the C++/C# production stack:

1. **Feature vector** must match training exactly — same 14 microstructure fields from Phase 2 + residuals + Betti numbers from Phase 3
2. **CB score** in production = actual CatBoost model output (not the momentum proxy used in the sim)
3. **ATR** computed over the last 14 × 1000ms windows in Redis
4. **Spread** from live tick spread_mean in the current window, not historical average
5. **Gate 3 (EMA5)** uses the same EMA decay: `ema5 = ema5 * 0.667 + close * 0.333`
6. **Non-FX nodes** contribute to graph signal but never receive orders — enforce at routing layer
