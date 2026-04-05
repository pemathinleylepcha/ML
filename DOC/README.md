# Algo C2 — Quantitative Trading Pipeline

**Status:** Research complete · Real tick data validated · Sim engine built  
**Data:** Mon 2 – Fri 6 Mar 2026 · 35 pairs · ~7175 1-min bars  
**Account:** $50 · 1:500 leverage · FX orders only

---

## Current Blockers

For active project issues, runtime failures, data-coverage gaps, and resolved operational incidents, use:
- [PROJECT_ISSUES_LOG.md](/d:/Algo-C2-Codex/DOC/PROJECT_ISSUES_LOG.md)

Current high-priority issues logged there:
- `staged_v4` signal is still not robust across folds
- remote training host stalls at `import torch` for real JIT runs
- `repository-codex` historical coverage is still too incomplete for staged_v4 use

Documentation rule:
- Add a dated issue entry with evidence, affected files, and next action whenever a material blocker is discovered or resolved.

---

## What was built in this session

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Architecture blueprint | `algo_c2_blueprint.docx` | ✅ |
| 2 | 35-pair CSV processor | `process_fx_csv_35.py` | ✅ |
| 3 | 5-day backtest + pair selector | `algo_c2_backtest_v2.html` | ✅ |
| 4 | Real tick sim engine | `algo_c2_tick_sim.html` | ✅ |
| 5 | Analysis engine (6 models + 3 outlier detectors) | `algo_c2_analysis.py` | ✅ |
| 6 | 35-node graph visualiser | inline widget | ✅ |

---

## Repository structure

```
algo_c2/
├── README.md                     ← this file
├── ARCHITECTURE.md               ← full 5-phase pipeline spec
├── SIGNAL_PIPELINE.md            ← 35-node graph + 7 indicators + 5 gates
├── DATA_FORMAT.md                ← CSV format, JSON schema, pip sizes
├── MODELS.md                     ← CatBoost-equiv + ensemble suite
├── BACKTEST_RESULTS.md           ← real tick sim results + interpretation
├── data/
│   └── algo_c2_5day_data.json    ← 35-pair real tick OHLC (upload here)
├── scripts/
│   ├── process_fx_csv_35.py      ← Step 1: CSV → JSON
│   └── algo_c2_analysis.py       ← Step 2+3: Report + Outliers
└── sim/
    ├── algo_c2_backtest_v2.html  ← 5-day backtest (open in browser)
    └── algo_c2_tick_sim.html     ← real tick sim (open in browser)
```

---

## Quick start

```bash
# Step 1 — Convert tick CSVs to aligned 1-min JSON
python process_fx_csv_35.py --input_dir /path/to/csvs --output algo_c2_5day_data.json

# Step 2+3 — Run analysis report + outlier detection
python algo_c2_analysis.py --data algo_c2_5day_data.json --output results.json

# Step 4 — Open the real tick sim in browser
open algo_c2_tick_sim.html
```

---

## Key results (Mar 2–6 2026)

| Metric | Value |
|--------|-------|
| Pairs | 35 (28 FX tradeable + 7 graph anchors) |
| Spectral gap λ₂ | 0.899 |
| Betti β₀ | 1 (single connected component) |
| Betti β₁ | 34 (dense cycle structure) |
| Regime | TRENDING |
| Best model AUC | 0.506 (GradientBoosting) |
| Consensus outliers | 1 bar (≥2/3 detectors) |
| Top mispricing | NZDUSD ε=−0.00124 |
