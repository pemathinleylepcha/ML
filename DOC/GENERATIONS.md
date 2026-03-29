# Algo C2 — Model Generation Registry

Each training run that produces a deployable model gets a generation tag.
New training must pass `--model-version <tag>` to `train_catboost_v2.py`.
All tags are registered here **before** training starts.

---

## Generation naming convention

```
v{major}_{minor}_c{correction}
```

| Part | Meaning |
|------|---------|
| `major` | Architecture version (1 = CatBoost dual-subnet) |
| `minor` | Dataset or feature set revision |
| `5` | Current minor — 5 = regime memory bridge features |
| `c{n}` | Correction number within this minor |

Examples: `v1_5_c1`, `v1_5_c2`, `v1_6_c1`

---

## Registered generations

### v2_legacy (baseline — DO NOT OVERWRITE)

| Field | Value |
|-------|-------|
| Path | `models/v2_legacy/` |
| Files | `catboost_btc_v2.cbm`, `catboost_fx_v2.cbm`, `catboost_v2_feature_names.json` |
| Source | Copied from remote `D:\DATASET-ML\models\catboost_v2_full3\` on 2026-03-27 |
| Trained | 2026-03-27 08:18 on remote (run: `train_catboost_v2_full3`) |
| Data | 9 years MT5 M5 bars (2018–2026), 41 instruments, 146345 bars after alignment |
| Samples | BTC: 103,528 — FX: 1,793,546 (28 pairs × 64,055 bars) |
| Features | BTC: 20 selected / 37 total — FX: 40 selected / 140 total |
| Bridge features | 20 (9 BTC→FX gated, 6 FX→BTC, 5 shared) |
| BTC PBO | 0.40 — LOW OVERFIT \| WFE=-0.09 \| DSR=1.00 \| ECE=0.32 |
| FX PBO | 0.70 — HIGH OVERFIT \| WFE=-3.87 \| DSR=0.10 \| ECE=0.09 |
| BTC acc (CV) | F1=52% F2=87% F3=65% F4=92% F5=— |
| FX acc (CV) | F1=44% F2=44% F3=41% F4=38% F5=— |
| BTC Sharpe (CV) | F1=0.00 F2=0.92 F3=-2.06 F4=0.00 F5=— |
| FX Sharpe (CV) | F1=0.00 F2=-3.79 F3=-3.60 F4=0.00 F5=— |
| Notes | Live paper trading on remote since 2026-03-27. Remote remains the repo. |

---

### v1_5_c1 — Correction 1: Regime Memory

| Field | Value |
|-------|-------|
| Path | `models/v1_5_c1/` |
| Files | `catboost_btc_v1_5_c1.cbm`, `catboost_fx_v1_5_c1.cbm` |
| Trained | _pending_ |
| Data | Same 16-month dataset |
| Bridge features | 24 (9 BTC→FX, 6 FX→BTC, **9 shared** — +4 regime memory) |
| Changes | See [CHANGELOG.md](CHANGELOG.md) — Correction 1 |
| Status | **NOT YET TRAINED** — retrain required to activate regime features |

**Train command (local machine):**
```bash
python src/train_catboost_v2.py \
  --data-dir d:/Algo-C2/data/DataExtractor \
  --model-version v1_5_c1 \
  --n-folds 5 --horizon 5 --lookback 120 \
  --iterations 500 --depth 6 --purge-bars 5
```

**After training — update this table:**
- Trained date
- PBO scores (BTC + FX)
- CV Sharpe per fold
- ECE calibration

---

## Future planned corrections

| Tag | Planned change |
|-----|---------------|
| `v1_5_c2` | Probability calibration (Platt/Isotonic on CV probabilities) |
| `v1_5_c3` | Short-term features (last 3-5 bar micro-patterns) |
| `v1_5_c4` | Correlation cap: max 2 open positions per JPY cluster |
| `v1_6_c1` | STGNN integration — replace CatBoost-FX with STGNN-FX head |

---

## Model compatibility

Models are **not forward/backward compatible across minor versions** — the
feature count in `shared_features()` changed between `v2_legacy` (5) and
`v1_5_c1` (9). Always load the feature names JSON that ships with each model:

```python
with open("models/v1_5_c1/catboost_feature_names_v1_5_c1.json") as f:
    meta = json.load(f)
# meta["version"] tells you which bridge feature set the model expects
```

The live engine (`live_mt5.py`) loads the model path from config — point it at
the new generation path when ready to deploy.
