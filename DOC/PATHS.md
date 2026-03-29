# Algo C2 — Resource Paths

Quick reference for all data, model, code, and log locations.

---

## Local machine (primary dev — this machine)

| Resource | Path |
|----------|------|
| Project root | `d:\Algo-C2\` |
| Source code | `d:\Algo-C2\src\` |
| Documentation | `d:\Algo-C2\DOC\` |
| Models (all generations) | `d:\Algo-C2\models\` |
| Live model (v2_legacy) | `d:\Algo-C2\models\v2_legacy\` |
| v1_5_c1 model (after training) | `d:\Algo-C2\models\v1_5_c1\` |
| Training dataset (local copy) | `d:\Algo-C2\data\DataExtractor\` |
| 5-day test data | `d:\Algo-C2\data\algo_c2_5day_data.json` |

---

## Remote server (repo + live trading)

| Resource | Path |
|----------|------|
| Host | `algoc2` (SSH alias) — `172.25.46.56`, user `thinley` |
| SSH key | `~/.ssh/id_ed25519` |
| SSH config | `~/.ssh/config` → `Host algoc2` |
| Working dir | `D:\DATASET-ML\` |
| Training dataset | `D:\DATASET-ML\DataExtractor\` (9 years, 41 instruments, 1.63 GB) |
| Models dir | `D:\DATASET-ML\models\` |
| Live production model | `D:\DATASET-ML\models\catboost_v2_full3\` |
| v1_5_c1 model (after training) | `D:\DATASET-ML\models\v1_5_c1\` |
| Training scripts | `D:\DATASET-ML\train_catboost_v2.py` |
| bridge.py (synced) | `D:\DATASET-ML\bridge.py` |
| Training log (v1_5_c1) | `D:\DATASET-ML\train_v1_5_c1.log` |
| Live engine | `C:\Users\Thinley\...` (live_mt5.py) |
| Dashboard | port `9876` — `http://172.25.46.56:9876` |

---

## Key files — source code

| File | Purpose |
|------|---------|
| `src/bridge.py` | Cross-learning bridge BTC↔FX, regime memory (v1_5_c1+) |
| `src/train_catboost_v2.py` | Training pipeline — `--model-version` for generation tagging |
| `src/live_mt5.py` | Live/paper trading engine |
| `src/live_dashboard.py` | WebSocket dashboard (port 9876) |
| `src/feature_engine.py` | Feature computation per bar |
| `src/subnet_btc.py` | BTC subnet features |
| `src/subnet_fx.py` | FX subnet features |
| `src/math_engine.py` | Graph Laplacian + TDA |
| `src/signal_pipeline.py` | Signal → order logic |
| `src/pbo_analysis.py` | PBO / walk-forward efficiency |
| `src/calibration.py` | ECE / Brier calibration report |

---

## Training commands

```bash
# Train new generation on REMOTE (preferred — data already there)
ssh algoc2 "python D:\\DATASET-ML\\train_catboost_v2.py \
  --data-dir D:\\DATASET-ML\\DataExtractor \
  --model-version <tag> \
  --n-folds 5 --horizon 5 --lookback 120 \
  --iterations 500 --depth 6 --purge-bars 5 \
  > D:\\DATASET-ML\\train_<tag>.log 2>&1"

# Watch training log on remote
ssh algoc2 "powershell -command \"Get-Content D:\\\\DATASET-ML\\\\train_<tag>.log -Wait\""

# Pull trained models from remote to local after training
scp -i ~/.ssh/id_ed25519 \
  "algoc2:D:/DATASET-ML/models/v1_5_c1/*" \
  "d:/Algo-C2/models/v1_5_c1/"

# Push code changes to remote before training
scp -i ~/.ssh/id_ed25519 \
  d:/Algo-C2/src/bridge.py \
  d:/Algo-C2/src/train_catboost_v2.py \
  "algoc2:D:/DATASET-ML/"
```

---

## Generation registry

See [GENERATIONS.md](GENERATIONS.md) for full model version history and scores.
See [CHANGELOG.md](CHANGELOG.md) for per-correction change details.
