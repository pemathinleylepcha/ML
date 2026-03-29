# Market-Neutral Alpha Model — Implementation Plan

## Overview

Build a PyTorch neural network that learns market-neutral portfolio weights from the 568-feature matrix. The model ingests backtest feature snapshots, outputs dollar-neutral weights across 35 pairs, and is trained via backpropagation to maximize a differentiable Sharpe ratio. Once trained, it replaces the CatBoost confidence proxy in `signal_pipeline.py`.

## Architecture

### New file: `src/market_neutral_model.py`

**Network: `AlphaNet`**
```
Input (568) → Linear(568, 256) → LayerNorm → LeakyReLU → Dropout(0.3)
           → Linear(256, 128)  → LayerNorm → LeakyReLU → Dropout(0.2)
           → Linear(128, 64)   → LayerNorm → LeakyReLU
           → Linear(64, 35)    → MarketNeutralHead
```

**MarketNeutralHead** (enforces dollar-neutrality):
```python
raw_weights = output  # shape (batch, 35)
weights = raw_weights - raw_weights.mean(dim=-1, keepdim=True)  # sum → 0
weights = weights / (weights.abs().sum(dim=-1, keepdim=True) + eps)  # normalize gross exposure to 1
```

This guarantees: `sum(weights) ≈ 0` (dollar neutral), `sum(|weights|) = 1` (unit leverage).

### Loss Function: Differentiable Sharpe Ratio

```python
portfolio_returns = (weights * forward_returns).sum(dim=-1)  # per-bar portfolio return
loss = -(mean(portfolio_returns) / (std(portfolio_returns) + eps)) * sqrt(252)
```

Plus regularization:
- **L2 on weights**: `lambda_l2 * sum(param^2)` — prevent overfitting
- **Turnover penalty**: `lambda_turnover * mean(|w_t - w_{t-1}|)` — reduce churn
- **Concentration penalty**: `lambda_conc * max(|w_i|)` — diversification

### Walk-Forward Training

```
|---- Train window (expanding) ----|-- Val --|-- Test --|
|  t=0 ... t=split_train           | val_end | test_end |
```

- **Expanding window**: train on bars `[0, split_train)`, validate on `[split_train, val_end)`, test on `[val_end, test_end)`
- **Fold schedule**: 5 walk-forward folds, each advancing the split by `n_bars / 6`
- **Early stopping**: patience=15 epochs on validation Sharpe
- **Batch size**: 32 bars (sequential, no shuffle — time series!)
- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4
- **Epochs**: up to 100 per fold (early stop expected ~30-50)

### Data Pipeline

1. Load JSON data → run `build_feature_matrix()` to get 568-feature DataFrame
2. Compute forward returns: `r_{t+1} = log(close_{t+1}/close_t)` for all 35 pairs
3. Create `SequentialDataset`:
   - X: feature matrix (N, 568)
   - R: forward returns (N, 35)
   - Sequential DataLoader (no shuffle)
4. StandardScaler on train split only, transform val/test

### Integration with Existing Pipeline

**Step 1**: Train AlphaNet on historical data (via `algo_c2_analysis.py`)
- Add `--train-alpha-model` flag to analysis orchestrator
- Save trained model to `models/alpha_net.pt`

**Step 2**: Replace CatBoost proxy in `signal_pipeline.py`
- New function: `compute_cb_from_model(features, model) → dict[pair, confidence]`
- The model's weight for pair_i becomes the confidence:
  - `confidence_i = sigmoid(raw_weight_i * scale)` mapped to [0, 1]
  - Positive weight → LONG confidence, negative → SHORT confidence
  - Magnitude → gate G2 confidence score (replaces `compute_cb_proxy`)

**Step 3**: Backtester uses model predictions
- In `backtester.py`, add optional `alpha_model` parameter
- When provided, replace `compute_cb_proxy()` calls with model inference
- The model sees the same features the backtester computes per bar

### Output: What the Model Produces

Per bar, for each of the 35 pairs:
1. **Weight** (float, sums to 0): dollar-neutral allocation signal
2. **Confidence** (float, 0-1): mapped from weight magnitude for gate G2
3. **Direction** (LONG/SHORT/FLAT): from weight sign

These replace the current `compute_cb_proxy()` → `cb_scores` dict and `directions_map` in the backtester loop.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/market_neutral_model.py` | **CREATE** | AlphaNet, loss, training loop, walk-forward |
| `src/algo_c2_analysis.py` | MODIFY | Add `--train-alpha-model` flag, call training |
| `src/signal_pipeline.py` | MODIFY | Add `compute_cb_from_model()`, keep proxy as fallback |
| `src/backtester.py` | MODIFY | Accept optional `alpha_model` for inference |
| `requirements.txt` | MODIFY | Add `torch>=2.0` |

## Implementation Order

1. Create `market_neutral_model.py` with AlphaNet, loss, dataset, training loop
2. Add smoke test with synthetic data
3. Wire into `algo_c2_analysis.py` (training path)
4. Add `compute_cb_from_model()` to `signal_pipeline.py`
5. Wire into `backtester.py` (inference path)
6. Run smoke tests on all modified files
