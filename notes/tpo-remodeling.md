Instructions to Re‑architect Existing Code to Latest Model Design

Goal: Transform the current codebase (which uses a pooled FX table, flat bridge, and timeframe‑separated STGNN) into the final architecture:

· Dual‑subnet (BTC 24/7, FX 24/5) with staged training.
· Conditional bridge (BTC → FX only during market overlap).
· Hierarchical collaborative learning across 11 timeframes (ticks, M1, …, MN1) inside each subnet, with each timeframe being an STGNN.
· TPO features integrated with volatility gating.
· All critical fixes applied (validity masks, probability calibration, adaptive thresholds, etc.).

---

1. Restructure Feature Engineering

Existing: research_features.py builds one pooled FX table with BTC as context.

Change:

· Split into three modules:
  · data/btc_features.py: extracts BTC‑only features (24/7) at all timeframes. Returns tensors with validity masks.
  · data/fx_features.py: extracts FX‑only features (24/5) at all timeframes. Also returns session codes and overlap flag.
  · data/bridge_features.py: given BTC and FX feature sets, builds the conditional bridge matrix only during overlap. This will be called during training.
· Keep existing research_dataset.py but adapt to load these separate datasets.
· Implement validity masks (as discussed) for all candles and ticks.

---

2. Create STGNN Modules per Timeframe

Existing: train_research_stgnn_dual.py has a single STGNN with fast/slow branches.

Change:

· Build a reusable STGNNBlock class (in models/stgnn_block.py) that:
  · Takes node features (batch, time, nodes, features).
  · Applies graph convolution(s) (use existing heterogeneous conv code).
  · Applies temporal attention (or temporal conv) to produce node embeddings.
  · Outputs both the embeddings and the final prediction head.
· For each timeframe, instantiate a separate STGNNBlock. The hyperparameters (hidden dims, attention heads, etc.) can be shared or tuned separately.
· In models/btc_subnet.py, create a BTCSubnet class that contains a list of STGNNBlocks (one per timeframe) and implements the hierarchical cooperation logic (alternating training, knowledge transfer between adjacent blocks).
· Similarly, create models/fx_subnet.py with the same structure, plus an input for bridge context.

---

3. Implement Hierarchical Cooperation (Within Subnet)

Existing: No such mechanism.

Add:

· In each subnet class, define a cooperative_step method that takes a batch of data and a active_idx (which timeframe is currently active). It:
  · Runs the active STGNNBlock on its respective data (with its sequence length).
  · Stores the output node embeddings.
  · After K batches (hyperparameter), switches active_idx to the next timeframe and transfers embeddings to the new active block (by concatenating to its input features or adding as a bias).
· Implement bidirectional transfer (both up and down). Use a loop over all timeframes, alternating.
· This logic must be integrated into the training loop.

---

4. Conditional Bridge (BTC → FX)

Existing: research_bridge.py builds a flat bridge matrix for every bar.

Change:

· In models/bridge.py, create a ConditionalBridge class:
  · Takes the frozen BTC subnet’s output embeddings (for a given timeframe) and a binary overlap mask.
  · Projects the BTC embedding to FX input dimension via an MLP.
  · Returns bridge_context * overlap_mask (zero when overlap is false).
· In the FX subnet, modify the forward pass of each STGNNBlock to accept an optional bridge context tensor. If provided and overlap flag is 1, concatenate (or add) it to the input node features before the graph convolution.

---

5. Add TPO Features with Volatility Gating

Existing: Not present.

Add:

· A new module data/tpo_features.py that, for each asset and each timeframe, computes TPO metrics (POC, VA, skew, etc.) from higher‑frequency data. Use rolling windows aligned to candle boundaries.
· In the dataset, load these TPO features as additional node features.
· Implement a volatility gate:
  · Compute rolling volatility for each asset at each timeframe (e.g., standard deviation of returns over last N bars).
  · Define a gate g = torch.sigmoid(alpha * (threshold - volatility)) (alpha, threshold are hyperparameters).
  · In the STGNNBlock input, multiply the TPO features by g (so they only contribute when volatility is low).
  · Optionally, make alpha and threshold trainable or optimize via GA.

---

6. Training Pipeline (Staged)

Existing: train_research_compact.py and train_research_stgnn_dual.py train a single model.

Replace with train_staged.py:

· Stage 1 – Train BTC Subnet:
  · Load BTC data (all timeframes) using btc_features.py.
  · Instantiate BTCSubnet.
  · Run the hierarchical cooperative training loop (alternating active timeframes).
  · Save best model (based on validation metric, e.g., directional accuracy).
· Stage 2 – Train FX Subnet with Frozen BTC:
  · Load BTC subnet from Stage 1, freeze all parameters.
  · Load FX data (all timeframes) using fx_features.py.
  · Instantiate FXSubnet and ConditionalBridge.
  · During training, pass bridge context (from frozen BTC) only when overlap flag is true.
  · Run the hierarchical cooperative training loop for FX subnet, updating both FX subnet weights and the bridge MLP.
· Stage 3 (optional) – Fine‑tune Bridge:
  · Unfreeze the bridge MLP and a small part of FX subnet (e.g., last layer).
  · Continue training with same data but smaller learning rate.

---

7. Incorporate All Critical Fixes

Ensure the following fixes are integrated throughout the code:

· Validity masks for candles/ticks: pass through all operations and ignore invalid steps.
· No forward fill: use masks instead.
· Probability calibration: after training, apply Platt scaling on validation set.
· Adaptive thresholds for trade signals: replace fixed threshold with base + coeff * volatility.
· Cooldown/direction lock after stop loss: implement in execution engine.
· Correlation exposure cap for multiple FX trades: in trade ranking.
· Rolling adaptive graph (edges updated with rolling windows).
· Feature dimensionality reduction via GA or SHAP.
· Short‑horizon features (tick momentum, acceleration) in micro model.
· Regime duration/transition features.
· Session‑separate training or loss weighting.
· Probability spread filter: only trade when |2p-1| > threshold.
· Execution‑aware backtest with limit fills and latency.

---

8. Testing & Validation

· Modify the existing evaluation scripts to work with the new architecture.
· Use walk‑forward validation with monthly folds.
· Compute PBO on the final set of candidates.
· Ensure all metrics are computed out‑of‑sample.

---

9. Summary of New File Structure

```
project/
├── data/
│   ├── btc_features.py
│   ├── fx_features.py
│   ├── bridge_features.py
│   ├── tpo_features.py
│   └── dataset.py (updated)
├── models/
│   ├── stgnn_block.py
│   ├── btc_subnet.py
│   ├── fx_subnet.py
│   └── bridge.py
├── training/
│   └── train_staged.py
├── evaluation/
│   ├── backtest.py
│   └── metrics.py
├── utils/
│   └── critical_fixes.py (all fixes implemented)
└── configs/
    └── ga_config.py
```

---

Actionable Next Steps for AI:

1. Analyze existing code to locate feature extraction, dataset, and model files.
2. Create new files as listed above, migrating relevant parts and discarding outdated ones.
3. Implement each component step‑by‑step, ensuring the architecture matches the described design.
4. Test with small data to verify shapes and cooperation.
5. Run staged training and compare performance against the old model.
6. Validate with walk‑forward and PBO to confirm overfitting is reduced.