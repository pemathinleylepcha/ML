Below is a detailed expansion of each “critical fix” one‑liner, explaining why it matters and how it can be integrated into the previously outlined cooperative multi‑asset, multi‑timeframe FX trading model.

1. Replace forward fill for candles with NaN + validity mask to avoid cross‑session leakage.
Problem: Forward filling missing candles (e.g., during weekends or holidays) can incorrectly imply continuity across sessions, introducing look‑ahead bias or making the model believe that prices persisted when markets were closed.

Solution:

Store raw candle data with explicit timestamps; leave missing periods as NaN.

Add a validity mask (binary feature) to each input that indicates whether a candle exists at that step.

In temporal attention or convolution, mask out invalid steps so the model does not attend to them.

For higher timeframes (e.g., D1, W1), align to calendar days, not trading days, to maintain true time gaps.

2. Add regime memory across sessions to prevent resetting behavior between Tokyo/London/NY.
Problem: Session labels are categorical; the model may treat each session independently, losing continuity of market state across session boundaries.

Solution:

Use a recurrent or gated component (e.g., GRU) that carries hidden state across session transitions.

Alternatively, add a session transition embedding that captures the relationship between consecutive sessions (e.g., Tokyo → London).

Maintain a short‑term “regime memory” vector that is updated at each time step and passed across sessions, allowing the model to remember volatility, trend, or order‑flow patterns from the previous session.

3. Separate macro (direction) and execution (entry timing) into two explicitly trained layers.
Problem: The macro model (candle) and micro model (tick) already have different objectives, but mixing them in a single loss may lead to suboptimal specialization.

Solution:

Keep the macro model solely responsible for directional probability (next‑bar up/down). Train it with binary cross‑entropy.

The micro model (tick) is trained to predict a short‑term “entry opportunity” label, which could be derived from future short‑horizon profitability.

During inference, the macro filter is applied first; only when the macro signal is present does the micro model decide the precise entry moment. This creates a clear separation of concerns.

4. Add confidence calibration (Platt/Isotonic) before using probabilities in Kelly sizing.
Problem: Neural network probabilities are often miscalibrated; using raw probabilities in Kelly can lead to over‑betting or under‑betting.

Solution:

After training the macro model (and/or the meta‑classifier), apply probability calibration on a hold‑out validation set.

Platt scaling (logistic regression on logits) or isotonic regression can transform the output probabilities to better reflect true win frequencies.

Calibrated probabilities are then used for fractional Kelly.

5. Replace static thresholds (0.55–0.7) with adaptive thresholds based on regime/volatility.
Problem: A fixed threshold for macro probability may be too aggressive in high‑volatility regimes and too conservative in low‑volatility ones.

Solution:

Compute a dynamic threshold 
T
(
t
)
=
T
b
a
s
e
+
α
⋅
σ
(
t
)
T(t)=T 
base
​
 +α⋅σ(t), where 
σ
(
t
)
σ(t) is recent volatility (e.g., ATR).

Alternatively, learn the threshold as a function of regime features (e.g., using a small neural network or a rule‑based mapping).

During backtesting, the GA can optimize the parameters of the adaptive threshold formula.

6. Introduce cooldown + direction lock after SL to prevent repeated losses.
Problem: After a stop‑loss, the model might immediately take another trade in the same direction, compounding losses (common in momentum‑chasing systems).

Solution:

Implement a cooldown period (e.g., 30 minutes) during which no new trades are taken for that FX pair.

Additionally, a direction lock: after a loss in a certain direction, block new trades in that same direction for a certain number of bars or until a regime change is detected.

This prevents emotional/over‑trading after a losing streak.

7. Rank trades by score (probability × confidence × regime weight) before applying MAX_OPEN.
Problem: If multiple FX pairs generate signals, the system may open too many positions, increasing correlation risk.

Solution:

Compute a trade score for each candidate:
Score = p_{calibrated} \times (1 - \text{model_uncertainty}) \times \text{regime_weight}
where regime_weight could be, for example, higher during high‑liquidity sessions.

Rank all candidates by score, then open positions in descending order until the maximum number of open positions (MAX_OPEN) is reached.

This ensures capital is allocated to the most promising opportunities.

8. Add correlation exposure cap to avoid stacking similar FX trades (e.g., JPY cluster).
Problem: Taking multiple positions in highly correlated pairs (e.g., USD/JPY, GBP/JPY, EUR/JPY) creates concentrated risk.

Solution:

Calculate the portfolio correlation matrix using rolling returns of the selected pairs.

Before opening a new position, check whether the incremental exposure would cause the sum of absolute correlation‑adjusted exposures to exceed a cap (e.g., 2.0).

If the cap is exceeded, skip the trade with the lowest score among the correlated group.

This is similar to a risk‑budgeting approach.

9. Use rolling adaptive graph (not static) with session‑aware edge masking.
Problem: A static graph cannot capture time‑varying relationships (e.g., correlations that change across sessions or market regimes).

Solution:

Recompute edge weights (e.g., correlation) on a rolling window (e.g., every 1 hour for tick model, daily for candle model).

Add session‑aware edge masking: during a given session (e.g., Tokyo), you can mask out edges that are known to be weak during that session (e.g., cross‑asset links that only become significant during London overlap).

This can be implemented by having separate edge sets per session and using gating to combine them.

10. Reduce feature dimensionality via GA or SHAP pruning to improve generalization.
Problem: Including many features (technical indicators, derived metrics) can lead to overfitting, especially with limited data.

Solution:

Use the genetic algorithm to perform feature selection: each chromosome includes a binary mask for which features are used.

Alternatively, use SHAP to compute feature importance after training, and prune features with low importance.

This reduces model complexity and improves out‑of‑sample robustness.

11. Add short‑horizon features (tick momentum, acceleration) to fix signal stickiness.
Problem: The tick model may rely solely on price levels, leading to “sticky” signals that fail to adapt quickly to rapid changes.

Solution:

Incorporate momentum (e.g., return over last 5 ticks) and acceleration (change in momentum) as explicit features.

Also include order flow imbalance (if L2 data available) and volume‑weighted mid‑price to capture short‑term supply/demand dynamics.

These features give the model more responsive short‑term indicators.

12. Include regime duration and transition features to model persistence.
Problem: The model may not know how long a regime (e.g., high volatility) has persisted or when a transition is likely.

Solution:

Add features such as:

“Time since last regime change”

“Expected remaining duration” (based on historical averages)

“Regime persistence indicator” (e.g., 1 if same regime for last N bars)

These help the model understand whether a pattern is likely to continue or reverse.

13. Separate training for different sessions or include session‑conditioned loss weighting.
Problem: The model trained on all sessions may not capture session‑specific dynamics well, especially if the data distribution differs.

Solution:

Option A: Train separate models for each session (Tokyo, London, New York). During inference, use the appropriate model.

Option B: Use a single model but apply loss weighting – assign higher weight to samples from sessions that are historically more important for profitability.

Option C: Use session‑conditioned layers where the model learns different weights for each session (e.g., via conditional batch normalization).

14. Add probability spread filter (|p_buy - p_sell|) to remove weak signals.
Problem: When the macro model’s predicted probability is near 0.5, the direction is ambiguous, and acting on it may cause unnecessary trades.

Solution:

Compute the absolute difference between the probability of up and down (or between long and short). For binary classification, it’s simply 
∣
2
p
−
1
∣
∣2p−1∣.

Only trade if this spread exceeds a threshold (e.g., 0.2). This eliminates low‑conviction signals.

The threshold can be made adaptive (e.g., higher during uncertain regimes).

15. Validate with execution‑aware backtest (limit fills + latency), not just predictions.
Problem: Evaluating only on prediction accuracy ignores the real‑world costs of slippage, partial fills, and latency.

Solution:

Build a simulation engine that models:

Latency: time from signal generation to order placement.

Limit order fills: only filled if price trades through the limit.

Market impact: for large orders, approximate impact based on volume.

Spread costs: always pay half the bid‑ask spread.

Backtest with the same walk‑forward windows, using realistic assumptions, to obtain credible performance metrics.

Integration into the Proposed Model
All these fixes can be incorporated into the previously defined architecture without changing its core cooperative graph structure. They act as refinements in data preprocessing, model training, signal generation, risk management, and backtesting. The GA can be extended to optimise the hyperparameters of these fixes (e.g., adaptive threshold coefficients, cooldown duration, correlation exposure cap) alongside the neural network hyperparameters.

By implementing these fixes, you will significantly increase the robustness, realism, and out‑of‑sample performance of the FX trading system.