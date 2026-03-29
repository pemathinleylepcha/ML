# Algo C2 — Signal Pipeline

## 35-Node Graph

### Node composition

| Type | Count | Role |
|------|-------|------|
| FX tradeable pairs | 28 | Signal + orders |
| Non-FX graph anchors | 7 | Signal only (no orders) |
| **Total** | **35** | |

**28 FX pairs:**
`AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY NZDUSD USDCAD USDCHF USDJPY`

**7 non-FX anchors:**
`BTCUSD US30 USDMXN USDZAR XAGUSD XAUUSD XBRUSD`

### Why non-FX nodes matter

Non-FX assets shift the Laplacian eigenvectors and improve residual signal quality for FX nodes. They act as macro regime anchors — when BTC volatility spikes, it propagates through the adjacency matrix and adjusts the expected returns of correlated FX pairs.

**Key non-FX → FX correlations (real Mar 2026 data):**

| Non-FX | Strongest FX link | Corr | Interpretation |
|--------|------------------|------|----------------|
| USDMXN | AUDUSD | −0.749 | EM risk-off moves AUD hardest |
| USDZAR | NZDUSD | −0.667 | Same EM risk channel |
| US30 | AUDUSD | +0.594 | Risk-on equity = AUD bid |
| XAGUSD | NZDUSD | +0.367 | Commodity currency link |
| XAUUSD | AUDUSD | +0.353 | Safe haven / AUD gold correlation |
| BTCUSD | AUDUSD | +0.329 | Risk appetite proxy |
| XBRUSD | EURCAD | −0.231 | Oil → CAD → EUR cross |

---

## 7 Technical Indicators

Computed per pair per 1-min bar:

| # | Indicator | Params | Output |
|---|-----------|--------|--------|
| 1 | RSI | 14 | 0–100, overbought/oversold |
| 2 | MACD | 12/26/9 | line, signal, histogram (pips) |
| 3 | Bollinger Bands | 20 bars, 2σ | %B position, bandwidth |
| 4 | ATR | 14 | True range in pips |
| 5 | Stochastic | 14/3/3 | %K and %D |
| 6 | CCI | 20 | Commodity channel index |
| 7 | Williams %R | 14 | −100 to 0 |

**7-indicator composite signal vote:**

```python
votes = 0
votes += 1 if rsi > 55  else (-1 if rsi < 45  else 0)
votes += 1 if macd > 0  else (-1 if macd < 0  else 0)
votes += 1 if bb > 0.6  else (-1 if bb < 0.4  else 0)
votes += 1 if stoch > 60 else (-1 if stoch < 40 else 0)
votes += 1 if cci > 50  else (-1 if cci < -50  else 0)
votes += 1 if willr > -40 else (-1 if willr < -60 else 0)
votes += 1 if mom > 0.3 else (-1 if mom < -0.3  else 0)

signal = LONG if votes >= 3 else SHORT if votes <= -3 else FLAT
```

---

## 5-Gate Filter

All 5 gates must pass for orders to be placed:

| Gate | Condition | Description |
|------|-----------|-------------|
| G1 | `|netScore| ≥ 0.15` | Minimum directional consensus from 35 nodes |
| G2 | `avgCb ≥ CB_THR` (default 0.65) | CatBoost confidence floor across all nodes |
| G3 | EMA5 slope matches direction | Trend alignment — prevents counter-trend entries |
| G4 | ATR ratio 0.3–2.8× avg | Volatility window — avoids dead markets and spikes |
| G5 | `agreeR ≥ 0.50` | High-CB nodes agree with direction |

**Net score computation:**
```python
netScore = (longCount − shortCount) / 35
# longCount: nodes with res > 0.05
# shortCount: nodes with res < −0.05 (dead-band prevents noise votes)
```

---

## Order Placement Logic

1. Gates 1–5 all pass
2. Filter FX-only nodes: `isFX AND cb > CB_THR AND res aligned with direction AND atr > avg×0.3`
3. Rank candidates: `cb × |res|` (conviction × signal strength)
4. Select top 3 candidates
5. Margin check: skip if used margin > 85% of balance

**Trade parameters:**
```
TP = entry ± atrAvg × 2.0 × pip_size    (×2.0 ATR)
SL = entry ∓ atrAvg × 1.5 × pip_size    (×1.5 ATR)
R:R = 2.0 / 1.5 = 1.33
Break-even win rate = 1/(1+1.33) ≈ 43%
Timeout = 10 bars if neither TP nor SL hit
```

**Entry price (with real spread):**
```
Buy  entry = mid + (spread / 2) × pip_size
Sell entry = mid − (spread / 2) × pip_size
Exit: additional half-spread deducted on close
```

---

## Position Sizing ($50 / 1:500)

```python
risk_usd   = balance × RISK_PCT        # default 2% = $1/trade
lot        = risk_usd / (sl_pips × pip_value_usd)
margin     = (lot × 100000 × price) / 500
```

**Lot bounds:** min 0.001, max 10.0

**Pip values (USD per pip per standard lot):**

| Pair | Pip value |
|------|-----------|
| EURUSD, GBPUSD, AUDUSD, NZDUSD | $10.00 |
| USDJPY, EURJPY, GBPJPY | $6.70 |
| USDCHF, EURCHF, GBPCHF | $11.10 |
| EURGBP | $12.70 |
| USDCAD, AUDCAD, NZDCAD | $7.30–7.40 |
| CADCHF, CADJPY, CHFJPY | $6.70–7.50 |

---

## Session Whitelists (default)

| Session | UTC hours | Default pairs |
|---------|-----------|---------------|
| Sydney | 00–03 | AUDUSD, AUDNZD, AUDCAD, NZDUSD, AUDCHF |
| Tokyo | 03–08 | USDJPY, EURJPY, GBPJPY, AUDJPY, CADJPY |
| London | 08–17 | EURUSD, GBPUSD, EURGBP, GBPCHF, EURCHF |
| NY | 17–24 | EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY |

Session whitelists are fully configurable in the backtest and sim. Enable/disable pairs globally, per-session, or by currency group (Majors, EUR, GBP, JPY, AUD/NZD, CHF, CAD, Crypto, Metals, Energy, EM).
