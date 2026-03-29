# Algo C2 — System Architecture

## Overview

5-phase pipeline: tick ingestion → 1-second resampling → graph/topology → state management → inference + execution.

```
Tick feed (35 pairs)
        │
        ▼
┌─────────────────┐
│  Phase 1        │  Kafka + Protobuf
│  Tick Ingestion │  Partitioned by pair
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 2        │  C# + Rx.NET
│  1000ms Window  │  Tumbling buffer
│  Resampling     │  OHLCV + microstructure biases
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 3        │  C++ workers
│  Graph Laplacian│  Eigen 3.4
│  + TDA          │  GUDHI 3.8 / Ripser
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 4        │  Redis HSET (O(1) retrieval)
│  State + Routing│  RabbitMQ trigger
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 5        │  C++ CatBoost native API
│  Inference +    │  ModelCalcerWrapper
│  Execution      │  FIX protocol → Nginx proxy
└─────────────────┘
```

---

## Phase 1 — Tick Ingestion

**Stack:** Kafka + Protobuf  
**Topic:** `ticks.{PAIR}` (e.g. `ticks.EURUSD`)  
**Partitioning:** by pair symbol  
**Durability:** `acks=1` for throughput, `acks=all` for strict

**Tick fields:**
```
DATE  TIME         BID       ASK       LAST  VOLUME  FLAGS
2026.03.02  00:05:00.718  1.17688  1.17728  —     —       6
```
- `FLAGS=6`: full bid+ask quote  
- `FLAGS=2`: bid only  
- `FLAGS=4`: ask only

---

## Phase 2 — 1000ms Resampling (C# / Rx.NET)

```csharp
Observable.Buffer(TimeSpan.FromMilliseconds(1000))
```

**Output per window per pair (14 features):**

| Feature | Description |
|---------|-------------|
| `bid_open` | First bid in window |
| `bid_high` | Highest bid |
| `bid_low` | Lowest bid |
| `bid_close` | Last bid |
| `ask_close` | Last ask |
| `tick_count` | Total ticks in buffer |
| `tick_direction_bias` | Up-ticks minus down-ticks |
| `spread_mean` | Mean(Ask − Bid) |
| `spread_std` | StdDev(Ask − Bid) |
| `velocity_mean_ms` | Mean time delta between ticks (ms) |
| `local_residual` | From Phase 3 |
| `betti_h0` | From Phase 3 |
| `betti_h1` | From Phase 3 |
| `max_h1_lifespan` | From Phase 3 |

---

## Phase 3 — Graph Laplacian + TDA (C++)

### Normalised Graph Laplacian

```
L = I − D^{−1/2} A D^{−1/2}
```

1. Compute log-return matrix for all 35 pairs over last N bars
2. Pearson correlation → Mantegna distance: `d_ij = sqrt(2(1 − corr_ij))`
3. Gaussian kernel adjacency: `A_ij = exp(−d²/2σ²)` where `σ = median(D)`
4. Degree matrix: `D_ii = sum(A_ij)`
5. Spectral gap λ₂ = algebraic connectivity (Fiedler value)

**Local residual per pair:**
```
ε_i = r_i − (A_i · r) / degree_i
```
Where `r` is the last-bar return vector. Positive ε = underperformed vs network peers.

**Adaptive σ guard:** if `median(D) < 1e-6`, use `σ = mean(D)` to avoid degenerate kernels.

### Persistent Homology (TDA)

- Library: GUDHI 3.8 / Ripser
- Complex: Vietoris-Rips on distance matrix
- Output: H₀ and H₁ Betti numbers, `max_h1_lifespan`

| Feature | Meaning |
|---------|---------|
| `betti_h0` | Connected components (β₀=1 = single cluster) |
| `betti_h1` | Cycle count (β₁=34 in current data = dense loops) |
| `max_h1_lifespan` | Longest-lived H₁ feature = regime fragmentation signal |

---

## Phase 4 — State Management

**Redis HSET key:** `PAIR:TIMESTAMP_MS`  
**Trigger:** RabbitMQ topic exchange `routing.key = state.ready.1000ms.{PAIR}`

C# service and C++ workers write to the same Redis hash. Once complete, a lightweight trigger fires the inference engine.

---

## Phase 5 — Inference + Execution (C++)

```cpp
ModelCalcerWrapper model;
model.LoadFullModelFromFile("catboost_model.cbm");
double prediction = model.CalcModelPrediction(features, cat_features);
```

**Libraries:**
- CatBoost native C++ API
- hiredis (Redis client)
- librdkafka
- FIX protocol → Nginx proxy

---

## Production vs Research stack

| Layer | Production | Research |
|-------|-----------|---------|
| Language | C++ / C# | Python / Polars |
| Streaming | Kafka / Rx.NET | Batch simulation |
| Graph | Eigen 3.4 | scipy.linalg |
| TDA | GUDHI 3.8 | Ripser |
| ML | CatBoost C++ API | sklearn ensembles |
| State | Redis / RabbitMQ | In-memory dict |
| Execution | FIX / Nginx | HTML sim engine |
