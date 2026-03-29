"""
Algo C2 v2 — Universe Definition
Single source of truth for all 43-node instrument constants.
"""

# ── Canonical 43-node instrument list (order is fixed — index = node ID) ────

ALL_INSTRUMENTS = [
    # 24x7 tradeable (1) — node 0
    "BTCUSD",
    # 24x5 tradeable — majors (7) — nodes 1-7
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    # 24x5 tradeable — minors (21) — nodes 8-28
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
    "NZDCAD", "NZDCHF", "NZDJPY",
    # 24x5 signal-only — indices (8) — nodes 29-36
    "AUS200", "US30", "GER40", "UK100", "NAS100", "EUSTX50", "JPN225", "SPX500",
    # 24x5 signal-only — energy (2) — nodes 37-38
    "XTIUSD", "XBRUSD",
    # 24x5 signal-only — metals (2) — nodes 39-40
    "XAUUSD", "XAGUSD",
    # 24x5 signal-only — exotic FX (2) — nodes 41-42
    "USDMXN", "USDZAR",
]

N_NODES = len(ALL_INSTRUMENTS)  # 43

# ── Subsets ──────────────────────────────────────────────────────────────────

TRADEABLE = ALL_INSTRUMENTS[:29]          # First 29 generate trade signals
SIGNAL_ONLY = ALL_INSTRUMENTS[29:]        # Last 14 are feature-only, never trade
SUBNET_24x7 = ["BTCUSD"]
SUBNET_24x5_TRADEABLE = ALL_INSTRUMENTS[1:29]   # 28 FX pairs
SUBNET_24x5_SIGNAL = ALL_INSTRUMENTS[29:]        # 14 signal-only
SUBNET_24x5 = ALL_INSTRUMENTS[1:]               # Everything except BTCUSD

# Node index lookup
NODE_IDX = {pair: i for i, pair in enumerate(ALL_INSTRUMENTS)}

# ── Classification sets ───────────────────────────────────────────────────────

FX_PAIRS = [                              # 28 tradeable FX (no crypto)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
    "NZDCAD", "NZDCHF", "NZDJPY",
]

JPY_PAIRS = {p for p in ALL_INSTRUMENTS if p.endswith("JPY") or p.startswith("JPY")}
INDICES = ["AUS200", "US30", "GER40", "UK100", "NAS100", "EUSTX50", "JPN225", "SPX500"]
ENERGY = ["XTIUSD", "XBRUSD"]
METALS = ["XAUUSD", "XAGUSD"]
EXOTIC_FX = ["USDMXN", "USDZAR"]

# ── Pip sizes ─────────────────────────────────────────────────────────────────

PIP_SIZES: dict[str, float] = {
    # FX
    "AUDCAD": 1e-4, "AUDCHF": 1e-4, "AUDJPY": 0.01, "AUDNZD": 1e-4, "AUDUSD": 1e-4,
    "CADCHF": 1e-4, "CADJPY": 0.01, "CHFJPY": 0.01,
    "EURAUD": 1e-4, "EURCAD": 1e-4, "EURCHF": 1e-4, "EURGBP": 1e-4, "EURJPY": 0.01,
    "EURNZD": 1e-4, "EURUSD": 1e-4,
    "GBPAUD": 1e-4, "GBPCAD": 1e-4, "GBPCHF": 1e-4, "GBPJPY": 0.01, "GBPNZD": 1e-4,
    "GBPUSD": 1e-4,
    "NZDCAD": 1e-4, "NZDCHF": 1e-4, "NZDJPY": 0.01, "NZDUSD": 1e-4,
    "USDCAD": 1e-4, "USDCHF": 1e-4, "USDJPY": 0.01,
    # Crypto
    "BTCUSD": 1.0,
    # Indices (index point)
    "AUS200": 1.0, "US30": 1.0, "GER40": 1.0, "UK100": 1.0,
    "NAS100": 1.0, "EUSTX50": 1.0, "JPN225": 1.0, "SPX500": 0.1,
    # Energy
    "XTIUSD": 0.01, "XBRUSD": 0.01,
    # Metals
    "XAUUSD": 0.1, "XAGUSD": 0.01,
    # Exotic FX
    "USDMXN": 1e-4, "USDZAR": 1e-4,
}

# ── Pip values (USD per pip per standard lot) ─────────────────────────────────
# Used for position sizing. Approximate mid-market values.

PIP_VALUES: dict[str, float] = {
    "AUDUSD": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "NZDUSD": 10.0,
    "USDCAD": 7.7,  "USDCHF": 10.9, "USDJPY": 6.8,
    "AUDCAD": 7.7,  "AUDCHF": 10.9, "AUDJPY": 6.8, "AUDNZD": 10.0,
    "CADCHF": 10.9, "CADJPY": 6.8,  "CHFJPY": 6.8,
    "EURAUD": 10.0, "EURCAD": 7.7,  "EURCHF": 10.9, "EURGBP": 12.5,
    "EURJPY": 6.8,  "EURNZD": 10.0,
    "GBPAUD": 10.0, "GBPCAD": 7.7,  "GBPCHF": 10.9, "GBPJPY": 6.8,
    "GBPNZD": 10.0,
    "NZDCAD": 7.7,  "NZDCHF": 10.9, "NZDJPY": 6.8,
    "BTCUSD": 1.0,
    # Signal-only: not traded, pip value irrelevant but set to 1.0
    "AUS200": 1.0, "US30": 1.0, "GER40": 1.0, "UK100": 1.0,
    "NAS100": 1.0, "EUSTX50": 1.0, "JPN225": 1.0, "SPX500": 1.0,
    "XTIUSD": 1.0, "XBRUSD": 1.0,
    "XAUUSD": 1.0, "XAGUSD": 1.0,
    "USDMXN": 1.0, "USDZAR": 1.0,
}

# ── Session sets (tradeable pairs by FX session) ──────────────────────────────

SESSION_TOKYO = frozenset([
    "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "AUDUSD", "NZDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF",
])
SESSION_LONDON = frozenset(FX_PAIRS)  # All FX active during London
SESSION_NY = frozenset([
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "EURJPY", "GBPJPY", "EURGBP", "EURCAD", "GBPCAD",
    "AUDCAD", "NZDCAD", "CADJPY", "CADCHF",
])
SESSION_SYDNEY = frozenset([
    "AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "AUDNZD",
    "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF",
])

# ── Regional correlation map (signal-only → FX pairs they influence) ─────────

REGIONAL_CORRELATION_MAP: dict[str, list[str]] = {
    "AUS200":  ["AUDUSD", "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD"],
    "GER40":   ["EURUSD", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD"],
    "UK100":   ["GBPUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "EURGBP"],
    "JPN225":  ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
    "US30":    ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
    "NAS100":  ["EURUSD", "GBPUSD", "USDJPY"],
    "SPX500":  ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
    "EUSTX50": ["EURUSD", "EURGBP", "EURJPY", "EURCHF"],
    "XTIUSD":  ["USDCAD", "AUDCAD", "EURCAD", "GBPCAD", "NZDCAD", "CADCHF", "CADJPY"],
    "XBRUSD":  ["USDCAD", "AUDCAD", "EURCAD"],
    "XAUUSD":  ["USDJPY", "USDCHF", "EURJPY", "CHFJPY", "GBPJPY"],
    "XAGUSD":  ["USDJPY", "USDCHF"],
    "USDMXN":  ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "USDZAR":  ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
}

# ── ATR baselines (approx daily ATR in price units, for risk normalisation) ───

ATR_BASELINE: dict[str, float] = {
    # FX (in price)
    "EURUSD": 0.0080, "GBPUSD": 0.0110, "USDJPY": 0.80, "USDCHF": 0.0075,
    "AUDUSD": 0.0065, "NZDUSD": 0.0060, "USDCAD": 0.0085,
    "AUDCAD": 0.0070, "AUDCHF": 0.0065, "AUDJPY": 0.65, "AUDNZD": 0.0050,
    "CADCHF": 0.0060, "CADJPY": 0.75, "CHFJPY": 0.80,
    "EURAUD": 0.0120, "EURCAD": 0.0090, "EURCHF": 0.0060, "EURGBP": 0.0060,
    "EURJPY": 1.10, "EURNZD": 0.0130,
    "GBPAUD": 0.0150, "GBPCAD": 0.0120, "GBPCHF": 0.0110, "GBPJPY": 1.50,
    "GBPNZD": 0.0160,
    "NZDCAD": 0.0065, "NZDCHF": 0.0060, "NZDJPY": 0.65,
    # Crypto
    "BTCUSD": 3000.0,
    # Indices (in index points)
    "AUS200": 45.0, "US30": 350.0, "GER40": 150.0, "UK100": 60.0,
    "NAS100": 200.0, "EUSTX50": 40.0, "JPN225": 300.0, "SPX500": 40.0,
    # Energy (in USD)
    "XTIUSD": 1.5, "XBRUSD": 1.5,
    # Metals (in USD)
    "XAUUSD": 25.0, "XAGUSD": 0.50,
    # Exotic FX
    "USDMXN": 0.30, "USDZAR": 0.40,
}

# ── Timeframe definitions ────────────────────────────────────────────────────

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]
TIMEFRAME_FREQ = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "H12": "12h", "D1": "1D", "W1": "1W", "MN1": "1ME",
}
TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "H12": 720, "D1": 1440, "W1": 10080, "MN1": 43200,
}

# ── TDA thresholds (43-node graph) ────────────────────────────────────────────
# Scaled from v1 (35-node): FRAGMENTED 20→25, TRANSITIONAL 4→5
# HIGH_STRESS and LOW_VOL thresholds are topological (don't scale with N)

TDA_THRESHOLDS = {
    "FRAGMENTED":   {"b0_gt": 25},
    "HIGH_STRESS":  {"h1_life_gt": 0.8},
    "TRANSITIONAL": {"b1_gt": 5},
    "LOW_VOL":      {"b1_le": 1, "h1_life_lt": 0.2},
}

SPECTRAL_GAP_WARN = 0.004   # Slightly lower than v1's 0.005 for 43-node graph


def is_tradeable(instrument: str) -> bool:
    return instrument in TRADEABLE


def is_signal_only(instrument: str) -> bool:
    return instrument in SIGNAL_ONLY


def node_idx(instrument: str) -> int:
    return NODE_IDX[instrument]


def get_subnet(instrument: str) -> str:
    if instrument == "BTCUSD":
        return "24x7"
    return "24x5"
