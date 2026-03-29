import MetaTrader5 as mt5
from datetime import datetime, timedelta
import sys

mt5.initialize(timeout=10000)
print(f"Version: {mt5.version()}", flush=True)

# Test with symbols we KNOW have tick data already (from existing files)
# Try very recent data (last 3 days) first
test_cases = [
    ("EURUSD",  datetime(2026, 3, 20)),
    ("AUDUSD",  datetime(2026, 3, 20)),
    ("AUDCAD",  datetime(2026, 3, 20)),   # missing pair - recent
    ("AUDCAD",  datetime(2025, 12, 1)),   # missing pair - older
    ("AUDCAD",  datetime(2025, 3, 3)),    # missing pair - start of target range
]

for sym, from_dt in test_cases:
    mt5.symbol_select(sym, True)
    ticks = mt5.copy_ticks_from(sym, from_dt, 20, mt5.COPY_TICKS_ALL)
    err = mt5.last_error()
    n = len(ticks) if ticks is not None else 0
    print(f"  {sym}  from={from_dt.date()}  ticks={n}  err={err}", flush=True)

# Also check account and connection status
acc = mt5.account_info()
print(f"\nAccount: {acc}", flush=True)

mt5.shutdown()
