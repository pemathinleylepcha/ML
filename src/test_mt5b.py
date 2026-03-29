import MetaTrader5 as mt5
from datetime import datetime
import sys

print("Initializing MT5...", flush=True)
r = mt5.initialize(timeout=5000)
print(f"init: {r}  error: {mt5.last_error()}", flush=True)

if not r:
    sys.exit(1)

print(f"Version: {mt5.version()}", flush=True)
mt5.symbol_select("AUDCAD", True)

print("Requesting 50 ticks from 2025-12-01...", flush=True)
ticks = mt5.copy_ticks_from("AUDCAD", datetime(2025, 12, 1), 50, mt5.COPY_TICKS_ALL)
print(f"Result: {len(ticks) if ticks is not None else 'None'}  error: {mt5.last_error()}", flush=True)

if ticks is not None and len(ticks) > 0:
    print(f"dtype: {ticks.dtype.names}", flush=True)
    print(f"First tick: {ticks[0]}", flush=True)

mt5.shutdown()
print("Done.", flush=True)
