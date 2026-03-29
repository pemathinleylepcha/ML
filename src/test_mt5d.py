import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize(timeout=10000)
print(f"Version: {mt5.version()}", flush=True)

# Check terminal info
info = mt5.terminal_info()
print(f"Connected: {info.connected if info else 'N/A'}", flush=True)
print(f"Trade allowed: {info.trade_allowed if info else 'N/A'}", flush=True)

# Try candle data (H1) - simpler than ticks
for sym in ["EURUSD", "AUDCAD"]:
    mt5.symbol_select(sym, True)
    # Try copy_rates_from (candles)
    rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_H1, datetime(2026,3,20), 5)
    err = mt5.last_error()
    n = len(rates) if rates is not None else 0
    print(f"  {sym} H1 rates from 2026-03-20: n={n}  err={err}", flush=True)

    # Try copy_ticks_from with very recent date
    ticks = mt5.copy_ticks_from(sym, datetime(2026,3,25,0,0), 5, mt5.COPY_TICKS_ALL)
    err2 = mt5.last_error()
    n2 = len(ticks) if ticks is not None else 0
    print(f"  {sym} ticks from 2026-03-25: n={n2}  err={err2}", flush=True)

mt5.shutdown()
print("Done.", flush=True)
