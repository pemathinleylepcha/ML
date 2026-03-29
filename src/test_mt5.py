import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()
mt5.symbol_select("AUDCAD", True)
ticks = mt5.copy_ticks_range("AUDCAD", datetime(2025,3,3), datetime(2025,3,5), mt5.COPY_TICKS_ALL)
if ticks is not None:
    print(f"AUDCAD ticks: {len(ticks):,}")
    t = ticks[0]
    print(f"First: time_msc={t['time_msc']} bid={t['bid']} ask={t['ask']} flags={t['flags']}")
    print(f"dtype: {ticks.dtype}")
else:
    print("None", mt5.last_error())
mt5.shutdown()
