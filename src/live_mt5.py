"""
live_mt5.py — MT5 Live Trading Loop for Algo C2 v2

Connects MetaTrader5 to LiveEngine, polls M5 bars every bar close,
places/closes positions based on CatBoost v2 signals.

Usage:
    python src/live_mt5.py --model-dir C:/Algo-C2/models/catboost_v2 --balance 50
    python src/live_mt5.py --model-dir C:/Algo-C2/models/catboost_v2 --paper   (paper trading, no orders)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: pip install MetaTrader5")
    sys.exit(1)

from universe import ALL_INSTRUMENTS, FX_PAIRS, PIP_SIZES
from live_engine import LiveEngine, Signal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEFRAME    = None          # set after mt5.initialize()
WARMUP_FETCH = 370           # M5 bars to fetch per instrument for warmup
RISK_PCT     = 0.01          # 1% risk per trade
LEVERAGE     = 500
ATR_MULT_TP  = 2.0
ATR_MULT_SL  = 1.5
MIN_SL_SPREAD = 3.0          # SL must be at least 3× spread
LOT_MIN      = 0.01
LOT_MAX      = 10.0
MAX_OPEN          = 3     # max concurrent positions
CONF_MIN          = 0.12  # |p_buy - p_sell| minimum — skip weak trades
SL_COOLDOWN_BARS  = 4     # bars to skip after SL hit on same symbol
BAR_SECONDS  = 300           # M5 = 300s
BAR_BUFFER   = 5             # wait N seconds after bar close before reading

PIP_VALUES = {               # pip value per standard lot in USD
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY": 6.70, "EURJPY": 6.70, "GBPJPY": 6.70, "AUDJPY": 6.70,
    "CADJPY": 6.70, "CHFJPY": 6.70, "NZDJPY": 6.70,
    "USDCHF": 11.10, "EURCHF": 11.10, "GBPCHF": 11.10, "AUDCHF": 11.10,
    "CADCHF": 11.10, "NZDCHF": 11.10,
    "EURGBP": 12.70,
    "USDCAD": 7.40, "AUDCAD": 7.40, "EURCAD": 7.40, "GBPCAD": 7.40,
    "NZDCAD": 7.30,
    "AUDNZD": 10.0, "EURNZD": 10.0, "GBPNZD": 10.0,
    "EURAUD": 10.0, "GBPAUD": 10.0,
    "BTCUSD": 1.0, "XAUUSD": 1.0, "XAGUSD": 1.0,
    "US30": 1.0, "USDMXN": 10.0, "USDZAR": 10.0, "XBRUSD": 1.0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(__file__).parent.parent / "live_mt5.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("live_mt5")

# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def mt5_connect() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    if info is None or not info.connected:
        log.error("MT5 terminal not connected to broker")
        return False
    log.info(f"MT5 connected — version {mt5.version()}, trade_allowed={info.trade_allowed}")
    return True


def fetch_m5_bars(sym: str, count: int) -> dict | None:
    """Fetch last `count` M5 bars for a symbol. Returns ohlc dict or None."""
    if not mt5.symbol_select(sym, True):
        log.debug(f"fetch_m5_bars: symbol_select({sym}) failed — {mt5.last_error()}")
        return None
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        log.debug(f"fetch_m5_bars: no data for {sym} — {mt5.last_error()}")
        return None
    o  = rates["open"].astype(np.float64)
    h  = rates["high"].astype(np.float64)
    l  = rates["low"].astype(np.float64)
    c  = rates["close"].astype(np.float64)
    sp = rates["spread"].astype(np.float64)
    tk = rates["tick_volume"].astype(np.float64)
    dt = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M")
          for t in rates["time"]]
    return {"o": o, "h": h, "l": l, "c": c, "sp": sp, "tk": tk, "dt": dt}


def get_current_bar(sym: str) -> dict | None:
    """Get the last completed M5 bar (index 1 = one bar ago = last closed)."""
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 1)
    if rates is None or len(rates) == 0:
        log.debug(f"get_current_bar({sym}): no data — {mt5.last_error()}")
        return None
    r = rates[0]
    return {
        "o":  float(r["open"]),
        "h":  float(r["high"]),
        "l":  float(r["low"]),
        "c":  float(r["close"]),
        "sp": float(r["spread"]),
        "tk": float(r["tick_volume"]),
    }


def get_account_balance() -> float:
    ai = mt5.account_info()
    return float(ai.balance) if ai else 50.0


def get_open_positions() -> dict[str, dict]:
    """Returns {symbol: position_info} for all open positions."""
    positions = mt5.positions_get()
    if positions is None:
        return {}
    result = {}
    for p in positions:
        result[p.symbol] = {
            "ticket":    p.ticket,
            "direction": 1 if p.type == 0 else -1,  # 0=BUY, 1=SELL in MT5
            "lot":       p.volume,
            "entry":     p.price_open,
            "sl":        p.sl,
            "tp":        p.tp,
        }
    return result


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def compute_atr(ohlc: dict, window: int = 14) -> float:
    """ATR from last `window` bars."""
    c = ohlc["c"]
    h = ohlc["h"]
    l = ohlc["l"]
    n = min(window, len(c) - 1)
    if n < 1:
        return float(c[-1]) * 0.002
    tr = np.maximum(h[-n:] - l[-n:],
         np.maximum(abs(h[-n:] - c[-n-1:-1]),
                    abs(l[-n:] - c[-n-1:-1])))
    return float(tr.mean())


def compute_lot(balance: float, sl_pips: float, pair: str, price: float) -> float:
    pip_value = PIP_VALUES.get(pair, 10.0)
    risk_usd  = balance * RISK_PCT
    if sl_pips < 1e-10:
        return 0.0
    lot = risk_usd / (sl_pips * pip_value)
    return float(np.clip(lot, LOT_MIN, LOT_MAX))


def place_order(sym: str, direction: int, atr: float, spread: float,
                balance: float, paper: bool = False) -> bool:
    """Place a market order with ATR-based TP/SL. Returns True on success."""
    pip  = PIP_SIZES.get(sym, 0.0001)
    tick = mt5.symbol_info(sym)
    if tick is None:
        log.warning(f"  symbol_info({sym}) = None, skipping")
        return False

    # Current bid/ask
    bid  = tick.bid
    ask  = tick.ask
    spread_price = ask - bid

    # ATR-based TP/SL (in price units)
    tp_dist = max(atr * ATR_MULT_TP, spread_price * MIN_SL_SPREAD)
    sl_dist = max(atr * ATR_MULT_SL, spread_price * MIN_SL_SPREAD)

    if direction == 1:    # BUY
        price  = ask
        tp     = price + tp_dist
        sl     = price - sl_dist
        otype  = mt5.ORDER_TYPE_BUY
    else:                 # SELL
        price  = bid
        tp     = price - tp_dist
        sl     = price + sl_dist
        otype  = mt5.ORDER_TYPE_SELL

    # Position sizing
    sl_pips = sl_dist / pip
    lot     = compute_lot(balance, sl_pips, sym, price)
    if lot == 0.0:
        log.warning(f"  {sym}: lot=0 (SL too small?), skipping")
        return False

    log.info(f"  ORDER {sym} {'BUY' if direction==1 else 'SELL'} lot={lot:.3f} "
             f"@{price:.5f} SL={sl:.5f} TP={tp:.5f}")

    if paper:
        return True

    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   sym,
        "volume":   lot,
        "type":     otype,
        "price":    price,
        "sl":       round(sl, tick.digits),
        "tp":       round(tp, tick.digits),
        "deviation": 20,
        "magic":    202600,
        "comment":  "algoc2v2",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        log.error(f"  {sym}: order_send failed — retcode={code}")
        return False

    log.info(f"  {sym}: order placed — ticket={result.order}")
    return True


def close_position(ticket: int, sym: str, direction: int,
                   lot: float, paper: bool = False) -> bool:
    """Close an open position by ticket."""
    log.info(f"  CLOSE {sym} ticket={ticket}")
    if paper:
        return True

    tick = mt5.symbol_info(sym)
    if tick is None:
        return False

    close_type  = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if direction == 1 else tick.ask

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    sym,
        "volume":    lot,
        "type":      close_type,
        "position":  ticket,
        "price":     close_price,
        "deviation": 20,
        "magic":     202600,
        "comment":   "algoc2v2_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        log.error(f"  Close {sym}: failed — retcode={code}")
        return False
    return True


# ---------------------------------------------------------------------------
# Bar timing
# ---------------------------------------------------------------------------

def next_m5_close() -> datetime:
    """Return the datetime of the next M5 bar close (UTC)."""
    now  = datetime.utcnow()
    mins = now.minute
    next_min = ((mins // 5) + 1) * 5
    if next_min >= 60:
        base = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        base = now.replace(minute=next_min, second=0, microsecond=0)
    return base


def wait_for_next_bar():
    """Sleep until BAR_BUFFER seconds after the next M5 close."""
    target = next_m5_close() + timedelta(seconds=BAR_BUFFER)
    now    = datetime.utcnow()
    delta  = (target - now).total_seconds()
    if delta > 0:
        log.info(f"  Waiting {delta:.0f}s until next M5 bar ({target.strftime('%H:%M:%S')} UTC)")
        time.sleep(delta)


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def warmup_engine(engine: LiveEngine, instruments: list[str]) -> dict[str, dict]:
    """Fetch historical bars for all instruments and feed into engine."""
    log.info(f"Fetching {WARMUP_FETCH} warmup bars per instrument...")
    ohlc_history: dict[str, dict] = {}

    for sym in instruments:
        try:
            data = fetch_m5_bars(sym, WARMUP_FETCH)
        except Exception as e:
            log.error(f"  [WARMUP] {sym}: fetch failed — {e}")
            data = None
        if data is None:
            log.warning(f"  [WARMUP] {sym}: no history (mt5={mt5.last_error()}), will be forward-filled")
            continue
        ohlc_history[sym] = data

    if not ohlc_history:
        log.error("No instruments loaded — check MT5 connection")
        return {}

    # Use BTCUSD as the timeline reference
    ref = ohlc_history.get("BTCUSD", next(iter(ohlc_history.values())))
    n_bars = len(ref["dt"])
    log.info(f"  Feeding {n_bars} warmup bars into engine...")

    for i in range(n_bars):
        dt_str = ref["dt"][i] if "dt" in ref else f"2026-01-01 {i*5//60:02d}:{(i*5)%60:02d}"
        bars = {}
        for sym, data in ohlc_history.items():
            if i < len(data["c"]):
                bars[sym] = {
                    "o":  float(data["o"][i]),
                    "h":  float(data["h"][i]),
                    "l":  float(data["l"][i]),
                    "c":  float(data["c"][i]),
                    "sp": float(data["sp"][i]),
                    "tk": float(data["tk"][i]),
                }
        engine.on_bar(dt_str, bars)

    log.info(f"  Warmup complete. Engine warmed={engine.is_warmed_up}, bars={engine.bar_count}")
    return ohlc_history


# ---------------------------------------------------------------------------
# Per-second positions writer (background thread)
# ---------------------------------------------------------------------------

_POSITIONS_JSON = Path(__file__).parent.parent / "live_positions.json"
_pos_lock       = threading.Lock()
_pos_snapshot:  dict  = {}    # {sym: position dict} — updated from run_live
_equity_snap:   float = 50.0  # updated from run_live


def _positions_writer() -> None:
    """Write current mark-to-market prices to live_positions.json every second."""
    while True:
        try:
            with _pos_lock:
                positions = dict(_pos_snapshot)
                base_eq   = _equity_snap

            result: list[dict] = []
            for sym, pos in positions.items():
                tick = mt5.symbol_info_tick(sym)
                if tick is None:
                    continue
                now_px = tick.bid if pos["direction"] == 1 else tick.ask
                upnl   = _paper_pnl(sym, pos["direction"], pos["lot"], pos["entry"], now_px)
                result.append({
                    "sym":      sym,
                    "side":     "BUY" if pos["direction"] == 1 else "SELL",
                    "lot":      pos["lot"],
                    "entry":    pos["entry"],
                    "sl":       pos["sl"],
                    "tp":       pos["tp"],
                    "now":      now_px,
                    "upnl":     round(upnl, 2),
                    "bar_time": pos.get("bar_time", ""),
                })

            total_upnl = round(sum(p["upnl"] for p in result), 2)
            payload = {
                "positions":   result,
                "total_upnl":  total_upnl,
                "equity":      round(base_eq + total_upnl, 2),
                "ts":          datetime.utcnow().strftime("%H:%M:%S UTC"),
            }
            with open(str(_POSITIONS_JSON), "w") as f:
                json.dump(payload, f)
        except Exception:
            pass
        time.sleep(1.0)


def _paper_pnl(sym: str, direction: int, lot: float, entry: float, exit_price: float) -> float:
    """Approximate USD P&L for a paper trade."""
    pip      = PIP_SIZES.get(sym, 0.0001)
    pip_val  = PIP_VALUES.get(sym, 10.0)
    return (exit_price - entry) * direction / pip * pip_val * lot


def run_live(model_dir: str, balance_override: float | None,
             paper: bool, entry_threshold: float, exit_threshold: float):

    if not mt5_connect():
        sys.exit(1)

    balance = balance_override or get_account_balance()
    log.info(f"Account balance: ${balance:.2f}  paper={paper}")

    # Load engine
    engine = LiveEngine(
        model_dir,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    log.info(f"LiveEngine loaded — BTC feat={len(engine.btc_feat_names)} "
             f"FX feat={len(engine.fx_feat_names)} pairs={len(engine.fx_pairs)}")

    instruments = engine.inst_ordered

    # Warmup
    warmup_engine(engine, instruments)

    log.info("=== Entering live loop ===")
    consecutive_errors = 0

    # Paper-mode virtual book
    paper_positions: dict[str, dict] = {}   # {sym: {direction,lot,entry,sl,tp,bar_time}}
    paper_equity: float = balance            # starts at balance, tracks realised P&L
    sl_cooldown:  dict[str, int]     = {}   # {sym: bars_remaining} after SL hit

    # Start per-second price writer (paper mode only)
    if paper:
        global _equity_snap
        _equity_snap = balance
        t = threading.Thread(target=_positions_writer, daemon=True, name="pos-writer")
        t.start()
        log.info("  Per-second positions writer started")

    while True:
        try:
            wait_for_next_bar()

            # Collect current closed bar for all instruments
            dt_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            bars = {}
            for sym in instruments:
                b = get_current_bar(sym)
                if b is not None:
                    bars[sym] = b

            # ── Paper: check SL/TP hits & log unrealised P&L ──────────────
            if paper and paper_positions:
                for sym, pos in list(paper_positions.items()):
                    tick_info = mt5.symbol_info(sym)
                    if tick_info is None:
                        continue
                    now_px = tick_info.bid if pos["direction"] == 1 else tick_info.ask
                    upnl   = _paper_pnl(sym, pos["direction"], pos["lot"], pos["entry"], now_px)
                    sign   = "+" if upnl >= 0 else ""
                    log.info(f"  PAPER_POS {sym} {'BUY' if pos['direction']==1 else 'SELL'} "
                             f"entry={pos['entry']:.5f} now={now_px:.5f} "
                             f"upnl={sign}{upnl:.2f}")

                    # SL/TP auto-close
                    if pos["direction"] == 1:
                        hit = "TP" if now_px >= pos["tp"] else ("SL" if now_px <= pos["sl"] else None)
                    else:
                        hit = "TP" if now_px <= pos["tp"] else ("SL" if now_px >= pos["sl"] else None)

                    if hit:
                        exit_px  = pos["tp"] if hit == "TP" else pos["sl"]
                        rpnl     = _paper_pnl(sym, pos["direction"], pos["lot"], pos["entry"], exit_px)
                        paper_equity += rpnl
                        sign2 = "+" if rpnl >= 0 else ""
                        log.info(f"  PAPER_CLOSE {sym} {'BUY' if pos['direction']==1 else 'SELL'} "
                                 f"lot={pos['lot']:.3f} entry={pos['entry']:.5f} exit={exit_px:.5f} "
                                 f"pnl={sign2}{rpnl:.2f} reason={hit} equity=${paper_equity:.2f}")
                        del paper_positions[sym]
                        if hit == "SL":
                            sl_cooldown[sym] = SL_COOLDOWN_BARS  # Fix #5
                            log.info(f"  COOLDOWN {sym} {SL_COOLDOWN_BARS} bars after SL")
                        with _pos_lock:
                            _pos_snapshot.pop(sym, None)
                            _equity_snap = paper_equity

            # Run signal engine
            signals = engine.on_bar(dt_str, bars)

            if not signals:
                log.info(f"  {dt_str}  no signals (warmup={not engine.is_warmed_up})")
                consecutive_errors = 0
                continue

            # Refresh balance
            if not paper:
                balance = get_account_balance()

            # Get open positions
            if paper:
                open_positions = paper_positions
            else:
                open_positions = get_open_positions()
            n_open = len(open_positions)

            # Tick down SL cooldown counters
            sl_cooldown = {s: n-1 for s, n in sl_cooldown.items() if n > 1}

            display_bal = paper_equity if paper else balance
            log.info(f"  {dt_str}  signals={len(signals)}  "
                     f"open={n_open}  balance=${display_bal:.2f}  "
                     f"regime={engine._last_regime}")

            # Fix #7: sort by conviction score so highest-probability pairs trade first
            ranked = sorted(
                signals.items(),
                key=lambda kv: max(kv[1].p_buy, kv[1].p_sell),
                reverse=True,
            )

            for pair, sig in ranked:
                sym = pair

                # Fix #4: skip low-confidence signals
                confidence = abs(sig.p_buy - sig.p_sell)
                log.info(f"    {pair:12s}  dir={sig.direction:+d}  "
                         f"P(B/H/S)={sig.p_buy:.3f}/{sig.p_hold:.3f}/{sig.p_sell:.3f}  "
                         f"conf={confidence:.3f}  gate={sig.gate:.2f}  regime={sig.regime}")

                pos = open_positions.get(sym)

                # ── Close if direction reversed or signal went flat ────────
                if pos is not None:
                    if sig.direction == 0 or sig.direction != pos["direction"]:
                        if paper:
                            tick_info = mt5.symbol_info(sym)
                            exit_px   = (tick_info.bid if pos["direction"] == 1 else tick_info.ask) \
                                        if tick_info else pos["entry"]
                            rpnl      = _paper_pnl(sym, pos["direction"], pos["lot"], pos["entry"], exit_px)
                            paper_equity += rpnl
                            sign3 = "+" if rpnl >= 0 else ""
                            log.info(f"  PAPER_CLOSE {sym} {'BUY' if pos['direction']==1 else 'SELL'} "
                                     f"lot={pos['lot']:.3f} entry={pos['entry']:.5f} exit={exit_px:.5f} "
                                     f"pnl={sign3}{rpnl:.2f} reason=SIGNAL equity=${paper_equity:.2f}")
                            del paper_positions[sym]
                            with _pos_lock:
                                _pos_snapshot.pop(sym, None)
                                _equity_snap = paper_equity
                        else:
                            close_position(pos["ticket"], sym, pos["direction"],
                                           pos["lot"], paper=False)
                            open_positions.pop(sym, None)
                        n_open = max(0, n_open - 1)

                # ── Open new position ─────────────────────────────────────
                if (sig.direction != 0
                        and sym not in open_positions
                        and n_open < MAX_OPEN
                        and sig.gate > 0.0
                        and sig.size > 0.05
                        and confidence >= CONF_MIN          # Fix #4: skip weak signals
                        and sl_cooldown.get(sym, 0) == 0):  # Fix #5: SL cooldown

                    recent = fetch_m5_bars(sym, 20)
                    if recent is not None:
                        atr    = compute_atr(recent, window=14)
                        spread = float(recent["sp"][-1])
                    else:
                        atr, spread = 0.0, 0.0

                    if atr > 0:
                        if paper:
                            tick_info = mt5.symbol_info(sym)
                            if tick_info is None:
                                continue
                            pip       = PIP_SIZES.get(sym, 0.0001)
                            entry_px  = tick_info.ask if sig.direction == 1 else tick_info.bid
                            sp_px     = tick_info.ask - tick_info.bid
                            tp_dist   = max(atr * ATR_MULT_TP, sp_px * MIN_SL_SPREAD)
                            sl_dist   = max(atr * ATR_MULT_SL, sp_px * MIN_SL_SPREAD)
                            tp        = entry_px + tp_dist if sig.direction == 1 else entry_px - tp_dist
                            sl        = entry_px - sl_dist if sig.direction == 1 else entry_px + sl_dist
                            lot       = compute_lot(paper_equity, sl_dist / pip, sym, entry_px)
                            if lot == 0.0:
                                continue
                            log.info(f"  PAPER_OPEN {sym} {'BUY' if sig.direction==1 else 'SELL'} "
                                     f"lot={lot:.3f} @{entry_px:.5f} SL={sl:.5f} TP={tp:.5f}")
                            paper_positions[sym] = {
                                "direction": sig.direction,
                                "lot":       lot,
                                "entry":     entry_px,
                                "sl":        sl,
                                "tp":        tp,
                                "ticket":    0,
                                "bar_time":  dt_str,
                            }
                            with _pos_lock:
                                _pos_snapshot[sym] = paper_positions[sym].copy()
                                _equity_snap = paper_equity
                            n_open += 1
                        else:
                            ok = place_order(sym, sig.direction, atr, spread, balance)
                            if ok:
                                n_open += 1

            consecutive_errors = 0

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            break
        except ConnectionError as e:
            consecutive_errors += 1
            log.error(f"[MT5_CONN] connection lost #{consecutive_errors}: {e} — mt5={mt5.last_error()}")
            if consecutive_errors >= 5:
                log.error("[MT5_CONN] too many connection errors — attempting reconnect")
                mt5.shutdown()
                time.sleep(10)
                if not mt5_connect():
                    log.error("[MT5_CONN] reconnect failed — stopping")
                    break
                consecutive_errors = 0
            else:
                time.sleep(15)
        except ValueError as e:
            # usually a bad bar or NaN from MT5 data — log and skip
            log.error(f"[DATA_ERR] bad value at bar {dt_str}: {e}", exc_info=True)
            consecutive_errors += 1
            if consecutive_errors >= 10:
                log.error("[DATA_ERR] too many bad bars — stopping")
                break
        except Exception as e:
            consecutive_errors += 1
            log.error(f"[LOOP_ERR] #{consecutive_errors} type={type(e).__name__} bar={dt_str}: {e}",
                      exc_info=True)
            if consecutive_errors >= 10:
                log.error("[LOOP_ERR] too many consecutive errors — stopping")
                break
            time.sleep(30)

    mt5.shutdown()
    log.info("MT5 shutdown. Live loop ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  default="C:/Algo-C2/models/catboost_v2",
                        help="Directory with .cbm models + feature_names.json")
    parser.add_argument("--balance",    type=float, default=None,
                        help="Override account balance (default: read from MT5)")
    parser.add_argument("--paper",      action="store_true",
                        help="Paper mode: log signals but place no real orders")
    parser.add_argument("--entry-threshold", type=float, default=0.40)
    parser.add_argument("--exit-threshold",  type=float, default=0.30)
    args = parser.parse_args()

    run_live(
        model_dir=args.model_dir,
        balance_override=args.balance,
        paper=args.paper,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
    )
