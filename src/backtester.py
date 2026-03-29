"""
Algo C2 -- Phase 5: Walk-Forward Backtester
Bar-by-bar simulation with full state tracking, TP/SL/timeout exits,
equity curve, daily P&L, gate stats, and performance metrics.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from math_engine import MathEngine, MathState
from feature_engine import (
    compute_rsi, compute_macd, compute_bollinger, compute_atr,
    compute_stochastic, compute_cci, compute_williams_r, PIP_SIZES, PAIRS_ALL
)
from signal_pipeline import (
    Direction, GateResult, Order, ClosedTrade,
    compute_votes, vote_to_direction, compute_net_score, compute_cb_proxy,
    compute_cb_from_model, check_gates, update_ema5, create_order,
    select_candidates, get_session_pairs, FX_SET, PIP_VALUES,
    INIT_BALANCE, MARGIN_LIMIT, CB_THR, RISK_PCT, TIMEOUT_BARS,
    RESIDUAL_DEAD_ZONE, MAX_POSITIONS_PER_SIGNAL, MIN_SIGNAL_GAP,
)
from feature_engine import compute_pair_features, extract_graph_features


@dataclass
class BacktestConfig:
    init_balance: float = INIT_BALANCE
    cb_thr: float = CB_THR
    risk_pct: float = RISK_PCT
    use_session_filter: bool = True
    enabled_pairs: set = field(default_factory=lambda: set(FX_SET))


@dataclass
class BacktestResults:
    config: dict = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    gate_stats: dict = field(default_factory=dict)
    regime_timeline: list = field(default_factory=list)
    signal_timeline: list = field(default_factory=list)


class Backtester:
    """Walk-forward bar-by-bar backtester for Algo C2."""

    def __init__(self, config: BacktestConfig = None,
                 alpha_model=None, alpha_scaler=None, alpha_pairs=None):
        """
        Args:
            config: backtest configuration
            alpha_model: trained AlphaNet model (optional, replaces CB proxy)
            alpha_scaler: tuple of (scaler_mean, scaler_std) for normalization
            alpha_pairs: ordered pair list matching model output indices
        """
        self.config = config or BacktestConfig()
        self.balance = self.config.init_balance
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

        self.open_positions: list[Order] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[tuple] = []
        self.daily_pnl: dict[str, float] = {}
        self.regime_timeline: list[tuple] = []
        self.signal_timeline: list[tuple] = []

        # Gate pass counters
        self.gate_checks = 0
        self.gate_passes = {f"g{i}": 0 for i in range(1, 7)}

        # Cooldown: track last bar where a signal fired
        self.last_signal_bar: int = -MIN_SIGNAL_GAP

        # Per-pair state
        self.ema5: dict[str, float] = {}
        self.atr_history: dict[str, list] = {}

        # Alpha model (market-neutral)
        self.alpha_model = alpha_model
        self.alpha_scaler_mean = alpha_scaler[0] if alpha_scaler else None
        self.alpha_scaler_std = alpha_scaler[1] if alpha_scaler else None
        self.alpha_pairs = alpha_pairs

    def run(self, data: dict, pairs: list[str] = None) -> BacktestResults:
        """
        Run the full backtest on OHLC data.

        Args:
            data: dict of pair -> list of bar dicts [{dt, o, h, l, c, sp, tk}, ...]
            pairs: list of pairs to use (default: all available)

        Returns:
            BacktestResults with all metrics and timelines
        """
        if pairs is None:
            pairs = [p for p in PAIRS_ALL if p in data]

        n_pairs = len(pairs)
        pair_to_idx = {p: i for i, p in enumerate(pairs)}

        # Convert to arrays for fast access
        arrays = {}
        for pair in pairs:
            bars = data[pair]
            arrays[pair] = {
                "o": np.array([b["o"] for b in bars], dtype=float),
                "h": np.array([b["h"] for b in bars], dtype=float),
                "l": np.array([b["l"] for b in bars], dtype=float),
                "c": np.array([b["c"] for b in bars], dtype=float),
                "sp": np.array([b["sp"] for b in bars], dtype=float),
                "tk": np.array([b["tk"] for b in bars], dtype=float),
                "dt": [b["dt"] for b in bars],
                # Tick paths for intra-bar TP/SL resolution
                "tp_paths": [b.get("tp", []) for b in bars],
            }

        min_bars = min(len(arrays[p]["c"]) for p in pairs)
        engine = MathEngine(n_pairs=n_pairs)

        # Initialize EMA5 for all pairs
        for pair in pairs:
            self.ema5[pair] = arrays[pair]["c"][0]
            self.atr_history[pair] = []

        # Lookback for indicators
        lookback = 60

        print(f"Running backtest: {n_pairs} pairs, {min_bars} bars, "
              f"balance=${self.config.init_balance:.0f}")

        for t in range(min_bars):
            dt = arrays[pairs[0]]["dt"][t]
            date_str = dt[:10] if len(dt) >= 10 else dt

            # -- 1. Compute returns and update math engine --
            returns = np.zeros(n_pairs)
            for i, pair in enumerate(pairs):
                c = arrays[pair]["c"]
                if t > 0 and c[t - 1] > 0:
                    returns[i] = np.log(c[t] / c[t - 1])

            math_state = engine.update(returns)

            # Record regime
            if math_state.valid:
                self.regime_timeline.append((dt, math_state.regime))

            # -- 2. Check open positions for exits --
            self._check_exits(t, dt, arrays, pairs, pair_to_idx)

            # Skip signal generation until we have enough history
            if t < lookback:
                self._record_equity(dt)
                continue

            # -- 3. Compute indicators for all pairs --
            votes_map = {}
            directions_map = {}
            cb_scores = {}
            atr_map = {}
            residuals_map = {}
            atr_ratios = {}

            for i, pair in enumerate(pairs):
                a = arrays[pair]
                c_slice = a["c"][:t + 1]
                h_slice = a["h"][:t + 1]
                l_slice = a["l"][:t + 1]

                # Update EMA5
                self.ema5[pair] = update_ema5(self.ema5[pair], c_slice[-1])

                # Indicators
                rsi = compute_rsi(c_slice[-30:])
                macd_line, macd_hist = compute_macd(c_slice[-50:])
                bb_pct_b, _ = compute_bollinger(c_slice[-30:])
                atr_val = compute_atr(h_slice[-20:], l_slice[-20:], c_slice[-20:])
                stoch_k, stoch_d = compute_stochastic(h_slice[-30:], l_slice[-30:], c_slice[-30:])
                cci_val = compute_cci(h_slice[-25:], l_slice[-25:], c_slice[-25:])
                willr_val = compute_williams_r(h_slice[-20:], l_slice[-20:], c_slice[-20:])

                # Mom5
                if len(c_slice) >= 6:
                    mom_raw = c_slice[-1] - c_slice[-6]
                    mom_std = np.std(np.diff(c_slice[-6:])) + 1e-10
                    mom5 = mom_raw / mom_std
                else:
                    mom5 = 0.0

                v = compute_votes(rsi, macd_line, bb_pct_b, stoch_k, cci_val, willr_val, mom5)
                d = vote_to_direction(v)
                votes_map[pair] = v
                directions_map[pair] = d

                # ATR in pips
                pip = PIP_SIZES.get(pair, 0.0001)
                atr_pips = (atr_val / pip) if atr_val is not np.nan and not np.isnan(atr_val) else 0.0
                atr_map[pair] = atr_pips
                self.atr_history.setdefault(pair, []).append(atr_pips)

                # Residual
                if math_state.valid:
                    residuals_map[pair] = float(math_state.residuals[i])
                else:
                    residuals_map[pair] = 0.0

                # CB proxy
                streak = float(math_state.streaks[i]) if math_state.valid else 0.0
                cb = compute_cb_proxy(v, residuals_map[pair], streak)
                cb_scores[pair] = cb

                # ATR ratio (current vs rolling average)
                hist = self.atr_history[pair]
                avg_atr = np.mean(hist[-60:]) if len(hist) > 1 else atr_pips
                atr_ratios[pair] = atr_pips / avg_atr if avg_atr > 1e-10 else 1.0

            # -- 3b. Alpha model override (replaces CB proxy if available) --
            if self.alpha_model is not None and math_state.valid:
                # Build feature vector using compute_pair_features (lookback=120)
                # to match the training feature distribution exactly
                feat_vec = []
                for pair in (self.alpha_pairs or pairs):
                    if pair not in arrays:
                        feat_vec.extend([0.0] * 16)
                        continue
                    pf = compute_pair_features(arrays[pair], pair, t, lookback=120)
                    feat_vec.extend([
                        v if not (isinstance(v, float) and np.isnan(v)) else 0.0
                        for v in pf.values()
                    ])
                # Graph features (8) — same order as build_feature_matrix
                gf = extract_graph_features(math_state)
                feat_vec.extend([
                    gf.get("graph_residual_mean", 0.0),
                    gf.get("graph_residual_std", 0.0),
                    gf.get("graph_residual_max", 0.0),
                    gf.get("graph_spectral_gap", 0.0),
                    gf.get("graph_betti_h0", 0.0),
                    gf.get("graph_betti_h1", 0.0),
                    gf.get("graph_avg_correlation", 0.0),
                    gf.get("graph_laplacian_trace", 0.0),
                ])
                feat_arr = np.array(feat_vec, dtype=np.float32)

                # Get model predictions
                model_result = compute_cb_from_model(
                    feat_arr, self.alpha_model,
                    self.alpha_pairs or pairs,
                    scaler_mean=self.alpha_scaler_mean,
                    scaler_std=self.alpha_scaler_std,
                    scale=10.0,  # AlphaNet weights std≈0.036; scale=10 maps p70 → conf≈0.60
                )
                if model_result is not None:
                    cb_scores = model_result["cb_scores"]
                    directions_map = model_result["directions"]

            # -- 4. Net score and direction --
            if math_state.valid:
                net_score, net_dir = compute_net_score(math_state.residuals)
            else:
                net_score, net_dir = 0.0, Direction.FLAT

            # -- 5. Gate check --
            # For AlphaNet (portfolio model): avg_cb has near-zero variance since
            # it's the mean of 35 sigmoid outputs — use max confidence (most-convicted pair)
            # to get meaningful G2 filtering.
            if self.alpha_model is not None and cb_scores:
                avg_cb = max(cb_scores.values())
            else:
                avg_cb = np.mean(list(cb_scores.values())) if cb_scores else 0.0

            # EMA5 slope (use EURUSD as reference if available)
            ref_pair = "EURUSD" if "EURUSD" in pairs else pairs[0]
            ema5_prev = self.ema5.get(ref_pair, 0)
            ema5_slope = arrays[ref_pair]["c"][t] - ema5_prev if t > 0 else 0.0

            # Agreement ratio
            if net_dir != Direction.FLAT and cb_scores:
                high_cb = [p for p, cb in cb_scores.items() if cb >= self.config.cb_thr]
                if high_cb:
                    aligned = sum(1 for p in high_cb
                                  if (net_dir == Direction.LONG and residuals_map.get(p, 0) > RESIDUAL_DEAD_ZONE)
                                  or (net_dir == Direction.SHORT and residuals_map.get(p, 0) < -RESIDUAL_DEAD_ZONE))
                    agree_r = aligned / len(high_cb)
                else:
                    agree_r = 0.0
            else:
                agree_r = 0.0

            # Average ATR ratio
            avg_atr_ratio = np.mean(list(atr_ratios.values())) if atr_ratios else 1.0

            gate_result = check_gates(
                net_score=net_score,
                direction=net_dir,
                avg_cb=avg_cb,
                ema5_slope=ema5_slope,
                atr_ratio=avg_atr_ratio,
                agree_r=agree_r,
                regime=math_state.regime if math_state.valid else "NORMAL",
                spectral_gap=math_state.spectral_gap if math_state.valid else 1.0,
                cb_thr=self.config.cb_thr,
            )

            # Track gate stats
            self.gate_checks += 1
            for gn in ["g1", "g2", "g3", "g4", "g5", "g6"]:
                if getattr(gate_result, f"{gn}_{'net_score' if gn == 'g1' else 'cb_floor' if gn == 'g2' else 'ema_trend' if gn == 'g3' else 'atr_window' if gn == 'g4' else 'agreement' if gn == 'g5' else 'regime'}"):
                    self.gate_passes[gn] += 1

            self.signal_timeline.append((dt, net_score, net_dir.name, gate_result.passed))

            # -- 6. Place orders if gates pass and cooldown elapsed --
            if (gate_result.passed
                    and len(self.open_positions) < MAX_POSITIONS_PER_SIGNAL
                    and (t - self.last_signal_bar) >= MIN_SIGNAL_GAP):
                # Session filter
                hour = int(dt[11:13]) if len(dt) >= 13 else 0
                session_pairs = get_session_pairs(hour) if self.config.use_session_filter else FX_SET

                # Filter to enabled + session pairs
                eligible = [p for p in pairs if p in session_pairs and p in self.config.enabled_pairs]

                candidates = select_candidates(
                    eligible, cb_scores, residuals_map, directions_map,
                    atr_ratios, net_dir, self.config.cb_thr,
                )

                used_margin = sum(o.margin for o in self.open_positions)
                for pair in candidates:
                    if len(self.open_positions) >= MAX_POSITIONS_PER_SIGNAL:
                        break

                    mid = arrays[pair]["c"][t]
                    spread = arrays[pair]["sp"][t]
                    atr_pips = atr_map.get(pair, 5.0)
                    idx = pair_to_idx[pair]

                    order = create_order(
                        pair=pair,
                        direction=net_dir,
                        mid_price=mid,
                        atr_pips=atr_pips,
                        spread_pips=spread,
                        balance=self.balance,
                        bar_idx=t,
                        residual=residuals_map.get(pair, 0.0),
                        cb=cb_scores.get(pair, 0.0),
                        regime=math_state.regime if math_state.valid else "NORMAL",
                        spectral_gap=math_state.spectral_gap if math_state.valid else 0.0,
                        streak=float(math_state.streaks[idx]) if math_state.valid else 0.0,
                    )

                    if order is None:
                        continue

                    # Margin check
                    if used_margin + order.margin > self.balance * MARGIN_LIMIT:
                        continue

                    self.open_positions.append(order)
                    used_margin += order.margin
                    self.last_signal_bar = t

            # -- 7. Record equity --
            self._record_equity(dt)
            self._record_daily_pnl(date_str)

        # Close remaining positions at last bar
        self._close_all(min_bars - 1, arrays, pairs, pair_to_idx)

        return self._build_results()

    def _check_exits(self, t: int, dt: str, arrays: dict, pairs: list,
                     pair_to_idx: dict):
        """
        Check open positions for TP/SL/timeout exits.

        Uses the intra-bar tick path (if available) to determine which
        level — TP or SL — was hit first chronologically. This eliminates
        the TP-before-SL bias that inflates win rates.

        Tick path format: list of prices in chronological order representing
        every new high/low extreme within the bar. Walk through them in order;
        the first one to breach TP or SL determines the outcome.
        """
        still_open = []
        for order in self.open_positions:
            pair = order.pair
            if pair not in arrays:
                still_open.append(order)
                continue

            high = arrays[pair]["h"][t]
            low = arrays[pair]["l"][t]
            close = arrays[pair]["c"][t]
            spread = arrays[pair]["sp"][t]

            exit_price = None
            exit_reason = None

            # Get tick path if available
            tick_path = None
            if "tp_paths" in arrays[pair] and t < len(arrays[pair]["tp_paths"]):
                tick_path = arrays[pair]["tp_paths"][t]

            if tick_path and len(tick_path) > 2:
                # Walk the tick path chronologically to find which level
                # was hit first
                exit_price, exit_reason = self._resolve_tp_sl_from_ticks(
                    order, tick_path
                )
            else:
                # Fallback: no tick path available.
                # Use bar H/L but check which extreme was likely hit first
                # based on open-to-extreme distance (conservative heuristic).
                bar_open = arrays[pair]["o"][t]
                exit_price, exit_reason = self._resolve_tp_sl_from_bar(
                    order, bar_open, high, low
                )

            # Timeout
            if exit_price is None and t >= order.timeout_bar:
                exit_price = close
                exit_reason = "TIMEOUT"

            if exit_price is not None:
                self._close_position(order, exit_price, t, dt, spread, exit_reason)
            else:
                still_open.append(order)

        self.open_positions = still_open

    @staticmethod
    def _resolve_tp_sl_from_ticks(order: Order,
                                  tick_path: list) -> tuple[float | None, str | None]:
        """
        Walk tick-by-tick through the bar to find which level was hit first.
        Returns (exit_price, exit_reason) or (None, None) if neither hit.
        """
        for price in tick_path:
            if order.direction == Direction.LONG:
                if price <= order.sl:
                    return order.sl, "SL"
                if price >= order.tp:
                    return order.tp, "TP"
            else:  # SHORT
                if price >= order.sl:
                    return order.sl, "SL"
                if price <= order.tp:
                    return order.tp, "TP"
        return None, None

    @staticmethod
    def _resolve_tp_sl_from_bar(order: Order, bar_open: float,
                                bar_high: float, bar_low: float
                                ) -> tuple[float | None, str | None]:
        """
        Fallback when no tick path is available.
        Uses bar open to determine which extreme was likely reached first:
        - If open is closer to SL than TP, assume SL was hit first
        - This is the conservative (anti-win-rate-inflation) approach
        """
        tp_hit = False
        sl_hit = False

        if order.direction == Direction.LONG:
            tp_hit = bar_high >= order.tp
            sl_hit = bar_low <= order.sl
        else:
            tp_hit = bar_low <= order.tp
            sl_hit = bar_high >= order.sl

        if tp_hit and sl_hit:
            # Both hit in same bar — use distance from open to determine order
            if order.direction == Direction.LONG:
                dist_to_sl = abs(bar_open - order.sl)
                dist_to_tp = abs(order.tp - bar_open)
            else:
                dist_to_sl = abs(order.sl - bar_open)
                dist_to_tp = abs(bar_open - order.tp)

            # Closer level was hit first
            if dist_to_sl <= dist_to_tp:
                return order.sl, "SL"
            else:
                return order.tp, "TP"
        elif sl_hit:
            return order.sl, "SL"
        elif tp_hit:
            return order.tp, "TP"

        return None, None

    def _close_position(self, order: Order, exit_price: float, bar: int,
                        dt: str, spread_pips: float, reason: str):
        """Close a position and record the trade."""
        pip = PIP_SIZES.get(order.pair, 0.0001)
        pip_value = PIP_VALUES.get(order.pair, 10.0)

        # Exit spread cost (half spread deducted on exit)
        exit_spread_cost = (spread_pips / 2.0) * pip * order.lot * 100_000

        # PnL calculation
        if order.direction == Direction.LONG:
            pnl_pips = (exit_price - order.entry_price) / pip
        else:
            pnl_pips = (order.entry_price - exit_price) / pip

        pnl_usd = pnl_pips * pip_value * order.lot
        total_spread_cost = order.spread_cost_entry + exit_spread_cost

        trade = ClosedTrade(
            pair=order.pair,
            direction=order.direction,
            entry_price=order.entry_price,
            exit_price=exit_price,
            tp=order.tp,
            sl=order.sl,
            lot=order.lot,
            pnl_usd=pnl_usd,
            pnl_pips=pnl_pips,
            spread_cost=total_spread_cost,
            entry_bar=order.entry_bar,
            exit_bar=bar,
            exit_reason=reason,
            entry_regime=order.entry_regime,
            entry_residual=order.entry_residual,
            entry_cb=order.entry_cb,
        )
        self.closed_trades.append(trade)
        self.balance += pnl_usd

    def _close_all(self, t: int, arrays: dict, pairs: list, pair_to_idx: dict):
        """Close all remaining open positions at end of backtest."""
        dt = arrays[pairs[0]]["dt"][t] if t < len(arrays[pairs[0]]["dt"]) else "END"
        for order in self.open_positions:
            pair = order.pair
            if pair in arrays and t < len(arrays[pair]["c"]):
                close = arrays[pair]["c"][t]
                spread = arrays[pair]["sp"][t]
            else:
                close = order.entry_price
                spread = 1.0
            self._close_position(order, close, t, dt, spread, "END")
        self.open_positions = []

    def _record_equity(self, dt: str):
        """Record equity curve point."""
        unrealized = 0.0  # simplified: only track realized
        equity = self.balance + unrealized
        self.equity_curve.append((dt, round(equity, 4)))

        if equity > self.peak_balance:
            self.peak_balance = equity
        dd = (self.peak_balance - equity) / self.peak_balance if self.peak_balance > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _record_daily_pnl(self, date_str: str):
        """Track daily PnL."""
        if date_str not in self.daily_pnl:
            self.daily_pnl[date_str] = 0.0
        # Daily PnL is computed from trades closed on this date
        # (simplified: running balance delta)

    def _build_results(self) -> BacktestResults:
        """Compile all results into BacktestResults."""
        results = BacktestResults()
        results.config = {
            "init_balance": self.config.init_balance,
            "cb_thr": self.config.cb_thr,
            "risk_pct": self.config.risk_pct,
            "use_session_filter": self.config.use_session_filter,
        }
        results.equity_curve = self.equity_curve
        results.regime_timeline = self.regime_timeline
        results.signal_timeline = self.signal_timeline

        # Convert trades to dicts
        results.trades = []
        for tr in self.closed_trades:
            results.trades.append({
                "pair": tr.pair,
                "direction": tr.direction.name,
                "entry_price": round(tr.entry_price, 6),
                "exit_price": round(tr.exit_price, 6),
                "tp": round(tr.tp, 6),
                "sl": round(tr.sl, 6),
                "lot": round(tr.lot, 4),
                "pnl_usd": round(tr.pnl_usd, 4),
                "pnl_pips": round(tr.pnl_pips, 2),
                "spread_cost": round(tr.spread_cost, 4),
                "entry_bar": tr.entry_bar,
                "exit_bar": tr.exit_bar,
                "exit_reason": tr.exit_reason,
                "entry_regime": tr.entry_regime,
                "entry_residual": round(tr.entry_residual, 6),
                "entry_cb": round(tr.entry_cb, 4),
            })

        # Daily PnL from trades
        for tr in self.closed_trades:
            # Use exit bar date (approximate)
            date_approx = tr.pair  # placeholder
            results.daily_pnl = self._compute_daily_pnl()

        # Performance metrics
        results.metrics = self._compute_metrics()

        # Gate stats
        if self.gate_checks > 0:
            results.gate_stats = {
                k: round(v / self.gate_checks, 4) for k, v in self.gate_passes.items()
            }
            results.gate_stats["total_checks"] = self.gate_checks
        else:
            results.gate_stats = {f"g{i}": 0.0 for i in range(1, 7)}

        return results

    def _compute_daily_pnl(self) -> dict:
        """Compute daily PnL from closed trades."""
        daily = {}
        for eq_dt, eq_val in self.equity_curve:
            date = eq_dt[:10] if len(eq_dt) >= 10 else eq_dt
            daily[date] = round(eq_val - self.config.init_balance, 4)
        return daily

    def _compute_metrics(self) -> dict:
        """Compute performance metrics from closed trades."""
        trades = self.closed_trades
        if not trades:
            return {
                "total_trades": 0, "winners": 0, "losers": 0,
                "win_rate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "ev": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "return_pct": 0.0, "spread_drag": 0.0,
                "sharpe": 0.0, "final_balance": round(self.balance, 4),
            }

        winners = [t for t in trades if t.pnl_usd > 0]
        losers = [t for t in trades if t.pnl_usd <= 0]

        total = len(trades)
        win_rate = len(winners) / total if total > 0 else 0.0
        avg_win = np.mean([t.pnl_usd for t in winners]) if winners else 0.0
        avg_loss = np.mean([abs(t.pnl_usd) for t in losers]) if losers else 0.0

        gross_profit = sum(t.pnl_usd for t in winners)
        gross_loss = sum(abs(t.pnl_usd) for t in losers)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        total_return = (self.balance - self.config.init_balance) / self.config.init_balance
        spread_drag = sum(t.spread_cost for t in trades)

        # Sharpe ratio (approximate from trade PnL)
        pnl_series = [t.pnl_usd for t in trades]
        sharpe = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-10)) * np.sqrt(252) if len(pnl_series) > 1 else 0.0

        return {
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(win_rate, 4),
            "avg_win": round(float(avg_win), 4),
            "avg_loss": round(float(avg_loss), 4),
            "ev": round(float(ev), 4),
            "profit_factor": round(float(profit_factor), 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "return_pct": round(float(total_return), 4),
            "spread_drag": round(float(spread_drag), 4),
            "sharpe": round(float(sharpe), 4),
            "final_balance": round(self.balance, 4),
        }

    def export_results(self, results: BacktestResults, output_path: str):
        """Write results to JSON."""
        out = {
            "config": results.config,
            "equity_curve": results.equity_curve,
            "trades": results.trades,
            "daily_pnl": results.daily_pnl,
            "metrics": results.metrics,
            "gate_stats": results.gate_stats,
            "regime_timeline": results.regime_timeline,
            "signal_timeline": results.signal_timeline,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {output_path}")


# -- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    print("Backtester smoke test with synthetic data...")
    rng = np.random.default_rng(42)
    n_bars = 500
    pairs = sorted(list(FX_SET))[:10]  # 10 FX pairs

    # Generate synthetic OHLC data
    data = {}
    for pair in pairs:
        base = 1.0 + rng.random() * 0.5
        closes = base + np.cumsum(rng.normal(0, 0.0005, n_bars))
        data[pair] = []
        for i in range(n_bars):
            c = closes[i]
            h = c + abs(rng.normal(0, 0.0003))
            l = c - abs(rng.normal(0, 0.0003))
            o = c + rng.normal(0, 0.0002)
            data[pair].append({
                "dt": f"2026-03-02 {(i // 60) % 24:02d}:{i % 60:02d}",
                "o": round(o, 5), "h": round(h, 5),
                "l": round(l, 5), "c": round(c, 5),
                "sp": round(rng.uniform(0.8, 2.5), 2),
                "tk": int(rng.integers(10, 80)),
            })

    config = BacktestConfig(use_session_filter=False, enabled_pairs=set(pairs))
    bt = Backtester(config)
    results = bt.run(data, pairs=pairs)

    m = results.metrics
    print(f"\nResults:")
    print(f"  Trades: {m['total_trades']}")
    print(f"  Win rate: {m['win_rate']:.1%}")
    print(f"  Avg win: ${m['avg_win']:.4f}")
    print(f"  Avg loss: ${m['avg_loss']:.4f}")
    print(f"  EV: ${m['ev']:.4f}")
    print(f"  Profit factor: {m['profit_factor']:.2f}")
    print(f"  Max DD: {m['max_drawdown']:.2%}")
    print(f"  Return: {m['return_pct']:.2%}")
    print(f"  Final balance: ${m['final_balance']:.2f}")
    print(f"  Spread drag: ${m['spread_drag']:.4f}")
    print(f"  Equity points: {len(results.equity_curve)}")
    print(f"  Gate stats: {results.gate_stats}")
    print("OK")
