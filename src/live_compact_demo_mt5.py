from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("MetaTrader5 package is required for live compact demo trading") from exc

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from math_engine import MathEngine
from research_dataset import encode_session_codes
from research_backtester import BinaryHysteresisFilter
from research_inference import CompactFXInference
from tpo_normal_layer import build_tpo_normal_decision
from universe import FX_PAIRS, PIP_SIZES, PIP_VALUES


DEFAULT_REPORT_PATH = Path("data/remote_clean_2025_runs/clean_2025_no_bridge_optk10_report.json")
DEFAULT_MODEL_DIR = Path("models/live_compact_no_bridge_optk10")
DEFAULT_SIGNAL_FILENAME = "algo_c2_demo_signals.csv"
DEFAULT_STATUS_FILENAME = "algo_c2_demo_status.json"
DEFAULT_MAGIC = 20260330
DEFAULT_COMMENT = "algoc2_compact_demo"

ATR_MULT_TP = 2.0
ATR_MULT_SL = 1.5
MIN_TP_SPREAD = 5.0
MIN_SL_SPREAD = 3.0
RISK_PCT = 0.01
LOT_MIN = 0.01
LOT_MAX = 10.0
WARMUP_BARS = 90


@dataclass(slots=True)
class PairSignal:
    symbol: str
    direction: int
    confidence: float
    p_buy: float
    p_sell: float
    lot: float
    sl_distance: float
    tp_distance: float
    source: str = "legacy_compact"
    legacy_direction: int = 0
    legacy_confidence: float = 0.0
    protector_blocked: bool = False
    protector_reason: str = ""
    tpo_poc: float = 0.0
    tpo_value_area_low: float = 0.0
    tpo_value_area_high: float = 0.0


def _common_files_dir() -> Path:
    appdata = Path(os.environ.get("APPDATA", Path.home()))
    return appdata / "MetaQuotes" / "Terminal" / "Common" / "Files"


def _estimate_spread_cost(pair: str, spread_raw: float) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    if spread_raw <= 0.0:
        return 0.0
    if spread_raw < pip * 50.0:
        return float(spread_raw)
    return float(spread_raw) * (pip / 10.0)


def _latest_zscore_from_past(values: np.ndarray, window: int) -> float:
    if len(values) < 3:
        return 0.0
    hist = values[max(0, len(values) - 1 - window): len(values) - 1].astype(np.float64, copy=False)
    if len(hist) < 2:
        return 0.0
    std = float(hist.std(ddof=0))
    if std < 1e-8:
        return 0.0
    return float((float(values[-1]) - float(hist.mean())) / std)


def _latest_atr_proxy(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int = 24) -> float:
    if len(close) < 2:
        return float(abs(high[-1] - low[-1])) if len(close) else 0.0
    tr = np.zeros(len(close), dtype=np.float64)
    tr[1:] = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ]
    )
    tail = tr[max(1, len(tr) - window):]
    if len(tail) == 0:
        return float(abs(high[-1] - low[-1]))
    return float(np.mean(tail))


def _compute_lot(balance: float, sl_distance: float, pair: str) -> float:
    pip = float(PIP_SIZES.get(pair, 1e-4))
    pip_value = float(PIP_VALUES.get(pair, 10.0))
    if sl_distance <= 0.0:
        return 0.0
    sl_pips = sl_distance / max(pip, 1e-8)
    if sl_pips <= 0.0:
        return 0.0
    risk_usd = float(balance) * RISK_PCT
    lot = risk_usd / max(sl_pips * pip_value, 1e-8)
    return float(np.clip(lot, LOT_MIN, LOT_MAX))


def _next_m5_close_utc() -> datetime:
    now = datetime.now(timezone.utc)
    next_min = ((now.minute // 5) + 1) * 5
    if next_min >= 60:
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return now.replace(minute=next_min, second=0, microsecond=0)


def _wait_for_next_bar(buffer_seconds: int) -> None:
    target = _next_m5_close_utc() + timedelta(seconds=buffer_seconds)
    delay = (target - datetime.now(timezone.utc)).total_seconds()
    if delay > 0:
        time.sleep(delay)


class CompactDemoSignalService:
    def __init__(
        self,
        model_dir: Path,
        report_path: Path,
        common_files_dir: Path,
        mt5_path: str | None = None,
        reference_symbol: str = "EURUSD",
    ):
        self.model_dir = model_dir
        self.report_path = report_path
        self.common_files_dir = common_files_dir
        self.mt5_path = mt5_path
        self.reference_symbol = reference_symbol
        self.model = CompactFXInference(str(model_dir))
        self.legacy_filters: dict[str, BinaryHysteresisFilter] = {}
        self.normal_filters: dict[str, BinaryHysteresisFilter] = {}
        self._load_report_config()

    def _load_report_config(self) -> None:
        with open(self.report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        backtest = report["config"]["backtest"]
        self.entry_threshold = float(backtest["entry_threshold"])
        self.exit_threshold = float(backtest["exit_threshold"])
        self.confidence_threshold = float(backtest["confidence_threshold"])
        self.feature_names = list(report["feature_names_selected"])
        for pair in FX_PAIRS:
            self.legacy_filters[pair] = BinaryHysteresisFilter(
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
                confidence_threshold=self.confidence_threshold,
            )
            self.normal_filters[pair] = BinaryHysteresisFilter(
                entry_threshold=max(0.53, self.entry_threshold - 0.04),
                exit_threshold=max(0.49, self.exit_threshold - 0.02),
                confidence_threshold=max(0.05, self.confidence_threshold - 0.04),
            )

    def connect(self, demo_only: bool = True) -> None:
        kwargs = {"timeout": 10000}
        if self.mt5_path:
            kwargs["path"] = self.mt5_path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        info = mt5.terminal_info()
        account = mt5.account_info()
        if info is None or not info.connected:
            raise RuntimeError("MT5 terminal not connected")
        if account is None:
            raise RuntimeError("MT5 account info unavailable")
        if demo_only and int(account.trade_mode) != int(mt5.ACCOUNT_TRADE_MODE_DEMO):
            raise RuntimeError("Refusing to run: account is not a demo account")
        self.account_info = account
        logging.info(
            "MT5 connected: login=%s server=%s balance=%.2f demo=%s",
            getattr(account, "login", "n/a"),
            getattr(account, "server", "n/a"),
            float(account.balance),
            int(account.trade_mode) == int(mt5.ACCOUNT_TRADE_MODE_DEMO),
        )

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    def _fetch_closed_m5(self, symbol: str, count: int) -> pd.DataFrame | None:
        if not mt5.symbol_select(symbol, True):
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, count)
        if rates is None or len(rates) == 0:
            return None
        frame = pd.DataFrame(rates)
        frame["dt"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.rename(
            columns={
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "spread": "sp",
                "tick_volume": "tk",
            }
        )
        return frame[["dt", "o", "h", "l", "c", "sp", "tk"]].sort_values("dt").drop_duplicates("dt").set_index("dt")

    def _collect_histories(self, count: int) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
        frames: dict[str, pd.DataFrame] = {}
        common_index: pd.DatetimeIndex | None = None
        for symbol in FX_PAIRS:
            frame = self._fetch_closed_m5(symbol, count=count)
            if frame is None or len(frame) < 16:
                continue
            frames[symbol] = frame
            common_index = frame.index if common_index is None else common_index.intersection(frame.index)
        if common_index is None or len(common_index) < 16:
            raise RuntimeError("Insufficient common M5 history across FX pairs")
        common_index = common_index.sort_values()
        return common_index, frames

    def _build_math_state(
        self,
        common_index: pd.DatetimeIndex,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
        aligned = {symbol: frame.loc[common_index] for symbol, frame in frames.items() if symbol in frames}
        available_pairs = [symbol for symbol in FX_PAIRS if symbol in aligned]
        if len(available_pairs) < 8:
            raise RuntimeError("Too few aligned FX pairs for Laplacian live state")

        close_matrix = np.column_stack([aligned[symbol]["c"].to_numpy(dtype=np.float64) for symbol in available_pairs])
        returns = np.zeros_like(close_matrix, dtype=np.float64)
        if close_matrix.shape[0] > 1:
            returns[1:] = np.diff(np.log(np.maximum(close_matrix, 1e-10)), axis=0)

        engine = MathEngine(n_pairs=len(available_pairs))
        last_state = None
        for idx in range(close_matrix.shape[0]):
            last_state = engine.update(returns[idx])
        residuals = {symbol: 0.0 for symbol in FX_PAIRS}
        if last_state is not None and last_state.valid:
            for idx, symbol in enumerate(available_pairs):
                residuals[symbol] = float(last_state.residuals[idx])
        return aligned, residuals

    def _build_feature_row(
        self,
        symbol: str,
        frame: pd.DataFrame,
        residual: float,
        session_code: int,
    ) -> np.ndarray:
        close = frame["c"].to_numpy(dtype=np.float64)
        high = frame["h"].to_numpy(dtype=np.float64)
        low = frame["l"].to_numpy(dtype=np.float64)
        spread_raw = frame["sp"].to_numpy(dtype=np.float64)
        ticks = frame["tk"].to_numpy(dtype=np.float64)
        safe_close = np.maximum(np.abs(close), 1e-10)
        range_now = high - low
        spread_cost = np.asarray([_estimate_spread_cost(symbol, float(value)) for value in spread_raw], dtype=np.float64)
        row = {
            "local_ret_1": float(np.log(safe_close[-1] / safe_close[max(0, len(close) - 2)])) if len(close) > 1 else 0.0,
            "local_ret_3": float(np.log(safe_close[-1] / safe_close[max(0, len(close) - 4)])) if len(close) > 3 else 0.0,
            "local_ret_6": float(np.log(safe_close[-1] / safe_close[max(0, len(close) - 7)])) if len(close) > 6 else 0.0,
            "local_atr_norm": float(_latest_atr_proxy(close, high, low) / safe_close[-1]),
            "local_range_norm": float(range_now[-1] / safe_close[-1]),
            "local_body_ratio": float(abs(close[-1] - frame["o"].to_numpy(dtype=np.float64)[-1]) / max(range_now[-1], 1e-10)),
            "local_spread_z": _latest_zscore_from_past(spread_cost.astype(np.float32), window=64),
            "local_tick_z": _latest_zscore_from_past(ticks.astype(np.float32), window=64),
            "local_liquidity_stress": float((spread_cost[-1] / safe_close[-1]) + (1.0 / np.sqrt(max(ticks[-1], 0.0) + 1.0))),
            "local_lap_residual": float(residual),
            "local_residual_streak": 0.0,
            "local_regime": 1.0,
            "local_session_code": float(session_code),
        }
        return np.asarray([row[name] for name in self.feature_names], dtype=np.float32)

    def generate_snapshot(self) -> tuple[str, list[PairSignal]]:
        common_index, frames = self._collect_histories(count=WARMUP_BARS)
        aligned, residuals = self._build_math_state(common_index, frames)
        latest_ts = common_index[-1].to_pydatetime()
        session_code = int(encode_session_codes(np.asarray([np.datetime64(latest_ts.replace(tzinfo=None))]))[0])
        rows: list[tuple[PairSignal, float]] = []
        balance = float(getattr(self.account_info, "balance", 50.0))

        for symbol in FX_PAIRS:
            frame = aligned.get(symbol)
            if frame is None or len(frame) < 16:
                continue
            features = self._build_feature_row(symbol, frame, residuals.get(symbol, 0.0), session_code)
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            p_sell = float(proba[0])
            p_buy = float(proba[1])
            legacy_confidence = abs(p_buy - p_sell)
            legacy_direction = self.legacy_filters[symbol].step(p_sell, p_buy)

            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info is None:
                continue
            ask = float(tick_info.ask)
            bid = float(tick_info.bid)
            spread_price = max(ask - bid, 0.0)
            close_values = frame["c"].to_numpy(dtype=np.float64)
            high_values = frame["h"].to_numpy(dtype=np.float64)
            low_values = frame["l"].to_numpy(dtype=np.float64)
            atr_price = _latest_atr_proxy(close_values, high_values, low_values)

            tpo_decision = build_tpo_normal_decision(
                close=close_values,
                high=high_values,
                low=low_values,
                atr_price=atr_price,
                spread_price=spread_price,
                legacy_direction=legacy_direction,
                legacy_confidence=legacy_confidence,
                legacy_p_buy=p_buy,
                legacy_p_sell=p_sell,
            )

            tpo_buy = 0.5
            tpo_sell = 0.5
            if tpo_decision.direction > 0:
                tpo_buy = 0.5 + 0.5 * tpo_decision.confidence
                tpo_sell = 1.0 - tpo_buy
            elif tpo_decision.direction < 0:
                tpo_sell = 0.5 + 0.5 * tpo_decision.confidence
                tpo_buy = 1.0 - tpo_sell

            direction = self.normal_filters[symbol].step(tpo_sell, tpo_buy)
            confidence = tpo_decision.confidence if direction != 0 else 0.0
            sl_distance = max(tpo_decision.sl_distance, spread_price * MIN_SL_SPREAD)
            tp_distance = max(tpo_decision.tp_distance, spread_price * MIN_TP_SPREAD)
            base_lot = _compute_lot(balance, sl_distance, symbol) if direction != 0 else 0.0
            lot = float(np.clip(base_lot * max(tpo_decision.lot_scale, 0.0), 0.0, LOT_MAX))

            signal = PairSignal(
                symbol=symbol,
                direction=int(direction),
                confidence=float(confidence),
                p_buy=p_buy,
                p_sell=p_sell,
                lot=float(lot),
                sl_distance=float(sl_distance),
                tp_distance=float(tp_distance),
                source="tpo_normal",
                legacy_direction=int(legacy_direction),
                legacy_confidence=float(legacy_confidence),
                protector_blocked=bool(tpo_decision.protector_blocked),
                protector_reason=tpo_decision.protector_reason,
                tpo_poc=float(tpo_decision.profile.poc),
                tpo_value_area_low=float(tpo_decision.profile.value_area_low),
                tpo_value_area_high=float(tpo_decision.profile.value_area_high),
            )
            rows.append((signal, signal.confidence))

        rows.sort(key=lambda item: item[1], reverse=True)
        return latest_ts.strftime("%Y-%m-%d %H:%M:%S"), [item[0] for item in rows]

    def write_signal_files(
        self,
        bar_time_utc: str,
        signals: list[PairSignal],
        signal_filename: str,
        status_filename: str,
        magic: int,
        comment: str,
    ) -> tuple[Path, Path]:
        self.common_files_dir.mkdir(parents=True, exist_ok=True)
        signal_path = self.common_files_dir / signal_filename
        status_path = self.common_files_dir / status_filename

        with open(signal_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp_utc",
                    "symbol",
                    "direction",
                    "confidence",
                    "p_buy",
                    "p_sell",
                    "lot",
                    "sl_distance",
                    "tp_distance",
                    "magic",
                    "comment",
                ]
            )
            for signal in signals:
                writer.writerow(
                    [
                        bar_time_utc,
                        signal.symbol,
                        signal.direction,
                        f"{signal.confidence:.6f}",
                        f"{signal.p_buy:.6f}",
                        f"{signal.p_sell:.6f}",
                        f"{signal.lot:.4f}",
                        f"{signal.sl_distance:.8f}",
                        f"{signal.tp_distance:.8f}",
                        magic,
                        comment,
                    ]
                )

        signal_payloads = [
            {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "p_buy": signal.p_buy,
                "p_sell": signal.p_sell,
                "lot": signal.lot,
                "sl_distance": signal.sl_distance,
                "tp_distance": signal.tp_distance,
                "source": signal.source,
                "legacy_direction": signal.legacy_direction,
                "legacy_confidence": signal.legacy_confidence,
                "protector_blocked": signal.protector_blocked,
                "protector_reason": signal.protector_reason,
                "tpo_poc": signal.tpo_poc,
                "tpo_value_area_low": signal.tpo_value_area_low,
                "tpo_value_area_high": signal.tpo_value_area_high,
            }
            for signal in signals
        ]

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "bar_time_utc": bar_time_utc,
            "model_dir": str(self.model_dir),
            "report_path": str(self.report_path),
            "account_balance": float(getattr(self.account_info, "balance", 50.0)),
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "confidence_threshold": self.confidence_threshold,
            "signals": signal_payloads,
        }
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return signal_path, status_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact benchmark demo signal service for MT5 EA automation")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--mt5-path", help="Optional explicit terminal64.exe path for MetaTrader5.initialize()")
    parser.add_argument("--common-files-dir", default=str(_common_files_dir()))
    parser.add_argument("--signal-file", default=DEFAULT_SIGNAL_FILENAME)
    parser.add_argument("--status-file", default=DEFAULT_STATUS_FILENAME)
    parser.add_argument("--reference-symbol", default="EURUSD")
    parser.add_argument("--buffer-seconds", type=int, default=5)
    parser.add_argument("--magic", type=int, default=DEFAULT_MAGIC)
    parser.add_argument("--comment", default=DEFAULT_COMMENT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--allow-live-account", action="store_true")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - runtime integration
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    service = CompactDemoSignalService(
        model_dir=Path(args.model_dir),
        report_path=Path(args.report_path),
        common_files_dir=Path(args.common_files_dir),
        mt5_path=args.mt5_path,
        reference_symbol=args.reference_symbol,
    )

    try:
        service.connect(demo_only=not args.allow_live_account)
        if args.once:
            bar_time_utc, signals = service.generate_snapshot()
            signal_path, status_path = service.write_signal_files(
                bar_time_utc,
                signals,
                signal_filename=args.signal_file,
                status_filename=args.status_file,
                magic=args.magic,
                comment=args.comment,
            )
            logging.info("Wrote %s and %s", signal_path, status_path)
            return

        while True:
            _wait_for_next_bar(buffer_seconds=args.buffer_seconds)
            bar_time_utc, signals = service.generate_snapshot()
            signal_path, status_path = service.write_signal_files(
                bar_time_utc,
                signals,
                signal_filename=args.signal_file,
                status_filename=args.status_file,
                magic=args.magic,
                comment=args.comment,
            )
            logging.info("Bar %s -> wrote %d signals to %s", bar_time_utc, len(signals), signal_path)
            logging.debug("Status file updated at %s", status_path)
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
