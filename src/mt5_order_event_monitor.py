from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError("MetaTrader5 package is required for MT5 order polling") from exc


DEFAULT_STATE_PATH = Path("data/live_demo_monitor/state.json")
DEFAULT_EVENTS_PATH = Path("data/live_demo_monitor/events.jsonl")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _serialize(item) for key, item in value._asdict().items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_order_ticket": 0,
            "last_position_ticket": 0,
            "last_deal_ticket": 0,
            "last_poll_utc": None,
        }
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _append_events(path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, default=_serialize) + "\n")


def _max_ticket(records: list[Any]) -> int:
    if not records:
        return 0
    return max(int(getattr(record, "ticket", 0) or 0) for record in records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll MT5 for new demo orders/deals/positions")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--mt5-path", default="", help="Optional explicit terminal64.exe path")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - runtime integration
    args = parse_args()
    state_path = Path(args.state_path)
    events_path = Path(args.events_path)
    state = _load_state(state_path)

    init_kwargs: dict[str, Any] = {"timeout": 10000}
    if args.mt5_path:
        init_kwargs["path"] = args.mt5_path

    if not mt5.initialize(**init_kwargs):
        payload = {
            "polled_at_utc": _utc_now().isoformat(),
            "type": "poll_error",
            "error": mt5.last_error(),
        }
        _append_events(events_path, [payload])
        raise SystemExit(1)

    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        now_utc = _utc_now()
        history_from = now_utc - timedelta(hours=max(args.lookback_hours, 1))

        orders = list(mt5.orders_get() or [])
        positions = list(mt5.positions_get() or [])
        deals = list(mt5.history_deals_get(history_from, now_utc) or [])

        last_order_ticket = int(state.get("last_order_ticket", 0) or 0)
        last_position_ticket = int(state.get("last_position_ticket", 0) or 0)
        last_deal_ticket = int(state.get("last_deal_ticket", 0) or 0)

        new_orders = [order for order in orders if int(getattr(order, "ticket", 0) or 0) > last_order_ticket]
        new_positions = [position for position in positions if int(getattr(position, "ticket", 0) or 0) > last_position_ticket]
        new_deals = [deal for deal in deals if int(getattr(deal, "ticket", 0) or 0) > last_deal_ticket]

        base_meta = {
            "polled_at_utc": now_utc.isoformat(),
            "terminal_trade_allowed": bool(getattr(terminal, "trade_allowed", False)) if terminal else False,
            "connected": bool(getattr(terminal, "connected", False)) if terminal else False,
            "account_login": int(getattr(account, "login", 0) or 0) if account else 0,
            "account_server": str(getattr(account, "server", "")) if account else "",
            "balance": float(getattr(account, "balance", 0.0) or 0.0) if account else 0.0,
            "equity": float(getattr(account, "equity", 0.0) or 0.0) if account else 0.0,
        }

        events: list[dict[str, Any]] = []
        for order in sorted(new_orders, key=lambda item: int(getattr(item, "ticket", 0) or 0)):
            events.append({**base_meta, "type": "new_order", "ticket": int(order.ticket), "data": _serialize(order)})
        for position in sorted(new_positions, key=lambda item: int(getattr(item, "ticket", 0) or 0)):
            events.append({**base_meta, "type": "new_position", "ticket": int(position.ticket), "data": _serialize(position)})
        for deal in sorted(new_deals, key=lambda item: int(getattr(item, "ticket", 0) or 0)):
            events.append({**base_meta, "type": "new_deal", "ticket": int(deal.ticket), "data": _serialize(deal)})

        _append_events(events_path, events)

        state.update(
            {
                "last_order_ticket": max(last_order_ticket, _max_ticket(orders)),
                "last_position_ticket": max(last_position_ticket, _max_ticket(positions)),
                "last_deal_ticket": max(last_deal_ticket, _max_ticket(deals)),
                "last_poll_utc": now_utc.isoformat(),
                "terminal_trade_allowed": bool(getattr(terminal, "trade_allowed", False)) if terminal else False,
                "connected": bool(getattr(terminal, "connected", False)) if terminal else False,
                "positions_total": len(positions),
                "orders_total": len(orders),
                "new_events_written": len(events),
            }
        )
        _save_state(state_path, state)

        print(
            json.dumps(
                {
                    "state_path": str(state_path),
                    "events_path": str(events_path),
                    "new_events": len(events),
                    "positions_total": len(positions),
                    "orders_total": len(orders),
                    "last_deal_ticket": state["last_deal_ticket"],
                }
            )
        )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
