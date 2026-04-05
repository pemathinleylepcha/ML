from __future__ import annotations

from pathlib import Path

from universe import ALL_INSTRUMENTS, TIMEFRAMES

RAW_TIMEFRAMES = tuple(TIMEFRAMES)
RESEARCH_REQUIRED_RAW_TIMEFRAMES = ("M5",)
RESEARCH_DERIVED_TIMEFRAMES = ("M15", "H1", "H4")

# Known historical directory aliases observed in the existing DataExtractor tree.
SYMBOL_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "XTIUSD": ("X",),
    "EUSTX50": ("EuSTX50",),
    "JPN225": ("JN225",),
}


def canonical_symbols() -> tuple[str, ...]:
    return tuple(ALL_INSTRUMENTS)


def symbol_dir_candidates(symbol: str) -> tuple[str, ...]:
    aliases = SYMBOL_DIR_ALIASES.get(symbol, ())
    ordered = [symbol]
    ordered.extend(alias for alias in aliases if alias != symbol)
    return tuple(ordered)


def resolve_symbol_dir(quarter_dir: Path, symbol: str) -> tuple[Path | None, str | None]:
    for candidate in symbol_dir_candidates(symbol):
        candidate_dir = quarter_dir / candidate
        if candidate_dir.is_dir():
            return candidate_dir, candidate
    return None, None


def resolve_candle_path(
    quarter_dir: Path,
    symbol: str,
    timeframe: str,
) -> tuple[Path | None, str | None]:
    symbol_dir, actual_symbol = resolve_symbol_dir(quarter_dir, symbol)
    if symbol_dir is None:
        return None, None
    candle_path = symbol_dir / f"candles_{timeframe}.csv"
    if not candle_path.exists():
        return None, actual_symbol
    return candle_path, actual_symbol
