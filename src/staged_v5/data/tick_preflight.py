from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class TickSymbolCoverage:
    symbol: str
    path: str | None
    first_timestamp: str | None
    last_timestamp: str | None
    covers_start: bool
    covers_end: bool
    exists: bool


@dataclass(slots=True)
class TickRootPreflightResult:
    tick_root: str
    start: str
    end: str
    required_symbols: tuple[str, ...]
    complete: bool
    coverage: tuple[TickSymbolCoverage, ...]

    @property
    def missing_symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.coverage if not item.exists)

    @property
    def insufficient_symbols(self) -> tuple[str, ...]:
        return tuple(
            item.symbol
            for item in self.coverage
            if item.exists and (not item.covers_start or not item.covers_end)
        )


def _find_tick_file(tick_root: Path, symbol: str) -> Path | None:
    candidates = (
        tick_root / f"{symbol}_1000ms.csv",
        tick_root / f"{symbol}_1000ms.parquet",
        tick_root / f"{symbol}.csv",
        tick_root / f"{symbol}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_csv_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    first_frame = pd.read_csv(path, nrows=1)
    if len(first_frame) == 0:
        raise ValueError(f"tick csv is empty: {path}")
    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        step = min(file_size, 8192)
        buffer = b""
        while file_size > 0:
            file_size = max(0, file_size - step)
            handle.seek(file_size)
            buffer = handle.read(step) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 2:
                last_line = lines[-1].decode("utf-8", errors="ignore")
                break
            if file_size == 0:
                last_line = lines[-1].decode("utf-8", errors="ignore")
                break
    first_ts = pd.to_datetime(first_frame.iloc[0]["dt"], errors="coerce")
    last_ts = pd.to_datetime(last_line.split(",")[0], errors="coerce")
    if pd.isna(first_ts) or pd.isna(last_ts):
        raise ValueError(f"failed to parse tick csv bounds: {path}")
    return first_ts, last_ts


def _read_tick_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["dt"])
        if len(frame) == 0:
            raise ValueError(f"tick parquet is empty: {path}")
        return pd.to_datetime(frame.iloc[0]["dt"]), pd.to_datetime(frame.iloc[-1]["dt"])
    return _read_csv_bounds(path)


def inspect_tick_root_preflight(
    tick_root: str | Path,
    symbols: tuple[str, ...],
    start: str,
    end: str,
) -> TickRootPreflightResult:
    root = Path(tick_root)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    coverage: list[TickSymbolCoverage] = []
    for symbol in symbols:
        path = _find_tick_file(root, symbol)
        if path is None:
            coverage.append(
                TickSymbolCoverage(
                    symbol=symbol,
                    path=None,
                    first_timestamp=None,
                    last_timestamp=None,
                    covers_start=False,
                    covers_end=False,
                    exists=False,
                )
            )
            continue
        first_ts, last_ts = _read_tick_bounds(path)
        coverage.append(
            TickSymbolCoverage(
                symbol=symbol,
                path=str(path),
                first_timestamp=str(first_ts),
                last_timestamp=str(last_ts),
                covers_start=first_ts <= start_ts,
                covers_end=last_ts >= end_ts,
                exists=True,
            )
        )
    result = TickRootPreflightResult(
        tick_root=str(root),
        start=str(start_ts),
        end=str(end_ts),
        required_symbols=tuple(symbols),
        complete=all(item.exists and item.covers_start and item.covers_end for item in coverage),
        coverage=tuple(coverage),
    )
    return result


def assert_tick_root_ready(
    tick_root: str | Path,
    symbols: tuple[str, ...],
    start: str,
    end: str,
) -> TickRootPreflightResult:
    result = inspect_tick_root_preflight(
        tick_root=tick_root,
        symbols=symbols,
        start=start,
        end=end,
    )
    if result.complete:
        return result
    parts: list[str] = []
    if result.missing_symbols:
        parts.append(f"missing={list(result.missing_symbols)}")
    if result.insufficient_symbols:
        parts.append(f"insufficient_window={list(result.insufficient_symbols)}")
    raise ValueError(
        "Tick root preflight failed for the requested window. "
        + " ".join(parts)
    )
