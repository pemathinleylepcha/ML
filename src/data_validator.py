"""
Algo C2 v2 — Data Validator
Audits data completeness for all 43 instruments defined in universe.py.

Usage:
    python data_validator.py --data_dir ./data/csvs
    python data_validator.py --data_dir ./data/csvs --json_file ./data/algo_c2_43_data.json

Output:
    A summary table: instrument | role | subnet | csv_found | csv_date_range | json_bars
    Coverage percentage across the 43-node universe.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from universe import (
    ALL_INSTRUMENTS,
    SIGNAL_ONLY,
    TRADEABLE,
    get_subnet,
)

# ── Tabulate import (optional) ────────────────────────────────────────────────

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ── CSV inspection helpers ────────────────────────────────────────────────────

def find_csv_for_instrument(data_dir: Path, instrument: str) -> Path | None:
    """
    Find a CSV file whose stem contains the instrument name (case-insensitive).
    Returns the first match by sorted filename, or None if not found.
    """
    for f in sorted(data_dir.glob("*.csv")):
        if instrument.upper() in f.stem.upper():
            return f
    return None


def _read_first_last_lines(filepath: Path, n: int = 5) -> tuple[list[str], list[str]]:
    """
    Read the first n non-empty lines and last n non-empty lines of a text file
    without loading the whole file into memory.
    """
    first_lines: list[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                first_lines.append(stripped)
            if len(first_lines) >= n + 1:  # +1 for header
                break

    last_lines: list[str] = []
    # Use a rolling buffer to get the last n lines efficiently
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        buffer: list[str] = []
        for line in fh:
            stripped = line.strip()
            if stripped:
                buffer.append(stripped)
                if len(buffer) > n:
                    buffer.pop(0)
        last_lines = buffer

    return first_lines, last_lines


def _parse_datetime_from_line(line: str) -> str | None:
    """
    Attempt to extract a datetime from the beginning of a CSV data line.
    Handles both tab-separated tick format (DATE\\tTIME) and comma-separated
    candle format (datetime,...).
    """
    if not line:
        return None
    # Tab-separated tick format: "2026.01.02\t00:00:01.234\t..."
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                dt = pd.to_datetime(parts[0].strip() + " " + parts[1].strip(),
                                    format="%Y.%m.%d %H:%M:%S.%f")
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            # Fallback: try first field as date only
            try:
                dt = pd.to_datetime(parts[0].strip(), format="%Y.%m.%d")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    # Comma-separated candle format: "2019-01-02 00:00,1.1450,..."
    parts = line.split(",")
    if parts:
        try:
            dt = pd.to_datetime(parts[0].strip(), infer_datetime_format=True)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    return None


def inspect_csv(filepath: Path) -> dict:
    """
    Inspect a CSV file and return metadata without fully loading it.

    Returns a dict with keys:
        size_mb, line_count_est, first_dt, last_dt, date_range
    """
    size_bytes = filepath.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Estimate line count from file size and a sample of first 1000 lines
    line_count_est: int = 0
    try:
        sample_bytes = 0
        sample_lines = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                sample_bytes += len(line.encode("utf-8"))
                sample_lines += 1
                if sample_lines >= 1000:
                    break
        if sample_lines > 0 and sample_bytes > 0:
            bytes_per_line = sample_bytes / sample_lines
            line_count_est = max(sample_lines, int(size_bytes / bytes_per_line))
        else:
            line_count_est = 0
    except Exception:
        line_count_est = 0

    # Read first and last few lines to extract date range
    first_dt: str = "unknown"
    last_dt: str = "unknown"
    try:
        first_lines, last_lines = _read_first_last_lines(filepath, n=5)
        # Skip header line (first_lines[0] is likely the header)
        data_start = 1 if len(first_lines) > 1 else 0
        if len(first_lines) > data_start:
            parsed = _parse_datetime_from_line(first_lines[data_start])
            if parsed:
                first_dt = parsed
        if last_lines:
            parsed = _parse_datetime_from_line(last_lines[-1])
            if parsed:
                last_dt = parsed
    except Exception:
        pass

    # Estimate bar count: line_count minus header(s)
    bar_count_est = max(0, line_count_est - 1)

    date_range = f"{first_dt} -> {last_dt}" if (first_dt != "unknown" or last_dt != "unknown") else "unknown"

    return {
        "size_mb": size_mb,
        "bar_count_est": bar_count_est,
        "first_dt": first_dt,
        "last_dt": last_dt,
        "date_range": date_range,
    }


# ── JSON inspection ───────────────────────────────────────────────────────────

def inspect_json(json_file: Path) -> dict[str, dict]:
    """
    Load the output JSON and return per-instrument bar counts and date ranges.

    Returns:
        {instrument: {bar_count, first_dt, last_dt, date_range}}
    """
    with open(json_file, "r") as f:
        data: dict = json.load(f)

    results: dict[str, dict] = {}
    for instrument, bars in data.items():
        if not bars:
            results[instrument] = {
                "bar_count": 0,
                "first_dt": "empty",
                "last_dt": "empty",
                "date_range": "empty",
            }
            continue
        first_dt = bars[0].get("dt", "unknown")
        last_dt = bars[-1].get("dt", "unknown")
        results[instrument] = {
            "bar_count": len(bars),
            "first_dt": first_dt,
            "last_dt": last_dt,
            "date_range": f"{first_dt} -> {last_dt}",
        }
    return results


# ── Table rendering ───────────────────────────────────────────────────────────

def _format_size(mb: float) -> str:
    if mb >= 1000:
        return f"{mb/1024:.1f} GB"
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{mb*1024:.0f} KB"


def render_table(rows: list[dict], use_tabulate: bool = True) -> str:
    """
    Render the summary table as a string.
    Uses tabulate if available; falls back to fixed-width plain text.
    """
    headers = ["#", "Instrument", "Role", "Subnet", "CSV", "CSV Date Range", "CSV Bars Est", "JSON Bars"]

    table_data = []
    for i, row in enumerate(rows, 1):
        csv_info = _format_size(row["csv_size_mb"]) if row["csv_found"] else "-"
        csv_range = row["csv_date_range"] if row["csv_found"] else "NOT FOUND"
        csv_bars = f"{row['csv_bars_est']:,}" if row["csv_found"] else "-"
        json_bars = f"{row['json_bars']:,}" if row["json_bars"] is not None else "-"

        table_data.append([
            i,
            row["instrument"],
            row["role"],
            row["subnet"],
            csv_info,
            csv_range,
            csv_bars,
            json_bars,
        ])

    if use_tabulate and _HAS_TABULATE:
        return _tabulate(table_data, headers=headers, tablefmt="simple")

    # Plain text fallback
    col_widths = [len(h) for h in headers]
    for row_data in table_data:
        for j, cell in enumerate(row_data):
            col_widths[j] = max(col_widths[j], len(str(cell)))

    sep = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(h.ljust(col_widths[j]) for j, h in enumerate(headers))

    lines = [header_line, sep]
    for row_data in table_data:
        line = "  ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row_data))
        lines.append(line)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algo C2 v2: Data completeness audit for all 43 instruments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory to scan for CSV files",
    )
    parser.add_argument(
        "--json_file", default=None,
        help="Path to processed JSON file (e.g. algo_c2_43_data.json) for bar count audit",
    )
    parser.add_argument(
        "--no_tabulate", action="store_true",
        help="Force plain text output even if tabulate is installed",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    json_data: dict[str, dict] = {}
    if args.json_file:
        json_path = Path(args.json_file)
        if not json_path.is_file():
            print(f"Warning: JSON file not found: {json_path}", file=sys.stderr)
        else:
            print(f"Loading JSON: {json_path} ...")
            try:
                json_data = inspect_json(json_path)
                print(f"  Found {len(json_data)} instruments in JSON\n")
            except Exception as exc:
                print(f"Warning: failed to read JSON: {exc}", file=sys.stderr)

    print(f"Scanning CSV directory: {data_dir}")
    print(f"Universe: {len(ALL_INSTRUMENTS)} instruments\n")

    rows: list[dict] = []
    csv_found_count = 0
    json_found_count = 0

    for instrument in ALL_INSTRUMENTS:
        role = "signal-only" if instrument in SIGNAL_ONLY else "tradeable"
        subnet = get_subnet(instrument)

        csv_path = find_csv_for_instrument(data_dir, instrument)
        csv_found = csv_path is not None

        if csv_found:
            csv_found_count += 1
            try:
                meta = inspect_csv(csv_path)
                csv_size_mb = meta["size_mb"]
                csv_date_range = meta["date_range"]
                csv_bars_est = meta["bar_count_est"]
            except Exception as exc:
                csv_size_mb = 0.0
                csv_date_range = f"read error: {exc}"
                csv_bars_est = 0
        else:
            csv_size_mb = 0.0
            csv_date_range = ""
            csv_bars_est = 0

        json_bars: int | None = None
        if instrument in json_data:
            json_bars = json_data[instrument]["bar_count"]
            json_found_count += 1

        rows.append({
            "instrument":    instrument,
            "role":          role,
            "subnet":        subnet,
            "csv_found":     csv_found,
            "csv_path":      str(csv_path) if csv_found else None,
            "csv_size_mb":   csv_size_mb,
            "csv_date_range": csv_date_range,
            "csv_bars_est":  csv_bars_est,
            "json_bars":     json_bars,
        })

    use_tabulate = _HAS_TABULATE and not args.no_tabulate
    print(render_table(rows, use_tabulate=use_tabulate))

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(ALL_INSTRUMENTS)
    csv_pct = 100.0 * csv_found_count / total
    tradeable_total = len(TRADEABLE)
    tradeable_found = sum(
        1 for r in rows if r["role"] == "tradeable" and r["csv_found"]
    )
    signal_total = len(SIGNAL_ONLY)
    signal_found = sum(
        1 for r in rows if r["role"] == "signal-only" and r["csv_found"]
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total instruments    : {total}")
    print(f"  CSV files found      : {csv_found_count} / {total}  ({csv_pct:.1f}%)")
    print(f"    Tradeable          : {tradeable_found} / {tradeable_total}")
    print(f"    Signal-only        : {signal_found} / {signal_total}")

    if json_data:
        json_pct = 100.0 * json_found_count / total
        print(f"  JSON bars available  : {json_found_count} / {total}  ({json_pct:.1f}%)")

        # Date range across all JSON instruments
        all_first = [json_data[i]["first_dt"] for i in json_data if json_data[i]["bar_count"] > 0]
        all_last  = [json_data[i]["last_dt"]  for i in json_data if json_data[i]["bar_count"] > 0]
        if all_first:
            print(f"  JSON date range      : {min(all_first)} -> {max(all_last)}")
        total_json_bars = sum(v["bar_count"] for v in json_data.values())
        print(f"  Total JSON bars      : {total_json_bars:,}")

    # List missing tradeable instruments explicitly (most impactful)
    missing_tradeable = [r["instrument"] for r in rows if r["role"] == "tradeable" and not r["csv_found"]]
    if missing_tradeable:
        print()
        print(f"  Missing tradeable ({len(missing_tradeable)}): {', '.join(missing_tradeable)}")

    missing_signal = [r["instrument"] for r in rows if r["role"] == "signal-only" and not r["csv_found"]]
    if missing_signal:
        print(f"  Missing signal-only ({len(missing_signal)}): {', '.join(missing_signal)}")

    print("=" * 60)

    if not _HAS_TABULATE:
        print("\nTip: install tabulate for prettier output: pip install tabulate")


if __name__ == "__main__":
    main()
