# Algo C2 — Data Format

## Source CSV format (broker tick data)

Tab-separated, one file per pair, same format for all 35 instruments:

```
<DATE>      <TIME>           <BID>    <ASK>    <LAST>  <VOLUME>  <FLAGS>
2026.03.02  00:05:00.718     1.17688  1.17728          —         6
2026.03.02  00:05:00.843     1.17685  1.17737          —         6
2026.03.02  00:05:01.390     1.17679  —                —         2
2026.03.02  00:05:10.108     —        1.17728          —         4
```

**FLAGS values:**
- `6` = full bid+ask quote
- `2` = bid only (ask forward-filled from last valid quote)
- `4` = ask only (bid forward-filled from last valid quote)

**Expected filenames:**
```
EURUSD_202603020005_202603202259.csv
GBPUSD_202603020005_202603202259.csv
BTCUSD_202603020005_202603202259.csv
XAUUSD_202603020005_202603202259.csv
...
```

---

## Pip sizes per instrument

| Pair | Pip size | Price decimals |
|------|----------|----------------|
| All JPY crosses (USDJPY, EURJPY…) | 0.01 | 3 |
| Standard FX (EURUSD, GBPUSD…) | 0.0001 | 5 |
| BTCUSD | 1.0 | 2 |
| US30 | 1.0 | 2 |
| XAUUSD | 0.1 | 2 |
| XAGUSD | 0.01 | 3 |
| XBRUSD | 0.01 | 3 |
| USDMXN, USDZAR | 0.0001 | 5 |

---

## Output JSON schema (algo_c2_5day_data.json)

```json
{
  "EURUSD": [
    {
      "dt": "2026-03-02 00:05",
      "o": 1.17708,
      "h": 1.17714,
      "l": 1.17696,
      "c": 1.17708,
      "sp": 5.17,
      "tk": 44
    },
    ...
  ],
  "GBPUSD": [...],
  "BTCUSD": [...],
  ...
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `dt` | string | `YYYY-MM-DD HH:MM` UTC |
| `o` | float | Open (mid price) |
| `h` | float | High (mid price) |
| `l` | float | Low (mid price) |
| `c` | float | Close (mid price) |
| `sp` | float | Mean spread in pips for that bar |
| `tk` | int | Tick count in that 1-min bar |

Mid price = (bid + ask) / 2  
Spread = (ask − bid) / pip_size

---

## Processing pipeline

```bash
# Install dependencies
pip install pandas numpy scipy

# Process all 35 CSVs → JSON
python process_fx_csv_35.py \
  --input_dir /path/to/csv/folder \
  --output algo_c2_5day_data.json \
  --start 2026-03-02 \
  --end 2026-03-06
```

The script:
1. Auto-matches CSV files by pair name in filename
2. Parses DATE + TIME columns, forward-fills partial ticks (FLAGS 2/4)
3. Resamples to 1-min OHLC using mid price
4. Computes mean spread per bar in pips
5. Reports missing pairs and expected filenames
6. Outputs compact JSON with separators `(',', ':')`

**Expected output size:** ~15–30 MB for 35 pairs × 5 days × ~1435 bars each

---

## Real data stats (Mar 2–6 2026)

| Pair | Bars | Session start | Open | Close |
|------|------|---------------|------|-------|
| EURUSD | 7173 | 00:05 | 1.17708 | 1.16176 |
| GBPUSD | 7173 | 00:05 | 1.34074 | 1.34120 |
| USDJPY | 7111 | 00:05 | 156.172 | 157.786 |
| BTCUSD | 7175 | 00:05 | 65750.76 | 68331.50 |
| XAUUSD | 6900 | 01:00 | 5378.60 | 5172.63 |
| XAGUSD | 6896 | 01:00 | 96.143 | 84.455 |
| US30 | 6825 | 01:00 | 48412.5 | 47461.5 |
| XBRUSD | 6300 | 03:00 | 78.245 | 90.945 |
| USDMXN | 7170 | 00:05 | 17.343 | 17.799 |
| USDZAR | 6000 | 04:00 | 16.011 | 16.544 |

Non-FX pairs start later due to thinner overnight liquidity. Missing bars are forward-filled in the sim engine.

---

## Master timeline

- All 35 pairs are aligned to a single master timestamp index (union of all bar timestamps)
- Missing bars for any pair are forward-filled from the last close
- Total unique timestamps: **7175** (Mon 00:05 → Fri 23:59 UTC)
- GBPJPY and USDJPY have 63–64 missing bars due to thin liquidity windows — forward-filled
