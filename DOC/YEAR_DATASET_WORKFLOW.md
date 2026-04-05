## Clean Year Dataset Workflow

Target use case: build a clean one-year `DataExtractor` slice for the research stack.

### 1. Audit the current year

```powershell
python src\audit_year_dataset.py `
  --root data\DataExtractor `
  --year 2025 `
  --timeframes all `
  --detect-gaps `
  --json-out data\year_audit_2025.json `
  --csv-out data\year_audit_2025.csv
```

This produces:

- a JSON contract report with quarter summaries and a refetch plan
- a flat CSV for filtering in Excel / pandas

### 2. Inspect alias mismatches

Known historical aliases are handled by the research loader:

- `X -> XTIUSD`
- `EuSTX50 -> EUSTX50`
- `JN225 -> JPN225`

Dry-run alias normalization:

```powershell
python src\normalize_dataextractor_aliases.py --root data\DataExtractor
```

Apply alias normalization:

```powershell
python src\normalize_dataextractor_aliases.py --root data\DataExtractor --apply
```

### 3. Refetch a clean one-year M5 panel

The downloader now supports exact `--start` / `--end` ranges and chunked MT5 requests.

Example:

```powershell
python src\mt5_m5_download.py `
  --out data\clean_2025 `
  --start 2025-01-01 `
  --end 2025-12-31 `
  --instruments BTCUSD,AUS200,GER40,UK100,NAS100,EUSTX50,JPN225,SPX500,XTIUSD
```

For a full clean rebuild of the entire canonical universe:

```powershell
python src\mt5_m5_download.py `
  --out data\clean_2025 `
  --start 2025-01-01 `
  --end 2025-12-31
```

### 4. Train against the clean root

```powershell
python src\train_research_compact.py `
  --data-dir data\clean_2025 `
  --end 2025-12-31 `
  --fx-top-k 24
```

Remote wrapper after the clean 2025 rebuild:

```powershell
.\scripts\remote_train_clean_2025.ps1
.\scripts\remote_train_clean_2025.ps1 -WithBridge
```
