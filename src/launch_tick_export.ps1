# launch_tick_export.ps1
# Launches MT5 Strategy Tester to export tick data for missing/short pairs
# using the existing MLBatchExporterEA.

param(
    [string]$Root          = 'D:\dataset-ml\DataExtractor',
    [string]$TerminalPath  = 'C:\Program Files\MetaTrader 5\terminal64.exe',
    [string]$FromDate      = '2025.03.01',
    [string]$ToDate        = '2026.03.26',
    [string]$SymbolsFile   = 'symbols_ticks.txt',
    [int]   $TickChunkMins = 720,          # 12h chunks — safe for any broker
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'

# ── Symbols to download ──────────────────────────────────────────────────────
# Missing entirely from tick folder
$missing = @(
    'AUDCAD','AUDCHF','AUDJPY','AUDNZD',
    'CADCHF','CADJPY','CHFJPY',
    'NZDCAD','NZDCHF','NZDJPY',
    'USDMXN','USDZAR'
)
# Pairs with short history — extend back to 2025-03-01
$short = @(
    'GBPUSD','USDJPY',
    'GBPJPY','GBPCAD','EURGBP','EURAUD',
    'US30','SPX500',
    'XAUUSD','XAGUSD',
    'BTCUSD','NAS100'
)

$allSymbols = ($missing + $short) | Select-Object -Unique | Sort-Object

# ── Write symbols file ───────────────────────────────────────────────────────
$symFile = Join-Path $Root $SymbolsFile
$allSymbols | Set-Content -LiteralPath $symFile -Encoding ASCII
Write-Host "Symbols file : $symFile"
Write-Host "Symbols      : $($allSymbols -join ', ')"
Write-Host "Date range   : $FromDate -> $ToDate"
Write-Host "Chunk        : $TickChunkMins minutes ($([math]::Round($TickChunkMins/60,1))h per request)"
Write-Host ""

# ── Build ini config ─────────────────────────────────────────────────────────
$iniPath    = Join-Path $Root 'tick_export.strategy.ini'
$expertPath = 'DataExtractor\MLBatchExporterEA.ex5'

$inputs = [ordered]@{
    SymbolsCsv             = ''
    SymbolsFileName        = $SymbolsFile
    TimeframesCsv          = 'M1'               # unused when ExportCandles=false
    ExportTicks            = 'true||false||0||true||N'
    ExportCandles          = 'false||false||0||true||N'
    DateMode               = 'custom'
    RollingYears           = '15||1||1||30||N'
    CustomStart            = "$FromDate 00:00:00"
    CustomEnd              = "$ToDate 23:59:59"
    UseCommonFiles         = 'true||false||0||true||N'
    OverwriteExisting      = 'false||false||0||true||N'
    ResumeFromCheckpoint   = 'true||false||0||true||N'
    TickChunkDays          = '0||1||1||30||N'    # 0 = use TickChunkMinutes
    TickChunkMinutes       = "$TickChunkMins||1||1||1440||N"
    HistoryRetryAttempts   = '60||1||1||200||N'
    HistoryRetryDelayMs    = '500||0||0||5000||N'
    TesterTasksPerPass     = '0||0||1||1000||N'
    TesterStopWhenComplete = 'true||false||0||true||N'
    DatasetRootName        = 'DataExtractor'
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('; MLBatchExporterEA Tick Export config')
$lines.Add("; Downloading missing/short tick pairs: $FromDate -> $ToDate")
$lines.Add('; Model=4 = Every tick based on real ticks (downloads actual broker tick data)')
$lines.Add('[Tester]')
$lines.Add("Expert=$expertPath")
$lines.Add('Symbol=EURUSD')     # pivot symbol for tester (irrelevant — EA loops all symbols)
$lines.Add('Period=M1')
$lines.Add('Optimization=0')
$lines.Add('Model=4')           # Every tick based on real ticks
$lines.Add("FromDate=$FromDate")
$lines.Add("ToDate=$ToDate")
$lines.Add('ForwardMode=0')
$lines.Add('Deposit=10000')
$lines.Add('Currency=USD')
$lines.Add('ProfitInPips=0')
$lines.Add('Leverage=100')
$lines.Add('ExecutionMode=0')
$lines.Add('OptimizationCriterion=0')
$lines.Add('Visual=0')
$lines.Add('[TesterInputs]')

foreach ($kv in $inputs.GetEnumerator()) {
    $lines.Add("$($kv.Key)=$($kv.Value)")
}

Set-Content -LiteralPath $iniPath -Value $lines -Encoding ASCII
Write-Host "INI written  : $iniPath"

# ── Estimated size warning ────────────────────────────────────────────────────
$nSymbols = $allSymbols.Count
$days = ([datetime]::ParseExact($ToDate,'yyyy.MM.dd',$null) -
         [datetime]::ParseExact($FromDate,'yyyy.MM.dd',$null)).Days
$chunksPerSym = [math]::Ceiling($days * 24 * 60 / $TickChunkMins)
$totalChunks  = $nSymbols * $chunksPerSym
Write-Host ""
Write-Host "Estimated chunks : $totalChunks ($nSymbols symbols x $chunksPerSym chunks/sym)"
Write-Host "Approx disk      : ~$([math]::Round($nSymbols * $days * 0.5, 0)) MB (rough estimate)"
Write-Host ""

# ── Launch ────────────────────────────────────────────────────────────────────
if ($Launch) {
    if (-not (Test-Path -LiteralPath $TerminalPath)) {
        throw "MT5 terminal not found at: $TerminalPath"
    }

    # Kill any existing terminal instance first to avoid config conflicts
    $running = Get-Process -Name 'terminal64' -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "MT5 is already running (PID $($running.Id)) — launching new instance with tick config"
        Write-Host "NOTE: MT5 will open in Strategy Tester mode automatically."
    }

    Write-Host "Launching MT5 with tick export config..."
    Start-Process -FilePath $TerminalPath -ArgumentList "/config:$iniPath"
    Write-Host "MT5 launched. Monitor progress in:"
    Write-Host "  Log  : $Root\run.log"
    Write-Host "  Data : $Root\YEAR\Q*\SYMBOL\ticks_*.csv"
} else {
    Write-Host "Dry run complete. Add -Launch to start MT5."
    Write-Host ""
    Write-Host "To launch:"
    Write-Host "  launch_tick_export.ps1 -Launch"
}
