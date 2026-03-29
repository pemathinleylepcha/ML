param(
    [string]$Root = 'D:\dataset-ml\DataExtractor',
    [string]$TerminalPath = 'C:\Program Files\MetaTrader 5\terminal64.exe',
    [string]$TesterSymbol = 'EURUSD',
    [string]$TesterPeriod = 'M1',
    [string]$FromDate = '2018.01.01',
    [string]$ToDate = '2026.03.25',
    [int]$Model = 1,
    [string]$SymbolsCsv = '',
    [string]$SymbolsFileName = 'symbols.txt',
    [string]$TimeframesCsv = 'M1,M5,M15',
    [int]$HistoryRetryAttempts = 40,
    [int]$HistoryRetryDelayMs = 0,
    [int]$TesterTasksPerPass = 0,
    [string]$DatasetRootName = 'DataExtractor',
    [switch]$ExportTicks,
    [switch]$Visual,
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'

$expertPath = 'DataExtractor\MLBatchExporterEA.ex5'
$configPath = Join-Path $Root 'MLBatchExporterEA.strategy.ini'
$customStartValue = "$FromDate 00:00:00"
$customEndValue = if ($ToDate -match '\d{2}:\d{2}:\d{2}$') { $ToDate } else { "$ToDate 23:59:59" }
$effectiveSymbolsFileName = if ([string]::IsNullOrWhiteSpace($SymbolsCsv)) { $SymbolsFileName } else { '' }

$tickChunkMinutesVal = if ($ExportTicks) { '720||1||1||1440||N' } else { '60||1||1||1440||N' }
$tickChunkDaysVal    = if ($ExportTicks) { '0||1||1||30||N'    } else { '7||1||1||30||N'    }
$exportTicksVal      = if ($ExportTicks) { 'true||false||0||true||N'  } else { 'false||false||0||true||N' }
$exportCandlesVal    = if ($ExportTicks) { 'false||false||0||true||N' } else { 'true||false||0||true||N'  }

$inputs = [ordered]@{
    SymbolsCsv             = $SymbolsCsv
    SymbolsFileName        = $effectiveSymbolsFileName
    TimeframesCsv          = $TimeframesCsv
    ExportTicks            = $exportTicksVal
    ExportCandles          = $exportCandlesVal
    DateMode               = 'custom'
    RollingYears           = '15||1||1||30||N'
    CustomStart            = $customStartValue
    CustomEnd              = $customEndValue
    UseCommonFiles         = 'true||false||0||true||N'
    OverwriteExisting      = 'false||false||0||true||N'
    ResumeFromCheckpoint   = 'true||false||0||true||N'
    TickChunkDays          = $tickChunkDaysVal
    TickChunkMinutes       = $tickChunkMinutesVal
    HistoryRetryAttempts   = "$HistoryRetryAttempts||1||1||200||N"
    HistoryRetryDelayMs    = "$HistoryRetryDelayMs||0||0||5000||N"
    TesterTasksPerPass     = "$TesterTasksPerPass||0||1||1000||N"
    TesterStopWhenComplete = 'true||false||0||true||N'
    DatasetRootName        = $DatasetRootName
}

$mode = if ($ExportTicks) { 'Tick export' } else { 'Candle export' }
$content = New-Object System.Collections.Generic.List[string]
$content.Add("; MLBatchExporterEA Strategy Tester export config")
$content.Add("; $mode from $FromDate through $ToDate")
$content.Add('[Tester]')
$content.Add("Expert=$expertPath")
$content.Add("Symbol=$TesterSymbol")
$content.Add("Period=$TesterPeriod")
$content.Add('Optimization=0')
$content.Add("Model=$Model")
$content.Add("FromDate=$FromDate")
$content.Add("ToDate=$ToDate")
$content.Add('ForwardMode=0')
$content.Add('Deposit=10000')
$content.Add('Currency=USD')
$content.Add('ProfitInPips=0')
$content.Add('Leverage=100')
$content.Add('ExecutionMode=0')
$content.Add('OptimizationCriterion=0')
$content.Add(('Visual=' + ($(if ($Visual) { '1' } else { '0' }))))
$content.Add('[TesterInputs]')

foreach ($entry in $inputs.GetEnumerator()) {
    $content.Add("$($entry.Key)=$($entry.Value)")
}

Set-Content -LiteralPath $configPath -Value $content -Encoding ASCII

[pscustomobject]@{
    ConfigPath    = $configPath
    TerminalPath  = $TerminalPath
    LaunchPlanned = [bool]$Launch
} | Format-List

if ($Launch) {
    if (-not (Test-Path -LiteralPath $TerminalPath)) {
        throw "terminal64.exe was not found at $TerminalPath"
    }
    Start-Process -FilePath $TerminalPath -ArgumentList "/config:$configPath"
}
