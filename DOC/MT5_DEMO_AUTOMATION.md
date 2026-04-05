# MT5 Demo Automation

This setup uses the best currently deployable model in the repo:

- Benchmark report: `data/remote_clean_2025_runs/clean_2025_no_bridge_optk10_report.json`
- Model family: compact FX, no bridge
- Outer-holdout win rate: `52.70%`
- Outer-holdout Sharpe: `7.92`
- PBO: `0.1104`

The higher-win-rate cooperative models are not the default live path because they are still flagged with `PBO=1.00`.

## Components

- Python signal service: `src/live_compact_demo_mt5.py`
- EA source: `mt5/Experts/AlgoC2CompactDemoBridgeEA.mq5`
- Demo runner: `scripts/run_live_compact_demo.ps1`
- EA installer helper: `scripts/install_mt5_demo_bridge.ps1`

## How It Works

1. The Python service connects to MT5, fetches the last closed `M5` bars for the 28 FX pairs, and computes the compact benchmark features.
2. It loads the compact CatBoost + isotonic calibrator and emits one desired position row per pair.
3. It writes the signal file into the MT5 common files folder:
   `C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\algo_c2_demo_signals.csv`
4. The EA reads that file with `FILE_COMMON` and executes on the demo account only.

## Safety

- The Python service refuses to run on a non-demo account unless `--allow-live-account` is passed.
- The EA also refuses to run on a non-demo account when `DemoOnly=true`.
- Max open trades are capped inside the EA.
- Orders are only opened from the compact benchmark path.

## Model Package

The default live model directory is:

- `models/live_compact_no_bridge_optk10`

It should contain:

- `research_compact_fx.cbm`
- `research_compact_fx_isotonic.pkl`
- `research_compact_fx_meta.json`

## Run Once

```powershell
.\scripts\run_live_compact_demo.ps1 -Once
```

This writes the latest desired positions into the MT5 common files folder.

## Run Continuously

```powershell
.\scripts\run_live_compact_demo.ps1
```

The service waits for each new `M5` close and refreshes the signal file.

## Install EA

If you know the MT5 terminal data directory:

```powershell
.\scripts\install_mt5_demo_bridge.ps1 -TerminalDataDir "C:\Path\To\MT5\Data\Folder"
```

If you only want the source path:

```powershell
.\scripts\install_mt5_demo_bridge.ps1
```

## Manual MT5 Steps

1. Open MetaEditor from the same MT5 demo terminal.
2. Open `AlgoC2CompactDemoBridgeEA.mq5`.
3. Compile it.
4. Attach it to any chart.
5. Keep `Algo Trading` enabled.
6. Start the Python service.

## Notes

- This machine currently has the `MetaTrader5` Python package installed, but `mt5.initialize()` reported that the MT5 x64 terminal is not discoverable. If MT5 lives in a nonstandard path, pass `--mt5-path`.
- The signal exchange uses the common files folder, so the Python service and EA do not need the same terminal hash folder.
