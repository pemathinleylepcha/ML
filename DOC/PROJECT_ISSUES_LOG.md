# Algo C2 — Project Issues Log

Operational, runtime, data, and validation blockers should be logged here with a timestamp, evidence, affected files, and next action.

Documentation rule:
- Add a new dated entry whenever a material blocker or resolved operational issue is discovered.
- Include exact file paths, report paths, probe logs, and remote paths when relevant.
- Keep model-code issues separate from host/runtime and data-coverage issues.
- Treat this file as the default memory log for ongoing problems, not just a one-off note.
- Every entry should include:
  - date and timestamp
  - status
  - evidence
  - affected files and paths
  - impact
  - next action

---

## 2026-04-05 07:30:17 +05:30

### 1. `staged_v4` signal is still not robust across folds

Status:
- Active

Summary:
- The latest honest no-GA, fixed-threshold validation still has one negative fold.
- The model is promising, but not stable enough to trust yet.

Evidence:
- [staged_v4_weekly_exec_v14_cpu_fixed_noga_t060_report.json](/d:/Algo-C2-Codex/data/remote_runs/staged_v4_weekly_exec_v14_cpu_fixed_noga_t060_report.json)
- [MULTIFOLD_RESULTS_AND_NEXT.md](/d:/Algo-C2-Codex/DOC/MULTIFOLD_RESULTS_AND_NEXT.md)

Current read:
- Mean Sharpe: `7.84`
- Fold 0 Sharpe: `-4.47`
- Fold 1 Sharpe: `20.15`
- Mean AUC: `0.7040`
- Mean trade count: `1873`

Affected files and paths:
- [config.py](/d:/Algo-C2-Codex/src/staged_v4/config.py)
- [backtest.py](/d:/Algo-C2-Codex/src/staged_v4/evaluation/backtest.py)
- [train_staged.py](/d:/Algo-C2-Codex/src/staged_v4/training/train_staged.py)
- [BACKTEST_EXECUTION_REWRITE.md](/d:/Algo-C2-Codex/DOC/BACKTEST_EXECUTION_REWRITE.md)
- [MULTIFOLD_VALIDATION_PLAN.md](/d:/Algo-C2-Codex/DOC/MULTIFOLD_VALIDATION_PLAN.md)

Impact:
- Training/eval pipeline works, but the strategy is not yet robust enough for promotion or live use.

Next action:
- Extend the training window beyond March only and validate on more folds after the runtime host is stable.

### 2. Remote training host runtime is blocking real JIT training

Status:
- Active

Summary:
- The no-cache JIT training path is code-complete and locally validated, but the remote host still stalls at `import torch`.
- This is now a host/runtime problem, not a repo logic problem.

Evidence:
- Remote probe log:
  - `D:\work\Algo-C2-Codex\data\remote_runs\probe_remote_imports_cpuvenv_clean\probe.log`
  - latest behavior: log stops at `import=start`
- Repo probe helpers:
  - [probe_remote_imports.py](/d:/Algo-C2-Codex/src/probe_remote_imports.py)
  - [probe_remote_python_basic.py](/d:/Algo-C2-Codex/src/probe_remote_python_basic.py)
  - [run_probe_remote_imports_cpuvenv_clean.cmd](/d:/Algo-C2-Codex/scripts/run_probe_remote_imports_cpuvenv_clean.cmd)
  - [run_probe_remote_imports_portable_python_safeenv.cmd](/d:/Algo-C2-Codex/scripts/run_probe_remote_imports_portable_python_safeenv.cmd)

Affected files and paths:
- [run_staged_v4_real_jit_smoke.py](/d:/Algo-C2-Codex/src/run_staged_v4_real_jit_smoke.py)
- [dataset.py](/d:/Algo-C2-Codex/src/staged_v4/data/dataset.py)
- [jit_sequences.py](/d:/Algo-C2-Codex/src/staged_v4/data/jit_sequences.py)
- [train_staged.py](/d:/Algo-C2-Codex/src/staged_v4/training/train_staged.py)

What is already fixed in code:
- JIT minibatch feature construction exists.
- `tick` CSV loading was hardened to avoid the earlier pandas threaded crash.
- The smoke runner was reduced to low thread pressure and small batch size.

Impact:
- Full real no-cache training cannot currently be validated on the remote box.

Next action:
- Repair or replace the remote Python/runtime environment, then rerun the JIT smoke before any longer real training job.

### 3. `repository-codex` does not yet have enough historical coverage

Status:
- Active

Summary:
- The MT5-backed collector path is real and functioning, but the historical backfill is still far too incomplete for staged_v4 use.

Evidence:
- Remote coverage report:
  - `D:\COLLECT-TICK-MT5\metadata\repository_codex_coverage.json`

Current read:
- Total expected symbol-days: `23386`
- Total available symbol-days: `58`
- Overall coverage ratio: `0.00248`
- `usable_for_staged_v4_m1 = false`

Affected files and paths:
- [mt5_backfill_pipeline.py](/d:/Algo-C2-Codex/tmp_remote_collector/mt5_backfill_pipeline.py)
- [mt5_terminal_pipeline.py](/d:/Algo-C2-Codex/tmp_remote_collector/mt5_terminal_pipeline.py)
- [repository_coverage_report.py](/d:/Algo-C2-Codex/tmp_remote_collector/repository_coverage_report.py)
- [backfill_pipeline.bat](/d:/Algo-C2-Codex/tmp_remote_collector/backfill_pipeline.bat)
- [run_backfill_scheduled.ps1](/d:/Algo-C2-Codex/tmp_remote_collector/run_backfill_scheduled.ps1)

Impact:
- The new collector output cannot be used as the staged_v4 real-data floor yet.

Next action:
- Keep the scheduled backfill running and use the coverage report as the gate before switching training to `repository-codex`.

### 4. MT5 local tick-store pressure on `C:` was fixed

Status:
- Resolved on 2026-04-05

Summary:
- The heavy MT5 terminal tick cache was consuming `C:` space.
- It has been moved to `D:` and replaced with a junction so MT5 still sees the original path.

Evidence:
- Repo migration helper:
  - [migrate_mt5_ticks_to_d.ps1](/d:/Algo-C2-Codex/tmp_remote_collector/migrate_mt5_ticks_to_d.ps1)
- Remote live path after migration:
  - `C:\Users\Thinley\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\bases\OctaFX-Demo\ticks`
  - junction target:
    `D:\COLLECT-TICK-MT5\mt5-terminal-store\OctaFX-Demo\ticks`

Resolved state:
- Moved size: `41.67 GB`
- File count: `3647`
- MT5 terminal restarted after migration

Impact:
- Future growth for this tick-store path no longer consumes `C:` directly.

---

## 2026-04-05 07:32:10 +05:30

### Documentation policy update

Status:
- Active documentation rule

Summary:
- Active Algo C2 blockers should always be recorded in `DOC/PROJECT_ISSUES_LOG.md`.
- This log should be updated whenever a material issue is discovered, confirmed, resolved, or re-scoped.

Affected files and paths:
- [PROJECT_ISSUES_LOG.md](/d:/Algo-C2-Codex/DOC/PROJECT_ISSUES_LOG.md)
- [README.md](/d:/Algo-C2-Codex/DOC/README.md)
- [CHANGELOG.md](/d:/Algo-C2-Codex/DOC/CHANGELOG.md)

Impact:
- The repo gets a durable issue-memory trail instead of relying on chat history or ad hoc notes.

Next action:
- Keep appending dated issue entries here and link to this file from summary docs whenever new blockers appear.

---

## 2026-04-05 10:38:39 +05:30

### `repository-codex` is still not ready for `2024 -> today` use

Status:
- Active

Summary:
- Checked the live MT5 collector output to decide whether we can start using `repository-codex` as a real-data source from `2024` to the present.
- The answer is still no.
- Coverage is far below the minimum needed for staged_v4 or any dependable `M1` floor use.

Evidence:
- Remote coverage report:
  - `D:\COLLECT-TICK-MT5\metadata\repository_codex_coverage.json`
- Remote scheduler state:
  - `COLLECT-TICK-MT5-Codex-Backfill`
- Remote checkpoint:
  - `D:\COLLECT-TICK-MT5\metadata\mt5_backfill_checkpoint.json`

Current read:
- Coverage report generation time: `2026-04-05T04:44:03.032350+00:00`
- Required window in the live report: `2024-04-05 -> 2026-04-05`
- Pair count: `44`
- Total expected symbol-days: `23344`
- Total available symbol-days: `58`
- Overall coverage ratio: `0.0024845784784098698`
- Complete symbols: `0`
- `usable_for_staged_v4_m1 = false`
- Scheduler last run result: `1`
- Scheduler last run time: `2026-04-05 10:14:01`
- Checkpoint cursor is still near the beginning of history:
  - `pair_index = 30`
  - `day_start_utc = 2018-01-04T00:00:00+00:00`

Affected files and paths:
- [mt5_backfill_pipeline.py](/d:/Algo-C2-Codex/tmp_remote_collector/mt5_backfill_pipeline.py)
- [mt5_terminal_pipeline.py](/d:/Algo-C2-Codex/tmp_remote_collector/mt5_terminal_pipeline.py)
- [repository_coverage_report.py](/d:/Algo-C2-Codex/tmp_remote_collector/repository_coverage_report.py)
- [backfill_pipeline.bat](/d:/Algo-C2-Codex/tmp_remote_collector/backfill_pipeline.bat)
- [run_backfill_scheduled.ps1](/d:/Algo-C2-Codex/tmp_remote_collector/run_backfill_scheduled.ps1)
- Remote data root:
  - `D:\COLLECT-TICK-MT5\repository-codex`

Impact:
- We should not switch Algo C2 training or validation to `repository-codex` yet.
- The collector is producing real data, but the historical backfill remains too sparse for `2024 -> today` use.

Next action:
- Keep the backfill running, but treat the live coverage report as the gate.
- Do not use `repository-codex` for staged_v4 until coverage becomes materially complete across the required window.
