# Remote SCP Workflow

This project now has a source-only `scp` workflow for the remote Windows host configured as `algoc2`.

Remote paths:

- Bare repo: `D:\repos\Algo-C2-Codex.git`
- Sync target: `D:\work\Algo-C2-Codex`

The sync scripts send the current non-ignored working tree, not `data/`, `models/`, `.json`, `.env`, or Python cache files.

Commands:

```powershell
.\scripts\scp_push.ps1
.\scripts\scp_run.ps1 -Command "python --version"
.\scripts\scp_run.ps1 -Command "python src\test_research_stack.py"
.\scripts\scp_pull.ps1
```

Optional pull target:

```powershell
.\scripts\scp_pull.ps1 -DestinationPath "D:\temp\Algo-C2-Codex-remote"
```

Notes:

- `push` uploads a zip with `scp`, then expands it on the remote machine into `D:\work\Algo-C2-Codex`.
- `pull` creates a filtered zip on the remote machine, downloads it with `scp`, and overlays it onto the chosen local destination.
- `run` executes a PowerShell command inside `D:\work\Algo-C2-Codex` over `ssh`.
- The sync is additive and overwrite-based. It does not mirror-delete remote files that disappeared locally.
- Use `-RemoteHost other-host` if you want to target a different SSH host entry.
