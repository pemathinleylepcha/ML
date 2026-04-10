# Algo C2

Working repository for code, experiments, and operational tooling.

## What `DOC/` Stores

The `DOC/` tree is the thinking layer of the repository.

It is where the project keeps:
- the gist of ideas before they become implementation
- design intent and the reasoning behind decisions
- tradeoffs, doubts, critiques, and course corrections
- action lists, investigation notes, and workflow rules
- project memory that helps future work start with context instead of guesswork

In short, `DOC/` stores how the work is being thought through, not just what code exists.

## What Kind of Material Lives There

The documents typically capture:
- hypotheses and proposed directions
- bundles of work and execution plans
- postmortems and autopsies after failures
- operational runbooks and deployment workflows
- review notes, objections, and alternative paths
- persistent notes about recurring concerns, expectations, and ways of working

Some files are stable references. Others are working notes. Together they form the memory of the project.

## What `DOC/` Is Not

`DOC/` is not meant to be:
- a polished marketing description
- a guaranteed up-to-date summary of the exact live system state
- the only source of truth for implementation details

The codebase, tests, and runtime artifacts remain the source of truth for behavior. The documents explain the intent, the reasoning, and the path taken to get there.

## How To Read It

Read `DOC/` as a map of thinking:
- use it to understand why certain choices were made
- use it to recover context after interruptions
- use it to see which ideas were tried, rejected, or deferred

Do not read it as a single linear manual. It is closer to a project notebook than a product brochure.

## Subfolders

Some subfolders hold persistent memory for specific workflows or collaborators. For example:
- `DOC/CODEX/` stores durable notes about working expectations, trust, workflow quality, and recurring lessons
- `DOC/CLAUDE/` stores action lists, planning notes, and human-facing thought traces that are useful to preserve

These are not separate products. They are simply different shelves in the same memory system.

## Repository Note

This repository keeps the git-facing overview intentionally lightweight.
The detailed thinking, planning, and historical context live under `DOC/`.
