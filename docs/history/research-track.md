# Historical: research track (removed)

The working tree no longer contains the France GDELT / warehouse / heterogeneous GNN / learned graph-builder / world-model program. Git history still has it.

## Last commits that still contained that tree

| Ref | SHA | Date |
|-----|-----|------|
| `origin/main` (untouched) | `b50ae186` | 2026-05-19 |
| Testbed `calibration-subgraph` (source of this harness-only branch) | `b8ac2d66` | 2026-06-23 |
| Nested `graph-vault` repo at copy time | `cd1231c` | (no remotes; files are now tracked in this repo) |

This branch (`harness-only`) was cut from `b8ac2d66`, then research packages and the v2 cognitive/tournament/GNN-calibration stack were deleted.

## What was removed

- `baselines/`, `evals/`, DuckDB warehouse, GDELT/ACLED ingest, Wikidata grounding
- Graph-builder contract implementation and WM ablation code
- Docs: France benchmark, warehouse v1, builder contract, reviewers-guide as front door
- `harness/orchestrator_v2.py`, `outside_view.py`, tournament watcher, tag/mechanism calibration

Do not revive that program unless explicitly un-parked.

## Parked checkouts (not deleted)

- `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/testbed` — `calibration-subgraph`
- `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/gnn-agent` — `spec/gnn-agent-architecture`
