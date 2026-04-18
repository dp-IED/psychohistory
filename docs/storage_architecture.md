# Storage Architecture

This document describes **where data lives** and how workspaces share it. Program goals, **roadmap**, and **what to avoid** are in [`project.md`](../project.md), [`roadmap.md`](../roadmap.md), and [`reviewers-guide.md`](reviewers-guide.md).

Persistent event data lives in one DuckDB warehouse outside repo worktrees by
default:

```text
/Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb
```

Override the root with `PSYCHOHISTORY_DATA_ROOT` or per command with
`--data-root`. JSONL tapes remain supported for import/export and compatibility,
but they are not the primary store. Avoid keeping large raw folders, mixed tapes,
weekly snapshots, or experiment outputs under repo-local `data/**`.

## Workspace Setup (Symlink)

Run this once per git workspace so all worktrees share the same warehouse file:

```bash
mkdir -p /Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse

# If this workspace already has a local DB, move it to shared storage.
if [ -f data/warehouse/events.duckdb ] && [ ! -L data/warehouse/events.duckdb ]; then
  mv data/warehouse/events.duckdb \
    /Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb
fi

# Point the workspace to the shared DB.
ln -sfn /Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb \
  data/warehouse/events.duckdb
```

This keeps the warehouse accessible across future workspaces without refetching.

## Initialize

```bash
python -m ingest.event_warehouse init
```

## Import Existing GDELT

Import the shared GDELT tape without refetching:

```bash
python -m ingest.event_warehouse import-tape \
  --input /Users/darenpalmer/conductor/shared-data/psychohistory-v2/gdelt/tape/france_protest/events.jsonl
```

Inspect the warehouse:

```bash
python -m ingest.event_warehouse audit
```

## Fetch ACLED Into Warehouse

```bash
source ~/.config/psychohistory/acled.env

python -m ingest.acled_raw fetch-france-protests \
  --event-start 2019-01-01 \
  --event-end 2026-01-04 \
  --limit 5000 \
  --max-pages 100 \
  --raw-retention none \
  --normalize-to-warehouse \
  --availability-policy event_date_lag \
  --availability-lag-days 7
```

`--raw-retention none` writes fetch metadata and manifest rows, but does not
persist raw API pages. Use `compressed` or `full` only when raw fragment
retention is explicitly needed for debugging or reproducibility.

## Run Source Experiments

```bash
python -m baselines.backtest source-experiments \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
  --snapshot-mode in-memory \
  --predictions-format jsonl.gz \
  --out-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2/runs/source_experiments/full_2026_04_16 \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --experiments gdelt_only acled_only gdelt_plus_acled gdelt_plus_acled_no_source_identity gdelt_plus_acled_no_event_attributes \
  --epochs 30 \
  --hidden-dim 64
```

Weekly graph snapshots are built in memory by default. Add
`--snapshot-mode materialize` only when snapshot artifacts are needed; the
default materialized format is `.json.gz`.

## Export Compatibility Tapes

```bash
python -m ingest.event_warehouse export-tape \
  --source-names gdelt_v2_events,acled \
  --out /tmp/france_protest_events.jsonl.gz
```

Use exported JSONL tapes for compatibility with legacy tooling, not as the
canonical long-term store.

## Disk Inspection

```bash
python -m ingest.data_gc report
```

## Safe Pruning

Preview:

```bash
python -m ingest.data_gc prune --raw --smoke-runs --dry-run
```

Apply:

```bash
python -m ingest.data_gc prune --raw --materialized-snapshots --smoke-runs
```

Pruning never deletes `warehouse/events.duckdb`. Raw pruning removes fragment
directories while leaving manifests and metadata in place.
