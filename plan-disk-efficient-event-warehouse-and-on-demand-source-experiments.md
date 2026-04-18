# Plan: Disk-Efficient Event Warehouse And On-Demand Source Experiments

**Status (2026-04):** The **canonical DuckDB warehouse**, shared data root, and **in-memory default** for source experiments are implemented; operational commands and workspace setup are documented in [`docs/storage_architecture.md`](docs/storage_architecture.md). The remainder of this file is **design history and detailed checklist**—read the storage doc first for day-to-day use.

---

## Summary

Refactor the ingestion and experiment pipeline away from duplicated JSONL/raw/snapshot trees and toward a compact canonical event warehouse outside repo workspaces.

The new default architecture:

- Stores persistent normalized events in one DuckDB database under a central data root.
- Keeps raw source API/download data ephemeral by default.
- Avoids writing full weekly graph snapshots unless explicitly requested.
- Runs source experiments directly from filtered warehouse queries.
- Writes only compact audits and compressed prediction files by default.
- Adds disk usage reporting and pruning commands.

This solves the current disk-space problem and prepares the project for more event sources without multiplying storage by source count, experiment count, and origin count.

## Core Decisions

- Use **DuckDB** as the canonical event warehouse.
- Add dependency: `duckdb>=1.0`.
- Use standard-library `gzip` for compressed JSONL/JSON artifacts.
- Put persistent data outside the repo by default:
  - Default data root: `/Users/darenpalmer/conductor/shared-data/psychohistory-v2`
  - Override with `PSYCHOHISTORY_DATA_ROOT`
  - Override per command with `--data-root`
- Make raw retention opt-in:
  - Default: `--raw-retention none`
  - Optional: `compressed`
  - Optional: `full`
- Make snapshot materialization opt-in:
  - Default source experiment mode: in-memory snapshots
  - Optional: `--materialize-snapshots`
  - Optional snapshot format: `.json.gz`
- Keep existing JSONL tape compatibility for now, but treat JSONL as import/export format, not the primary store.

## Goals

1. Reduce persistent disk footprint.
2. Avoid duplicated source tapes and mixed tapes.
3. Avoid per-experiment snapshot duplication.
4. Keep point-in-time forecasting semantics unchanged.
5. Preserve source-aware graph artifacts when requested.
6. Keep existing tests passing.
7. Add migration-compatible commands so existing shared GDELT JSONL can be imported without refetching.

## Non-Goals

- Do not redesign the forecasting task.
- Do not change the core `EventTapeRecord` fields.
- Do not implement cross-source deduplication.
- Do not add source-node GNN message passing.
- Do not require raw GDELT or raw ACLED retention for normal runs.
- Do not build a general lakehouse or cloud storage layer.

## Current State To Account For

The repo currently has:

- `ingest.event_tape.EventTapeRecord`
- GDELT raw fetch + GDELT JSONL normalization
- ACLED raw fetch + ACLED JSONL normalization
- `ingest.tape_merge` that writes a physical mixed JSONL tape
- `ingest.snapshot_export.build_snapshot_payload`
- `ingest.snapshot_export.export_weekly_snapshots`
- `baselines.source_experiments.run_source_layer_experiments`
- Backtests that load full JSONL tapes into memory
- Source experiments that currently write per-experiment weekly snapshots

The refactor should retain compatibility with those paths while adding a warehouse-backed path.

## Public API And Interface Changes

### 1. New Dependency

Update `pyproject.toml`:

```toml
dependencies = [
  "numpy>=1.26",
  "pydantic>=2.5",
  "xgboost>=2.0",
  "duckdb>=1.0",
]
```

No pandas dependency.

### 2. New Data Root Helper

Add `ingest/paths.py`.

Public functions:

```python
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/Users/darenpalmer/conductor/shared-data/psychohistory-v2")

def resolve_data_root(cli_value: str | Path | None = None) -> Path:
    ...
```

Behavior:

1. If `cli_value` is provided, return `Path(cli_value).expanduser().resolve()`.
2. Else if `PSYCHOHISTORY_DATA_ROOT` is set, return that expanded/resolved path.
3. Else return `/Users/darenpalmer/conductor/shared-data/psychohistory-v2`.
4. Create the directory only in commands that write; do not create it in pure read helpers.

Also add:

```python
def warehouse_path(data_root: Path) -> Path:
    return data_root / "warehouse" / "events.duckdb"

def runs_root(data_root: Path) -> Path:
    return data_root / "runs"
```

### 3. New Warehouse Module

Add `ingest/event_warehouse.py`.

DuckDB schema:

```sql
CREATE TABLE IF NOT EXISTS events (
  source_name TEXT NOT NULL,
  source_event_id TEXT NOT NULL,
  event_date DATE NOT NULL,
  source_available_at TIMESTAMPTZ NOT NULL,
  retrieved_at TIMESTAMPTZ NOT NULL,
  country_code TEXT NOT NULL,
  admin1_code TEXT NOT NULL,
  location_name TEXT,
  latitude DOUBLE,
  longitude DOUBLE,
  event_class TEXT NOT NULL,
  event_code TEXT NOT NULL,
  event_base_code TEXT NOT NULL,
  event_root_code TEXT NOT NULL,
  quad_class INTEGER,
  goldstein_scale DOUBLE,
  num_mentions INTEGER,
  num_sources INTEGER,
  num_articles INTEGER,
  avg_tone DOUBLE,
  actor1_name TEXT,
  actor1_country_code TEXT,
  actor2_name TEXT,
  actor2_country_code TEXT,
  source_url TEXT,
  raw_json TEXT NOT NULL,
  inserted_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (source_name, source_event_id)
);
```

Also create indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_name);
CREATE INDEX IF NOT EXISTS idx_events_event_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_available ON events(source_available_at);
CREATE INDEX IF NOT EXISTS idx_events_admin1 ON events(admin1_code);
```

Public functions:

```python
def init_warehouse(db_path: Path) -> None: ...

def upsert_records(
    *,
    db_path: Path,
    records: list[EventTapeRecord],
) -> dict[str, Any]: ...

def import_tape(
    *,
    db_path: Path,
    tape_path: Path,
    source_names: set[str] | None = None,
) -> dict[str, Any]: ...

def query_records(
    *,
    db_path: Path,
    source_names: set[str] | None = None,
    event_start: dt.date | None = None,
    event_end: dt.date | None = None,
    available_before: dt.datetime | None = None,
    country_code: str | None = None,
    event_class: str | None = None,
) -> list[EventTapeRecord]: ...

def source_counts(db_path: Path) -> dict[str, int]: ...

def export_tape(
    *,
    db_path: Path,
    out_path: Path,
    source_names: set[str] | None = None,
    gzip_output: bool | None = None,
) -> dict[str, Any]: ...
```

Implementation details:

- Use parameterized DuckDB queries.
- Serialize `raw` as compact JSON in `raw_json`.
- Reconstruct `EventTapeRecord` from rows in `query_records`.
- `event_end` is inclusive for CLI ergonomics, but query internally as `event_date <= event_end`.
- `export_tape` supports `.jsonl` and `.jsonl.gz`.
- `import_tape` uses current `load_event_tape`, which should be upgraded to read gzip too.

### 4. Extend Event Tape IO To Support Gzip

Update `ingest/event_tape.py`.

Changes:

```python
def open_text_auto(path: Path, mode: str):
    # .gz => gzip.open(..., "rt"/"wt")
    # else => path.open(...)
```

Update:

```python
load_event_tape(...)
write_event_tape(...)
```

Behavior:

- `load_event_tape` reads both `.jsonl` and `.jsonl.gz`.
- `write_event_tape` writes gzip if `out_path.suffix == ".gz"` or if explicit `compress=True`.
- Existing non-gzip behavior remains unchanged.

### 5. Add Warehouse CLI

Add parser to `ingest/event_warehouse.py`.

Commands:

```bash
python -m ingest.event_warehouse init \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2

python -m ingest.event_warehouse import-tape \
  --input /path/to/events.jsonl \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2

python -m ingest.event_warehouse export-tape \
  --source-names gdelt_v2_events,acled \
  --out /path/to/events.jsonl.gz \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2

python -m ingest.event_warehouse audit \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2
```

Audit output shape:

```json
{
  "warehouse_path": ".../warehouse/events.duckdb",
  "total_event_count": 87780,
  "source_counts": {
    "gdelt_v2_events": 51101,
    "acled": 36679
  },
  "earliest_event_date": "2019-01-01",
  "latest_event_date": "2026-01-04",
  "earliest_source_available_at": "...",
  "latest_source_available_at": "...",
  "admin1_count": 22,
  "database_bytes": 123456789
}
```

### 6. Raw Retention Flags

Update `ingest/acled_raw.py`.

Add CLI option:

```bash
--raw-retention none|compressed|full
```

Default:

```bash
--raw-retention none
```

Behavior:

- `full`: current behavior; write `fragments/page_*.jsonl`.
- `compressed`: write `fragments/page_*.jsonl.gz`.
- `none`: do not write raw fragment pages; write only:
  - `fetch_metadata.json`
  - `fetch_manifest.jsonl`
  - optional normalized direct output if `--normalize-to-warehouse` is provided.

Add optional flags:

```bash
--normalize-to-warehouse
--data-root ...
--availability-policy event_date_lag|timestamp|retrieved_at
--availability-lag-days 7
```

When `--normalize-to-warehouse` is set:

- Convert fetched rows to `EventTapeRecord` in memory page-by-page.
- Upsert into warehouse.
- Do not require raw fragments.
- Manifest rows include only counts, page number, endpoint path, status.
- Manifest must not include credentials, tokens, or request bodies.

Recommended full ACLED command after refactor:

```bash
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

Update `ingest/gdelt_raw.py` similarly, but do not require full direct-to-warehouse streaming in the first implementation if too invasive.

Minimum GDELT support:

- Add `--raw-retention full|compressed|none`.
- For `none`, write only manifest/metadata and do not write raw fragments.
- Leave current `ingest.event_tape normalize-france-protests` path for existing raw folders.
- Prefer importing the existing shared GDELT tape into the warehouse instead of refetching.

### 7. Replace Physical Mixed Tape With Warehouse Queries

Keep `ingest.tape_merge` for compatibility, but source experiments should no longer require:

```text
data/mixed/tape/france_protest/events.jsonl
```

Update `baselines/source_experiments.py`.

Add arguments to `run_source_layer_experiments`:

```python
def run_source_layer_experiments(
    *,
    tape_path: Path | None = None,
    warehouse_path: Path | None = None,
    data_root: Path | None = None,
    ...
    snapshot_mode: Literal["in_memory", "materialize"] = "in_memory",
    snapshot_format: Literal["json", "json.gz"] = "json.gz",
    predictions_format: Literal["jsonl", "jsonl.gz"] = "jsonl.gz",
) -> dict[str, Any]:
    ...
```

Resolution rules:

1. If `warehouse_path` is provided, load from warehouse.
2. Else if `data_root` is provided, use `warehouse_path(data_root)`.
3. Else if `tape_path` is provided, use current JSONL loading.
4. Else use default data root warehouse.

CLI changes:

```bash
python -m baselines.backtest source-experiments \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
  --snapshot-mode in-memory \
  --predictions-format jsonl.gz
```

Keep `--tape` for backward compatibility.

### 8. On-Demand Snapshot Construction

Update `baselines/source_experiments.py`.

Current behavior:

- Builds and writes weekly snapshots under `snapshots_root / experiment.name`.
- GNN backtest reads snapshots from disk.

New behavior:

- For each experiment and origin, build snapshot payload in memory with:
  ```python
  build_snapshot_payload(records=records, origin_date=origin, ...)
  ```
- Pass payload directly to GNN graph construction.
- Do not write snapshot files unless `snapshot_mode == "materialize"`.

Add helper:

```python
@dataclass
class OriginInputs:
    origin: dt.date
    snapshot: dict[str, Any]
    feature_rows: list[FeatureRow]
```

Add GNN runner variant:

```python
def run_gnn_backtest_from_payloads(
    *,
    train_inputs: list[OriginInputs],
    eval_inputs: list[OriginInputs],
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
    out_path: Path,
    epochs: int,
    hidden_dim: int,
    gnn_ablation: GNNGraphAblation | None = None,
    predictions_format: Literal["jsonl", "jsonl.gz"] = "jsonl.gz",
    progress: bool = False,
) -> dict[str, Any]:
    ...
```

This avoids writing snapshots just to read them back.

Keep existing `run_gnn_backtest` for compatibility with already materialized snapshots.

### 9. Prediction Compression

Add utility module `baselines/io.py` or `ingest/io_utils.py`.

Public function:

```python
def open_text_auto(path: Path, mode: str):
    ...
```

Use it for:

- prediction JSONL
- audit JSON if `.gz`
- event tape gzip
- snapshot gzip

Update `run_recurrence_backtest`, `run_tabular_backtest`, `run_gnn_backtest`, and source experiment runner to write `.jsonl.gz` when requested.

Default for source experiments:

```text
predictions/*.jsonl.gz
```

Audit remains uncompressed by default because it is small and human-inspectable.

### 10. Disk Usage And Pruning CLI

Add `ingest/data_gc.py`.

Commands:

```bash
python -m ingest.data_gc report \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2

python -m ingest.data_gc prune \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
  --raw \
  --materialized-snapshots \
  --smoke-runs \
  --dry-run

python -m ingest.data_gc prune \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
  --raw \
  --materialized-snapshots \
  --smoke-runs
```

Report output:

```json
{
  "data_root": "...",
  "total_bytes": 123,
  "paths": [
    {
      "path": "warehouse/events.duckdb",
      "bytes": 123,
      "category": "warehouse"
    },
    { "path": "raw/gdelt", "bytes": 123, "category": "raw" },
    {
      "path": "runs/source_experiments_smoke",
      "bytes": 123,
      "category": "run"
    },
    { "path": "snapshots", "bytes": 123, "category": "snapshots" }
  ]
}
```

Prune rules:

- `--raw`: delete raw source fragments only, not normalized warehouse.
- `--materialized-snapshots`: delete snapshot directories under runs.
- `--smoke-runs`: delete run directories whose name contains `smoke`.
- `--older-than-days N`: optional filter by mtime.
- `--dry-run`: default false, but tests should cover it.
- Never delete:
  - `warehouse/events.duckdb`
  - `manifests`
  - audits unless explicitly `--audits`

### 11. Documentation Updates

Add `docs/storage_architecture.md`.

Include:

- Why JSONL tapes remain import/export only.
- Default central data root.
- How to import existing shared GDELT tape:
  ```bash
  python -m ingest.event_warehouse import-tape \
    --input /Users/darenpalmer/conductor/shared-data/psychohistory-v2/gdelt/tape/france_protest/events.jsonl
  ```
- How to fetch ACLED directly into warehouse:

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

- How to run source experiments from warehouse:
  ```bash
  python -m baselines.backtest source-experiments \
    --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
    --snapshot-mode in-memory \
    --out-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2/runs/source_experiments/full_2026_04_16 \
    --train-origin-start 2021-01-04 \
    --train-origin-end 2024-12-30 \
    --eval-origin-start 2025-01-06 \
    --eval-origin-end 2025-12-29 \
    --experiments gdelt_only acled_only gdelt_plus_acled gdelt_plus_acled_no_source_identity gdelt_plus_acled_no_event_attributes \
    --epochs 30 \
    --hidden-dim 64
  ```
- How to inspect disk:
  ```bash
  python -m ingest.data_gc report
  ```
- How to prune safely:
  ```bash
  python -m ingest.data_gc prune --raw --smoke-runs --dry-run
  ```

Update `docs/acled_ingestion.md` to point to warehouse-first workflow.

Update `docs/source_layer_experiments.md` to make `--tape` legacy and `--data-root` preferred.

## Detailed Implementation Steps

### Step 1: Add IO Utilities

Files:

- `ingest/io_utils.py`
- tests in `tests/test_io_utils.py`

Implement:

- `open_text_auto`
- `write_json_atomic`
- `write_jsonl_records`
- gzip handling based on suffix

Acceptance:

- Can read/write `.jsonl`.
- Can read/write `.jsonl.gz`.
- Existing JSONL tests continue to pass.

### Step 2: Add Data Root Helper

Files:

- `ingest/paths.py`
- tests in `tests/test_paths.py`

Acceptance:

- CLI value wins.
- Env var wins over default.
- Default is `/Users/darenpalmer/conductor/shared-data/psychohistory-v2`.

### Step 3: Add Warehouse Module

Files:

- `ingest/event_warehouse.py`
- tests in `tests/test_event_warehouse.py`

Tests:

1. `init_warehouse` creates DB and table.
2. `upsert_records` inserts records.
3. Duplicate `(source_name, source_event_id)` upserts do not duplicate rows.
4. `query_records(source_names={"acled"})` filters correctly.
5. `query_records(available_before=...)` applies point-in-time filtering.
6. `import_tape` imports a JSONL tape.
7. `export_tape` writes `.jsonl.gz` and reloads correctly.
8. `audit` returns expected source counts.

### Step 4: Upgrade Event Tape Gzip Compatibility

Files:

- `ingest/event_tape.py`
- tests in `tests/test_event_tape.py`

Acceptance:

- Existing tests pass.
- New test verifies `load_event_tape` reads `.jsonl.gz`.

### Step 5: Add Warehouse Import Path For Existing GDELT

No data fetch.

Command expected to work:

```bash
python -m ingest.event_warehouse import-tape \
  --input /Users/darenpalmer/conductor/shared-data/psychohistory-v2/gdelt/tape/france_protest/events.jsonl
```

Acceptance:

- Warehouse audit reports `gdelt_v2_events: 51101` when run against the current shared GDELT tape.

This can be an optional manual verification, not a unit test relying on local data.

### Step 6: Add ACLED Direct-To-Warehouse

Files:

- `ingest/acled_raw.py`
- `ingest/acled_tape.py`
- `ingest/event_warehouse.py`
- tests in `tests/test_acled_ingest.py`

Changes:

- Add `--raw-retention`.
- Add `--normalize-to-warehouse`.
- Add `--data-root`.
- Add `--warehouse-path` optional override.
- When `--normalize-to-warehouse` is set, normalize rows page-by-page and upsert records into DuckDB.

Acceptance:

- Unit test monkeypatches ACLED fetch pages and confirms:
  - no raw fragments written for `--raw-retention none`
  - warehouse receives normalized ACLED rows
  - manifest has no token/password/request body
- Existing raw fragment mode still works.

### Step 7: Refactor Source Experiments To Use Warehouse

Files:

- `baselines/source_experiments.py`
- `baselines/backtest.py`
- tests in `tests/test_source_experiments.py`

Implementation:

- Add `--data-root`, `--warehouse-path`, `--snapshot-mode`, `--predictions-format`.
- If warehouse mode is active, query all needed records once per experiment:
  - Source filter from experiment spec.
  - Event date range should cover:
    - earliest needed feature history before `train_origin_start`
    - through `eval_origin_end + WINDOW_DAYS`
  - Safe implementation may query all records for selected source names first; optimize later.
- Compute scoring universe from selected records.
- Build snapshots in memory by default.
- Build feature rows using selected records.
- Compute target lookup from in-memory snapshot payloads.
- Write prediction JSONL gzip by default.

Acceptance:

- Existing tape-based tests still pass.
- New test runs source experiments from a temporary DuckDB warehouse.
- New test asserts no snapshot files are written in `snapshot_mode="in_memory"`.
- New test asserts snapshot files are written as `.json.gz` in `snapshot_mode="materialize"`.

### Step 8: Add In-Memory GNN Runner

Files:

- `baselines/backtest.py` or new `baselines/gnn_backtest.py`
- tests in `tests/test_gnn.py` and `tests/test_source_experiments.py`

Implementation:

- Add `run_gnn_backtest_from_payloads`.
- Reuse `build_graph_from_snapshot`.
- Preserve metrics shape.
- Support `gnn_ablation`.

Acceptance:

- Existing disk-snapshot GNN tests pass.
- New in-memory GNN test passes.
- Source experiment tests use in-memory snapshots by default.

### Step 9: Add Data GC

Files:

- `ingest/data_gc.py`
- tests in `tests/test_data_gc.py`

Acceptance:

- `report` computes sizes for temp tree.
- `prune --dry-run` deletes nothing.
- `prune --raw` deletes only raw directories.
- `prune --smoke-runs` deletes only smoke run directories.
- Never deletes warehouse.

### Step 10: Documentation

Files:

- `docs/storage_architecture.md`
- `docs/acled_ingestion.md`
- `docs/source_layer_experiments.md`

Acceptance:

- Docs show warehouse-first flow.
- Docs warn against storing large generated artifacts in repo worktrees.
- Docs clearly label JSONL mixed tapes as legacy/compatibility.

## Recommended New Full Workflow

After implementation, the preferred full workflow is:

```bash
cd /Users/darenpalmer/conductor/workspaces/psychohistory-v2/brisbane
source ~/.config/psychohistory/acled.env

python -m ingest.event_warehouse init

python -m ingest.event_warehouse import-tape \
  --input /Users/darenpalmer/conductor/shared-data/psychohistory-v2/gdelt/tape/france_protest/events.jsonl

python -m ingest.acled_raw fetch-france-protests \
  --event-start 2019-01-01 \
  --event-end 2026-01-04 \
  --limit 5000 \
  --max-pages 100 \
  --raw-retention none \
  --normalize-to-warehouse \
  --availability-policy event_date_lag \
  --availability-lag-days 7

python -m ingest.event_warehouse audit

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

## Edge Cases And Failure Modes

### Missing Warehouse

If a warehouse-backed command cannot find `events.duckdb`, fail with:

```text
missing event warehouse: <path>; run python -m ingest.event_warehouse init and import data first
```

### Missing Source

If an experiment requests `acled` but warehouse has only GDELT, fail with:

```text
source experiment 'acled_only' requested missing sources ['acled']; available=['gdelt_v2_events']; warehouse=<path>
```

### Empty Raw Retention

For `--raw-retention none`, manifests must still contain enough information to audit:

- run ID
- endpoint path
- page
- row count
- status
- retrieved timestamp
- query filters excluding credentials/token

### ACLED Availability

Keep default:

```text
event_date_lag, 7 days
```

Audit must include:

```json
{
  "availability_policy": "event_date_lag",
  "availability_lag_days": 7
}
```

### Existing JSONL Compatibility

Do not remove:

- `load_event_tape`
- `tape_merge`
- tape-based `source-experiments --tape`

Warehouse mode is preferred, not mandatory.

### No Snapshot Materialization

When `snapshot_mode="in_memory"`:

- Do not create `snapshots_root`.
- Combined audit should still include:
  - source counts
  - label counts
  - feature counts
  - model metrics

### Materialized Snapshots

When `snapshot_mode="materialize"`:

- Write snapshots only under the requested run directory.
- Default format: `.json.gz`.
- Never write source experiment snapshots under repo `data/` unless explicitly requested.

## Test Plan

Run default tests:

```bash
pytest
```

Run torch tests:

```bash
pytest -m torch_train tests/test_gnn.py tests/test_source_experiments.py
```

Targeted new tests:

```bash
pytest \
  tests/test_io_utils.py \
  tests/test_paths.py \
  tests/test_event_warehouse.py \
  tests/test_data_gc.py \
  tests/test_acled_ingest.py \
  tests/test_source_experiments.py
```

Manual smoke after implementation:

```bash
python -m ingest.event_warehouse init

python -m ingest.event_warehouse import-tape \
  --input /Users/darenpalmer/conductor/shared-data/psychohistory-v2/gdelt/tape/france_protest/events.jsonl

python -m ingest.event_warehouse audit
```

Expected if shared GDELT tape has been restored:

```json
"source_counts": {
  "gdelt_v2_events": 51101
}
```

Then ACLED direct-to-warehouse smoke:

```bash
source ~/.config/psychohistory/acled.env

python -m ingest.acled_raw fetch-france-protests \
  --event-start 2023-03-20 \
  --event-end 2023-04-03 \
  --limit 5000 \
  --max-pages 5 \
  --raw-retention none \
  --normalize-to-warehouse \
  --availability-policy event_date_lag \
  --availability-lag-days 7

python -m ingest.event_warehouse audit
```

Then source experiment smoke:

```bash
python -m baselines.backtest source-experiments \
  --snapshot-mode in-memory \
  --predictions-format jsonl.gz \
  --out-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2/runs/source_experiments/smoke \
  --train-origin-start 2023-03-27 \
  --train-origin-end 2023-03-27 \
  --eval-origin-start 2023-04-03 \
  --eval-origin-end 2023-04-03 \
  --experiments gdelt_only gdelt_plus_acled \
  --epochs 2 \
  --hidden-dim 16
```

## Acceptance Criteria

- Full GDELT tape can be imported into DuckDB without refetching.
- ACLED can be fetched and normalized directly into DuckDB with no raw fragment retention.
- Source experiments can run from warehouse without writing weekly snapshot files.
- Prediction files can be gzip-compressed.
- JSONL tape compatibility remains intact.
- Disk report/prune commands work and do not delete warehouse data.
- Default full workflow stores persistent data outside repo worktrees.
- No credentials, bearer tokens, raw API request bodies, or generated large artifacts are committed.
- Default tests and torch-marked tests pass.
