# Calgary PIT subgraph audit checkpoint — 2026-05-05

## Scope

Motion task: `[Workspace: Calgary] PIT subgraph audit stabilization + gate enforcement`.

This checkpoint covers:

1. In-flight code/test stabilization for the warehouse build + gate path.
2. Artifact policy: tracked source/summaries vs ignored generated artifacts.
3. Re-run of the local warehouse quality gate and a fixed-checkpoint PIT retrieval audit without rebuilding the warehouse.

## Code changes in this checkpoint

- `scripts/build_arab_spring_warehouse.py`
  - Defaults output to `artifacts/warehouse_validation/` instead of `shared_data/`.
  - Adds `--quality-gate` and `--quality-gate-strict` to wire warehouse contract checks into the normal build path.
  - Adds explicit `--allow-duckdb-fallback`; JSONL remains required by default for v1 builds.

- `baselines/node_warehouse_build_v0.py`
  - Fills missing `entity_hint_keys` deterministically from row-local context.
  - Records `fallback_filled_entity_hint_rows` in build output.
  - Uses true per-key event counts for v1 count features.
  - Makes v1 DuckDB fallback explicit instead of implicit.

- `scripts/warehouse_quality_gate.py`
  - Validates country/admin1 inference, `first_seen`, entity hints, and unmapped admin labels.

- `scripts/pit_subgraph_audit.py`
  - Emits PIT retrieval diagnostics with time, country, hint, precursor, lift, horizon, and grouped summaries.

- `.gitignore` + `docs/artifact_policy.md`
  - `artifacts/**`, raw PIT audit JSON, raw quality-gate JSON, and generated ablation JSON are ignored/regenerable.
  - Tracked artifacts should be source, tests, policy docs, and compact markdown decision/checkpoint notes.

## Verification commands run

```bash
pytest -q tests/test_node_warehouse_build_v0.py \
  tests/test_arab_spring_node_warehouse_v0.py \
  tests/test_arab_spring_node_warehouse_v1.py
```

Result: passed, 23 tests.

```bash
pytest -q
```

Result: passed, 323 tests. Only existing `torch_geometric` deprecation warnings were emitted.

```bash
python scripts/warehouse_quality_gate.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --output docs/warehouse_quality_gate_2026-05-05.current.json \
  --strict
```

Result: passed.

Quality summary:

- row_count: 29
- country_inference_complete: true
- entity_hint_keys_complete: true
- first_seen_complete: true
- no_unmapped_admin1_labels: true
- invalid_country_rows: 0
- missing_entity_hint_rows: 0
- missing_first_seen_rows: 0
- mixed_format_rows: 0
- unmapped_admin1_rows: 0

```bash
PYTHONPATH=. python scripts/pit_subgraph_audit.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --mmap artifacts/warehouse_validation/node_warehouse_v0.mmap \
  --checkpoint /Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt \
  --output docs/pit_subgraph_audit_2026-05-05.stage1_v1_fixed.json \
  --k 10
```

Result: audit report generated.

Audit summary for `stage1_v1_fixed/query_encoder_epoch_049.pt` on the local v0 validation warehouse:

- probe_count: 243
- top1_time_ok_rate: 1.0
- topk_time_ok_ratio_mean: 1.0
- top1_country_ok_rate: 0.4074074074
- topk_country_ok_ratio_mean: 0.3160493827
- topk_hint_hit_ratio_mean: 0.0
- precursor_hit_rate: 0.0
- future_rank_lift_mean: 0.0
- horizon_consistency_rate: 1.0

By geography:

- Egypt: top1_country_ok_rate 0.4321; topk_country_ok_ratio_mean 0.4370
- Libya: top1_country_ok_rate 0.2963; topk_country_ok_ratio_mean 0.1333
- Tunisia: top1_country_ok_rate 0.4938; topk_country_ok_ratio_mean 0.3778

## Interpretation

The gate-enforcement path is green for the small local validation warehouse: temporal/horizon consistency and quality contract checks pass cleanly.

The retrieval audit is still not a promotion-quality result:

- Geography is non-collapsed but weak, especially Libya.
- Hint and precursor diagnostics are zero, so the current checkpoint is useful as a plumbing/audit baseline, not a scientific promotion.
- Next work should prioritize geo-normalization regression coverage and gate integration in the default warehouse/run flow before any long training run.

## Raw generated artifacts

Raw JSON outputs are intentionally ignored by git and can be regenerated:

- `docs/warehouse_quality_gate_2026-05-05.current.json`
- `docs/pit_subgraph_audit_2026-05-05.stage1_v1_fixed.json`
- `artifacts/warehouse_validation/node_warehouse_v0_manifest.json`
- `artifacts/warehouse_validation/node_warehouse_v0.mmap`
