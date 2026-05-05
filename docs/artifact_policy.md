# Artifact policy — PIT subgraph audit

Purpose: keep the Calgary PIT audit reproducible without committing regenerable warehouse matrices or shared-data outputs.

## Tracked

- Source code and tests that define the artifact contracts:
  - `baselines/node_warehouse_build_v0.py`
  - `scripts/build_arab_spring_warehouse.py`
  - `scripts/warehouse_quality_gate.py`
  - `scripts/pit_subgraph_audit.py`
  - regression tests under `tests/`
- Small, human-reviewable run reports under `docs/`, especially:
  - concise markdown checkpoint notes with command lines, metric summaries, and raw artifact paths.
- Compact ablation plans/decision notes under `ablations/` when needed for decision provenance.

## Ignored / regenerated

- Generated warehouse validation matrices and manifests under `artifacts/`.
- Raw PIT audit JSON reports matching `docs/pit_subgraph_audit_*.json`.
- Raw warehouse quality-gate JSON reports matching `docs/warehouse_quality_gate_*.json`.
- Generated ablation JSON summaries matching `ablations/*.json`.
- Large shared-data outputs, checkpoints, DuckDB files, JSONL event tapes, caches, and logs.
- Any local smoke artifacts that can be reproduced from tracked scripts plus shared-data inputs.

## Reproduction boundary

Use tracked scripts to regenerate local artifacts instead of committing them:

```bash
PYTHONPATH=. python scripts/build_arab_spring_warehouse.py \
  --recipe v0 \
  --quality-gate \
  --quality-gate-strict

PYTHONPATH=. python scripts/warehouse_quality_gate.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --output docs/warehouse_quality_gate_YYYY-MM-DD.current.json \
  --strict

PYTHONPATH=. python scripts/pit_subgraph_audit.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --mmap artifacts/warehouse_validation/node_warehouse_v0.mmap \
  --checkpoint <checkpoint.pt> \
  --output docs/pit_subgraph_audit_YYYY-MM-DD.<checkpoint_label>.json \
  --k 10
```

## Current policy decision

`artifacts/` is ignored. If a generated manifest becomes a canonical benchmark fixture, copy a minimized report/summary into `docs/` rather than tracking the generated artifact directory directly.
