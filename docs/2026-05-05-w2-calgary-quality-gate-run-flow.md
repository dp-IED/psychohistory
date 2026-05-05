# W2 Calgary 4/4 checkpoint — quality gate enforced in run flow

Motion task: `[W2 Calgary 4/4] Enforce quality gate in run flow`.

## Code changes

- `scripts/build_arab_spring_warehouse.py`
  - Refactored CLI into `parse_args`, `run_build`, and `main` for testable run-flow semantics.
  - Made the warehouse quality gate run by default.
  - Added explicit `--no-quality-gate` escape hatch for local debugging.
  - Kept `--quality-gate-strict` as the promotion/CI hard-fail mode.
  - Strict failures now preserve and print the quality-gate payload once before exiting with code 2.

- `tests/test_build_arab_spring_warehouse_cli.py`
  - Covers default quality-gate embedding.
  - Covers strict quality-gate failure returning exit code 2.
  - Covers explicit skip via `quality_gate=False` / `--no-quality-gate` semantics.

## Verification

Commands run:

```bash
pytest -q tests/test_build_arab_spring_warehouse_cli.py tests/test_pit_geo_quality_gates.py
pytest -q
python -m py_compile scripts/build_arab_spring_warehouse.py
python scripts/warehouse_quality_gate.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --strict \
  --output docs/warehouse_quality_gate_2026-05-05.enforced_flow.json
```

Results:

- focused tests: PASS, 20 tests.
- full suite: PASS, 343 tests.
- py_compile: PASS.
- strict warehouse quality gate: PASS.

## Artifact policy

Raw gate report generated for local verification:

- `docs/warehouse_quality_gate_2026-05-05.enforced_flow.json`

It is intentionally ignored by `.gitignore`; this markdown note is the tracked record.

## Decision

Quality gate is now in the normal build run flow. Any default warehouse build returns a `quality_gate` payload, and strict mode is available for CI/promotion blocking.
