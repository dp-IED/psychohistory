# W2 Calgary 1/4 checkpoint — geo normalization regression tests

Motion task: `[W2 Calgary 1/4] Add geo normalization regression tests`.

## Added

- `tests/test_pit_geo_quality_gates.py`

Coverage:

1. Warehouse quality-gate country inference source classification:
   - native ISO2 country code (`EG`)
   - prefixed admin code (`EG-C`)
   - known admin label aliases (`Cairo`, `North Sinai`)
   - unmapped labels (`Tripolitania`)
   - missing/blank admin labels

2. PIT audit country correctness contract:
   - alias-mapped admin labels count as correct only for the expected country.
   - unknown expected country returns `None` instead of a false failure.

3. Manifest-level quality gate behavior:
   - passes with alias-mapped labels when `first_seen` and `entity_hint_keys` are complete.
   - fails closed when any of country inference, unmapped labels, `first_seen`, or entity hints are incomplete.

## Verification

```bash
pytest -q tests/test_pit_geo_quality_gates.py
```

Result: 17 passed.

```bash
pytest -q
```

Result: 340 passed. Only existing `torch_geometric` deprecation warnings were emitted.

## Notes

These tests lock the post-hoc geo-normalization contract used by both:

- `scripts/warehouse_quality_gate.py`
- `scripts/pit_subgraph_audit.py`

They directly protect the previous failure mode where mixed `admin1_code` semantics made country metrics look like model collapse rather than a data/eval contract issue.
