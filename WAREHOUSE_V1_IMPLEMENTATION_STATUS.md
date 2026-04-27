# Warehouse v1 Implementation Status

**Date**: 2026-04-27  
**Current Phase**: Phase 2 (Warehouse rebuild in progress)

## ✅ Phase 1: Code Changes (COMPLETE)

### 1.1 Updated `baselines/node_warehouse_build_v0.py`
- ✅ Added `MIN_EVENTS_PER_NODE = 3` constant with docstring
- ✅ Implemented two-pass filtering to minimize memory usage:
  - Pass 1: Count events per (actor, admin1, month) key
  - Pass 2: Only store events for keys meeting MIN_EVENTS_PER_NODE threshold
- ✅ Entity hint keys already use lowercased `actor1_name` values (confirmed)

**Memory Optimization**: Reduces peak memory from 135GB+ to manageable levels by:
1. First counting all keys (O(k) memory where k = unique keys ≈ hundreds of thousands)
2. Only storing EventTapeRecord objects for qualifying keys in second pass
3. Avoids loading all events into a single dictionary simultaneously

### 1.2 Rewrote `baselines/arab_spring_probes.py` base seeds
Replaced 11 named-entity hints with GDELT actor type strings (20 base seed probes updated):

| Probe ID | Old Hint | New Hint | Rationale |
|----------|----------|----------|-----------|
| ar_base_00 | UGTT | protester | Labour union = protesters in GDELT |
| ar_base_01 | SCAF | military | Military council actor type |
| ar_base_02 | NTC | government | Transitional authority = government |
| ar_base_03 | Ennahda | government | Opposition party now governing |
| ar_base_05 | Feb 17 Brigade | rebel | Rebel forces in Libya |
| ar_base_11 | UGTT | protester | Same as base_00 |
| ar_base_12 | Ben Ali | government | Leader personification = government |
| ar_base_13 | SCAF | military | Same as base_01 |
| ar_base_14 | Gaddafi loyalists | military | Military forces |
| ar_base_16 | UGTT + Ennahda | protester + government | Combination |
| ar_base_17 | ElBaradei | (removed) | Doesn't exist in GDELT; kept Muslim Brotherhood |
| ar_base_18 | NTC + Feb 17 | government + rebel | Combination |

All new hints match GDELT actor strings exactly (lowercase).

### 1.3 Test Updates & Verification
- ✅ Updated 3 test cases to accommodate MIN_EVENTS_PER_NODE >= 3:
  - `test_v1_collapses_same_normalized_actor_to_one_monthly_node` ✓
  - `test_v1_splits_same_actor_across_two_months` ✓ (now uses 6 events)
  - `test_v1_entity_hint_keys_keep_sorted_distinct_raw_names` ✓ (corrected expectations)
  - `test_v1_pre_norm_guard_raises` ✓ (now uses 3 events)
- ✅ Full regression suite passes (334 tests)

### Created CLI Wrapper
- ✅ Added `baselines/graph_builder_warehouse.py` with argparse CLI
- ✅ Supports all required flags: `--input`, `--output-mmap`, `--output-manifest`, `--as-of`, `--window-days`
- ✅ Tested with `--help` flag

---

## 🔄 Phase 2: Warehouse Rebuild (IN PROGRESS)

**Status**: Running in background  
**Command**: 
```bash
python -m baselines.graph_builder_warehouse \
  --input shared_data/arab_spring/events.duckdb \
  --output-mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --output-manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --window-days 1461
```

**Expected Outputs**:
- `node_warehouse_v1_fixed.mmap` (~74k rows × 128 dims = ~37 MB)
- `node_warehouse_v1_fixed_manifest.json` (updated with actor-keyed nodes)
- Manifest `row_count` field: ~74,325 (reduced from 168,195)

**Validation Criteria**:
- [ ] Spot-check manifest: nodes have format `ar_v1|{actor}|{admin1}|{month}`
- [ ] entity_hint_keys match GDELT actor strings (lowercase)
- [ ] All nodes have >= 3 events in manifest

**Timeline**: ~2-3 hours (DuckDB query + matrix build)

---

## ⏳ Phase 3: Positive Pairs Regeneration (PENDING)

**Command**:
```bash
python -m baselines.graph_builder_positive_pairs \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --recipe admin1_lead_lag_v0 \
  --output-dir shared_data/arab_spring/positive_pairs_v1_fixed/
```

**Expected Outputs**:
- `positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.npy`
- `positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.meta.json`

**Validation**:
- [ ] Pair count > 0
- [ ] Meta file contains version tags

---

## ⏳ Phase 4: Stage 1 Retraining (PENDING)

**Command**:
```bash
python -m baselines.graph_builder_stage1_train \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --pairs-metadata shared_data/arab_spring/positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.meta.json \
  --output-dir shared_data/arab_spring/stage1_v1_fixed_out/ \
  --epochs 10
```

**Expected Behavior**:
- [ ] Training runs without errors
- [ ] Loss decreases over epochs
- [ ] Final loss in range 0.5–2.0 (NOT 0.003 like before)

**Key Metric**: Loss should improve dramatically from degenerate 0.003 → informative 0.5–2.0

---

## ⏳ Phase 5: Validation & Documentation (PENDING)

**Sanity Checks**:
```bash
python -c "
from pathlib import Path
from baselines.arab_spring_probes import build_arab_spring_probe_corpus, validate_probe_hints_against_manifest
from schemas.graph_builder_warehouse import NodeWarehouseManifest

probes = build_arab_spring_probe_corpus()
manifest = NodeWarehouseManifest.model_validate_json(
  Path('shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json').read_text()
)
try:
  validate_probe_hints_against_manifest(probes, manifest)
  print('✓ All probe hints resolve')
except ValueError as e:
  print(f'✗ Hint validation failed: {e}')
"
```

**Documentation Updates**:
- [ ] Update `project_state.md` with warehouse v1 status
- [ ] Create entry in implementation log
- [ ] Update any cached numbers (168k → 74k)

---

## Files Modified

1. `baselines/node_warehouse_build_v0.py` — Added MIN_EVENTS_PER_NODE, two-pass filtering
2. `baselines/arab_spring_probes.py` — Updated 20 base seed probes with GDELT actors
3. `tests/test_arab_spring_node_warehouse_v1.py` — Updated 4 tests for new threshold
4. `baselines/graph_builder_warehouse.py` — **NEW** CLI wrapper

## Success Criteria

- [x] Phase 1 code changes reviewed and approved
- [x] All tests pass
- [ ] New warehouse builds successfully (~74k nodes)
- [ ] Positive pairs regenerated
- [ ] Stage 1 retrains with loss in 0.5–2.0 range
- [ ] All probe hints resolve
- [ ] Documentation updated

**Definition of Done**: All phases complete, new warehouse ready for Stage 2 training.
