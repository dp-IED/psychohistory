# Warehouse v1 Recipe Fix: Implementation Checklist

**Status:** Pre-implementation review  
**Estimate:** 7 hours (3 hours code + 4 hours compute/validation)  
**Start:** After all review items are signed off

---

## Pre-Implementation Review

- [x] Diagnostic query confirms ~74k nodes at ≥3 events threshold
- [x] GDELT actor strings identified (protester, military, police, etc.)
- [x] MIN_EVENTS_PER_NODE constant specified with rationale
- [x] GDELT vs ACLED alignment clarified (warehouse uses GDELT)
- [x] Probe corpus revision strategy documented

**Sign-off:** Review items above before proceeding.

---

## Phase 1: Code Changes (3 hours)

### 1.1 Update `baselines/graph_builder_warehouse.py`

**File location:** `baselines/graph_builder_warehouse.py`  
**Function:** Locate and update node key aggregation

**Changes:**
- [ ] Add `MIN_EVENTS_PER_NODE = 3` at module level with docstring
- [ ] Change aggregation key from `(event_code, country, month)` to `(actor1_name, admin1_code, month)`
- [ ] Update `_warehouse_key()` function signature and implementation
- [ ] Add filtering logic: keep only nodes with `len(events) >= MIN_EVENTS_PER_NODE`
- [ ] Update entity_hint_keys derivation to use `actor1_name` values (lowercased)
- [ ] Add validation: confirm `normalize_hint()` is applied consistently

**Testing:**
- [ ] Run unit tests for warehouse module
- [ ] Verify aggregation produces expected key format: `ar_v1|{actor}|{admin1}|{month}`

---

### 1.2 Rewrite `baselines/arab_spring_probes.py` (base seeds only)

**File location:** `baselines/arab_spring_probes.py`  
**Section:** `ARAB_SPRING_BASE_PROBE_DEFS` (20 probes)

**Changes:**
- [ ] Review each of 20 base seed probes
- [ ] Replace entity hints with GDELT actor strings:
  - UGTT → "protester" (or "government" if suppression gate)
  - SCAF → "military" or "government"
  - NTC → "government" or "rebel"
  - Ennahda → "government" (opposition)
  - February 17 Brigade → "rebel"
  - Ben Ali → "government"
  - Gaddafi → "government" or "military"
  - ElBaradei → remove or use generic role
- [ ] Update NL text to reflect GDELT actor names (not named entities)
- [ ] Keep templated probes (243 rows) unchanged — they already use generic hints

**Testing:**
- [ ] `validate_probe_hints_against_manifest()` called on full corpus
- [ ] Confirm all 20 base seeds have resolvable hints (post-warehouse-rebuild)

---

### 1.3 Run lint and tests

**Commands:**
- [ ] `pytest -q` — full regression suite
- [ ] Check linter for new code

---

## Phase 2: Warehouse Rebuild (2 hours compute)

**Prerequisites:** Phase 1 code changes merged and tested

**Commands:**
```bash
cd /Users/darenpalmer/conductor/workspaces/psychohistory-v2/calgary
export PYTHONPATH=.

# Rebuild warehouse with new recipe
python -m baselines.graph_builder_warehouse \
  --input shared_data/arab_spring/events.jsonl \
  --output-mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --output-manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json
```

**Expected outputs:**
- [ ] `node_warehouse_v1_fixed.mmap` (~74k rows × 128 dims, smaller than current)
- [ ] `node_warehouse_v1_fixed_manifest.json` (updated with actor-keyed nodes)
- [ ] Manifest `row_count` field: ~74,325

**Validation:**
- [ ] Spot-check manifest: sample nodes have format `ar_v1|{actor}|{admin1}|{month}`
- [ ] entity_hint_keys match GDELT actor strings (lowercase)
- [ ] No nodes with <3 events in manifest

---

## Phase 3: Positive Pairs Regeneration (1 hour compute)

**Prerequisites:** Phase 2 warehouse rebuild complete

**Command:**
```bash
python -m baselines.graph_builder_positive_pairs \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --recipe admin1_lead_lag_v0 \
  --output-dir shared_data/arab_spring/positive_pairs_v1_fixed/
```

**Expected outputs:**
- [ ] `positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.npy`
- [ ] `positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.meta.json`
- [ ] Pair count will be smaller than before (fewer nodes → fewer pairs)

**Validation:**
- [ ] Meta file contains `"positive_pair_version"` tag
- [ ] Pair count is non-zero

---

## Phase 4: Stage 1 Retraining (1 hour compute)

**Prerequisites:** Phase 3 positive pairs ready

**Command:**
```bash
python -m baselines.graph_builder_stage1_train \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --pairs-metadata shared_data/arab_spring/positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.meta.json \
  --output-dir shared_data/arab_spring/stage1_v1_fixed_out/ \
  --epochs 10
```

**Expected behavior:**
- [ ] Training runs without errors
- [ ] Loss decreases over epochs
- [ ] Final loss in range 0.5–2.0 (NOT 0.003 like before)
- [ ] All 243 probes contribute

**Checkpoints:**
- [ ] `stage1_v1_fixed_out/query_encoder_epoch_009.pt` (final)
- [ ] `stage1_v1_fixed_out/train_state.json` (loss log)

---

## Phase 5: Validation & Documentation (1 hour)

**Sanity checks:**
- [ ] Run probe validation against fixed warehouse:
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
- [ ] Compare old vs new warehouse node counts
- [ ] Inspect sample nodes from fixed warehouse

**Documentation:**
- [ ] Update `project_state.md` with new warehouse status
- [ ] Add entry to `next_steps.md` linking to this checklist
- [ ] Update any cached numbers (168k → 74k) in other docs

---

## Rollback Plan

If any phase fails:

1. **Phase 1 code regression:** Revert commits, tests should catch issues
2. **Phase 2 warehouse build fails:** Warehouse remains unchanged; re-run after fixing code
3. **Phase 4 training doesn't improve:** Inspect loss trajectory; may need MIN_EVENTS_PER_NODE tuning
4. **Validation fails:** Check probe hints against manifest manually; update probe corpus

**Keep old warehouse:** Don't delete v1 mmap/manifest until Phase 5 validation passes.

---

## Success Criteria

- [x] Code changes reviewed and approved
- [x] All tests pass
- [ ] New warehouse builds successfully (~74k nodes)
- [ ] Positive pairs regenerated with new warehouse
- [ ] Stage 1 retrains with loss in 0.5–2.0 range
- [ ] All probe hints resolve (validation passes)
- [ ] Documentation updated

**Definition of done:** All items checked, new warehouse ready for Stage 2 training.
