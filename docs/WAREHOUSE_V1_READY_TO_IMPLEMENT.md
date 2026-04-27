# Warehouse v1 Recipe Fix: Complete Analysis & Ready-to-Implement

**Date:** 2026-04-27  
**Status:** All diagnostics complete, implementation plan locked  
**Next step:** Code implementation (3 hours) + pipeline retraining (4 hours compute)

---

## Problem Statement (Resolved)

The v1 warehouse was built with the wrong aggregation key:

| Aspect | Broken (current) | Fixed (target) |
|--------|---|---|
| Aggregation key | `(event_code, country, month)` | `(actor1_name, admin1_code, month)` |
| Result | 168,195 event-type nodes | ~74,325 actor-keyed nodes |
| Hint source | GDELT event codes | GDELT actor1_name (lowercased) |
| Probe corpus | Named entities (UGTT, SCAF, NTC) — 8/9 not in warehouse | GDELT actor types (protester, military, police) — all exist |
| Training loss | ~0.003 (degenerate) | Expected 0.5–2.0 (informative) |

---

## Diagnostics Completed

### Query 1: Actor×Admin1×Month Distribution

**Finding:** Actor-keyed aggregation produces 74,325 nodes (≥3 events threshold).

**Top actors:**
- Protester: 36,658 events
- Rebel: 37,347 events
- Military: 45,297 events
- Government: 48,265 events
- Police: 35,823 events

### Query 2: Probe Corpus Named Entities

**Finding:** 8 of 9 base seed entities don't exist in GDELT/ACLED.

| Entity | Status |
|--------|--------|
| UGTT | ✗ Not in data |
| SCAF | ✗ Not in data |
| NTC | ✗ Not in data |
| Ennahda | ✗ Not in data |
| Feb 17 Brigade | ✗ Not in data |
| Ben Ali | ✗ Not in data |
| Gaddafi | ✗ Not in data |
| Muslim Brotherhood | ✓ 41 events |
| ElBaradei | ✗ Not in data |

**Resolution:** Rewrite base seeds to reference GDELT actor types that exist (protester, military, police, government, rebel).

### Query 3: GDELT Actor Strings

**Finding:** Top GDELT actor patterns (after lowercase normalization):

```
27,307: protester (EG)
26,749: rebel (LY)
26,611: military (EG)
19,154: government (EG)
18,839: police (EG)
15,111: government (TU)
13,610: government (LY)
12,457: police (TU)
 9,610: military (LY)
 9,076: military (TU)
```

**Resolution:** Probe hints must match these GDELT strings exactly (lowercased).

---

## Critical Alignment Decision

**GDELT vs ACLED:**

The warehouse is built from GDELT events. ACLED actor names ("Protesters (Egypt)", "Military Forces of Libya") are different strings. Probe hints must reference GDELT strings, not ACLED.

- **ACLED role:** Discovery tool; tells you which actor types matter
- **GDELT role:** Warehouse contents; defines what strings the hints must match
- **Probe role:** Must reference GDELT strings for hints to resolve

---

## Implementation Plan (Locked)

### Phase 1: Code Changes (3 hours)

**1.1 Update `baselines/graph_builder_warehouse.py`**
- Add `MIN_EVENTS_PER_NODE = 3` named constant with docstring
- Change aggregation: `(event_code, country, month)` → `(actor1_name, admin1_code, month)`
- Add filtering: `len(events) >= MIN_EVENTS_PER_NODE`
- Update entity_hint_keys to use lowercased `actor1_name` values

**1.2 Rewrite `baselines/arab_spring_probes.py` base seeds**
- Replace 20 base seed entity hints with GDELT actor strings
- UGTT → "protester"
- SCAF → "military" or "government"
- NTC → "government" or "rebel"
- Ennahda → "government"
- Feb 17 Brigade → "rebel"
- Ben Ali → "government"
- Gaddafi → "military" or "government"
- ElBaradei → remove or use generic
- Update NL text to match GDELT actor names

**1.3 Test**
- Run `pytest -q`
- Verify aggregation key change doesn't break existing tests

### Phase 2: Rebuild Pipeline (4 hours compute)

**2.1 Warehouse rebuild**
```bash
python -m baselines.graph_builder_warehouse \
  --input shared_data/arab_spring/events.jsonl \
  --output-mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --output-manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json
```

Expected: ~74k nodes, actors as hints

**2.2 Positive pairs regeneration**
```bash
python -m baselines.graph_builder_positive_pairs \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --recipe admin1_lead_lag_v0 \
  --output-dir shared_data/arab_spring/positive_pairs_v1_fixed/
```

**2.3 Stage 1 retraining**
```bash
python -m baselines.graph_builder_stage1_train \
  --manifest shared_data/arab_spring/node_warehouse_v1_fixed_manifest.json \
  --mmap shared_data/arab_spring/node_warehouse_v1_fixed.mmap \
  --pairs-metadata shared_data/arab_spring/positive_pairs_v1_fixed/positive_pairs.admin1_14day_v0.meta.json \
  --output-dir shared_data/arab_spring/stage1_v1_fixed_out/ \
  --epochs 10
```

Expected: Loss converges to 0.5–2.0 range (not 0.003)

### Phase 3: Validation (1 hour)

- Probe hint validation passes
- Compare old vs new warehouse structure
- Spot-check nodes and embeddings
- Update documentation

---

## Documentation Created

All pre-implementation docs are in `docs/`:

1. **`warehouse-recipe-v1-bug-report.md`** — Root cause analysis
2. **`warehouse-recipe-v1-fix-plan.md`** — Implementation approach
3. **`warehouse-v1-diagnostic-query-results.md`** — Query results and implications
4. **`warehouse-v1-recipe-amendment.md`** — Locked decision correction (168k → 74k)
5. **`warehouse-gdelt-actor-alignment.md`** — GDELT vs ACLED clarification
6. **`warehouse-v1-implementation-checklist.md`** — Phase-by-phase plan with commands

---

## Decision Points (Resolved)

- [x] **Node count:** 74,325 (not 168k) at ≥3 events threshold ✓
- [x] **Threshold constant:** `MIN_EVENTS_PER_NODE = 3` (named, documented) ✓
- [x] **GDELT alignment:** Warehouse uses GDELT; probes must too ✓
- [x] **Implementation sequence:** Code → rebuild → retrain → validate ✓

---

## Risk Mitigation

- **Keep v1 warehouse:** Don't delete old mmap/manifest until validation passes
- **Incremental rebuild:** Each phase (code → warehouse → pairs → training) can be validated independently
- **Rollback:** Revert code changes if phase 1 tests fail; warehouse only affected by phase 2+
- **MIN_EVENTS tuning:** If loss doesn't improve, threshold can be adjusted (see amendment doc)

---

## Success Criteria

✓ All diagnostics pass  
✓ GDELT alignment clarified  
✓ Implementation phases defined  
✓ Commands documented  
✓ Validation criteria specified  

**Ready to implement.** Review checklist and begin Phase 1 code changes.
