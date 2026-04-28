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

---

## Post-Implementation Findings (2026-04-27)

### Implementation Execution

**Phase 1–4 completed successfully:**
- ✓ Warehouse rebuilt: 98,798 nodes (actor×admin1×month keying, ≥3 events threshold)
- ✓ Memory efficiency: Peak RAM 4.4GB (from 85GB+) via JSONL streaming + single-pass aggregation
- ✓ Positive pairs: 101.5M pairs generated (admin1_lead_lag_v0 recipe)
- ✓ Training: 50 epochs converged, loss: 1.84 → 0.32 (informative improvement)

### Critical Issue: Retrieval Quality is Broken

**Query inspection (epoch 50 checkpoint) reveals fundamental geographic/temporal confusion:**

| Query | Geography | Time Period | Top Result | Issue |
|-------|-----------|-------------|-----------|-------|
| Tunisia Dec 2010 | Tunisia | 2010 precursor | **Syria Aug 2013** | Wrong country, 3+ years later |
| Egypt Jan 2011 | Egypt | 2011 propagation | **Syria Aug 2013** | Wrong country, 2+ years later |
| Libya Feb 2011 | Libya | 2011 suppression | **Syria Nov 2012** | Wrong country, 1+ year later |

**Scores are normalized** (0.46–0.48 cosine similarity, correct range), but the model learned to **ignore geography and temporal alignment** entirely.

### Root Cause: Flawed SSL Positive Pairs Recipe

The `admin1_lead_lag_v0` recipe pairs nodes by:
1. **Exact `admin1_code` match** (any governorate or country code)
2. **Temporal separation 32–90 days**

**Problem in Arab Spring data:**

- **Warehouse mixes geographic granularities:** Egypt has both country-level `EG` nodes AND governorate-level nodes (Cairo, Alexandria, Giza, etc.)
- **No country-level pairing:** Cairo nodes pair with Cairo; EG nodes pair with EG—they never cross-pair
- **Temporal leakage:** Syrian nodes (SY) from 2012–2013 overlap temporally with many Egyptian governorate nodes, causing false positive pairs
- **Deceptive loss:** Model learns to match the SSL pairs well (loss drops to 0.32), but those pairs teach it to confuse Syria with Egypt

**Example false pair:** 
- `ar_v1|protester|Cairo|2011-02` (first_seen: 2011-02-02) pairs with `ar_v1|protester|Cairo|2011-05` (first_seen: 2011-05-05) ✓ Same region, 32–90 day gap
- But model also sees: `ar_v1|government|SY|2012-05` (first_seen: 2012-05-08) in the corpus, and nothing prevents it from learning Cairo ≈ Syria in the embedding space during optimization

### Implication

**Loss improvement is necessary but not sufficient.** The encoder learned to encode queries consistently, but into a latent space that does not preserve geographic or temporal discrimination. This is a **data quality / recipe choice failure**, not a training failure.

### Next Steps Required

**Before declaring the warehouse "fixed":**

1. **Redesign the SSL positive pairs recipe** — Options:
   - `admin1_lead_lag_strict_v1`: Only pair nodes from the same exact admin1 AND exclude cross-country pairs entirely
   - `actor_type_consistency_v1`: Pair nodes that have similar actor type distributions (requires aggregation)
   - `supervised_pairs_v1`: Use ground-truth event co-occurrence as pairs (loses SSL unsupervised advantage)

2. **Retrain Stage 1** with corrected pairs and validate retrieval again

3. **Document the pattern:** Single-pass warehouse aggregation now works; next issue is SSL recipe design for heterogeneous geographic data
