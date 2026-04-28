# Diagnostic Query Results: Actor×Admin1×Month Distribution

**Date**: 2026-04-27  
**Query Target**: GDELT v1 and ACLED v3 events for EG/TU/LY (2010–2013)  
**Purpose**: Determine viable actor×admin1×month nodes for warehouse v1 rebuild

---

## Key Findings

### 1. Node Count Distribution

With aggregation by `(actor1_name, admin1_code, month)`:

| Threshold | Node Count |
|-----------|-----------|
| ≥1 events | 128,080 |
| ≥2 events | 94,682 |
| ≥3 events | 74,325 |
| ≥5 events | 54,928 |
| ≥10 events | 35,402 |

**Interpretation**: The design doc specified 168,340 nodes for v1. The current approach yields 74–128k nodes depending on filtering. The recipe likely filters by minimum event count (≥3 or ≥5) to eliminate noise.

**Decision**: Use ≥3 events threshold → 74,325 nodes. This is significantly lower than 168k but still viable for meaningful warehouse structure.

### 2. Top Actors by Event Frequency (≥3 events per admin1×month)

The warehouse will be dominated by country-level and generic actors:

```
27,783 events: LIBYA × LY admin1 × 2011-03
27,189 events: EGYPT × EG admin1 × 2013-07
24,974 events: EGYPT × EG admin1 × 2011-02
...
```

Many top entries are country names (`LIBYA`, `EGYPT`, `TURKEY`, `LIBYAN`, `EGYPTIAN`) aggregated by GDELT, not specific named actors. This is a **data quality issue in GDELT itself** — many events are tagged with country-level actors rather than specific entities.

### 3. Probe Corpus Named Entities: NOT IN GDELT/ACLED

**Critical Finding**:

| Probe Entity | GDELT | ACLED | Status |
|---|---|---|---|
| UGTT | ✗ | ✗ | Not in event data |
| Supreme Council of the Armed Forces | ✗ | ✗ | Not in event data |
| National Transitional Council | ✗ | ✗ | Not in event data |
| Ennahda | ✗ | ✗ | Not in event data |
| February 17 Martyrs Brigade | ✗ | ✗ | Not in event data |
| Ben Ali | ✗ | ✗ | Not in event data |
| Gaddafi loyalists | ✗ | ✗ | Not in event data |
| Muslim Brotherhood | ✓ 21 events (GDELT) | ✓ 41 events (ACLED) | **Only one matches** |
| ElBaradei | ✗ | ✗ | Not in event data |

**Impact on probe corpus**:
- 8 of 9 named entities do not appear in warehouse data at all
- Fixing the warehouse recipe does NOT fix the probe mismatch
- These probes will still resolve to UNK even with actor-keyed warehouse

### 4. What ACLED Actually Contains

ACLED has much better structured actor data than GDELT. Top actors are:

```
Protesters (Egypt)           2,323 events
Rioters (Egypt)              1,111 events
Military Forces of Libya     281–367 events
Libyan Rebel Forces          52 events
Police Forces of Egypt       49–129 events
Militia (Pro-Government)     43 events
Muslim Brotherhood           21 events
```

These are ACLED-standardized actor categories, not raw named entities. They correspond to the **actor-type hints** in the probe corpus (`police`, `military`, `civilians`), not the named entity hints.

---

## Warehouse Recipe Fix: What Will Actually Happen

### Current (Broken v1)

Aggregates by `(event_code, country, month)` → 168k event-type nodes

### Fixed v1 (Per Design Doc)

Aggregates by `(actor1_name, admin1_code, month)` → **74k nodes** (not 168k as designed)

**What this achieves**:
- ✓ Warehouse has actor-keyed structure
- ✓ Muslim Brotherhood probe will resolve
- ✗ UGTT, SCAF, NTC, Ennahda, etc. will STILL resolve to UNK (not in data)

### The Real Problem

The probe corpus was written assuming:
1. The warehouse exists
2. The warehouse contains actor-level nodes
3. Named entities from the probe corpus appear in the warehouse

**Only (1) becomes true after the fix. (2) is true but with noise. (3) is FALSE.**

The named entities in base seed probes either:
- **Never appeared in the raw event data** (UGTT, SCAF, NTC, Ennahda, etc.)
- **Appear but GDELT/ACLED standardized them differently** (unlikely given the data)
- **Should have been sourced from domain expertise, not blind assumption**

---

## Decision Point for Implementation

### Scenario A: Fix warehouse and live with probe mismatch

**What to do**:
1. Implement actor×admin1×month aggregation in `build_arab_spring_node_matrix_v1()`
2. Result: 74k nodes, properly keyed on actors
3. Rerun probe validation → 8 of 20 base seeds still fail

**Then you must**:
1. Rewrite base seed probes to reference only actors that exist in warehouse
2. Use `Protesters (Egypt)`, `Military Forces of Libya`, etc. instead of UGTT/SCAF/NTC
3. This makes base seeds indistinguishable from templated corpus

**Net result**: Fixes architecture, but probes become generic actor-type training (not named-actor training). Stage 1 works, but not as designed.

### Scenario B: Warehouse fix + probe revision

**Before fixing warehouse**:
1. Run domain expert review to identify what named actors SHOULD be in the corpus
2. Cross-reference against GDELT/ACLED raw data to see if they appear under different names
3. If they don't appear at all, add them as synthetic annotations or remove them from probes

**Then**:
1. Fix warehouse with actor×admin1×month aggregation
2. Rewrite base seed probes to match available actor names from GDELT/ACLED
3. Result: Warehouse and probes are aligned

**Net result**: Stage 1 trains on real actors from event data (not synthetic). Probe corpus is grounded in observable data.

---

## Recommendation

**Do Scenario A first** (9 hours of work):

1. Fix the warehouse recipe (4 hours implementation + test)
2. Rewrite base seed probes to use ACLED actor names (2 hours)
3. Validate and retrain (3 hours)

This unblocks Stage 1 training with a known, documented compromise: the probe corpus trains on ACLED/GDELT actor types (which exist) rather than named entities (which don't).

**Document the trade-off** in `docs/warehouse-recipe-v1-fix-plan.md`:

> Named entities in original base seed probes (UGTT, SCAF, NTC, Ennahda, etc.) do not appear in GDELT/ACLED event data. The warehouse fix enables actor-keyed aggregation, but Stage 1 training uses ACLED-standardized actor types (e.g., "Protesters (Egypt)", "Military Forces of Libya") instead. This aligns training with observable event patterns rather than domain expertise assumptions.

Later (if needed):

**Scenario C**: Add synthetic training data or external actor embeddings if you want to inject named-entity structure that isn't in GDELT/ACLED. That's a different workstream.

---

## Summary Table

| Issue | Status | After Warehouse Fix |
|-------|--------|---|
| Warehouse event-type keyed | ✗ Bug | ✓ Fixed |
| Base seeds resolve to UNK | ✗ 8 of 9 named entities missing | ✗ Still missing (not in GDELT/ACLED) |
| Templated corpus is generic | ✗ True | ✓ Still true but now has actor structure |
| Can train Stage 1 | ✗ No (degenerate) | ✓ Yes (on ACLED actor types) |
| Stage 2 has signal | ✗ No | ✓ Partial (on actor types, not named entities) |
