# Warehouse v1 Recipe: Locked Decisions Amendment

**Date:** 2026-04-27  
**Status:** Correction to embedded numbers in design documentation  
**Related:** `docs/graph-builder-contract-v0.1.md` (Training contexts section)

---

## Correction

The Arab Spring training context is implemented with a node warehouse (v1) built from GDELT + ACLED events. Previous documentation referenced "168k nodes" as a fixed design parameter.

**Corrected specification:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Warehouse recipe** | Monthly actor×admin1 aggregation | Per design: aggregate by `(actor1_name, admin1_code, month)` not event type |
| **Node count** | **~74,325** (at ≥3 events threshold) | Filters one-off noise while retaining meaningful actor×geography×time combinations |
| **Event threshold** | **≥3 events per node** | Minimum event count to qualify node for warehouse; tunable parameter, currently set to 3 |
| **Data source** | GDELT v1 events (4M+ rows) aggregated by actor | `actor1_name` field from GDELT; lowercased for hint matching |
| **Node format** | `ar_v1|{actor}|{admin1}|{month}` | E.g., `ar_v1|protester|EG|Cairo|2011-02` |

---

## Why the Number Changed

The original 168,340 figure was derived from a **different aggregation**: `(event_code, country, month)` rather than `(actor1_name, admin1_code, month)`. The former aggregates on GDELT event types (thousands of distinct event codes), producing more nodes. The design spec called for actor-keyed nodes; the broken implementation produced event-type nodes.

The fix implements the design spec:
- **Aggregation key change**: `(event_code, country, month)` → `(actor1_name, admin1_code, month)`
- **Threshold filter**: Add minimum event count (≥3) to reduce noise
- **Result**: ~74k actor-keyed nodes instead of 168k event-type nodes

Both warehouses are valid; the 74k version aligns with the design intent.

---

## GDELT Actor Names (After Lowercase Normalization)

Top actors by event frequency (sample):

```
27,307 events: protester (EG)
26,749 events: rebel (LY)
26,611 events: military (EG)
19,154 events: government (EG)
18,839 events: police (EG)
15,111 events: government (TU)
13,610 events: government (LY)
12,457 events: police (TU)
 9,610 events: military (LY)
 9,076 events: military (TU)
 5,004 events: protester (TU)
 4,981 events: rebel (TU)
 4,347 events: protester (LY)
 4,196 events: militia (LY)
 4,098 events: civilian (LY)
 3,527 events: police (LY)
 3,170 events: civilian (EG)
```

These actor names appear in the warehouse's `entity_hint_keys` after normalization. Probe corpus entity hints must match these strings.

---

## Implementation Impact

**Code locations**:
- `baselines/graph_builder_warehouse.py`: Change aggregation key in node ID generation
- `MIN_EVENTS_PER_NODE = 3`: Named constant for threshold
- `entity_hint_keys` derivation: Extract from `actor1_name` values, lowercased

**Testing**:
- Manifest validation: Confirm `node_count` is ~74k
- Probe validation: Ensure hints match GDELT actor strings (not ACLED names)
- Loss sanity: Training loss should move to 0.5–2.0 range (not 0.003)

---

## Decision Rationale

The **≥3 events threshold** is a tunable design decision. Lower thresholds (≥1) retain more nodes (128k) but introduce noise. Higher thresholds (≥5, ≥10) produce cleaner nodes (35k) but may drop meaningful low-frequency actor patterns.

**Threshold ≥3 chosen because:**
- Filters one-off noise (single-event actors)
- Retains meaningful patterns (3 events in a month across one actor×admin1 is signal)
- Produces 74k nodes (tractable for offline embedding + ANN index)
- Empirically tunable later if needed

---

## Downstream References

Update any documentation that quoted "168k nodes for Arab Spring warehouse v1" to use the corrected figure: **~74k nodes (≥3 events threshold)**.

Locations to check:
- Contract locked decisions (if any warehouse-specific row exists)
- Architecture docs (if node count appears)
- Capacity planning docs (if storage/ANN index sizing uses the 168k figure)
