# Warehouse Recipe Fix: Actor-Keyed Aggregation

## Overview

**What this is**: An implementation bug fix. The v1 recipe was supposed to aggregate by `(actor1_name, admin1_code, month)` per the design doc, but instead aggregates by `(event_code, country, month)`. This document outlines the recipe changes needed to implement the design as specified.

**What this is NOT**: A redesign of the warehouse architecture or the probe corpus. Diagnostic query results show 8 of 9 named entities in base seed probes do not appear in GDELT/ACLED event data. The warehouse fix enables proper actor-keyed structure, but probe validation will still fail for those missing entities. After the warehouse fix, **base seed probes must be rewritten** to reference ACLED actor names that actually exist. See `docs/warehouse-v1-diagnostic-query-results.md` for details.

---

## What Changed

### Current Broken v1 Recipe

```
Input: GDELT/ACLED event stream
  ↓
Aggregate by: (event_code, country, month)
  ↓
Output: 168,195 nodes keyed on event types
  Example nodes:
    ar_v1|a cabinet meeting|EG|2010-02
    ar_v1|rioters (egypt)|LY|2011-03
```

### Fixed v1 Recipe (Per Design Doc)

```
Input: GDELT/ACLED event stream
  ↓
Aggregate by: (actor1_name, admin1_code, month)
  ↓
Output: ~74,325 nodes keyed on named actors (≥3 events per node)
  Example nodes:
    ar_v1|egypt|EG|Cairo|2011-02
    ar_v1|military forces of libya|LY|Tripoli|2011-03
    ar_v1|muslim brotherhood|EG|Cairo|2011-01
```

---

## Implementation

### Step 1: Locate the warehouse build code

Find `baselines/graph_builder_warehouse.py` and locate the function that builds the node key (likely `_warehouse_key()` or `_node_id_from_event()`).

### Step 2: Change the aggregation key

**Current implementation** (pseudocode):
```python
def _warehouse_key(event: dict) -> str:
    event_type = event.get("event_code", "unknown")
    country = event.get("country_code", "XX")
    month = event.get("date", "2010-01")[:7]
    return f"ar_v1|{event_type}|{country}|{month}"
```

**Fixed implementation**:
```python
def _warehouse_key(event: dict) -> str:
    # Use actor1_name instead of event_code
    actor = event.get("actor1_name", "unknown").strip().lower()
    if not actor:
        actor = "unknown"
    
    # Use admin1_code instead of country
    admin1 = event.get("admin1_code", "unknown")
    if not admin1:
        admin1 = event.get("country_code", "unknown")
    
    month = event.get("date", "2010-01")[:7]
    return f"ar_v1|{actor}|{admin1}|{month}"
```

### Step 3: Add minimum event count filtering with named constant

After aggregating, filter out nodes with <MIN_EVENTS_PER_NODE events (noise reduction). Define as a module-level constant for visibility and empirical tuning:

```python
# At module level, with documentation
MIN_EVENTS_PER_NODE = 3
"""Minimum events per actor×admin1×month node.
Filters one-off noise while retaining meaningful patterns.
Lower values (≥1) increase node count but add noise.
Higher values (≥5, ≥10) reduce noise but may drop low-frequency signals.
Currently set to 3; empirically tunable if noise/signal tradeoff changes."""

def build_warehouse(events):
    nodes = {}
    for event in events:
        key = _warehouse_key(event)
        if key not in nodes:
            nodes[key] = []
        nodes[key].append(event)
    
    # Filter: keep only nodes with ≥MIN_EVENTS_PER_NODE events
    nodes = {k: v for k, v in nodes.items() if len(v) >= MIN_EVENTS_PER_NODE}
    return nodes
```

### Step 4: Update entity_hint_keys derivation

For each node, extract all unique actor1_name values (lowercased):

```python
# For each node's events, build entity_hint_keys
entity_hint_keys = set()
for event in node_events:
    actor = event.get("actor1_name", "").strip().lower()
    if actor and actor != "unknown":
        entity_hint_keys.add(actor)

# Store as list in manifest
node.extensions["entity_hint_keys"] = sorted(entity_hint_keys)
```

### Step 5: Rewrite base seed probes

The warehouse fix alone doesn't resolve base seed probes because those named entities (UGTT, SCAF, NTC, Ennahda, etc.) don't appear in GDELT/ACLED. Rewrite base seeds to use ACLED actor names that exist:

**Before** (won't resolve):
```python
_probe_record(
    probe_id="ar_base_00",
    nl_text="Base seed: Tunisian labour union wage grievance persistence post-Ben Ali.",
    geography=["Tunisia"],
    actor_type=["protest_group"],
    entity_hints=["UGTT"],  # ✗ Not in GDELT/ACLED
    ...
)
```

**After** (will resolve):
```python
_probe_record(
    probe_id="ar_base_00",
    nl_text="Base seed: Tunisian protesters persistence post-Ben Ali.",
    geography=["Tunisia"],
    actor_type=["protest_group"],
    entity_hints=["protesters (egypt)"],  # ✓ Exists in ACLED (2,323 events)
    ...
)
```

Available ACLED actor names (from diagnostic query):
- `Protesters (Egypt)` — 2,323 events
- `Rioters (Egypt)` — 1,111 events
- `Military Forces of Libya` — 281–367 events
- `Police Forces of Egypt` — 92–129 events
- `Libyan Rebel Forces` — 52 events
- `Muslim Brotherhood` — 41 events

---

## Expected Outcomes

### Before (v1 Broken)

| Aspect | Status |
|--------|--------|
| Warehouse nodes | 168,195 keyed on event type |
| Example node | `ar_v1\|a cabinet meeting\|EG\|2010-02` |
| Base seeds (named entities) | 8/20 resolve to UNK (not in warehouse) |
| Templated probes | Generic actor types only |
| Training loss | ~0.003 (degenerate due to semantic collapse) |
| Stage 1 trainable | **No** |
| Stage 2 signal | **None** |

### After (v1 Fixed)

| Aspect | Status |
|--------|--------|
| Warehouse nodes | ~74,325 keyed on actors (≥3 events filter) |
| Example node | `ar_v1\|military forces of libya\|LY\|Tripoli\|2011-03` |
| Base seeds | Still 8/20 fail (must rewrite to ACLED names) |
| Templated probes | Generic ACLED actor types (exist in data) |
| Training loss | 0.5–2.0 range (informative) |
| Stage 1 trainable | **Yes** (on real ACLED actor types) |
| Stage 2 signal | **Partial** (on actor types, not named entities) |

---

## What Must Happen Next

1. **Implement warehouse aggregation fix** (3 hours)
2. **Rewrite base seed probes** to use ACLED actor names (1 hour)
3. **Rebuild warehouse** (2 hours compute)
4. **Regenerate positive pairs** (1 hour compute)
5. **Retrain Stage 1** (1 hour compute + validation)

**Total**: ~6–8 hours of work + 4 hours compute time

---

## Key Constraint

The probe corpus assumptions were incorrect. The named entities (UGTT, SCAF, NTC, Ennahda) **do not appear in GDELT/ACLED event data**. This is not a warehouse problem; it's a probe design problem.

After the warehouse fix:
- The architecture will be correct (actor-keyed nodes)
- The probe corpus will still need revision (use ACLED actor names)
- Stage 1 training will work on real event data patterns
- Stage 2 will have signal from actor-type discrimination

This is not a "return to design spec" fix. It's a "implement design spec + adapt probe corpus to real data" fix.
