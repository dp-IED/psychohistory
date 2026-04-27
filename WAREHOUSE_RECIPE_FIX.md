# Warehouse Recipe Fix: Actor-Keyed Aggregation

## Overview

The v1 warehouse aggregates events by `(event_type, country, month)`. To fix the probe mismatch, we need to aggregate by `(named_actor, country, admin1, month)`.

This document outlines the recipe changes needed.

---

## Current v1 Recipe (Broken)

```
Input: GDELT/ACLED event stream
  ↓
Aggregate by: (GDELT.event_code, country, month)
  ↓
Output: 168,195 nodes keyed on event types
  - ar_v1|a cabinet meeting|EG|2010-02
  - ar_v1|rioters (egypt)|LY|2011-03
  ↓
Hint index: All GDELT/ACLED actors merged into single index
  - 6,673 unique hints (many are generic like "police", "government")
  - Probes cannot distinguish named actors
```

---

## Proposed v2 Recipe (Fixed)

```
Input: GDELT/ACLED event stream
  ↓
Extract named actors: 
  - GDELT: use actor1_name (actor)
  - ACLED: use actor (primary actor)
  ↓
Standardize/deduplicate actor names:
  - Lowercase, strip punctuation
  - Merge variant spellings (e.g., "UGTT", "Union Générale Tunisienne du Travail")
  - Create mapping table for Stage 2 label propagation
  ↓
Aggregate by: (named_actor, country, admin1, month)
  ↓
Output: ~500–1,000 nodes keyed on actors
  - ar_v2|ugtt|TU|Sfax|2011-02
  - ar_v2|scaf|EG|Cairo|2011-03
  - ar_v2|ntc|LY|Tripoli|2011-05
  ↓
Hint index: Actor names as primary keys
  - UGTT → ar_v2|ugtt|TU|... (all occurrences)
  - Supreme Council Armed Forces → ar_v2|scaf|EG|... (all occurrences)
  - Each named actor maps to one or more nodes
```

---

## Implementation Steps

### 1. Identify Named Actors in Raw Tape

**Input**: `shared_data/arab_spring/events.jsonl` (GDELT events)

**Script**: Extract and count unique actors

```python
import json
from collections import Counter

actors = Counter()
with open("shared_data/arab_spring/events.jsonl") as f:
    for line in f:
        event = json.loads(line)
        actor1 = event.get("actor1_name", "").strip().lower()
        if actor1:
            actors[actor1] += 1

# Keep top N by frequency (e.g., top 500 actors covering 90% of events)
total_events = sum(actors.values())
cumsum = 0
for actor, count in actors.most_common(1000):
    cumsum += count
    pct = 100 * cumsum / total_events
    if pct > 90:
        break
    print(f"{count:5d}x  {actor}")
```

**Expected output**: ~200–300 actors cover 80–90% of Arab Spring events. Manual review to merge variants (e.g., "muslim brotherhood", "muslim brothers").

### 2. Build Actor Standardization Mapping

Create a JSON mapping of raw → canonical actor names:

```json
{
  "ugtt": "ugtt",
  "union general tunisienne du travail": "ugtt",
  "general union tunisian labour": "ugtt",
  "scaf": "scaf",
  "supreme council armed forces": "scaf",
  "supreme military council": "scaf",
  "ntc": "ntc",
  "national transitional council": "ntc",
  "libyan ntc": "ntc",
  ...
}
```

### 3. Revise graph_builder_warehouse.py

Change the aggregation key:

**Before** (v1):
```python
# In graph_builder_warehouse.py
def _warehouse_key(event: dict) -> str:
    event_type = event.get("event_code", "unknown")
    country = event.get("country_code", "XX")
    month = event.get("date", "2010-01")[:7]
    return f"ar_v1|{event_type}|{country}|{month}"
```

**After** (v2):
```python
def _warehouse_key(event: dict, actor_map: dict[str, str]) -> str:
    # Standardize actor name
    raw_actor = event.get("actor1_name", "unknown").strip().lower()
    canonical_actor = actor_map.get(raw_actor, raw_actor)
    
    # Get geography
    country = event.get("country_code", "XX")
    admin1 = event.get("admin1_code", "unknown")
    month = event.get("date", "2010-01")[:7]
    
    return f"ar_v2|{canonical_actor}|{country}|{admin1}|{month}"
```

### 4. Rebuild Hint Index

**Before** (v1): All raw actor strings → no structure

**After** (v2): Canonical actor names → warehouse rows

```python
def build_actor_hint_index(manifest: NodeWarehouseManifest) -> dict[str, list[str]]:
    """Map canonical actor name to list of node_ids containing that actor."""
    index: dict[str, list[str]] = {}
    for row in manifest.rows:
        # Extract canonical actor from node_id
        # e.g., ar_v2|ugtt|TU|Sfax|2011-02 → "ugtt"
        parts = row.node_id.split("|")
        if len(parts) >= 2:
            actor = parts[1]
            if actor not in index:
                index[actor] = []
            index[actor].append(row.node_id)
    return index
```

### 5. Update Probe Corpus

No changes needed! The probes already reference the right named actors:
- UGTT ✓
- SCAF ✓
- NTC ✓
- Ennahda ✓
- February 17 Martyrs Brigade ✓

With the new warehouse, `validate_probe_hints_against_manifest` will pass because these actors will now exist.

---

## Expected Outcomes

### Before (v1 Broken)

- Warehouse: 168k nodes on event types
- Base seeds: 8/20 fail to resolve named actors → UNK embeddings
- Templated corpus: 243 probes on generic features only
- Training loss: ~0.003 (degenerate due to semantic collapse)
- Stage 2 signal: None (no named-actor embeddings to propagate)

### After (v2 Fixed)

- Warehouse: 500–1,000 nodes on named actors
- Base seeds: 20/20 resolve successfully
- Templated corpus: Meaningful discrimination on actor×geography×time
- Training loss: Higher but informative (0.5–2.0 range expected)
- Stage 2 signal: Named-actor embeddings that support label propagation

---

## Implementation Effort

- **Step 1** (Actor extraction): 2 hours (includes manual deduplication)
- **Step 2** (Mapping file): 1 hour (manual curation)
- **Step 3** (Code changes): 4 hours (update warehouse recipe, add new hint indexing)
- **Step 4** (Testing): 2 hours (regenerate warehouse, validate probes)
- **Total**: ~9 hours of work across 1–2 days

---

## Risk Mitigation

1. **Backup v1 warehouse**: Keep the old recipe for comparison.
2. **Validate coverage**: Ensure new actor list covers >80% of events.
3. **Rerun full pipeline**: Regenerate positive pairs, retrain Stage 1, verify loss sanity.
4. **Keep v0 as fallback**: v0 warehouse is still available if v2 has issues.

---

## Decision Point

This is not a small patch. It requires:
- Rebuilding the warehouse (~2 hours compute time)
- Rerunning positive pair generation (~1 hour)
- Retraining Stage 1 (~1 hour)
- **Total pipeline time: 4 hours**

**Proceed if**: You want Stage 1 to train meaningful embeddings and Stage 2 to have a signal for label propagation.

**Skip if**: You're comfortable with generic-actor-type embeddings and want to unblock downstream work quickly by rewriting the probe corpus to match v1.
