# Warehouse v1 Recipe: GDELT vs ACLED Actor Name Alignment

**Date:** 2026-04-27  
**Issue:** Probe corpus hints must match GDELT actor strings, not ACLED standardized names  
**Related:** `baselines/arab_spring_probes.py` (probe corpus)

---

## The Problem

After the warehouse recipe fix, `entity_hint_keys` are derived from GDELT `actor1_name` fields (lowercased). The diagnostic query reveals these GDELT strings are **different** from ACLED standardized actor names.

Example mismatch:

| GDELT String | ACLED String | Both refer to |
|---|---|---|
| `protester` | `Protesters (Egypt)` | Civilian protesters |
| `military` | `Military Forces of Libya` | Military actors |
| `police` | `Police Forces of Egypt` | Police actors |

If probe corpus hints reference ACLED names and the warehouse contains GDELT names, hint resolution will fail (same root cause as before, different strings).

---

## Solution: Write Probes Against GDELT Actor Strings

The warehouse contains GDELT actor strings. The probe corpus must reference GDELT actor strings. Use ACLED as a **discovery tool** to identify which actors matter, then look up their GDELT equivalents.

---

## GDELT Actor Strings: Common Patterns

From diagnostic query (4M+ GDELT events, EG/TU/LY 2010–2013):

### Generic role-type actors (occur in all countries)
- `government` — 48k+ events total
- `military` — 45k+ events total
- `police` — 35k+ events total
- `civilian` — 11k+ events total
- `militia` — 4k+ events total
- `rebel` — 37k+ events total

### Compound actors (role + descriptor)
- `government forces` — 2k events
- `government official` — 2k events
- `government spokesman` — 781 events
- `police officer` — 2.4k events
- `police force` — 834 events
- `military official` — 955 events
- `military force` — 665 events
- `riot police` — 1.6k events
- `rebel leader` — 892 events
- `rebel force` — 1.6k events
- `rebel commander` — 478 events

### Variant spellings (sample — check your country/period)
- `protester` vs `protest group` vs `protest leader`
- `militia` vs `militiaman`
- `policeman` vs `policemen` vs `police officer`

---

## How to Rewrite Base Seed Probes

**Step 1: Identify which real actors matter**

Use ACLED as ground truth: "Protesters (Egypt)", "Military Forces of Libya", etc. exist in real structured data.

**Step 2: Find GDELT equivalents**

Diagnostic query output shows GDELT names. For example:
- ACLED "Protesters (Egypt)" → GDELT `protester` (27k events in EG alone)
- ACLED "Military Forces of Libya" → GDELT `military` (9.6k events in LY)
- ACLED "Police Forces of Egypt" → GDELT `police` (18.8k events in EG)

**Step 3: Rewrite hints to match GDELT**

Before (won't resolve):
```python
entity_hints=["UGTT"]  # ✗ Not in any warehouse
```

After (will resolve):
```python
entity_hints=["protester"]  # ✓ 27k events in EG warehouse
```

---

## Base Seed Probe Revisions

Original base seeds attempt to reference named organizations. Since these don't exist in GDELT, rewrite to reference actor types that **do** exist:

### Persistence gate example

**Before:**
```python
_probe_record(
    probe_id="ar_base_00",
    nl_text="Base seed: Tunisian labour union wage grievance persistence post-Ben Ali.",
    geography=["Tunisia"],
    actor_type=["protest_group"],
    entity_hints=["UGTT"],  # ✗ Not in GDELT
    ...
)
```

**After:**
```python
_probe_record(
    probe_id="ar_base_00",
    nl_text="Base seed: Tunisian protester persistence post-Ben Ali.",
    geography=["Tunisia"],
    actor_type=["protest_group"],
    entity_hints=["protester"],  # ✓ 5k events in TU 2010–2013
    ...
)
```

---

## Cross-Check: Verify Hints Will Resolve

After rewriting probes to use GDELT actor strings:

```python
from pathlib import Path
from schemas.graph_builder_warehouse import NodeWarehouseManifest
from baselines.graph_builder_query_encoder import manifest_entity_hint_key_set, normalize_hint

manifest = NodeWarehouseManifest.model_validate_json(
    Path("shared_data/arab_spring/node_warehouse_v1_manifest.json").read_text()
)
warehouse_hints = manifest_entity_hint_key_set(manifest)

probe_hints = {"protester", "military", "police", "government", "rebel"}
for hint in probe_hints:
    normalized = normalize_hint(hint)
    if normalized in warehouse_hints:
        print(f"✓ {hint} will resolve")
    else:
        print(f"✗ {hint} WILL NOT resolve — fix required")
```

This check should pass for all hints after the warehouse fix.

---

## Summary

| Step | What | Result |
|------|------|--------|
| 1 | Fix warehouse recipe | 74k actor-keyed nodes, hints from GDELT |
| 2 | Rewrite probe hints | Reference GDELT actor strings (e.g., "protester", "military") |
| 3 | Validate hints | Run cross-check to confirm all hints exist |
| 4 | Retrain | Stage 1 trains on real actor patterns from GDELT |

**Do not** write probe hints against ACLED actor names. ACLED is useful for discovering which actors matter, but GDELT is what the warehouse contains.
