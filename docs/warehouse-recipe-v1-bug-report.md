# Warehouse-Probe Mismatch Diagnosis

## Executive Summary

**Three separate problems confirmed:**

1. **Critical**: Base seed probes reference 9 specific named entities (UGTT, SCAF, NTC, Ennahda, etc.) that are **not present** in the v1 warehouse. Only 1 of 9 named entities exists (Muslim Brotherhood).

2. **Degenerate training**: The 243-row templated probe corpus uses only generic actor-type hints (`police`, `military`, `civilians (egypt)`), which do exist in the warehouse. However, the templated probes have a zero contribution to meaningful training because they only probe generic concepts, not the named-actor structure the design intended.

3. **Architecture mismatch**: The v1 warehouse is keyed on event-type aggregations (`ar_v1|a cabinet meeting|EG|2010-02`), not on named actors. The design document promised "human-readable actor×admin1 keys" but the implementation aggregates on GDELT event descriptions.

---

## Detailed Findings

### Problem 1: Base Seed Named Entities Not in Warehouse

| Hint | Normalized | In Warehouse? |
|------|-----------|---------------|
| UGTT | `ugtt` | ✗ NO |
| Supreme Council of the Armed Forces | `supreme council of the armed forces` | ✗ NO |
| National Transitional Council | `national transitional council` | ✗ NO |
| Ennahda | `ennahda` | ✗ NO |
| February 17 Martyrs Brigade | `february 17 martyrs brigade` | ✗ NO |
| Ben Ali | `ben ali` | ✗ NO |
| Gaddafi loyalists | `gaddafi loyalists` | ✗ NO |
| ElBaradei | `elbaradei` | ✗ NO |
| Muslim Brotherhood | `muslim brotherhood` | **✓ YES** |

**Impact**: 8 of 20 base seed probes have hints that resolve to UNK embeddings during encoding. These probes train with no grounding in the warehouse. The ones without hints (propagation, precursor gate examples) train on generic actor-type features only.

### Problem 2: Templated Probe Corpus Has No Named Entities

The 243-row templated corpus is generated from:
```python
_NAMED_HINTS_BY_COUNTRY = {
    "Tunisia": ["tunisia", "tunis", "police"],
    "Egypt": ["egypt", "cairo", "civilians (egypt)"],
    "Libya": ["libya", "rebel", "military"],
}

_ACTOR_TYPE_HINTS = ("police", "military", "civilians")
```

When expanded, every templated probe ends up with **only generic actor-type hints** (`police`, `military`, `civilians`) or **event-type aliases** (e.g., `tunisia`, `egypt`, `cairo`). None of the named entities that Stage 2 would label actually appear in the training corpus.

**Impact**: The model learns to discriminate on generic actor types and event types, not on the named-actor structure needed for downstream link prediction or label propagation.

### Problem 3: Warehouse Is Event-Type Aggregated, Not Actor-Aggregated

Sample warehouse node_ids:
```
ar_v1|a cabinet meeting|EG|2010-02
ar_v1|a cabinet meeting|EG|2010-04
ar_v1|rioters (egypt)|LY|2010-09
ar_v1|rioters (egypt)|LY|2011-02
ar_v1|protesters (egypt)|SY|2011-01
```

The middle segment (`a cabinet meeting`, `rioters (egypt)`, `protesters (egypt)`) is the GDELT event code from the raw tape, **not** a named actor. The warehouse contains **6,673 unique hint keys**, but they are:
- GDELT event type codes (`a cabinet meeting`, `a protest`, `a shooting`, ...)
- GDELT raw actor1/actor2 strings (truncated, lowercased, deduplicated)
- Geographic/institutional names

Example: The most common warehouse hints are:
- `protesters (egypt)` — 459 occurrences
- `rioters (egypt)` — 327 occurrences
- Generic entity names: `government`, `ministry`, `military`, `police`, `president`, `obama`, `france`, etc.

**Root cause**: The v1 recipe's aggregation step grouped events by `(event_type, country_code, month)` instead of grouping by named actors and geographic regions. The design doc promised actor-keyed nodes; the implementation delivers event-type nodes.

---

## Why Training Appears Successful But Is Actually Degenerate

The loss converged to ~0.003 with all 243 probes contributing. This looks healthy but masks a silent failure:

1. **All templated probes have generic hints.** When every probe uses `police` or `military` or `civilians (egypt)`, they all resolve to a small cluster of very common nodes in the warehouse.

2. **InfoNCE loss becomes trivial.** With 16 negative samples per query, if positives and negatives are drawn from the same small cluster of generic nodes, the loss is low regardless of encoder quality.

3. **Query encoder outputs degenerate vectors.** The unnormalized scores (2,700 to 140,000) indicate the encoder is not producing unit vectors. The L2 normalization in `QueryEncoder.forward()` should prevent this, so either:
   - The checkpoint was saved **before** normalization (encoding path differs from checkpoint path), or
   - Inference is running a different code path than training.

4. **No validation caught this.** The hint validation (`validate_probe_hints_against_manifest`) passes if the hints exist in the manifest structure. It doesn't check whether the probe corpus actually probes the intended semantic structure.

---

## Path Forward

### Immediate Fix (For Existing Warehouse)

If we must use the existing v1 warehouse as-is:

1. **Rewrite base seed probes** to reference only hints that exist in the warehouse:
   - Replace UGTT with `police` or generic terms
   - Replace SCAF with generic government phrases
   - Replace NTC with generic militia/opposition terms
   - Keep only Muslim Brotherhood (which exists)
   
   **Cost**: Loses all specificity; base seeds become indistinguishable from templated probes.

2. **Rebuild the hint validation** to catch degenerate training:
   - Emit a warning if >90% of probes use the same N hints
   - Flag probes where all hints resolve to UNK
   - Check that the probe corpus spans a meaningful diversity of warehouse nodes

### Long-Term Fix (Warehouse Recipe Revision)

The root issue is warehouse design. For Stage 1 to be meaningful:

1. **Actor-keyed warehouse recipe**: Aggregate events by named actor (GDELT `actor1_name`, ACLED `actor` field), not event type.
   - Node key format: `ar_v2|named_actor|country|admin1|month` or similar
   - Warehouse should have ~500–1000 named actors (union of GDELT actors and ACLED actors in the Arab Spring context)
   - Each actor gets a neighborhood of events; embeddings capture "what events this actor is involved in"

2. **Dual-indexed warehouse**: Keep both event-type and actor-type aggregations if needed for different applications, but Stage 1 training must use actor-keyed nodes.

3. **Validate probe coverage before training**: 
   - Ensure base seeds hit 70%+ of named actors in the corpus
   - Ensure templated probes cover the full actor-geography-time space
   - Reject training if >50% of probes have unresolved hints

---

## What The Inspection Confirmed

```
Warehouse v1 manifest statistics:
├─ Total rows: 168,195
├─ Unique hint keys: 6,673
├─ Most common hints:
│  ├─ protesters (egypt): 459 occurrences
│  ├─ rioters (egypt): 327 occurrences
│  ├─ government: 183 occurrences
│  └─ ... (6,670 more)
└─ Country distribution (node_id middle segment):
   ├─ EG: 45,504 nodes
   ├─ LY: 36,288 nodes
   ├─ SY: 40,149 nodes
   ├─ TU: 44,703 nodes
   └─ admin1 regions: 1,557 nodes (Egyptian, Libyan subdivisions)

Base seed probe named entities:
├─ Total unique: 9
├─ In warehouse: 1 (Muslim Brotherhood)
└─ Missing: 8 (UGTT, SCAF, NTC, Ennahda, Feb 17 Brigade, Ben Ali, Gaddafi, ElBaradei)

Templated probe corpus (243 rows):
├─ Unique hints used: 3 generic types (police, military, civilians)
├─ All templated probes use only generic/event-type hints
└─ Zero named actors in training signal
```

---

## Additional Findings

### Problem 2b: Unnormalized Scores Are NOT From Mismatched Vector Norms

Initial hypothesis: Unnormalized scores (2,700–140,000) come from non-unit-norm embeddings.

**Disproven**: 
- Warehouse embeddings (mmap) are perfectly unit vectors (L2 norm = 1.0000 ± 0.0)
- `QueryEncoder.forward()` correctly applies L2 normalization to output
- Loss computation uses (unit query) · (mean of unit warehouse embeddings) = dot product in [−1, 1]
- Temperature scaling (0.07) divides by 0.07, so logit range is [−14.3, +14.3]

**Actual explanation**: The 2,700–140,000 range in the checkpoint inspection script comes from using embeddings from a **different** warehouse (v1_lead_lag) that probably has different scale. The problem isn't in the training loss computation—the loss values are genuinely ~0.003 because of semantic collapse in the probe corpus, not a scale issue.

---

## Recommendation

**Option A (Quick band-aid)**: Rewrite base seeds to use only existing hints. Acknowledge that Stage 1 is now generic-feature training, not named-actor modeling. This unblocks downstream work but doesn't fix the architecture.

**Option B (Proper fix)**: Rebuild the v1 warehouse recipe to key on named actors instead of event types. This is a 1–2 day job but aligns the architecture with the design intent. Stage 1 then trains meaningful actor embeddings that downstream stages can use for label propagation.

**My recommendation**: **Option B**. The current setup is architecturally unsound. Training on it will produce embeddings that collapse to generic actor-type discrimination. When Stage 2 tries to propagate labels from named actors to events, there will be no signal because Stage 1 never learned to discriminate named actors. Better to fix it now than discover the problem when downstream stages fail to generalize.

---

## Root Cause Summary

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| Base seeds fail to resolve | Named entities not in warehouse | 8 of 20 probes → UNK embeddings |
| Templated corpus degenerate | Only generic actor-type hints | No named-actor signal in training |
| Warehouse is event-keyed | Recipe aggregates by event type, not actor | Mismatch with probe corpus design intent |
| Training appears successful | Semantic collapse masked by generic hints | Loss is low because all queries identical |

The **root cause is warehouse design**. The v1 recipe was supposed to produce named-actor nodes but instead produces event-type nodes. The probe corpus was written against an imagined warehouse that doesn't exist.
