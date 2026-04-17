# Plan: Actor nodes in the hetero GNN + QID-aware merging

**Goal:** Today’s [`HeteroGNNModel`](../../baselines/gnn.py) ignores **Actor** nodes and `participates_in` edges. This plan adds an **Actor** tensor type and message passing so actor identity (via Wikidata QID) can influence location logits. Primary objective is to test whether **Wikidata-driven actor features** improve forecasting metrics, while merge logic stays robust to homonym collisions.

---

## Why this is a separate track from plan 02

- **02** adds features **without** new edges or node types.
- **03** changes **graph structure** (more edges, more parameters) → confound with QID unless ablated carefully.

---

## Design

### Graph construction

- **Nodes:** Keep `location` and `event`; add **`actor`** with `num_actors` rows.
- **Edges:**
  - Existing: `(event, occurs_in, location)`.
  - New: `(actor, participates_in, event)` or `(actor, participates_in, location)` depending on snapshot edge list in [`ingest/snapshot_export.py`](../../ingest/snapshot_export.py) (today: actor → event).

### Actor features

- **Option A:** Same hash/embed as plan 02, keyed by QID on Actor node.
- **Option B:** Scalar features only (role count, source count) if QID missing.
- **Option C:** Zero features for actor (structure-only ablation).

### Message passing

- **Minimal:** One additional `SAGEConv` or separate conv over `(actor, ?, event)` then fuse into event embedding before `occurs_in` to location.
- **Complexity:** Bipartite actor–event–location may need two-hop or heterogeneous conv stack; start with **one** extra conv layer.

---

## Ablations (factorial intent)

### A. Structure vs features

| ID | Actor nodes | Actor edges | Actor features |
|----|-------------|-------------|----------------|
| **A0** | No | — | — |
| **A1** | Yes | Yes | **Zero** (structure only) |
| **A2** | Yes | Yes | Hash QID |
| **A3** | Yes | Yes | Learned + QID |

**Interpretation:** If `A1 > A0`, **topology** matters. If `A2 > A1`, **QID** adds signal. If `A2 ≈ A1`, QID embedding is noisy or redundant.

### B. QID merge (collapse duplicate actors)

- **Problem:** Same QID, different `actor_id` strings → duplicate nodes.
- **Merge:** Before conv, **collapse** actor nodes with same `external_ids.wikidata` into one super-node (sum or mean features, multigraph edges).
- **Homonym guard:** do **not** merge unresolved actors by label alone. For unresolved actors, keep distinct nodes unless a stricter key matches exactly (normalized label + country/admin code + role/type + same week).

| ID | Merge policy |
|----|----------------|
| **B0** | No merge |
| **B1** | Merge by QID when present |
| **B2** | QID merge + strict unresolved-key merge (label+context), never label-only |
| **B3** | Label-only merge stress test (diagnostic only, not production candidate) |

**Interpretation:** `B1 > B0` supports identity unification; `B2` vs `B1` tests whether safe unresolved merging helps; `B3` quantifies homonym damage and should typically underperform.

### C. Interaction with plan 02

- Run **best** Location QID config from 02 **with** and **without** Actor track (A0 vs A2) to see if gains are additive or redundant.

---

## Metrics and success

- Same as France benchmark + comparability metrics.
- **Per-origin** stability: recall@5 variance across weeks (actors may be sparse early).
- Track merge diagnostics: `% actors with QID`, `% nodes collapsed by QID`, `% unresolved nodes merged by strict key`, and `% potentially ambiguous label collisions`.

**Failure mode:** If Actor count is tiny or highly imbalanced, conv may not train—monitor **actor count per snapshot** in metadata.

**Success criteria (priority):**

- `A2` or `A3` improves over `A1` on at least one primary forecasting metric with fixed splits and seeds.
- `B3` should not be selected even if superficially strong on a single metric; treat as leakage/collision diagnostic.

---

## Implementation sketch

1. Extend `build_graph_from_snapshot` to emit `data["actor"].x`, edge_index for `participates_in`.
2. Extend `HeteroGNNModel.forward` with actor branch; document tensor shapes.
3. Optional: `merge_actors_by_qid(snapshot)` preprocessing step.

---

## Risks

- **Parameter count** and overfitting; match hidden dims and keep actor dim small.
- **Latency:** Larger graphs per batch.

---

## Dependencies

- Grounding on **Actor** nodes (plan [`01`](01-point-in-time-grounding.md)) must be good enough; track **unresolved** rate.
- Prefer completing **02** first to reuse QID embedding machinery.
