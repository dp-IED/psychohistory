# Plan: Wikidata properties, hierarchy edges, and structured features

**Goal:** Go beyond **identity (QID)** to **structured properties** (e.g. instance-of, population, containment) and **optional graph edges** derived from Wikidata (e.g. admin hierarchy “part of”). This is **higher effort** and **higher leakage risk** than plans 01–02; ship only with strong PIT controls and ablations.

---

## Feature categories (in depth)

### 1. Scalar properties as node attributes

Examples (illustrative; PIDs must be validated for France admin regions):

- **P31** (instance of) → one-hot or embedding of “type” (department vs region).
- **P1082** (population) → log-scaled float; **must** be **as-of** population for date `t`, not current.
- **P2046** (area) → similar.

**Challenge:** Each property needs a **temporal snapshot** or **best available value at t** from Wikidata history (expensive) or from a **static dump** with statement qualifiers (point in time).

### 2. Hierarchy edges

- Add edges **Location → Location** (e.g. `admin1` **part of** `admin0`) from Wikidata.
- **Use:** second-hop message passing or auxiliary loss.

**Challenge:** Must align with **scoring universe** (admin1 codes); avoid nodes outside the tape.

### 3. Relation to events

- Link **Event** to **Actor** QIDs already in graph; optionally link Event to **topic** entities (e.g. protest) via weak supervision—**high ambiguity**; defer to later.

---

## Ablations (layered)

### Layer 1: Properties only (no new edges)

| ID | Properties |
|----|------------|
| **P0** | QID only (plan 02 baseline) |
| **P1** | + P31 one-hot (fixed taxonomy size) |
| **P2** | + log population (PIT) |
| **P3** | + area |

**Control:** **P0** must match **02** best run when property pipeline is “off.”

### Layer 2: Hierarchy edges only (no extra properties)

| ID | Edges |
|----|--------|
| **E0** | No hierarchy edges |
| **E1** | Directed Location->Location `part_of` edges (child to parent) |
| **E2** | Bidirectional `part_of` edges (child<->parent), with all baseline edges unchanged |

**Interpretation:** `E1 > E0` suggests **geographic structure** helps beyond tabular admin features.

### Layer 3: Combined (full)

| ID | Description |
|----|-------------|
| **C1** | P2 + E1 (best property + best edge from 1–2) |
| **C2** | C1 + regularization (dropout on property branch) |

---

## Leakage and PIT controls (mandatory)

- **Population / area:** Must use values valid for **forecast origin `t`** or earlier; document data source (dump revision or SPARQL query with `POINT_IN_TIME` qualifiers).
- **Ablation:** **P_fake** — inject **shuffled** property values across locations (same distribution) → metrics should **collapse**; if not, bug or leakage.

---

## Success criteria

- **P2** or **E1** improves over **P0** on Brier with **shuffled** control failing.
- No improvement on **P_fake** vs real properties.

---

## Implementation sketch

1. **Property store:** DuckDB table `(qid, pid, value, valid_from, valid_to)` from dump or selective SPARQL batch.
2. **Join** at `build_graph_from_snapshot` time: `location_qid` → feature vector.
3. **Hierarchy:** precompute adjacency for FR admin hierarchy once per dump version.

---

## Risks

- **Engineering cost** dominates small model gains.
- **Wrong PIT** invalidates benchmarks—review with domain expert for France admin units.

---

## Dependencies

- **01** (PIT grounding) for correct QIDs.
- **02** recommended first so **QID-only** lift is already measured.
