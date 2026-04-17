# Plan: Analog retrieval, explanations, and downstream use of QIDs

**Goal:** Use canonical QIDs as **anchors** for **subgraph similarity**, **historical analogs**, and **explainability** (“this forecast is similar to week W involving entity Q…”). This track is **orthogonal** to training a better Brier score; it can ship **without** changing the GNN loss if retrieval is **offline** or **post-hoc**.

**Deferred in** [`docs/2026-04-16-france-gdelt-benchmark-note.md`](../2026-04-16-france-gdelt-benchmark-note.md) until multi-source graph is stable.

---

## Use cases

1. **Analyst-facing:** Given a forecast origin, retrieve **past weeks** with similar **event subgraphs** (shared QIDs, similar event counts).
2. **Model debugging:** Highlight when predictions rely on **rare** QIDs or **conflicting** sources.
3. **Future training signal (optional):** Contrastive loss on subgraph embeddings — **not** required for v1.

---

## Representations

### A. Subgraph fingerprint

- **Vector:** Bag of QIDs (multi-hot) + event counts + admin1 counts per week.
- **Similarity:** cosine, Jaccard on QID set, or **weighted** by event importance.

### B. Embedding of subgraph

- Run a small GNN **encoder** on the weekly snapshot (frozen or trained on auxiliary task) → vector per week.
- **Analog:** k-NN in embedding space.

**Ablations:**

| ID | Method | k |
|----|--------|---|
| **R0** | Jaccard on QID set | 5 |
| **R1** | + weight by Goldstein / tone | 5 |
| **R2** | Learned encoder | 5 |

**Evaluation (retrieval quality, not Brier):**

- **Precision@k:** Among retrieved analog weeks, fraction with **same** coarse outcome bucket (e.g. protest intensity) as query—requires defining buckets.
- **Temporal holdout retrieval:** Evaluate on held-out origin weeks with fixed buckets and report precision@k/recall@k across seeds.

---

## Explainability hooks

- **Attention-free baseline:** List top-k **shared QIDs** between query week and retrieved week.
- **With GNN:** If model exposes attention or gradient-based saliency on nodes, map salient nodes → QIDs.

**Ablations:**

| ID | Explanation |
|----|-------------|
| **X0** | QID overlap list only |
| **X1** | + tabular feature delta (Δ in protest counts) |
| **X2** | + short natural language template (fill-in) |

---

## Dependencies

- **Stable QIDs** (plan [`01`](01-point-in-time-grounding.md)).
- **Rich graph** (Actors, optional hierarchy from [`04`](04-wikidata-properties-hierarchy.md)) improves analog quality; **minimum** viable is **Event + Location** + QIDs on Location.

---

## Risks

- **Privacy / sensitivity** of surfacing actor names; use QIDs + labels from public graph only.
- **Metric definition** for “good analog” is subjective—document clearly.

---

## Integration with forecasting

- **Default:** Retrieval **does not** affect `predicted_occurrence_probability`.
- **Optional later:** Feature = similarity score to “high-tension” historical weeks → concat to tabular or GNN (separate ablation in plan 02).
