# Plan: QID-derived input features for the existing GNN

**Goal:** Use `external_ids.wikidata` to add **fixed-size numeric vectors** to tensors that [`build_graph_from_snapshot`](../../baselines/gnn.py) already builds—**without** changing graph topology first. This is the smallest path to a **measurable** change on Brier / ranking metrics.

**Current fact:** Location rows use tabular [`FeatureRow`](../../baselines/features.py) features; Event rows use `EVENT_FEATURE_KEYS`. **No** QID signal is read today.

---

## Feature families (in depth)

### 1. Hash embedding (deterministic, no training data)

- Map QID string `Q123...` to a vector in `ℝ^d` via **multiple hash functions** (e.g. `d` independent hashes mod bucket count) or **SinusoidalHash**-style projection.
- **Pros:** No new parameters, reproducible, no Wikidata API at train time.
- **Cons:** Collisions; no semantic similarity.

**Ablations:**

| ID | Variant | `d` | Notes |
|----|---------|-----|--------|
| **H0** | No QID features | 0 | Current baseline |
| **H1** | Hash only Location | 16, 64 | Compare dims |
| **H2** | Hash Location + Event (if event has no QID, zero pad) | 64 | |
| **H3** | Hash only Event | 64 | Isolated signal test |

### 2. Learned embedding table (trainable)

- Maintain `nn.Embedding(num_qids_or_buckets, d)` with **hash bucket** or **vocabulary** of QIDs seen in training snapshots.
- **Cold QID at eval:** map to `UNK` bucket or fallback to hash.
- **Pros:** Can fit task-specific use of identity.
- **Cons:** Risk of overfitting; **must** freeze or regularize; needs enough QID diversity in train.

**Ablations:**

| ID | Variant | Regularization |
|----|---------|----------------|
| **L0** | Same as H0 | — |
| **L1** | Embedding dim 32, bucket 4096 | L2 on embedding matrix |
| **L2** | + dropout on QID vector before concat | |
| **L3** | Freeze embeddings after epoch 10 | Early stopping on validation split only (never eval/test) |

### 3. Concatenation point

- **Location:** `loc_x = concat(tabular_features, qid_vec)` → extend `loc_proj` input dim.
- **Event:** `evt_x = concat(event_attrs, qid_vec)` (only if events get QIDs—often they do not; may need grounding only on Location/Actor first).

**Recommendation:** Start **Location-only** (admin regions already have labels conducive to Wikidata admin items).

---

## Ablations (model + grounding interaction)

Hold **grounding cache fixed** (G1 from plan 01) so differences are from **features only**.

| Run | Model | QID feature |
|-----|--------|-------------|
| **M0** | Full graph GNN | Off |
| **M1** | Full graph GNN | Hash Location, d=32 |
| **M2** | Full graph GNN | Learned Location, d=32 |
| **M3** | `location_features_only` ablation | Hash Location — tests if QID substitutes for some tabular signal |
| **M4** | `no_event_features` | Hash Location — QID + topology only |

**Metrics:** Primary: Brier, MAE, recall@5; secondary: `pr_auc - positive_rate`, balanced accuracy (per source-experiment audit).

**Split discipline (mandatory):**

- Use a fixed validation slice carved from training origins for model selection and early stopping.
- Keep benchmark eval/test origins untouched until final scoring.
- Report final metrics on eval/test only once per run config and seed.

**Success criteria:**

- M1 or M2 beats M0 on **at least one** primary metric with **same** train/eval split and seeds (report mean ± std over 3 seeds if noisy).
- If M3 beats M0: QID carries **information** beyond tabular. If not: QID is redundant or too noisy.

---

## Implementation sketch

1. In `build_graph_from_snapshot`, read `external_ids.wikidata` from Location nodes (and optionally Event).
2. Add `qid_dim` to `HeteroGNNModel` constructor; `loc_proj` becomes `Linear(location_dim + qid_dim, hidden)`.
3. CLI flag: `--gnn-qid-features {off,hash,learned}` and `--qid-dim`.
4. Tests: unit tests for hash stability; small graph with fake QIDs.

---

## Risks

- **Label–QID mismatch:** Wrong region QID hurts; use PIT grounding (plan 01).
- **Leakage:** Do not use Wikidata **future** properties here—only identity slot for v1.

---

## Dependencies

- **Stable grounding** (plan [`01`](01-point-in-time-grounding.md)) for fair comparison across weeks.
- **03** (Actor hetero) is **optional**; can ship **02** without Actor nodes.
