# Next steps — execution plan (review handoff)

**Audience:** Reviewers and a future executor model with full-repo context.  
**Status:** Planning document only; not a commitment to run every line in this session.  
**Alignment:** [`project.md`](project.md), [`roadmap.md`](roadmap.md) Stages 5–6, [`forecast_charter.md`](forecast_charter.md), research synthesis [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md).

---

## 0. Realism and operating discipline

The program thesis (relational encoder + latent dynamics + optional market supervision under **PIT**) is coherent and literature-grounded; the France benchmark **de-risked GNN vs baselines**. What remains is **hard research**, not only engineering. The full seven-layer **target** architecture in [`docs/research/architecture.md`](docs/research/architecture.md) is **ambitious** relative to the repo: **Layers 0–1** (warehouse, builder/snapshots) and **Layer 3** (heterogeneous GNN) are in code; **Layer 5** is only **partially** realized (material backtests, not full multi-head product). **Layer 4 (world model)** and **Layer 2 (query lens)** are **highest risk**—minimal or no dedicated implementations yet—so **Track M** (markets) and **Track W** (WM) both need new infrastructure before they produce reliable signal.

**Core discipline:** **get gradients flowing on real event data early**—a thin, correct training loop on existing snapshots beats a perfect design that never runs. **Grow the architecture around a working loop** (`roadmap.md` execution reality). **Collapse** open-ended “Phase 0” charter work: **time-box** it and run it **in parallel** with a **dirty-but-runnable** training skeleton—not serial prerequisites that expand to fill the schedule.

---

## 1. Program goals (what this phase must achieve)

### 1.1 Primary outcomes

1. **World-model path (Stage 6) — ship v0 on event labels first**  
   **Time-then-space:** per-node temporal core (**GRU/SSM**) over a short history of weekly snapshots, **then** heterogeneous message passing—**not** an interleaved fusion where credit assignment is opaque. Add **multi-step auxiliary losses** (\(t{+}2\), \(t{+}4\) horizons) once the single-step loop works. Run **GRU-only vs GNN-only vs GRU+GNN** ablations early to satisfy the **spirit of G5** before the full stack exists.

2. **Prediction-market track (Stage 5) — after event loop is running**  
   Polymarket (etc.) becomes a **label-contract and ingestion** problem, not a “build everything at once” problem. Start with **terminal resolutions only**; short-horizon belief dynamics later. **Before** any model consumes real resolution labels: **adversarial PIT harness** on **synthetic tapes** (inject future labels/corrections; assert detectors fire). Treat resolution timestamps as **messy** (delayed reporting, retroactive fixes)—tests first.

3. **Engineering hygiene**  
   Reproducibility (seeds, audits); **coverage audit** for market supervision—which domains have **zero** listed markets so the model must not silently fail there (`forecast_charter.md` markets tier). **Aggressive** minus-variants before new node types (`roadmap.md`).

### 1.2 Explicit non-goals for this phase

- Blocking WM work on Polymarket ingestion; **event resolution** is the first label source for WM v0.
- Re-proving France on every unrelated change; France = **smoke** when shared ingestion/snapshot/backtest changes.
- Using **Iran** (or similar contested, sparse domains) as the **primary** optimization benchmark or headline eval **before** stable baselines, held-out contracts, and ablations are frozen on the **France** (or agreed) scaffold—Iran is a **shadow / red-team** lane until a labeled eval contract exists (**§2.3**).
- Productized constrained Q&A at scale before retrieval + forecast pieces exist.

### 1.3 Success criteria (gates)


| Gate | Criterion |
|------|-----------|
| **G4** (markets) | Masked runs + **coverage reporting**; non-market slices explicit. |
| **G5** (discovery / WM) | Held-out time/region + **ablations** (temporal vs MP vs both) with **clean** structural variants—not one fused block that cannot be ablated. |
| **Material tier** | Charter metrics; prevalence-aware scores where sparse. |

---

## 2. Immediate path (minimal sequencing — start here)

Only **blocking** data steps; then **model**. Target: **something trained end-to-end by the fifth major milestone** (skeleton + temporal encoder path), even if baselines are embarrassingly simple at first.

### 2.1 Data prerequisites (blocking only)

| Step | Deliverable |
|------|-------------|
| **A** | **Snapshot / tensor audit:** DuckDB → time-ordered sequences \((S_t, \text{label}_{t+k})\) with **zero leakage**. Run [`evals/graph_artifact_contract.py`](evals/graph_artifact_contract.py); add **one adversarial test** that injects a future label and **asserts** the pipeline rejects or flags it. |
| **B** | **Pin a training slice:** e.g. 2–3 years of event data for one domain (France is fine as **scaffold**; keep the model **domain-agnostic** in design). Fix **hold-out** now (e.g. last 12 months). |
| **C** | **Node feature contract:** enumerate per node type at snapshot \(t\) (degree, recency, type, optional **frozen** text embedding, etc.) so missing data is explicit **before** large training spend. |

### 2.2 Model training (order)

Sequence: **D → E → F → G**, then **I** (Iran shadow slice—**§2.3**), then **H** (Polymarket).

| Step | Deliverable |
|------|-------------|
| **D** | **Training loop skeleton** (e.g. `train.py` or module under `baselines/`): loader over \((S_t, \text{label}_{t+k})\), loss (Brier / BCE as appropriate), optimizer, **dummy linear** baseline. Surfaces batching, device, gradient bugs **before** model complexity. |
| **E** | **WM v0:** prepend per-node **GRU** over last \(k\) weekly snapshots to existing [`baselines/gnn.py`](baselines/gnn.py) hetero GNN; **time step then message passing** in a way that supports **separate** ablations. Train end-to-end on **event** targets. |
| **F** | **Multi-step losses:** secondary terms for \(t{+}2\), \(t{+}4\) (or charter horizons) from shared \(z_t\). |
| **G** | **First ablation matrix:** GRU-only (no MP) vs GNN-only (no GRU) vs GRU+GNN—evidence that **composition earns its keep**. |
| **I** | **Iran stress-test lane (shadow / red-team):** after **G** is green, stand up a parallel **Iran** dev slice per **§2.3**. Does **not** replace Step **E** or France as the regression-controlled training harness; no headline forecasting claims without a labeled eval contract. |
| **H** | **Then Polymarket:** ingest + resolution labels as **Phase M** below; label contracts and masking—not parallel “build the world” from scratch. |

### 2.3 Iran stress-test lane (parallel; after **G**)

**Positioning:** Iran is a strong **product-style scenario** (sparse timestamped evidence, multi-domain reasoning, explicit uncertainty, epistemological tier—competing interpretations, evidence pointers, limits under cutoff discipline). It is a **poor primary development benchmark right now** because it stacks confounds (ingest gaps, ontology pressure, sparse labels, entity ambiguity, multi-community timelines, politically contested interpretation) and invites **compelling narratives without enough labeled outcomes** to validate cleanly. **Do not** replace Step **E** with Iran.

**Split:**

| Track | Role |
|-------|------|
| **France** | Model-development **control**: cleaner metrics, ablations, and regression on the pinned scaffold (Steps **D–G**). |
| **Iran** | **System stress test**: failure modes—missing or contradictory sources, sparse temporal history, domain transfer, interpretive uncertainty, and reviewer inspection of ranked hypotheses + cited evidence + explicit limits. |

**Explicit non-goals for the Iran slice:** no new ontology unless a **documented** failure mode demands it; no **regime-naming** claims as model outputs; no **headline forecasting** claims without a **labeled eval contract** aligned to `forecast_charter.md` / `evals/`.

**Immediate audits (what to run, not what to optimize):**

1. **Ingest audit:** Layers 0/1 represent evidence cleanly under frozen cutoffs.  
2. **Lens audit:** Military, political, material, and epistemic subquestions can be scoped without silently dropping evidence.  
3. **Transfer audit:** Models trained on existing scaffolds show **calibrated** uncertainty rather than overconfident behavior on sparse, out-of-scaffold geography.  
4. **Reviewer workflow:** Humans can inspect ranked hypotheses, pointers to evidence, and stated limits in a domain where ambiguity is real (`docs/research/research.md`, epistemological tier).

**Summary:** **Yes, use Iran—as a shadow evaluation / red-team slice, not the main benchmark that drives model iteration until** baselines, held-out evaluation, and ablations are stable on the primary harness.

---

## 3. Prerequisites (time-boxed, parallel with §2)

**Goal:** Charter and PIT docs stay accurate, but **do not** balloon into a serial gate.


| Step | Deliverable | Notes |
|------|-------------|--------|
| P0.1 | **Charter slices** for WM + markets (horizons, tables) | Minimal edits to [`forecast_charter.md`](forecast_charter.md); revise when heads land. |
| P0.2 | **PIT documentation** next to loaders | Especially market **resolution time** semantics. |
| P0.3 | **Eval contracts** | `evals/`, `baselines/metrics.py` only as needed. |

**Touchpoints:** `ingest/`, `evals/`, `baselines/metrics.py`, `tests/`.

---

## 4. Track M — Markets (after §2 H)

**Dependency:** Event-based training loop (**§2 D–G**) running; **adversarial PIT harness** green for synthetic market-like tapes.


| Phase | Deliverable |
|-------|-------------|
| **M1** | Schema + ingestion: quotes, resolutions, metadata; versioned raw; loader tests. |
| **M2** | **Synthetic adversarial tests** before real labels: future resolution, retroactive correction—pipeline must fail closed or alarm. |
| **M3** | **Coverage audit:** table or report of domains/eras with **no** market coverage vs eval geography—explicit “no silent failure.” |
| **M4** | Label contracts: resolution head first; short-horizon later; masking ablations; baselines. |

**Touchpoints:** `schemas/`, `ingest/`, `baselines/`, `tests/`.

---

## 5. Track W — Lens and tooling (research-grade; after WM v0)

**Goal:** Faithful query-conditioned subgraphs—not blocking §2.


| Phase | Deliverable |
|-------|-------------|
| W1 | Scored retrieval / GNN-RAG-style ranking + budget (`perplexity.md` §3). |
| W2 | Mask / audit log vs full \(S_t\). |
| W3 | Optional iterative expansion (flag-guarded). |

---

## 6. Integration, reporting, smoke

| Step | Deliverable |
|------|-------------|
| I1 | Runbooks: commands, data pins, seeds. |
| I2 | France harness smoke when `ingest` / `snapshot_export` / `backtest` change. |
| I3 | Cross-track: WM + market exogenous vs WM-only on agreed slice. |
| I4 | **Iran shadow slice** (after step **G** in §2): run the **§2.3** audits (ingest, lens, transfer, reviewer workflow)—**not** a primary metric gate or replacement for the France scaffold. |

---

## 7. Risks and mitigations

### 7.1 Technical

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Label leakage** (market resolutions, messy timestamps) | Invalidates metrics | **Synthetic adversarial harness before real labels**; separate label builders; review `forecast_charter.md`. |
| **Credit assignment** (joint temporal + MP) | Unreviewable claims | **Time-then-space**, **separate** forward blocks for ablation; multi-step losses; multi-seed. **Do not** fuse recurrence and MP in one indistinguishable pass. |
| **Learned adjacency** shortcuts | Spurious structure | PIT on targets; bottlenecks / sparsity (`perplexity.md` §2). |
| **Schema creep** | Unmaintainable graph | Minus-variants **mandatory** before new node types; measurement plan. |

### 7.2 Data and operations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Polymarket selection bias** | Overfitting to “interesting” topics | **Coverage audit**; domain/liquidity slices; masking; train-without-market (`perplexity.md` §4). |
| **API/vendor churn** | Broken pipelines | Versioned snapshots; provider abstraction. |
| **Entity resolution gaps** | Biased joins | Explicit `unknown`; metrics conditional on link confidence. |

### 7.3 Evaluation and schedule

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Phase 0 expands forever** | No trained model | **Time-box**; parallel **§2** immediately. |
| **Parallel tracks starve** | Nothing ships | WM v0 on **events** does not wait for markets. |
| **Single headline metric** | Misleading story | Full charter table. |

---

## 8. Executor checklist

- Re-read [`roadmap.md`](roadmap.md) gates for the current change.  
- **Gradients-first:** prefer extending `schemas/`, `ingest/`, `baselines/`, `evals/`, `tests/` over spec-only work.  
- Any material claim: **metrics**, **slices**, **seeds**, **ablation** or **mask**.  
- **Markets:** adversarial tape tests **before** trusting production labels; **coverage** documented.  
- France smoke when shared snapshot/backtest contracts change—not a gate on WM v0.  
- **Iran** only as **§2.3** shadow / red-team after **G**; keep France (or agreed scaffold) as the ablation-controlled harness until eval contracts support stronger claims.

---

## 9. References

- [`roadmap.md`](roadmap.md) — stages, gates, execution reality  
- [`forecast_charter.md`](forecast_charter.md) — metrics, markets tier  
- [`docs/reviewers-guide.md`](docs/reviewers-guide.md)  
- [`docs/research/architecture.md`](docs/research/architecture.md) — target layers + implementation status  
- [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md)  
- [`docs/source_layer_experiments.md`](docs/source_layer_experiments.md)
