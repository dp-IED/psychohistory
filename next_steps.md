# Next steps — execution plan (review handoff)

**Audience:** Reviewers and a future executor model with full-repo context.  
**Status:** Planning document only; not a commitment to run every line in this session.  
**Alignment:** [`project.md`](project.md), [`roadmap.md`](roadmap.md) Stages 4–6 (builder + assumptions **before** WM depth), **[`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)** (locked builder + assumptions + contexts), [`forecast_charter.md`](forecast_charter.md), research synthesis [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md).

---

## 0. Realism and operating discipline

The program thesis (relational encoder + latent dynamics + optional market supervision under **PIT**) is coherent and literature-grounded; the France benchmark **de-risked GNN vs baselines**. What remains is **hard research**, not only engineering. The full seven-layer **target** architecture in [`docs/research/architecture.md`](docs/research/architecture.md) is **ambitious** relative to the repo: **Layers 0–1** (warehouse, deterministic snapshot builder) and **Layer 3** (heterogeneous GNN) are in code; **Layer 5** is only **partially** realized (material backtests, not full multi-head product). **Highest execution risk** is now **Layer 2 (query lens / query-conditioned subgraph)** and the **gap between deterministic builder and learned retrieval**, plus an **assumption layer** between subgraph and forecast—**before** treating **Layer 4 (world model)** depth as the main sequencing spine. WM remains **mandatory** in the program but is **scheduled after** a pinned subgraph + assumption interface so temporal ablations are interpretable. **Track M** (markets) still needs the **event** training spine through **WM v0** (see §2) before it produces reliable claims.

**Core discipline:** **get gradients flowing on real event data early**—a thin, correct training loop on existing snapshots beats a perfect design that never runs. The first “working loop” should validate **subgraph construction + assumptions + a minimal forecast head**; then grow **WM v0 (time-then-space)** and multi-step losses on that contract (`roadmap.md` execution reality). **Collapse** open-ended “Phase 0” charter work: **time-box** it and run it **in parallel** with a **dirty-but-runnable** training skeleton—not serial prerequisites that expand to fill the schedule.

---

## 1. Program goals (what this phase must achieve)

### 1.1 Primary outcomes

1. **Graph builder + query-conditioned subgraph v0 (Stages 1–2 / 6)** — **first**  
   Implement **[`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)**: **retrieval unit** = three-tier hierarchy (**actor-state**, **trend thread**, **historical analogue**); **v1 = actor-states only** with graph **slots** for the other two. **Two-stage retriever:** ANN **top-100** → reranker + edge enricher → **top-50** nodes, **≤200** edges, **static shapes** at inference. **Supervision:** **Stage 1** self-supervised on **GDELT + ACLED** tape (co-evolution, persistence, lead–lag, cross-source corroboration); **Stage 2** weak labels from **forecast improvement** on held-out slices—**no human retrieval labels**. **Primary training** on contract contexts (Arab Spring, European sovereign debt crisis, Latin America TBD); **France = validation harness + smoke**, not primary builder training.

2. **Assumption layer v0 (Stage 6)** — **second**  
   Five **soft gates** in \([0,1]\) per contract—**Persistence**, **Propagation**, **Precursor**, **Suppression**, **Coordination**—from lightweight MLPs on the retrieved subgraph; **soft** in training, **hard-thresholded** at inference; **modulate forecaster attention**; **architectural priors** (not a hand label schema). Supervision = **forecast improvement when a gate fires**, jointly with the stack—not separate gate labels.

3. **World-model path (Stage 6) — after builder + assumption baseline**  
   **Time-then-space:** per-node temporal core (**GRU/SSM**) over a short history of weekly snapshots, **then** heterogeneous message passing on the **pinned** subgraph interface—**not** an interleaved fusion where credit assignment is opaque. Add **multi-step auxiliary losses** (\(t{+}2\), \(t{+}4\) horizons) once the single-step loop works. Run **GRU-only vs GNN-only vs GRU+GNN** ablations to satisfy the **spirit of G5** for the **temporal + relational stack**, not as a substitute for builder/assumption ablations.

4. **Prediction-market track (Stage 5) — after §2 spine through WM v0 (`H`); contract Stage 3**  
   Polymarket (etc.): **forecaster calibration** and late validation per [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)—**not** supervision for the graph builder. Still a **label-contract and ingestion** problem: start with **terminal resolutions only**; short-horizon belief dynamics later. **Before** any model consumes real resolution labels: **adversarial PIT harness** on **synthetic tapes** (inject future labels/corrections; assert detectors fire). Treat resolution timestamps as **messy** (delayed reporting, retroactive fixes)—tests first.

5. **Engineering hygiene**  
   Reproducibility (seeds, audits); **coverage audit** for market supervision—which domains have **zero** listed markets so the model must not silently fail there (`forecast_charter.md` markets tier). **Aggressive** minus-variants before new node types (`roadmap.md`).

### 1.2 Explicit non-goals for this phase

- Blocking **builder / assumption** work on Polymarket ingestion; **event resolution** is the first label source for training loops on the scaffold.
- **Over-optimizing the forecast head** (including France-only score chasing) **before** retrieval ontology and assumption interfaces are stable enough for **reproducible ablations**—France stays **smoke**, not the program center.
- Blocking **WM v0** indefinitely on perfect builder specs; WM is **sequenced after** D–G (see §2.2) but remains **in scope**—ship a thin WM on a **pinned** subgraph contract rather than waiting for the final retriever.
- Re-proving France on every unrelated change; France = **smoke** when shared ingestion/snapshot/backtest changes; do not use France as the **primary builder training** corpus (**contract**).
- Using **Iran** (or similar contested, sparse domains) as the **primary** optimization benchmark or headline eval **before** stable baselines, held-out contracts, and ablations are frozen on the **France** (or agreed) scaffold—Iran is a **shadow / red-team** lane until a labeled eval contract exists (**§2.3**).
- Productized constrained Q&A at scale before retrieval + assumption + forecast pieces exist.

### 1.3 Success criteria (gates)


| Gate | Criterion |
|------|-----------|
| **G4** (markets) | Masked runs + **coverage reporting**; non-market slices explicit. |
| **G5** (discovery) | Held-out time/region + **ablations** across **builder vs encoder vs WM** as credited—**retrieval/sparsity/stability** plus temporal vs MP vs both with **clean** structural variants—not one fused block that cannot be ablated. |
| **Material tier** | Charter metrics; prevalence-aware scores where sparse. |

---

## 2. Immediate path (minimal sequencing — start here)

Only **blocking** data steps; then **model**. Target: **something trained end-to-end through subgraph + assumption + minimal forecast milestones (§2.2 `D`–`G`)**, then **WM v0 (`H`)** on the pinned interface—even if baselines are embarrassingly simple at first.

### 2.1 Data prerequisites (blocking only)

| Step | Deliverable |
|------|-------------|
| **A** | **Snapshot / tensor audit:** DuckDB → time-ordered sequences \((S_t, \text{label}_{t+k})\) with **zero leakage**. Run [`evals/graph_artifact_contract.py`](evals/graph_artifact_contract.py); add **one adversarial test** that injects a future label and **asserts** the pipeline rejects or flags it. |
| **B** | **Pin training + validation contexts:** per [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md), stand up **offline embedding stores** and splits for **primary builder training** (Arab Spring 2010–2013, European sovereign debt crisis 2009–2015, one Latin American commodity-politics sequence **TBD**). **France:** **validation harness + regression smoke** for the protest slice—not primary builder training. Fix **hold-out** rules per context. |
| **C** | **Node feature contract:** enumerate per node type at snapshot \(t\) (degree, recency, type, optional **frozen** text embedding, etc.) so missing data is explicit **before** large training spend. |
| **C′** | **Retrieval / supervision contract:** follow **[`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)** (locked). Implementation TBD: node embedding scheme, ANN library, reranker/gate MLP shapes—**after** contract-stable ingest. |

### 2.2 Model training (order)

Sequence: **D → E → F → G → H**, then **I** (Iran shadow slice—**§2.3**), then **M** (Polymarket—**§4**). **World model** depth lands in **`H`** so builder and assumption interfaces are pinned first.

| Step | Deliverable |
|------|-------------|
| **D** | **Training loop skeleton** (e.g. `train.py` or module under `baselines/`): loader over \((S_t, \text{label}_{t+k})\), loss (Brier / BCE as appropriate), optimizer, **dummy linear** baseline. Surfaces batching, device, gradient bugs **before** model complexity. |
| **E** | **Graph builder + query-conditioned subgraph v0:** implement **[`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)** — ANN **top-100**, rerank to **≤50** nodes, **≤200** edges, precomputed **memory-mapped** embeddings (**dim 128 or 256**), **no** variable-size graph ops at query time; wire into [`baselines/gnn.py`](baselines/gnn.py) (or adapter). **Ablations** on subgraph policy vs encoder. |
| **F** | **Assumption layer v0:** five **soft gates** (Persistence, Propagation, Precursor, Suppression, Coordination) per contract; modulate forecaster **attention**; thresholds hard at inference. |
| **G** | **Minimal forecast attachment:** lightweight head on **induced** subgraphs to validate upstream representations (Brier / BCE on event targets)—**forecast is the evaluator**, not the only training center. |
| **H** | **WM v0 + multi-step + ablation matrix:** prepend per-node **GRU** (or SSM) over last \(k\) weekly snapshots, **then** message passing—**time then space**; add **multi-step** terms (\(t{+}2\), \(t{+}4\) or charter horizons); run **GRU-only vs GNN-only vs GRU+GNN** on the **same** subgraph contract from **E**. |
| **I** | **Iran stress-test lane (shadow / red-team):** after **H** is green, stand up a parallel **Iran** dev slice per **§2.3**. Does **not** replace the **France** scaffold steps **D–G**; no headline forecasting claims without a labeled eval contract. |
| **M** | **Then Polymarket:** ingest + resolution labels as **Track M** below; label contracts and masking—not parallel “build the world” from scratch. |

### 2.3 Iran stress-test lane (parallel; after **H**)

**Positioning:** Iran is a strong **product-style scenario** (sparse timestamped evidence, multi-domain reasoning, explicit uncertainty, epistemological tier—competing interpretations, evidence pointers, limits under cutoff discipline). It is a **poor primary development benchmark right now** because it stacks confounds (ingest gaps, ontology pressure, sparse labels, entity ambiguity, multi-community timelines, politically contested interpretation) and invites **compelling narratives without enough labeled outcomes** to validate cleanly. **Do not** replace **France** scaffold work (**D–G**: through minimal forecast attachment) or the **WM ablation spine (`H`)** with Iran as the primary driver.

**Split:**

| Track | Role |
|-------|------|
| **France** | **Validation harness + regression smoke** on the protest slice; **not** primary **builder training** (`docs/graph-builder-contract-v0.1.md`). Engineering milestones **D–H** may still run on France **until** multi-context stores are wired—**tag runs** clearly. |
| **Iran** | **System stress test**: failure modes—missing or contradictory sources, sparse temporal history, domain transfer, interpretive uncertainty, and reviewer inspection of ranked hypotheses + cited evidence + explicit limits. |

**Explicit non-goals for the Iran slice:** no new ontology unless a **documented** failure mode demands it; no **regime-naming** claims as model outputs; no **headline forecasting** claims without a **labeled eval contract** aligned to `forecast_charter.md` / `evals/`.

**Immediate audits (what to run, not what to optimize):**

1. **Ingest audit:** Layers 0/1 represent evidence cleanly under frozen cutoffs.  
2. **Lens / builder audit:** Subgraphs are **faithful** to \(S_t\) under the lens contract; retrieval is not silently inventing facts; **stability** across nearby timesteps or query variants where expected.  
3. **Assumption audit:** Stated latent assumptions remain **tied to evidence** and inspectable—not unconstrained narrative.  
4. **Transfer audit:** Models trained on existing scaffolds show **calibrated** uncertainty rather than overconfident behavior on sparse, out-of-scaffold geography.  
5. **Reviewer workflow:** Humans can inspect ranked hypotheses, pointers to evidence, and stated limits in a domain where ambiguity is real (`docs/research/research.md`, epistemological tier).

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

**Dependency:** Event-based training spine (**§2 D–H**: through **WM v0** + ablation matrix on the pinned subgraph contract) running; **adversarial PIT harness** green for synthetic market-like tapes.


| Phase | Deliverable |
|-------|-------------|
| **M1** | Schema + ingestion: quotes, resolutions, metadata; versioned raw; loader tests. |
| **M2** | **Synthetic adversarial tests** before real labels: future resolution, retroactive correction—pipeline must fail closed or alarm. |
| **M3** | **Coverage audit:** table or report of domains/eras with **no** market coverage vs eval geography—explicit “no silent failure.” |
| **M4** | Label contracts: resolution head first; short-horizon later; masking ablations; baselines. |

**Touchpoints:** `schemas/`, `ingest/`, `baselines/`, `tests/`.

---

## 5. Track W — Query lens / builder tooling (research-grade; parallel with §2 **E–F**)

**Goal:** Faithful **query-conditioned subgraphs**—the same capability §2 step **E** must instantiate. This track is **not** deferred behind WM; it **feeds** the WM stage (`H`) with a stable \(S_t^{(q)}\) contract.


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
| I4 | **Iran shadow slice** (after step **H** in §2): run the **§2.3** audits (ingest, lens, assumptions, transfer, reviewer workflow)—**not** a primary metric gate or replacement for the France scaffold. |

---

## 7. Risks and mitigations

### 7.1 Technical

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Label leakage** (market resolutions, messy timestamps) | Invalidates metrics | **Synthetic adversarial harness before real labels**; separate label builders; review `forecast_charter.md`. |
| **Credit assignment** (joint temporal + MP) | Unreviewable claims | **Time-then-space**, **separate** forward blocks for ablation; multi-step losses; multi-seed. **Do not** fuse recurrence and MP in one indistinguishable pass. |
| **Builder supervision starvation** | Retrieval collapses to shortcuts | **Plural losses** per [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md) (Stage 1 SSL on tape, Stage 2 forecast-utility); staged training; **no** market signal in builder stages. |
| **Contract drift** | Claims mix France training with validation-only rule | **Tag** runs by training corpus; builder **primary** training = contract contexts; France = **validation / smoke** unless explicitly bridging. |
| **Short-horizon weak labels** | Misses long-lag precursors | Keep **long-horizon inductive bias** in retrieval targets and self-supervision; horizon-aware utility weighting. |
| **Frozen snapshot / teacher staleness** | Builder overfits stale encoder | **Refresh** backbone or embedding snapshots on a schedule when using indirect signals. |
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
| **Parallel tracks starve** | Nothing ships | Builder + assumptions on **events** do not wait for markets; **WM v0 (`H`)** does not wait for Polymarket. |
| **Single headline metric** | Misleading story | Full charter table. |

---

## 8. Executor checklist

- Re-read [`roadmap.md`](roadmap.md) gates for the current change.  
- **Gradients-first:** prefer extending `schemas/`, `ingest/`, `baselines/`, `evals/`, `tests/` over spec-only work.  
- Any material claim: **metrics**, **slices**, **seeds**, **ablation** or **mask**.  
- **Subgraph / assumption contract** changes: update the **§2.2** milestone map and rerun **France smoke** when shared snapshot/backtest contracts change.  
- **Markets:** adversarial tape tests **before** trusting production labels; **coverage** documented.  
- France smoke when shared snapshot/backtest contracts change—not a gate on every builder tweak, but required when contracts move.  
- **Iran** only as **§2.3** shadow / red-team after **H**; keep France (or agreed scaffold) as the ablation-controlled harness until eval contracts support stronger claims.

---

## 9. References

- [`roadmap.md`](roadmap.md) — stages, gates, execution reality  
- [`forecast_charter.md`](forecast_charter.md) — metrics, markets tier  
- [`docs/reviewers-guide.md`](docs/reviewers-guide.md)  
- [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md) — locked builder, assumptions, supervision, compute, training contexts  
- [`docs/research/architecture.md`](docs/research/architecture.md) — target layers + implementation status  
- [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md)  
- [`docs/source_layer_experiments.md`](docs/source_layer_experiments.md)
