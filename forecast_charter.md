# Forecast Charter

Evaluation contract and **non-goals**. **`roadmap.md`** is the full program; **`project.md`** states purpose and goals.

---

## Product direction

Represent geopolitical, economic, social, and narrative dynamics as **evidence on a graph over time**, with **interactive Q&A** that returns **ranked hypotheses**, **probabilities**, and **evidence** under an explicit **information cutoff**.

The **France protest** benchmark **validated** the pipeline and showed the GNN beats the main baselines there—it is a **reference eval**, not the product ceiling. **Ongoing France reruns are optional** (e.g. when ingestion or backtests change), not a requirement before work on markets or world-model tracks.

**Discovery claim:** slow-moving structure (regimes, coalitions, framings) should be **learned** where possible—**multi-step prediction**, **switching / mixture** models, **self-supervision**, **graph inference** with regularization—and **named** only after **stability / ablation** tests. **Do not** treat hand-coded historical priors as “discovered” without ablation.

---

## Validated benchmark target (France protest — reference, not gate)

> Given information visible at date `t`, forecast **event intensity** for the next 7 days by **location** and **event class** (France regional units).

The heterogeneous **GNN** beat recurrence and **XGBoost** on main **calibration / error** metrics for the documented holdout; ranking metrics remain a tuning dimension.

---

## Training curriculum: prediction markets

**Polymarket** (optional **Kalshi**, peers) are **training data**:

- resolutions and belief paths;
- cross-market structure where modeled.

Rules:

- **Frozen cutoffs** everywhere.
- **Masking ablations** when claiming **non-market** inference.
- If using **two heads** (e.g. terminal resolution + short-horizon dynamics), **label contracts** must not leak across heads.

---

## Success criteria

### Material tier

- Beat naive **recurrence** where applicable.
- Graph adds value vs **tabular** features from the **same** cutoff.
- Report **Brier**, **MAE**, ranking diagnostics, and **prevalence-aware** metrics (`positive_rate`, `pr_auc`, `balanced_accuracy`) when label density varies.
- **Ablations** for new nodes, edges, sources.
- **Reproducibility:** seeds and audits for GNN runs that drive decisions.

### Markets track (when active)

- Baselines: **market-implied**, naive, graph (or agreed stack).
- Slices: liquidity, domain, era.
- **Masked** runs for transfer claims to unlisted questions.
- **Coverage audit:** for each eval geography or domain, report whether **any** listed market exists; **do not** assume silent supervision where coverage is zero (listed markets are a **non-random**, interest-biased sample—see `roadmap.md` Stage 5, `docs/research/outputs/perplexity.md` §4).
- **PIT harness:** before trusting production resolution timestamps, run **adversarial tests on synthetic tapes** (injected future labels, retroactive corrections)—loaders and label builders must **fail closed** or alarm (`next_steps.md` Track M).

### Discovery / slow-structure track (research)

- **Held-out** time and geography; **multi-seed** stability.
- **Ablations:** removing latent / switching / graph-inference module hurts prediction when the module is claimed.
- Optional **post-hoc** alignment to named constructs—**not** the sole proof.

### Epistemological tier

- **Ranked hypotheses + evidence + uncertainty**—not one authoritative narrative.
- **HITL**, rubrics, or proxy evaluation—**not** fluency alone.

---

## Allowed inputs at prediction time (all as of `t`)

- Event records (GDELT-like, ACLED-like).
- Wikidata IDs/properties in the chosen snapshot.
- Source metadata and narrative clusters observed before `t`.
- Market quotes/metadata before `t` when in scope.

**Not allowed**

- Facts first observed after `t`.
- Future narrative clusters.
- Final resolutions before resolution time.
- Retrospective expert write-ups posed as contemporaneous evidence.

---

## Graph scope

Keep ontology **small but typed:** actors, locations, events, narratives, sources, markets. Add types only when a **metric** or **failure mode** requires them.

---

## Immediate work (sync with `next_steps.md`)

1. **Primary:** **training loop + WM v0** on **event** labels (time-then-space, multi-step losses, GRU/GNN ablations); then **prediction-market** ingestion with **adversarial PIT tests** and **coverage** reporting (`next_steps.md`, `roadmap.md` Stages 5–6).
2. **Parallel:** time-boxed charter/PIT doc updates; evidence expansion, grounding, metrics—as needed for the active eval.
3. **Constrained Q&A** prototype when retrieval + forecast stack exists.
4. **France benchmark:** rerun on a **cadence** or when **ingest/snapshot/backtest** contracts change—not a blocker on WM v0 or market ingest.

---

## Non-goals

- Universal ontology before measurement.
- Ungrounded open-ended QA.
- Epistemological accuracy claims without **HITL** or proxies.
- LLM as **sole** forecaster.
- Claiming **discovered** historical narratives from **fixed** priors in the core model without ablation.
