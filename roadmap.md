# Program roadmap

Single source of truth for **stages**, **decision gates**, **goals**, and **what to avoid**. **`project.md`** summarizes purpose; **`next_steps.md`** lists immediate execution order; **`forecast_charter.md`** locks eval contracts; **`docs/reviewers-guide.md`** defines review and discovery rigor.

### Execution reality

The **target** layered architecture (`docs/research/architecture.md`) is **intellectually coherent** but **ambitious** relative to current code: the **world model** (Stage 6) and **query lens** (precursor to Stage 8) are **not yet first-class modules**, while the **GNN backbone** (Stage 4) is validated. **Forward progress** should prioritize a **thin, correct training loop on existing event snapshots** (multi-step losses, ablatable time-then-space stack) **before** treating Polymarket ingestion as the main thread—markets then become **label contracts + coverage** on top of a **running** optimizer. **Do not** let open-ended charter or PIT documentation expand to fill the schedule **without** parallel training work (`next_steps.md` §0–2).

---

## North star

A system that:

1. **Forecasts** and **ranks** outcomes from **situation graphs** + time-varying evidence under **frozen cutoffs**.
2. **Trains** partly on **prediction markets** and other dense signals, but can **answer questions nothing lists as a contract**.
3. **Discovers** (or approximates) **slow-moving structure** via **neural estimators + multi-step losses + ablations**, instead of hard-coding named historical narratives as model priors.
4. **Explains** through **retrieved graph evidence** and **explicit uncertainty**, with **HITL** for the epistemological tier.

---

## Principles (non-negotiable)

| # | Principle |
|---|------------|
| P1 | **Forecastable ontology** — types must be applicable to real evidence by two annotators. |
| P2 | **Temporal cleanliness** — no post-`t` facts in inputs. |
| P3 | **Baselines first** — recurrence + tabular stay in every comparable audit. |
| P4 | **Graph earns complexity** — ablations, seeds, documented minus-variants. |
| P5 | **Markets supervise, don’t define the product** — masking / non-market queries matter. |
| P6 | **Discovery protocol** — named regimes and social-science labels require **post-hoc alignment** or probes, not baked-in switches, unless ablated as **optional** covariates. |
| P7 | **Q&A is constrained** — LLM routes/summarizes; structured stack does forecasting and evidence. |

---

## Stages

### Stage 0 — Charter

**Output:** `forecast_charter.md` (targets, horizons, metrics, non-goals).  
**Status:** baseline charter exists; revise as new tasks (markets, latents) land.

### Stage 1 — Small ontology, operationalized

Core classes: **actors, ideas/narratives, events, markets** (+ **locations**). “Ideas” attach only to **observable** evidence.  
**Gate:** sample hand-mapping with acceptable inter-annotator agreement.

### Stage 2 — Time-travel-safe replay

Immutable substrate: graph + tapes + (when in scope) **market state** visible by day `t`. Wikidata = identity layer; GDELT/ACLED = event layers; markets = **belief paths**, not world ground truth.  
**Gate:** arbitrary historical dates — **zero leakage** audit.

### Stage 3 — Non-neural baselines

Recurrence, seasonal patterns, tabular models on graph-derived features.  
**Status:** France harness established; **re-run** when task or graph changes.

### Stage 4 — Heterogeneous GNN (backbone)

Typed relations, temporal context, sparse regimes. **Status:** validated on the **France protest** benchmark vs baselines—treating further protest-only reruns as **optional** maintenance, not a prerequisite for Stages 5–6. **Requirements when extending the graph:** seeds, ablations for new node/edge types; **second event source** (e.g. ACLED) when multi-source evidence is in scope (`docs/source_layer_experiments.md`).  
**Gate:** new structure improves metrics **or** transfer **or** explains a **documented** failure mode.

### Stage 5 — Prediction-market training track

Ingest **Polymarket** (optional **Kalshi**, etc.): resolutions, price paths, metadata, cross-market links. Define **label contracts** (resolution vs short-horizon dynamics) so heads do not leak; short-horizon targets must be predictable from **state at $t$** only. Train with **frozen cutoffs**; run **market-feature masking** ablations and compare to a **no-market** training run to test generalization vs. contract-structure overfitting.  
**PIT de-risking:** resolution timestamps are **operationally messy** (delays, retroactive corrections)—run an **adversarial harness on synthetic tapes** before trusting production labels. **Coverage audit:** report which **domains or geographies lack any listed market** so evals do not **silently** assume supervision where none exists (listed markets are a **biased** sample of “interesting” questions).  
**Gate:** beats documented baselines on agreed slices; **masked** runs still usable for non-market eval.  
**Research note:** [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md) §4.

### Stage 6 — Learned slow structure & hypotheses (research)

**Intent:** latent **slow** factors + **switching / mixture** dynamics (world-model style) + optional **adaptive / latent graph** components—trained with **multi-step predictive losses**, self-supervision where useful, **sparsity / diversity** regularizers.  
**Implementation hints (not mandatory vendors):** **time-then-space** fusion (temporal GRU/SSM per node, then MP), exogenous heads for markets; **stochastic** structure losses where adjacency is learned (Manenti et al.); **GTGIB**-style bottlenecks for noisy temporal graphs; **query lens** with scored retrieval + **mask audit log** (GNN-RAG–style) and optional iterative expansion (RoE-style).  
**Rules:**

- **Do not** bake specific historical events (e.g. calendar shocks) as **fixed** train-time priors in the core model if the claim is “discovered structure”; use them only as **ablation covariates** or **post-hoc alignment**.
- **Do** report **stability** across seeds/splits and **ablations** (remove module → performance drops). **France harness:** run as **smoke** only if the WM code path still touches shared snapshot/backtest code—not as proof the WM is “good.”

**Gate:** held-out prediction gain **and** identifiable failure modes; expert naming optional and **separate**.  
**Research synthesis:** [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md).

### Stage 7 — Analog retrieval

Similar neighborhoods, trajectories, belief patterns — **before** open-ended user phrasing drives the core.  
**Gate:** human agreement that analogs are **plausibly comparable** above chance on a sample.

### Stage 8 — Constrained interactive Q&A

LLM **interprets** and **packages**; **retrieval + graph/world-model outputs + cutoff** ground answers. Faithfulness / historical QA audits on held-out questions.  
**Gate:** no systematic hallucinated evidence chains.

### Stage 9 — Cross-domain stress

One adjacent + one more distant domain; shared ontology **subset**; **per-domain** metrics. **Earlier shadow work:** a geography such as **Iran** may begin as a **red-team / ingest–lens–transfer–reviewer** slice **after** the GRU ablation driver is stable on the primary scaffold—**not** as the main optimization benchmark (`next_steps.md` **§2.3**); that split avoids collapsing the program into one contested domain before metrics and contracts are frozen.

### Stage 10 — Product identity

Choose flagship emphasis (forecasting vs analyst copilot vs disagreement analytics vs early warning) **after** evidence.

---

## Decision gates

| ID | Question |
|----|----------|
| G1 | World reconstructible as of `t` without leakage? |
| G2 | Do baselines justify the task for material targets? |
| G3 | Does the graph beat tabular features **from the same cutoff**? |
| G4 | With markets: does **masking** still yield a usable model for non-market queries (when claimed)? |
| G5 | For “discovered” structure: **ablations** and **held-out** time/regions support the claim? |
| G6 | Analog retrieval useful to humans (when in scope)? |
| G7 | Q&A grounded and temporally faithful? |
| G8 | Epistemological tier: **HITL** or proxy evaluation in place? |

---

## Goals (by layer)

| Layer | Goal |
|-------|------|
| **Engineering** | Green France harness; reproducible audits; warehouse path documented. |
| **Material forecasting** | Calibration + ranking on agreed slices; ablations published. |
| **Markets** | Ingestion + baselines + masking story; no single “beat close” vanity metric without context. |
| **Discovery** | Slow / switching latents or graph inference justified by prediction + ablations. |
| **Product** | Ranked hypotheses + evidence + cutoff; no solo-LLM forecasting. |

---

## What to avoid

### Architecture and research

- **Baking named history** (specific shocks, national narratives) as **immutable priors** in the core discovery model—then claiming the model “found” them. Prefer **learned** switches + **optional** covariate ablations.
- **Two opaque neural stages** (graph builder + forecaster) both trained **only** on one downstream loss **without** auxiliary objectives, masking, or staged training—credit assignment becomes unreviewable.
- **Universal ontology** before measurement or inter-annotator checks.
- **Ideas/trends** with no path to observables in the graph or text channels.

### Evaluation

- **Ranking-only** success without calibration where probabilities are claimed.
- **One global metric** across domains or eras when heterogeneity is expected.
- **Epistemological** claims validated by **fluency** only.
- **LLM as sole forecaster** or **evidence author** without structured retrieval and cutoff.

### Data and ethics

- Mixing **Wikidata / canonical truth** with **historically visible** evidence without an explicit rule.
- **Single national timeline** for all actors/communities when the product implies diversity of experience.
- **Stereotype-prone** group labels as primitives without mechanism-level care and bias review.

### Process

- Expanding `schemas/` or node types **without** a minus-variant or measurement plan.
- Skipping **reproducibility** (seeds, audit hashes) on GNN experiments that drive decisions.

---

## References (external)

- [Wikidata data access](https://www.wikidata.org/wiki/Wikidata%3AData_access)
- [ACLED API](https://acleddata.com/acled-api-documentation)
- [GDELT data](https://www.gdeltproject.org/data.html)
- [Polymarket docs](https://docs.polymarket.com/concepts/markets-events)

---

## Current summary

The **France GNN benchmark** established that the graph approach **earns its keep** on a clean protest forecast task. **Forward work** prioritizes a **working training loop** on **event** snapshots (time-then-space WM v0, multi-step losses, ablations), then **market-informed training** with **adversarial PIT tests** and **coverage reporting**, then **lens** and **constrained Q&A**; the France pipeline remains a **useful smoke test** when core ingestion or backtests change, not a perpetual validation gate. **Iran** (or similar) is planned as a **parallel stress-test lane** after ablations are real—**France (or agreed scaffold) stays the regression-controlled harness** until labeled eval contracts justify headline claims (**`next_steps.md` §2.3**).
