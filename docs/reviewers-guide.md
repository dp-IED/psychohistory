# Reviewer’s guide

What must be true for results and demos to **count**. Complements **`roadmap.md`** (gates, what to avoid) and **`forecast_charter.md`** (metrics and inputs).

---

## What this project is trying to be

An **adaptive graph forecaster**: **query-conditioned subgraphs** and **explicit assumptions** feed heterogeneous GNN(s) and **world-model-style** temporal cores on a **single situation-graph schema**, capturing **slow-moving structure** that manifests in **faster observables** (events, text proxies, markets)—without requiring every latent to be pre-named.

**Product shape:** interactive Q&A with **ranked hypotheses**, **probabilities**, **evidence**, and a **cutoff**—not unconstrained punditry.

---

## What is not claimed by default

- Raw latents equal psychoanalysis / sociology / historiography constructs **without** alignment probes or ablations.
- Beating **closing** market prices as the sole headline metric.
- LLM **fluency** as epistemological correctness.
- **Hand-coded** historical narratives in the **core** learning objective passed off as **unsupervised discovery** (see Discovery protocol).

---

## Core design commitments

### One graph schema

One vocabulary for training and deployment. **Markets optional** at inference for many questions; they remain **training signal** and **context** when present.

### Markets as supervision

Dense supervision from **Polymarket / Kalshi / peers**; **frozen cutoffs**; **masking** ablations for non-market claims. Report **with** and **without** market features when transfer is asserted.

### Material vs epistemological tiers

- **Material:** sharp observables; standard calibration metrics.
- **Epistemological:** ranked interpretations + evidence + uncertainty; **HITL** or rubrics required for trustworthy iteration.

### Discovery protocol (slow structure / regimes)

If the claim is that the model **uncovers** structure (regimes, switches, relational rewiring):

1. **Inductive bias** may be generic (mixtures, switching, sparsity)—not specific dated events as **immutable** train priors.
2. **Evidence** must include **held-out** time/regions and **multi-seed** stability.
3. **Ablations:** removing the module (latent, switch, graph learner) should **hurt** when the module is credited.
4. **Naming** (e.g. “post-9/11 framing”) is **post-hoc** or **small-sample** alignment—not proof of discovery alone.
5. Optional **covariate ablations:** compare learned switches to **hand-added** shock flags to show what is **data-driven** vs **ornamental**.

### Interactive Q&A

Answers decompose into **retrieved graph evidence**, **model outputs**, **cutoff**, **uncertainty**—no fabricated citations or post-cutoff facts.

---

## How to review a change

1. **Leakage** — inputs only as of stated `t`.
2. **Baselines** — recurrence, tabular, market-implied as appropriate.
3. **Ablations** — minus-variants for new structure.
4. **Metrics match claims** — Brier/log loss vs rubric/HITL.
5. **Scope** — per-domain / per-era slices for heterogeneous claims.
6. **Discovery** — ablations + held-out + seeds if structure is claimed.

---

## Regression harness (France protest — reference)

The France / GDELT pipeline is a **validated reference** for temporal hygiene and GNN vs baselines. **It is not the only eval:** markets and world-model work should be judged on **their own** tasks and protocols (`docs/research/outputs/perplexity.md`). Re-running France is **recommended** when shared ingestion or backtest code changes, not for every unrelated feature.

---

## Red flags

- Ontology expansion without a **measurement plan** or minus-variant.
- Epistemological outputs without **HITL** or proxies.
- Q&A without retrieval + cutoff discipline.
- Embedding similarity sold as **causal** mechanism without entity/topic checks.
- **Fixed** historical priors in the core model + claim of **pure discovery**.
- Single national timeline for all communities when the question is inherently **multi-experience**.

---

## Where to read next

| File | Content |
|------|---------|
| `project.md` | Purpose, goals, repo map |
| `roadmap.md` | Full stages, gates, **what to avoid** |
| `next_steps.md` | Actionable work order |
| `docs/graph-builder-contract-v0.1.md` | Locked builder retrieval, supervision, gates, compute, training contexts |
| `forecast_charter.md` | Metrics, inputs, non-goals |
| `docs/storage_architecture.md` | Warehouse layout |
| `docs/source_layer_experiments.md` | France source ablations |
| `docs/wikidata/00-overview.md` | Grounding workstream |
| `docs/research/architecture.md` | Target layered architecture (research) |
| `docs/research/research.md` | Deep-research handoff + conversation summary |
