# Deep research handoff

Use this file (plus `docs/research/architecture.md`) as **primary context** for a capable research model. **Paste the model’s outputs** back under `docs/research/`—for example `docs/research/outputs/<topic>-<date>.md` or a subfolder you create—so findings stay next to the target design.

---

## What to give the research model

1. **This file** (`docs/research/research.md`) — conversation summary and open questions below.  
2. **`docs/research/architecture.md`** — target layered architecture (evidence → situation → lens → encoder → world model → heads → Q&A).  
3. **`docs/graph-builder-contract-v0.1.md`** — locked builder retrieval, supervision stages, assumption gates, compute limits, training vs validation contexts.  
4. Optionally: `project.md`, `roadmap.md`, `docs/reviewers-guide.md` for program gates and what to avoid.

---

## Conversation summary (context for the AI)

### Project aim

Build a **temporally clean**, **graph-informed** forecasting and analysis system whose long-term bet is **relational evidence + latent temporal dynamics** (a **world model**), not a generic RAG or chat wrapper. **Psychohistory-style** product: **ranked hypotheses**, **probabilities**, **evidence**, explicit **information cutoffs**—not unconstrained punditry.

### What was rejected or deprioritized

- **Perplexity-style** product as the center of value: retrieval + fluency without a disciplined forecast object.  
- **Only** ingesting more news/event sources as the path to MVP without a clear **eval** and **model** thesis.  
- **Two jointly trained black-box stages** (learned graph builder + forecaster) optimized on **one** downstream loss without auxiliary losses, masking, or staged training—**credit assignment** and reviewability suffer.  
- **Baking named history** (specific shocks, national storylines) into **core** train-time priors while claiming **unsupervised discovery** of structure.

### Agreed product and training shape

- **Prediction markets (Polymarket; Kalshi optional)** as **dense training signal**—resolutions, belief paths, cross-market structure—under **frozen cutoffs** (for **forecaster / Stage 3**; they **do not** supervise the graph builder—[`docs/graph-builder-contract-v0.1.md`](../graph-builder-contract-v0.1.md)).  
- **Deployment** includes questions **never listed** as markets; **market features are optional** at inference; **masking ablations** validate non-market use.  
- **Dual material objectives** discussed: terminal outcomes + short-horizon belief dynamics; **human-facing “as of \(t\)”** explanation matches the **training cutoff regime**.  
- **Epistemological tier:** competing interpretations / norms / framings as **ranked hypotheses + evidence + uncertainty**—**HITL** or rubrics for trustworthy iteration, not fluency alone.

### Architecture direction (target, not only current code)

**Layered design:** **Evidence store** → **canonical situation \(S_t\)** (rules + optional learned completion) → **query lens** \(S_t^{(q)}\) (focus without forging facts) → **relational encoder** (GNN or successor) → **world model** (\(z_t\), transitions, mixture over experts, emissions to observables) → **task heads** (material, market curriculum, epistemological) → **constrained Q&A façade** (LLM routes/packages, does not solo-forecast).

**World model** is the **temporal** centerpiece for multi-step structure and hypothesis competition; **GNN/graph encoder** is the **relational** centerpiece for situation structure—they **compose**, not compete. Current repo emphasizes **baselines + hetero GNN on snapshots** as a **regression harness**; the **research** target elevates the **world model** relative to snapshot-only.

### Interpretability and “discovery”

- Some latents may **align** to social-science constructs via probes; others **discovered** then named **post hoc**.  
- **Rigor:** held-out prediction, **ablations**, **multi-seed** stability; optional **covariate ablations** comparing learned switches to hand-coded shocks.  
- **Historian-style critique** acknowledged: avoid **presentism**, **teleological** shock coding, **elite-only** periodization, thin evidence for ideational claims—document **coverage limits**.

### Literature threads (for deeper paper chase)

- **Adaptive / latent graph learning** for series (e.g. MTGNN line, discrete structure learning, **DGM**, surveys: Jin et al. GNN4TS TPAMI, Kumar ST-GNN SLR, dynamic GNN survey).  
- **Self-supervised temporal / dynamic graphs** (e.g. DySubC, GraphTNC, S2T, CLDG, contrastive + STG forecasting).  
- **World models** (RSSM / Dreamer lineage) for multi-step latent dynamics—**combine** with graph encoding rather than replace.

### Docs already in repo (engineering program)

`project.md`, `roadmap.md`, `next_steps.md`, `forecast_charter.md`, `docs/graph-builder-contract-v0.1.md`, `docs/reviewers-guide.md`, `README.md`—program gates, goals, what to avoid, discovery protocol, builder contract.

---

## Suggested research questions for the powerful AI

Prioritize **survey + gaps + concrete architectures + evaluation protocols**:

1. **Graph encoder ↔ world model:** best practices for **fusing** heterogeneous graph embeddings with **latent transition** models (RSSM-style, SSM, neural SDEs); **multi-horizon** losses and **identifiability**.  
2. **Learned graph structure** under **PIT constraints:** latent graph inference + regularization; failure modes (shortcut learning, dense correlation soup).  
3. **Query-conditioned** graph views: attention vs subgraph retrieval vs gated message passing—**faithfulness** and **no hidden evidence dropping**.  
4. **Prediction market** integration: label contracts, masking, baselines (market-implied), **selection bias**.  
5. **Epistemological / interpretive** outputs without hand-labeling everything: **mixture-of-experts** over dynamics, **probe-based** alignment, **HITL** design.  
6. **Bias and ethics:** surveillance risk, reification of groups, **multi-community** time—not single national clock.

---

## Where to write results

- Prefer `docs/research/outputs/` (create if needed) with descriptive filenames, e.g. `2026-04-17-world-model-graph-fusion.md`.  
- Link new files from this document or from `docs/research/architecture.md` **References / further reading** when stable.

### Outputs received

| File | Summary |
|------|---------|
| [`outputs/perplexity.md`](outputs/perplexity.md) | Deep research across six themes: Graph-RSSM / Graph Dreamer fusion, PIT-safe learned graphs (GTGIB, stochastic adjacency), query lens (GNN-RAG + logged masks), prediction markets (label contracts, bias), epistemological MoE + probes + HITL, bias/multi-community time; integration table + eval protocol. |
