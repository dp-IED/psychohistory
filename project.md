# Psychohistory Forecasting Project

## Purpose

Build a **temporally clean**, **graph-based forecasting system** whose primary technical bet is **learned query-conditioned subgraph construction** (retriever / builder + query lens) and **explicit latent assumptions** over evidence, with a **heterogeneous GNN** and **world-model-style temporal core** as **downstream encoders and dynamics** on that interface—not as the sole research frontier. The system should **predict and explain** under explicit information cutoffs: **ranked outcome hypotheses**, **probabilities**, and **grounded evidence**—not a generic chatbot.

France protest forecasting on GDELT was the **first controlled benchmark**: the heterogeneous **GNN beat the main baselines** there, so we do **not** need to keep re-proving that choice before advancing. That pipeline stays useful as an **optional smoke test** when touching ingestion, snapshots, or the event backtest stack—not a **gate** on every markets or world-model milestone. For the **learned graph builder**, France is a **validation harness**, not a primary **training** corpus—see [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md).

---

## Goals (what “done” means)

1. **Replayable history:** For any date `t`, reconstruct inputs from evidence and markets **visible at or before `t`** (no leakage).
2. **Material forecasting:** On sharp targets (events, resolutions, prices where defined), match or beat **strong baselines** with **calibrated** probabilities; show **when graph structure helps** vs tabular features from the same cutoff.
3. **Market-informed training:** Use **Polymarket / Kalshi / peers** as **dense supervision** (resolutions, belief paths, cross-market structure) with **masking ablations** so the backbone is not only “price memorization.”
4. **Deploy beyond listed markets:** Support questions **not** listed as contracts; markets are **optional context**, not a hard requirement at inference for every query.
5. **Discovered structure (not baked history):** Prefer **learned** slow factors, **switching / mixture** dynamics, **query-conditioned subgraphs**, **assumption proposal and belief updates**, and **graph inference** over the **retrieval hierarchy** in [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md) (**actor-state**, **trend thread**, **historical analogue**—v1 implements **actor-state** only with **slots** for the other two) that **earn their keep** on **held-out prediction** and **ablations**—not fixed priors on named historical events (see `roadmap.md` Stage 6).
6. **Epistemological tier (interpretations):** **Ranked hypotheses + evidence + uncertainty**; improvement judged by **HITL**, rubrics, or proxy tasks—not fluency alone.
7. **Constrained Q&A:** Natural language that **routes** to **retrieval + forecasts + cutoff**; the LLM does **not** forecast alone.

---

## Product surface

- **Interactive Q&A** (psychohistory-style): hypotheses, probabilities, evidence, limits.
- **Material + epistemological** outputs where scoped; epistemological claims require explicit evaluation discipline (`docs/reviewers-guide.md`).

---

## Current codebase (engineering reality)

- **DuckDB warehouse** + JSONL compatibility; weekly **graph snapshots** as of `t`.
- **Baselines:** recurrence, tabular (XGBoost), heterogeneous **GNN** (`baselines/`).
- **Contracts:** `evals/graph_artifact_contract.py`, metrics in `baselines/metrics.py`.
- **Grounding:** Wikidata / linking workstreams under `docs/wikidata/`.

**vs target architecture** (`docs/research/architecture.md`): **evidence**, **deterministic snapshot builder**, and **GNN encoder** are in code; a **learned graph builder / retriever**, **query lens**, **assumption layer**, and **named world-model module** are **not** first-class—together they are the main **research and execution-risk** pieces. **Locked builder design** (budgets, supervision stages, assumption gates, training contexts): [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md). **Ongoing:** implement that contract; **time-then-space WM v0** and multi-step losses **on the stabilized subgraph + assumption interface**; **Polymarket / Kalshi** for **Stage 3** forecaster calibration only (they **do not** supervise the builder); in parallel **reproducible audits**, **prediction-market ingestion** (schema TBD) with **adversarial PIT harness** and **coverage** reporting.

---

## Repository map

| Path | Role |
|------|------|
| `forecast_charter.md` | Targets, metrics, inputs, non-goals |
| `roadmap.md` | **Full** program stages, gates, **what to avoid** |
| `next_steps.md` | **Actionable** near-term work order |
| `docs/reviewers-guide.md` | How to review claims, discovery protocol, red flags |
| `docs/graph-builder-contract-v0.1.md` | **Locked** retrieval hierarchy, builder I/O, supervision stages, assumption gates, compute & training contexts |
| `docs/research/architecture.md` | **Target** layered architecture (evidence → world model → heads); not code-bound |
| `docs/research/research.md` | Deep-research handoff and conversation summary |
| `docs/storage_architecture.md` | Shared warehouse, symlinks |
| `docs/source_layer_experiments.md` | France harness source ablations |
| `schemas/` | Typed graph IR |
| `ingest/` | Tapes, warehouse, snapshot export |
| `baselines/` | Backtests and models |
| `tests/` | Contracts |

---

## Strategic themes (tie-breakers)

- **Gradients before grand design:** a **thin, correct training loop** on pinned event snapshots beats a perfect spec that never runs; the first loop should prove **builder / subgraph + assumption** artifacts and contracts, then attach a **minimal forecast head**, then extend **`baselines/`** until **WM v0** (time-then-space) trains on that interface—then add markets (`next_steps.md` §0–2).
- **Measurement over ontology:** expand types only when a metric or failure mode demands it; **minus-variants** before new node types.
- **Parallel tracks:** **Graph-builder and assumption work** trains on **GDELT + ACLED** staged supervision per **contract**; **Polymarket / Kalshi** are **Stage 3** for the **forecaster**, not builder supervision. **WM v0** is **sequenced after** a reproducible subgraph + assumption baseline so temporal ablations are not confounded with an unset retrieval target. Markets add **label contracts**, **masking**, and **coverage audits** once the event spine through WM is real. Rerun the **France** harness when shared code changes for **smoke / validation**—do not treat France as the primary **builder training** context or collapse the program into a single protest demo metric. **Iran** (or similar contested geographies) is a **stress-test / shadow** lane after the **WM + encoder ablation** spine is stable on the primary scaffold—not the primary benchmark that drives iteration until baselines and eval contracts are frozen (**`next_steps.md` §2.3**).
- **Discovery vs naming:** statistical structure first; **historical / expert naming** second, with independent checks.
