# Psychohistory Forecasting Project

## Purpose

Build a **temporally clean**, **graph-based forecasting system** whose primary technical bet is an **adaptive heterogeneous GNN** over a **single situation-graph schema**. The system should **predict and explain** under explicit information cutoffs: **ranked outcome hypotheses**, **probabilities**, and **grounded evidence**—not a generic chatbot.

France protest forecasting on GDELT was the **first controlled benchmark**: the heterogeneous **GNN beat the main baselines** there, so we do **not** need to keep re-proving that choice before advancing. That pipeline stays useful as an **optional smoke test** when touching ingestion, snapshots, or the event backtest stack—not a **gate** on every markets or world-model milestone.

---

## Goals (what “done” means)

1. **Replayable history:** For any date `t`, reconstruct inputs from evidence and markets **visible at or before `t`** (no leakage).
2. **Material forecasting:** On sharp targets (events, resolutions, prices where defined), match or beat **strong baselines** with **calibrated** probabilities; show **when graph structure helps** vs tabular features from the same cutoff.
3. **Market-informed training:** Use **Polymarket / Kalshi / peers** as **dense supervision** (resolutions, belief paths, cross-market structure) with **masking ablations** so the backbone is not only “price memorization.”
4. **Deploy beyond listed markets:** Support questions **not** listed as contracts; markets are **optional context**, not a hard requirement at inference for every query.
5. **Discovered structure (not baked history):** Prefer **learned** slow factors, **switching / mixture** dynamics, and **graph inference** that **earn their keep** on **held-out prediction** and **ablations**—not fixed priors on named historical events (see `roadmap.md` Stage 6).
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

**vs target architecture** (`docs/research/architecture.md`): **evidence, builder, and GNN encoder** are in code; a **named world-model module** and **query-lens module** are **not**—they are the main **research and execution-risk** pieces. **Ongoing:** **ACLED** and other sources, **reproducible audits**, **prediction-market ingestion** (schema TBD), **time-then-space WM v0** on event labels (train loop first), then market labels with **adversarial PIT harness** and **coverage** reporting.

---

## Repository map

| Path | Role |
|------|------|
| `forecast_charter.md` | Targets, metrics, inputs, non-goals |
| `roadmap.md` | **Full** program stages, gates, **what to avoid** |
| `next_steps.md` | **Actionable** near-term work order |
| `docs/reviewers-guide.md` | How to review claims, discovery protocol, red flags |
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

- **Gradients before grand design:** a **thin, correct training loop** on pinned event snapshots beats a perfect spec that never runs; extend `baselines/` and tests until **WM v0** trains end-to-end, then add markets (`next_steps.md` §0–2).
- **Measurement over ontology:** expand types only when a metric or failure mode demands it; **minus-variants** before new node types.
- **Parallel tracks:** **WM v0 on events** does not wait for Polymarket; markets add **label contracts**, **masking**, and **coverage audits** once the loop is real. Rerun the **France** reference when shared code changes—do not collapse the program into a single protest demo metric.
- **Discovery vs naming:** statistical structure first; **historical / expert naming** second, with independent checks.
