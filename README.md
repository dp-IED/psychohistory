# Psychohistory

Temporally clean **graph forecasting** with **query-conditioned subgraphs + assumptions** upstream of the encoder/WM; (roadmap) **market-informed training**, **learned slow structure**, and **constrained Q&A**.

## Start here

| Document | Purpose |
|----------|---------|
| [`project.md`](project.md) | Purpose, **goals**, repository map |
| [`roadmap.md`](roadmap.md) | **Full** program stages, gates, **what to avoid** |
| [`next_steps.md`](next_steps.md) | **Actionable** work order (gradients-first sequencing) |
| [`forecast_charter.md`](forecast_charter.md) | Metrics, inputs, non-goals |
| [`docs/reviewers-guide.md`](docs/reviewers-guide.md) | Reviews, discovery protocol, red flags |
| [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md) | **Locked** graph builder, assumptions, supervision stages, compute & training contexts |
| [`docs/research/architecture.md`](docs/research/architecture.md) | **Target** layered architecture (evidence → builder/lens → encoder/WM; not code-bound) |
| [`docs/research/research.md`](docs/research/research.md) | Deep-research handoff + conversation summary |

## Engineering

- **Data:** [`docs/storage_architecture.md`](docs/storage_architecture.md)
- **France benchmark / source ablations:** [`docs/source_layer_experiments.md`](docs/source_layer_experiments.md), [`docs/2026-04-16-france-gdelt-benchmark-note.md`](docs/2026-04-16-france-gdelt-benchmark-note.md)
- **Wikidata grounding:** [`docs/wikidata/00-overview.md`](docs/wikidata/00-overview.md)
