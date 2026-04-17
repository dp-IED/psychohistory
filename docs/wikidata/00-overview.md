# Wikidata roadmap — overview

This folder splits the Wikidata workstream into **independent plans**. Each plan is self-contained: goal, design, **ablation matrix**, risks, and dependencies. Implement and benchmark one track at a time; shared metrics keep results comparable to the France protest benchmark ([`docs/2026-04-16-france-gdelt-benchmark-note.md`](../2026-04-16-france-gdelt-benchmark-note.md)).

## Recommended order

```text
01 PIT grounding (reproducibility)  →  02 QID input features (first model lift)
         ↓                                      ↓
03 Actor hetero GNN (optional)        04 Properties / hierarchy (optional, heavier)
         ↓
05 Analog retrieval & explainability (mostly offline / product)
```

- **01** does not require model changes; it hardens experiments.
- **02** is the **smallest path** to measurable GNN impact (today’s [`baselines/gnn.py`](../../baselines/gnn.py) ignores QIDs).
- **03–04** increase graph and feature complexity; justify each with ablations.
- **05** can proceed in parallel once QIDs are stable; retrieval need not block forecasting accuracy work.

## Shared evaluation contract

Use the same task definition unless a plan explicitly scopes a smoke subset:

- Weekly Monday origins, 7-day horizon, admin1 France regional units.
- Holdout and metrics aligned with [`baselines/metrics.py`](../../baselines/metrics.py) and source-experiment audits: Brier, MAE, recall@5, top-5 hit rate, plus **`positive_rate`**, **`pr_auc`**, **`balanced_accuracy`** where applicable ([`docs/superpowers/plans/2026-04-17-source-experiments-metric-comparability.md`](../superpowers/plans/2026-04-17-source-experiments-metric-comparability.md)).

## Plan index

| File | Topic |
|------|--------|
| [`01-point-in-time-grounding.md`](01-point-in-time-grounding.md) | Reproducible grounding: API vs dumps, cache policy, audit |
| [`02-qid-input-features.md`](02-qid-input-features.md) | QID-derived vectors for the **existing** Location/Event GNN inputs |
| [`03-actor-hetero-gnn.md`](03-actor-hetero-gnn.md) | Actor node type, edges, QID-based merge |
| [`04-wikidata-properties-hierarchy.md`](04-wikidata-properties-hierarchy.md) | Structured properties, hierarchy edges, leakage controls |
| [`05-analog-retrieval-explainability.md`](05-analog-retrieval-explainability.md) | Subgraph retrieval, explanations; mostly non-training |

## Cross-cutting ablation principle

Every new feature ships with a **minus variant** (same graph and training recipe, feature off or zeroed) so lift is attributable. Prefer **one knob per PR** when possible.
