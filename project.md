# Psychohistory Forecasting Project

## Purpose

Build a temporally clean forecasting system over real-world historical evidence:
events, actors, locations, narratives, sources, and eventually market belief
signals. The product is not limited to protest forecasting. France protest
forecasting was the first controlled benchmark used to test whether the graph
forecasting approach earns its complexity.

That first gate has been crossed: the heterogeneous GNN now beats recurrence and
the tabular XGBoost baseline on the main calibration/error metrics for the 2025
holdout, with ranking still needing deeper ablation and tuning. The project can
now move from proving the method on one narrow event class to expanding the
evidence graph.

## Current Scope

The active MVP is a graph forecasting engine whose first validated task is:

> Given information visible at date `t`, forecast event intensity for the next
> 7 days by location and event class.

The current graph vocabulary is intentionally small but now eligible for
measured expansion:

- actors
- locations
- events
- narratives
- markets
- sources

## Repository Shape

The repository now keeps the pieces needed for the temporal forecasting loop:

- `forecast_charter.md` defines the first target, success metrics, allowed
  inputs, validation gate, and expansion rules.
- `roadmap.md` defines the broader program roadmap after the France protest
  benchmark.
- `schemas/` contains the typed graph IR schema and loader.
- `evals/graph_artifact_contract.py` validates versioned graph artifacts.
- `evals/wikidata_linking.py` provides lightweight Wikidata grounding helpers.
- `ingest/` builds point-in-time event tapes and graph snapshots.
- `baselines/` contains recurrence, tabular, and GNN backtests.
- `tests/` verifies schema, artifact, ingestion, and baseline contracts.

The previous autoresearch harness, broad historical probe pack, and schema
search machinery were removed from the active project because they encouraged
ontology expansion before the forecasting target and temporal replay substrate
were proven.

## Next Engineering Step

Freeze the France protest benchmark as the regression harness, then expand the
graph one layer at a time:

1. add a unified comparison audit for recurrence, XGBoost, and GNN on the same
   holdout windows;
2. add GNN seed control and ablations so improvements are reproducible;
3. ingest ACLED as a second event evidence layer;
4. ground actors and locations through point-in-time Wikidata snapshots;
5. add actor, source, and narrative nodes only when each addition improves
   backtest metrics or explains a concrete failure mode.

The next product milestone is not "more protest forecasting." It is a broader,
temporally clean graph forecaster that proves each added source and node type
against the frozen benchmark before moving toward analog retrieval and analyst
Q&A.
