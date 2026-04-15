# Psychohistory Forecasting Project

## Purpose

Build a temporally clean forecasting system for geopolitical contention,
starting with protests and conflict escalation. The project should first prove
that a narrow, measurable forecast target can beat simple recurrence baselines
before adding graph neural networks, broad historical analogies, or open-ended
question answering.

## Current Scope

The active MVP is:

> Given information visible at date `t`, forecast event intensity for the next
> 7 days by location and event class.

The first graph vocabulary is intentionally small:

- actors
- locations
- events
- narratives
- markets
- sources

## Repository Shape

The repository now keeps only the pieces needed for the first implementation
loop:

- `forecast_charter.md` defines the first target, success metrics, allowed
  inputs, and non-goals.
- `roadmap.md` remains the broader program roadmap.
- `schemas/` contains the typed graph IR schema and loader.
- `evals/graph_artifact_contract.py` validates versioned graph artifacts.
- `evals/wikidata_linking.py` provides lightweight Wikidata grounding helpers.
- `tests/` verifies the retained schema, artifact, and grounding contracts.

The previous autoresearch harness, broad historical probe pack, and schema
search machinery were removed from the active project because they encouraged
ontology expansion before the forecasting target and temporal replay substrate
were proven.

## Next Engineering Step

Build an immutable event tape and snapshot exporter:

1. ingest dated event records from one real source;
2. retain source IDs, retrieval timestamps, locations, event classes, and actors
   where available;
3. emit daily or weekly `graph_artifact_v1` snapshots;
4. implement recurrence and recent-window baselines;
5. run rolling historical backtests.

Only add richer ontology, graph features, or model complexity when those
baselines expose a concrete failure that the added structure can plausibly fix.
