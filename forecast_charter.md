# Forecast Charter

## Product Direction

Build a temporally clean forecasting system that can model geopolitical,
economic, social, and narrative dynamics as graph-structured evidence over time.
The first validation benchmark used geopolitical contention, specifically
France protest events, because it provided a narrow event class with measurable
targets, source timestamps, locations, and enough recurrence structure to test
whether a GNN adds value.

That benchmark is not the product boundary. It is the proof point that the
temporal graph approach is worth expanding.

## Validated First Target

The first forecast primitive was:

> Given information visible at date `t`, forecast event intensity for the next
> 7 days by location and event class.

The initial tabular target was:

- `country_or_region`
- `event_class`
- `week_start`
- `event_count_next_7d`
- optional severity aggregate

The France protest benchmark has now validated the next modeling step: a
heterogeneous GNN beat recurrence and the XGBoost tabular baseline on the main
calibration/error metrics, with mixed ranking results. That is enough to justify
expanding sources and node types while keeping the same temporal-cleanliness and
backtest discipline.

## Success Criteria

A useful system must beat naive recurrence on rolling historical backtests, then
show that graph structure adds value beyond engineered tabular features.
Minimum metrics:

- Brier score or log loss for thresholded event occurrence
- ranking quality for top-risk locations
- calibration by event class and region
- comparison against recent-window recurrence
- ablations that show which node types, edge types, and source layers actually
  improve the forecast

## Allowed Inputs At Prediction Time

Every feature must be reconstructible as of date `t`.

Allowed early sources:

- event records from ACLED-like or GDELT-like feeds
- Wikidata IDs and aliases available in the chosen snapshot
- source metadata and article clusters observed before `t`
- market probabilities observed before `t`, when a matching market exists

Not allowed:

- labels or entity facts first observed after `t`
- future article clusters
- final market resolutions
- hand-authored retrospective macro explanations

## Active Graph Scope

Keep the ontology small, but expand it now that the first GNN benchmark has
earned the added complexity:

- actors
- locations
- events
- narratives
- markets
- sources

Add new node or edge types only when a concrete feature, label, or failure case
requires them.

## Immediate Work

1. Freeze the France protest benchmark as a regression suite for temporal
   hygiene, recurrence baselines, tabular baselines, and GNN performance.
2. Add a second event source, preferably ACLED, to test whether the GNN lift
   survives cleaner conflict/protest data and source disagreement.
3. Add canonical entity grounding with Wikidata snapshots for actors and
   locations visible as of date `t`.
4. Add actor, source, and narrative edge types only behind ablations that
   measure their contribution against the current GNN.
5. Start historical analog retrieval from graph neighborhoods and event
   trajectories once the expanded graph has stable backtest artifacts.

## Explicit Non-Goals

- no broad historical probe expansion
- no open-ended QA layer
- no agent-driven schema search
- no macro-force layer until added sources and graph features improve forecasts
