# Forecast Charter

## First Target

Build the first version around geopolitical contention, protests, and conflict
escalation. The first forecast primitive is:

> Given information visible at date `t`, forecast event intensity for the next
> 7 days by location and event class.

Start with a tabular target such as:

- `country_or_region`
- `event_class`
- `week_start`
- `event_count_next_7d`
- optional severity aggregate

Actor-relation-target forecasting can come later, after the event tape and
rolling backtest are trustworthy.

## Success Criteria

The first useful system beats naive recurrence on rolling historical backtests.
Minimum metrics:

- Brier score or log loss for thresholded event occurrence
- ranking quality for top-risk locations
- calibration by event class and region
- comparison against recent-window recurrence

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

Keep the ontology small:

- actors
- locations
- events
- narratives
- markets
- sources

Add new node or edge types only when a concrete feature, label, or failure case
requires them.

## Immediate Work

1. Build an immutable event tape with source IDs, event dates, locations, event
   classes, actors where available, and retrieval timestamps.
2. Export daily or weekly `graph_artifact_v1` snapshots.
3. Implement recurrence and recent-window baselines.
4. Run rolling backtests over multiple historical windows.
5. Use baseline failures to decide whether richer graph features are justified.

## Explicit Non-Goals

- no full GNN yet
- no broad historical probe expansion
- no open-ended QA layer
- no agent-driven schema search
- no macro-force layer until measured features improve forecasts
