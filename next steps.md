# Next Steps

## Current Position

France protest forecasting is now the validation benchmark, not the product
boundary.

The benchmark has served its purpose: it proved that the temporally clean graph
forecasting pipeline can beat recurrence and the XGBoost tabular baseline on the
main calibration/error metrics. The GNN still needs ranking ablations and tuning,
especially around recall@5, but the result is strong enough to justify expanding
the graph.

The product direction is broader:

- build a temporally clean graph forecaster over real-world historical evidence;
- add sources and node types only when they improve measured forecasts or explain
  concrete benchmark failures;
- preserve point-in-time reconstruction as the central engineering constraint;
- move toward historical analog retrieval and analyst-facing explanations after
  the expanded graph is stable.

## Immediate Engineering Sequence

1. Freeze the France protest benchmark as the regression harness.

   Keep the current GDELT France protest event tape, weekly graph snapshots,
   recurrence baselines, XGBoost baseline, and GNN backtest as the standard
   comparison suite. Every future source or graph expansion should be measured
   against this harness before being accepted.

2. Add a unified comparison audit.

   Produce one audit/table that compares recurrence, XGBoost, and GNN on the
   same holdout windows with the same metrics:

   - Brier score
   - MAE
   - top-5 hit rate
   - recall@5
   - positive counts
   - origin counts

3. Make GNN results reproducible.

   Add seed control to training and record the seed in the audit. Run multi-seed
   evaluations and report mean/std so the GNN lift is known to be stable rather
   than a single favorable run.

4. Add ablations before adding complexity.

   Measure which parts of the current graph actually matter:

   - location features only
   - event-location edges only
   - no `admin1_code_idx`
   - event attributes removed
   - actor edges added
   - source edges added
   - narrative features added

5. Add ACLED as the second event evidence layer.

   ACLED is the next source because it is cleaner and more structured for
   conflict/protest settings. The goal is not simply more data; the goal is to
   test whether the GNN lift survives source disagreement and a cleaner event
   ontology.

6. Add point-in-time Wikidata grounding.

   Ground actors and locations through Wikidata snapshots or versioned dumps
   visible as of date `t`. This should improve canonical identity without
   leaking future entity facts into historical forecasts.

7. Promote new node and edge types only behind measured wins.

   Candidate additions:

   - actor nodes
   - source nodes
   - narrative/topic nodes
   - actor-event edges
   - source-event edges
   - narrative-event or narrative-region edges

   Each addition should survive backtest comparison or explain a concrete error
   class before becoming part of the default graph.

8. Start historical analog retrieval after graph expansion stabilizes.

   Once the graph contains useful actor/source/narrative structure, build analog
   retrieval over:

   - local subgraph topology
   - event trajectories
   - narrative states
   - source disagreement
   - market disagreement later

   This is the path from classifier to analyst tool.

## Product Milestone

The next milestone is not "more protest forecasting."

The next milestone is a broader, temporally clean graph forecaster that proves
each added source and node type against the frozen France protest benchmark, then
uses the expanded graph to support historical analog retrieval and grounded
forecast explanations.

