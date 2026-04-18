# France GDELT Benchmark Note

Date: 2026-04-16

This note **freezes** the France protest benchmark for regression testing. For **program direction**, **full roadmap**, **next steps**, and review criteria, see [`project.md`](../project.md), [`roadmap.md`](../roadmap.md), [`next_steps.md`](../next_steps.md), and [`reviewers-guide.md`](reviewers-guide.md).

## Scope

This note freezes the current France protest benchmark before moving to a
second event evidence layer. The benchmark uses the temporally clean GDELT event
tape and weekly graph snapshots for regional France protest forecasting.

Forecast task:

- origin cadence: weekly Mondays
- train origins: 2021-01-04 to 2024-12-30
- eval origins: 2025-01-06 to 2025-12-29
- horizon: next 7 days
- prediction unit: French regional `admin1_code`
- admin1 count: 22

Training configuration for the current GNN run:

- model family: `gnn_sage`
- epochs: 30
- hidden dimension: 64
- eval rows across ablations: 6864
- audit path: `data/gdelt/baselines/france_protest/gnn_ablation_predictions.audit.json`

## Current Result

The current full graph GNN remains the default model. It beats the stripped GNN
variants on Brier score, MAE, and recall@5, and the project documentation
already records that the GNN beats recurrence and XGBoost on the main
calibration/error metrics for the 2025 holdout.

| Variant | Brier | MAE | Top-5 Hit Rate | Recall@5 | Delta Brier vs Full | Delta Recall@5 vs Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_graph` | 0.165814 | 0.580501 | 0.769231 | 0.313725 | +0.000000 | +0.000000 |
| `location_features_only` | 0.170080 | 0.591568 | 0.788462 | 0.305882 | +0.004265 | -0.007843 |
| `event_layer_only` | 0.174594 | 0.610884 | 0.692308 | 0.250980 | +0.008780 | -0.062745 |
| `no_event_features` | 0.169198 | 0.584838 | 0.769231 | 0.286275 | +0.003384 | -0.027451 |
| `no_event_edges` | 0.169975 | 0.592242 | 0.692308 | 0.278431 | +0.004160 | -0.035294 |
| `no_location_features` | 0.174454 | 0.609806 | 0.692308 | 0.250980 | +0.008639 | -0.062745 |

## Interpretation

The result is strong enough to stop spending major compute on GNN micro-ablations.

The main read:

- `full_graph` is best on Brier, MAE, and recall@5.
- Location/history features still carry most of the signal.
- Event-location message passing adds useful calibration and positive-capture
  lift.
- Removing event features hurts less than removing event edges, which suggests
  topology/message passing matters more than the raw event attribute vector.
- Top-5 hit rate is noisy because it is binary per origin. Recall@5 is the more
  useful ranking metric for this benchmark.

Two caveats matter for future reporting:

- `event_layer_only` and `no_location_features` are duplicate configurations in
  the current ablation flag set.
- `location_features_only` and `no_event_edges` are functionally the same model
  class in the current architecture because disconnected event nodes cannot
  influence location logits. Differences between them should be treated as
  stochastic training noise.

The GNN ablation harness is useful as a smoke/regression tool, but it is not the
next high-value research direction.

## Data Caveat

The underlying GDELT fetch was partial: 91 source files failed out of roughly
250,000 selected files. This is small relative to the full fetch, but any
published or shared benchmark summary should disclose the partial fetch and
record the exact fetch audit metadata alongside model results.

## Next Work

Move from model-internal ablations to source-layer validation.

The next milestone should be ACLED as a second event evidence layer:

1. Add ACLED ingestion into the normalized event tape shape.
2. Preserve source identity in snapshots with separate `Source` nodes and
   source-aware event metadata.
3. Run source-layer experiments:
   - `gdelt_only`
   - `acled_only`
   - `gdelt_plus_acled`
   - `gdelt_plus_acled_no_source_identity`
   - `gdelt_plus_acled_no_event_attributes`
4. Add Wikidata grounding after ACLED creates real actor/location reconciliation
   pressure across sources.
5. Defer historical analog retrieval until the graph has stable multi-source
   evidence.

Keep the France/GDELT benchmark as the regression harness while expanding the
evidence graph one source at a time.
