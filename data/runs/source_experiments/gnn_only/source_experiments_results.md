# Source experiments (GNN-only) — results and notes

Run output directory: `data/runs/source_experiments/gnn_only/`  
Primary audit: `source_experiments.audit.json`  
Data warehouse: `data/warehouse/events.duckdb`

## Command (approximate)

```bash
python -m baselines.backtest source-experiments \
  --data-root …/chennai/data \
  --no-recurrence \
  --no-tabular \
  --snapshot-mode in-memory \
  --predictions-format jsonl.gz \
  --out-root …/data/runs/source_experiments/gnn_only \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --epochs 30 \
  --hidden-dim 64 \
  --progress
```

Recurrence and tabular baselines were **skipped**; only **`gnn_sage`** metrics are populated. Other model slots in the audit are zero-filled.

## Evaluation setup

- **Train origins:** 2021-01-04 through 2024-12-30 (weekly).  
- **Eval origins:** 2025-01-06 through 2025-12-29 (52 weeks).  
- **Regions:** 22 admin1 scoring units per week → **1,144** forecast rows per experiment (52 × 22).

## GNN (`gnn_sage`) metrics by experiment

Lower **Brier** and **MAE** are better. **Top-5 hit rate** and **recall@5** are ranking-style diagnostics.

| Experiment | Brier | MAE | Top-5 hit rate | Recall@5 | Positive labels (eval)\* |
|------------|-------|-----|----------------|----------|---------------------------|
| gdelt_only | 0.169 | 0.578 | 0.808 | 0.325 | 255 |
| acled_only | 0.610 | 2.170 | 0.288 | 0.244 | 308 |
| gdelt_plus_acled | 0.396 | 2.312 | 0.769 | 0.249 | 485 |
| gdelt_plus_acled_no_source_identity | 0.411 | 2.316 | 0.769 | 0.249 | 485 |
| gdelt_plus_acled_no_event_attributes | 0.467 | 2.348 | 0.731 | 0.229 | 485 |

\*From audit `positive_count`: count of positive-class labels in the eval slice for that experiment (differs by source filter).

## Event counts (filtered tape per experiment)

| Experiment | Events (`event_count`) |
|------------|-------------------------|
| gdelt_only | 51,101 |
| acled_only | 36,679 |
| Combined (three merged variants) | 87,780 |

## Interpretation

1. **GDELT-only is strongest on this run** — best calibration (Brier, MAE) and ranking (top-5), versus ACLED-only and merged configurations. Under this benchmark and model recipe, the **GDELT-only graph is the easiest to fit** and scores best on 2025 holdout weeks.

2. **ACLED-only is weak here** — much higher Brier/MAE and low top-5. That may reflect **coding/coverage differences**, **label construction**, or **calibration**, not necessarily intrinsic uselessness of ACLED; it does mean the current GNN pipeline does not extract a strong signal from ACLED alone on this task.

3. **Merging GDELT + ACLED does not beat GDELT-only** on these metrics; it sits between GDELT-only and ACLED-only on Brier. **Positive label counts differ** across experiments (e.g. 255 vs 485), so **absolute metrics are not strictly comparable** without accounting for **class balance**—but the overall ordering still suggests **adding ACLED does not improve** over GDELT-only in this setup.

4. **Collapsing source identity** (`gdelt_plus_acled_no_source_identity`) is **nearly identical** to `gdelt_plus_acled` → **explicit source nodes are not driving** the gap between merged and GDELT-only in this run.

5. **Zeroing event attributes** (`gdelt_plus_acled_no_event_attributes`) **hurts** vs the full combined model → **event-side attributes contribute** when both sources are present.

## Files per experiment

Under `gnn_only/<experiment_name>/`:

- `gnn_predictions.jsonl.gz` — holdout predictions  
- `gnn_predictions.jsonl.gz.audit.json` — per-run GNN audit (if present)

Recurrence/tabular prediction files were not written (`--no-recurrence`, `--no-tabular`).
