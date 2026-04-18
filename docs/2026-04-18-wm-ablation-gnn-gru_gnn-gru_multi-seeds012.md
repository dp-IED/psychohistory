# WM ablation: `gnn`, `gru_gnn`, `gru_multi` (seeds 0–2, France scaffold)

Date: 2026-04-18

This note records a **warehouse-backed** `baselines.wm_ablation_run` session: three variants, **three RNG seeds**, split learning rates for skeleton vs GNN paths, and **grad-norm logging** on GNN training. Program contracts: [`forecast_charter.md`](../forecast_charter.md), [`next_steps.md`](../next_steps.md).

## Command

```bash
python -m baselines.wm_ablation_run \
  --warehouse-path data/warehouse/events.duckdb \
  --seeds 0,1,2 \
  --variant gnn,gru_gnn,gru_multi \
  --lr-linear 0.01 \
  --lr-gnn 0.001 \
  --log-grad-norm-gnn \
  --epochs 5
```

The driver resolves `data/warehouse/events.duckdb` to the shared DuckDB (see below). One earlier attempt was interrupted during `import torch` (^C); the completed run used the command above.

**Output shape:** one **NDJSON line per seed** (`event: wm_ablation_seed`), then an **`wm_ablation_aggregate`** line on stdout.

## Data

| Item | Value |
| --- | --- |
| Warehouse (CLI) | `data/warehouse/events.duckdb` |
| Resolved (this machine) | `/Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb` |
| Rows loaded | 87,780 |
| Scoring universe (`admin1`) | 22 |
| Tabular features | 13 (`baselines.features.FEATURE_NAMES`) |
| Train masked rows | 2,288 (label prevalence 0.9060 on masked train rows) |
| Holdout masked rows | **1,144** (all variants / seeds) |
| Holdout label prevalence | ≈ 0.8741 |

**Splits** (`baselines/training_slice.py`, weekly Monday origins):

| Split | Start | End | Weeks |
| --- | --- | --- | ---: |
| Train | 2021-01-04 | 2022-12-26 | 104 |
| Holdout | 2023-01-02 | 2023-12-25 | 52 |

**Evaluation mask identity** (identical across seeds and variants for this run): `keys_sha256` = `0e2c40541160b5ef67f0993806bb13b4207d0282f9e35225b7db93d249a9723b` (`holdout_mask_identity` in JSON).

**Baselines** (same 1,144 holdout rows): `brier_always_positive` ≈ 0.125874, `brier_predict_train_prevalence` ≈ 0.111048.

## Training setup

| Parameter | Value |
| --- | --- |
| `variants` | `gnn`, `gru_gnn`, `gru_multi` |
| `seeds` | `0`, `1`, `2` |
| `epochs` | 5 (requested) |
| `lr-linear` | **0.01** (`linear` / **GRU** paths: `gru_multi` uses this) |
| `lr-gnn` | **0.001** (`gnn`, `gru_gnn`) |
| `batch_size` | 64 |
| `early_stop_patience` | 1 (linear and GNN; no separate override) |
| `history_weeks` | 8 |
| `device` | `cpu` |
| `log_grad_norm_gnn` | enabled |

Early stopping restores the **best holdout Brier** checkpoint. **`gru_multi`** is the multi-head occurrence GRU (primary 7d head + aux horizons; see `forecast_charter.md` WM skeleton section).

## Variants

| Variant | Meaning |
| --- | --- |
| `gnn` | Weekly graph snapshot → hetero GNN; `use_loc_temporal=False` |
| `gru_gnn` | Same stack with **location temporal** path: `use_loc_temporal=True` |
| `gru_multi` | `train_loop_skeleton` **GRU** path with `variant="gru_multi"` (multi-head / aux horizons) |

## Results: best holdout Brier by seed

Primary metric: **`best_holdout_brier`** after checkpoint selection (7d head for `gru_multi`).

| Seed | `gnn` | `gru_gnn` | `gru_multi` |
| ---: | ---: | ---: | ---: |
| 0 | 0.106132 | 0.155054 | 0.111885 |
| 1 | 0.125089 | 0.198920 | 0.110860 |
| 2 | 0.125414 | 0.168174 | 0.110003 |

## Aggregate (`wm_ablation_aggregate`)

| Variant | Mean Brier | Std (across seeds) | `collapse_count` (seeds with `collapse_detected`) |
| --- | ---: | ---: | ---: |
| `gnn` | 0.118879 | 0.009014 | 3 / 3 |
| `gru_gnn` | 0.174049 | 0.018384 | 0 / 3 |
| `gru_multi` | **0.110916** | **0.000769** | 0 / 3 |

(`collapse_detected` / `collapse_events` on **`gnn`** reflect `mean_holdout_prediction_saturated` warnings in logs; **`gru_gnn`** and **`gru_multi`** did not flag collapse in the emitted JSON.)

## Aux horizons (`gru_multi` only)

Example from seed 0 (`holdout_metrics_aux_horizons`): `brier_week2_7_14d` ≈ 0.115053, `brier_week4_21_28d` ≈ 0.114749. Other seeds are similar (see per-seed JSON lines).

## Observations (this run only)

- **`gru_multi`** has the **lowest mean Brier** and **lowest cross-seed variance** — stable under the three seeds.
- **`gnn`** achieves the **best single seed** (0.106 on seed 0) but **higher variance** and **collapse warnings on every seed**; mean Brier sits between `gru_multi` and naive baselines depending on seed.
- **`gru_gnn`** is **worse on mean holdout Brier** here than `gnn` / `gru_multi` with these LRs and early stopping; all three seeds ran full 5 epochs without early stop (`early_stopped: false` in JSON).

## Reproducibility

Re-run with the same CLI; warehouse byte-identical revision recommended. Seeds are passed explicitly so NDJSON + aggregate line are deterministic given the same code and data.
