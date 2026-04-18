# WM ablation run (France scaffold, all base variants)

Date: 2026-04-18

This note records a single **warehouse-backed** run of `baselines.wm_ablation_run` with default France scaffold splits and training hyperparameters, for **linear**, **GRU**, **GNN**, and **GRU+GNN** (`gru_gnn`). Program direction and evaluation contracts live in [`forecast_charter.md`](../forecast_charter.md), [`next_steps.md`](../next_steps.md), and [`roadmap.md`](../roadmap.md).

## Command

Run from a repo checkout with `baselines` importable (same as other baseline CLIs):

```bash
python -m baselines.wm_ablation_run --warehouse-path data/warehouse/events.duckdb --variant all
```

No extra flags were passed; defaults below apply. The process resolved the warehouse to the shared DuckDB on disk (see Data).

**Approximate wall time:** about **2 hours** on the machine used (CPU-bound GNN graph builds and training).

## Data

| Item | Value |
| --- | --- |
| Warehouse (logical CLI path) | `data/warehouse/events.duckdb` |
| Resolved path (this run) | `/Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb` |
| Rows loaded | 87,780 |
| Scoring universe (French regional `admin1`) | 22 units |
| Tabular feature count | 13 (`baselines.features.FEATURE_NAMES`) |
| Train label prevalence (masked train rows) | 0.9060 (2,073 / 2,288) |
| Holdout masked rows (evaluation) | 1,144 |

**Weekly origin cadence (France scaffold defaults in `baselines/training_slice.py`):**

| Split | Start (inclusive) | End (inclusive) | Weeks |
| --- | --- | --- | ---: |
| Train origins | 2021-01-04 | 2022-12-26 | 104 |
| Holdout origins | 2023-01-02 | 2023-12-25 | 52 |

**Task:** binary `target_occurs_next_7d` (next-7-day protest occurrence per `(forecast_origin, admin1)`), aligned with the skeleton / WM charter (`forecast_charter.md`).

## Training setup (shared)

| Parameter | Value |
| --- | --- |
| `variants` | `all` → `linear`, `gru`, `gnn`, `gru_gnn` |
| `epochs` (requested) | 5 |
| `lr` | 0.01 (shared default for linear/GRU and for GNN paths when not split) |
| `batch_size` | 64 |
| `early_stop_patience` | 1 (same for GNN unless overridden) |
| `history_weeks` | 8 |
| `device` | `cpu` |
| `seed` | 0 (default) |
| Excluded admin1 | `ingest.snapshot_export.EXCLUDED_REGIONAL_ADMIN1_CODES` (same as other France runs) |

Early stopping restores the **best holdout Brier** checkpoint (not the last epoch).

## Model variants (what each run means)

| Variant | Implementation | Notes |
| --- | --- | --- |
| `linear` | Tabular features per weekly row | `train_loop_skeleton.run_linear_skeleton_cli`, `variant="linear"` |
| `gru` | Past-only weekly sequences per location | `variant="gru"`, `history_weeks=8` |
| `gnn` | Weekly heterogeneous graph snapshot → GNN head | `wm_ablation_train.run_wm_gnn_training`, **`use_loc_temporal=False`** |
| `gru_gnn` | Same GNN backbone with **location temporal** path enabled | **`use_loc_temporal=True`** |

## Results (best holdout Brier)

Holdout metrics use the **same 1,144 masked holdout rows** across variants. Baselines on that slice:

- `brier_always_positive` ≈ **0.125874**
- `brier_predict_train_prevalence` ≈ **0.111048**
- `holdout_prevalence` ≈ **0.874126**

| Variant | Best holdout Brier | Best epoch | Epochs run | Early stopped |
| --- | ---: | ---: | ---: | ---: |
| `linear` | 0.125874 | 1 | 2 | yes |
| `gru` | **0.109705** | 2 | 3 | yes |
| `gnn` | 0.125873 | 1 | 2 | yes |
| `gru_gnn` | 0.111173 | 3 | 4 | yes |

JSON excerpt (stdout summary array) matches these values; `gnn` row used `use_loc_temporal: false`, `gru_gnn` used `use_loc_temporal: true`.

## Observations

- **GRU** achieved the **lowest** holdout Brier in this configuration.
- **`gru_gnn`** improved substantially over **`gnn`** alone and sits between GRU and the linear/GNN-always-positive tier on Brier.
- **Plain `gnn`** (`use_loc_temporal=False`) **did not beat** the trivial always-positive baseline on best checkpoint Brier. Training logs show a **severe holdout Brier spike** on epoch 2 (~0.84) with **mean predicted probability collapsing** toward ~0.02, after which early stopping reverted to the epoch-1 checkpoint (near-baseline Brier). Treat this as **optimization instability / collapse behaviour** on this run—not a claim about GNN capacity in general.
- **Linear** matched the always-positive baseline at the restored best epoch (epoch 1 saturates predictions), then worsened on epoch 2 before early stop.

## Reproducibility

Re-run with the same CLI and warehouse revision. For multi-seed reporting and aggregation, use `--seeds` and the aggregate line emitted by the driver (`wm_ablation_aggregate` in `wm_ablation_run.py`).
