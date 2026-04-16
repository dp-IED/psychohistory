# Source-Layer Experiments

This workflow compares France protest forecasting runs across source layers:

- `gdelt_only`
- `acled_only`
- `gdelt_plus_acled`
- `gdelt_plus_acled_no_source_identity`
- `gdelt_plus_acled_no_event_attributes`

The preferred input is the central DuckDB warehouse. Legacy mixed JSONL tapes
still work with `--tape`, but they are compatibility artifacts, not the primary
store.

## Credentials

Do not store ACLED credentials in this repository or in any git workspace. Use a
user-level environment file with restrictive permissions:

```bash
mkdir -p ~/.config/psychohistory
chmod 700 ~/.config/psychohistory

cat > ~/.config/psychohistory/acled.env <<'EOF'
export ACLED_USERNAME='your-email@example.com'
export ACLED_EMAIL='your-email@example.com'
export ACLED_PASSWORD='<redacted>'
EOF

chmod 600 ~/.config/psychohistory/acled.env
```

Validate without printing the password:

```bash
source ~/.config/psychohistory/acled.env
test -n "$ACLED_USERNAME"
test -n "$ACLED_EMAIL"
test -n "$ACLED_PASSWORD"
```

## Warehouse Run

```bash
python -m baselines.backtest source-experiments \
  --data-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2 \
  --snapshot-mode in-memory \
  --predictions-format jsonl.gz \
  --out-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2/runs/source_experiments/full_2026_04_16 \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --epochs 30 \
  --hidden-dim 64
```

Run a subset:

```bash
python -m baselines.backtest source-experiments \
  --experiments gdelt_only gdelt_plus_acled \
  --snapshot-mode in-memory \
  --epochs 2 \
  --hidden-dim 16
```

## Legacy Tape Run

```bash
python -m baselines.backtest source-experiments \
  --tape data/mixed/tape/france_protest/events.jsonl \
  --out-root /Users/darenpalmer/conductor/shared-data/psychohistory-v2/runs/source_experiments/legacy_tape \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --epochs 30 \
  --hidden-dim 64
```

## Outputs

Predictions are compressed by default:

```text
<out-root>/<experiment>/recurrence_predictions.jsonl.gz
<out-root>/<experiment>/tabular_predictions.jsonl.gz
<out-root>/<experiment>/gnn_predictions.jsonl.gz
```

The combined audit is:

```text
<out-root>/source_experiments.audit.json
```

When `--snapshot-mode in-memory` is used, no snapshot directory is created.
When `--snapshot-mode materialize` is used, snapshots are written under one
directory per experiment, using `.json.gz` by default:

```text
<snapshots-root>/<experiment>/as_of_YYYY-MM-DD.json.gz
```

Generated snapshots, predictions, raw API outputs, and logs are ignored by git.
Do not commit credential files, bearer tokens, run logs, raw fragments, mixed
tapes, or generated benchmark artifacts.
