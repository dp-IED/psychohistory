# Gap 3: Seed30 Full Run — All Domains

## Purpose

The current catalog has only 13 probes (12 with market anchors), all geopolitics-filtered. The gold dataset at `data/polymarket/gold_branch_dataset.json` contains 30 cases across 3 domains (10 politics/institutional, 10 economics/macro, 10 culture/entertainment). The `seed_from_gold_dataset()` function in `pit_market_probe.py` currently hardcodes a geopolitics-only filter via `_is_geopolitics_slug()`. This spec removes that filter and runs the full set.

## What to Build

### 1. Domain-Unfiltered Seeding

Modify `seed_from_gold_dataset()` in `harness/pit_market_probe.py`:

- Add a `domain_filter: str | None = None` parameter. When `None`, include ALL cases (no geopolitics filter).
- Add a `--domain` option to the seed subcommand: `--domain geopolitics` (default, current behavior), `--domain all` (no filter), `--domain economics`, `--domain culture`.
- Assign the `domain` field on each probe spec: classify from the gold dataset's metadata or infer from the full text/question keywords:
  - "geopolitics": keywords like "ceasefire", "war", "strike", "sanction", "election", "president"
  - "economics": keywords like "fed", "fomc", "interest", "bps", "rate", "gdp", "inflation", "unemployment"
  - "culture": keywords like "grammy", "billboard", "netflix", "oscar", "tennis", "nba", "nfl", "box office", "spotify"

### 2. Domain-Aware Probes

The probes from non-geopolitics domains need:
- `kind: "market_anchor"` (same as geopolitics)
- Proper `clob_yes_token_id` from the gold dataset's `clob_token_ids` or `clobTokenIds` field
- `domain` set to the correct category

### 3. Run Subcommand Support for Domain Filtering

Add `--domain` to the `run` subcommand too:
```
python scripts/pit_market_calibration.py run --domain economics --max-probes 5 --skip-existing
```
This only runs probes with `domain == "economics"`.

### 4. Scoring by Domain

Modify `summarize_results()` to report per-domain metrics:
```python
{
  "n_total": 30,
  "n_with_market": 30,
  "mean_market_abs_error": 0.037,
  "by_domain": {
    "geopolitics": {"n": 12, "mean_mae": 0.015, "pct_within_band": 91.7},
    "economics": {"n": 10, "mean_mae": 0.052, "pct_within_band": 70.0},
    "culture": {"n": 8, "mean_mae": 0.089, "pct_within_band": 50.0}
  }
}
```

### 5. Results for Existing Geopolitics Probes

If you run `--domain all`, the geopolitics probes that already exist should be skipped (via `--skip-existing`). Only new probes (economics, culture) should run.

## Implementation Plan

The simplest path:

**Step A: Seed catalog with all domains**
```
python scripts/pit_market_calibration.py seed --from-gold --domain all --max-gold 30
```
This populates `catalog.jsonl` with all 30 probes, domain-tagged.

**Step B: Run new probes**
```
python scripts/pit_market_calibration.py run --skip-existing --max-probes 30
```
This runs only the new (non-geopolitics) probes.

**Step C: Score with domain breakdown**
```
python scripts/pit_market_calibration.py score
```
This now shows per-domain metrics.

## File Changes

- `harness/pit_market_probe.py`:
  - `seed_from_gold_dataset()` — add `domain_filter` param, remove `_is_geopolitics_slug()` filter, add domain classification
  - `_classify_domain(slug, question_text) -> str` — heuristic classifier
  - `summarize_results()` — add `by_domain` section

- `scripts/pit_market_calibration.py`:
  - `cmd_seed()` — add `--domain` argument, pass to `seed_from_gold_dataset()`
  - `cmd_run()` — add `--domain` argument for filtering
  - `cmd_score()` — call updated `summarize_results()`

## Key Metrics

- Mean MAE per domain (geopolitics vs economics vs culture)
- Percentage within ±5% band per domain
- Which domains is the vault/agent weakest at?
- Are culture questions fundamentally harder (less structured knowledge, more noise)?
