# Source-Experiment Metric Comparability — Implementation Plan

**Program context:** `project.md`, `docs/reviewers-guide.md` (France harness metrics must stay comparable as the regression suite evolves.)

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cross-experiment reporting fair when `positive_count` differs (e.g. 255 vs 485 on the same row count) by adding supplemental metrics and clear reporting—without changing label construction or the forecast task.

**Architecture:** Add pure metric helpers in `baselines/metrics.py` (`positive_rate`, `balanced_accuracy`, `pr_auc`); extend `_metrics_for_model` in `baselines/source_experiments.py` to emit new keys alongside existing Brier/MAE/recall; validate with unit tests and optional offline recomputation from existing audits.

**Tech Stack:** Existing `ForecastRow` / row lists; prefer `sklearn.metrics` for PR-AUC if already a dependency, else manual trapezoid (verify in Step 0).

---

**Date:** 2026-04-17

## Why raw Brier alone misleads

Trivial always-negative Brier equals \(p(1-p)\) for prevalence \(p\). Different experiments can have different \(p\) on the same scoring grid, so raw Brier is prevalence-confounded. Example: `positive_rate` ≈ 255/1144 vs 485/1144 shifts the trivial baseline.

## Options

| Option | Summary | Verdict |
|--------|---------|---------|
| **A — Supplemental metrics** | Emit `positive_rate`, `pr_auc`, `balanced_accuracy` in audit | **Recommended** |
| **B — Stratified subsampling** | Equalize positives by dropping negatives | Harms data; awkward per-week structure; skip |
| **C — Unified label source** | e.g. always GDELT labels for all experiments | Conceptual change; out of scope |
| **D — Per-origin only** | Limited fix for Brier; recall@5 already origin-stable | Partial |
| **E — Brier skill score (BSS)** | \(1 - \mathrm{Brier}/(p(1-p))\) | **Reporting** (can compute from audit + new `positive_rate`) |

**Primary implementation:** Option A. **Reporting:** add BSS column in results tables (Option E), and interpret `pr_auc - positive_rate` as skill vs random PR baseline.

## What NOT to change

- `ForecastRow` fields and label semantics.
- `_update_target_lookup` / how `target_occurs_next_7d` is built from snapshots.
- Horizon, weekly origins, admin1 unit, experiment names.

## File Map

| File | Change |
|------|--------|
| `baselines/metrics.py` | Add `positive_rate`, `balanced_accuracy`, `pr_auc` |
| `baselines/source_experiments.py` | Extend `_metrics_for_model` (+ empty-row branch) |
| `tests/test_metrics.py` or `tests/test_baselines.py` | Unit tests for new functions |
| `tests/test_source_experiments.py` | Assert new keys in audit for `gnn_sage` |
| `data/runs/source_experiments/gnn_only/source_experiments_results.md` | Optional: comparability table after re-run or offline calc |

## Tasks (checkboxes)

### Step 0

- [ ] Confirm whether `sklearn` is a declared dependency (`pyproject.toml`); if not, implement PR-AUC without sklearn.

### Step 1 — Metrics

- [ ] `positive_rate(rows, model_name)`.
- [ ] `balanced_accuracy(rows, model_name, threshold=0.5)` with degenerate-label handling documented.
- [ ] `pr_auc(rows, model_name)`; return `0.0` if no positives.

### Step 2 — Wire-up

- [ ] Import and add keys in `_metrics_for_model` in `baselines/source_experiments.py` (both populated and zero-row branches).

### Step 3 — TDD

- [ ] Unit tests: perfect classifier, random 0.5, no positives, balanced accuracy edge cases.
- [ ] Integration: audit contains `positive_rate`, `pr_auc`, `balanced_accuracy` for gnn (existing monkeypatch pattern).

### Step 4 — Validate

- [ ] **4a:** Offline script/REPL: recompute new metrics from `source_experiments.audit.json` + `gnn_predictions.jsonl.gz`; confirm `positive_rate` matches `positive_count/row_count`.
- [ ] **4b:** Full re-run only if you need fresh audits with new keys embedded.

### Step 5 — Reporting

- [ ] Update `source_experiments_results.md` with comparability section: `positive_rate`, `pr_auc`, `balanced_accuracy`, BSS; note recall@5 as already ranking-stable.

### Step 6 — Quality

- [ ] `pytest` targeted tests; fix lints/types on touched files.

## Fair-footing checks (post-implementation)

1. `positive_rate` matches `positive_count / row_count` per experiment.
2. Compare **PR-AUC skill** `pr_auc - positive_rate` across experiments, not raw PR-AUC alone.
3. `balanced_accuracy` at 0.5 is comparable across experiments (separate sensitivity/specificity normalization).
4. BSS from stored Brier + `positive_rate` clarifies skill vs prevalence-only baseline.
