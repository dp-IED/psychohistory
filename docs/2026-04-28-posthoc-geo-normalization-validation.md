# Post hoc geo-normalization validation (2026-04-28)

Status: executed
Owner: Hermes agent run (CLI)

## Goal
Validate whether current geo gate failures are caused by model behavior vs warehouse/admin label normalization issues.

## What was executed
Using existing audit outputs (no retraining), we recomputed country correctness with post hoc normalization of `top1_admin1`:

- If `top1_admin1` is country-like code (`EG`, `TU`, `LY`, `SY`, etc.), use as-is.
- If `top1_admin1` is admin label, map to country:
  - Egypt labels -> `EG` (`North Sinai`, `South Sinai`, `Cairo`, `Giza`, `Alexandria`, etc.)
  - Libya coarse regions -> `LY` (`West`, `East`, `South`)

Compared against expected country from probe geography (`Egypt`/`Tunisia`/`Libya`/`Syria`).

## Core finding
The prior hard geo failure is largely a normalization artifact.

- Baseline (`stage1_v1_fixed`):
  - old `top1_country_ok_rate`: 0.2963
  - post hoc corrected: 0.2963 (unchanged)

- Hotfix (`stage1_predictive_rankchange_phase3_hotfix_smoke`):
  - old `top1_country_ok_rate`: 0.0000
  - post hoc corrected: 0.3004

Interpretation:
- Under canonicalized geo interpretation, hotfix is not a geo hard fail and is slightly above baseline on top1 country correctness.
- Therefore, the zero-country signal was mainly due to mixed-format warehouse/admin labels plus brittle audit matching.

## Additional evidence
Warehouse manifest has mixed semantics in `admin1_code`:
- Majority rows are code-like (`EG`, `TU`, `LY`, `SY`...),
- but 507 rows are name-like labels (`North Sinai`, `South Sinai`, `Cairo`, `West`, `East`, `South`, ...).

This mixed field contract contaminates both training/eval interpretation when country checks rely on prefix matching.

## Impact on current ablation read
Recomputed post hoc country rates for recent ablations are mostly around baseline (~0.30), not zero.
Seed variance still exists (e.g., one seed ~0.35), so instability remains real, but geo-zero hard fail does not.

## Action items
1) Short-term (immediate):
- [x] Patch `scripts/pit_subgraph_audit.py` country scoring to canonicalize admin labels/codes before country comparison.
- [x] Re-issue Phase 4 gate report from corrected metrics.
- [x] Implement warehouse quality gate script + build-script hook:
  - `scripts/warehouse_quality_gate.py`
  - `scripts/build_arab_spring_warehouse.py --quality-gate [--quality-gate-strict]`

2) Medium-term (warehouse fix):
- Enforce explicit geo fields in warehouse rows:
  - `country_code`
  - `admin1_name`
  - `admin1_code` (canonical)
- Add ingest/build-time assertions to block mixed-format regressions.

## Repro artifacts
- Validation summary JSON:
  - `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/calgary-pit-subgraph-audit/ablations/posthoc_geo_fix_validation_2026-04-28.json`
- Source audits:
  - `.../docs/pit_subgraph_audit_2026-04-28.phase4_gates.json`
  - `.../docs/pit_subgraph_audit_2026-04-28.ablation_first6.json`
  - `.../docs/pit_subgraph_audit_2026-04-28.ablation_pairs_seed.json`
- Patched audit script:
  - `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/calgary-pit-subgraph-audit/scripts/pit_subgraph_audit.py`
- Re-issued Phase 4 gate report (geo-normalized country scoring):
  - `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/calgary-pit-subgraph-audit/docs/pit_subgraph_audit_2026-04-28.phase4_gates_geo_fixed.json`

## Re-issued Phase 4 key metrics (geo-normalized country scoring)
- Baseline (`stage1_v1_fixed/query_encoder_epoch_049.pt`):
  - `top1_time_ok_rate`: 0.1564
  - `top1_country_ok_rate`: 0.2963
- Phase3 smoke (`stage1_predictive_rankchange_phase3_smoke/query_encoder_epoch_002.pt`):
  - `top1_time_ok_rate`: 0.6091
  - `top1_country_ok_rate`: 0.2428 (delta vs baseline: -0.0535)
- Hotfix smoke (`stage1_predictive_rankchange_phase3_hotfix_smoke/query_encoder_epoch_000.pt`):
  - `top1_time_ok_rate`: 0.5638
  - `top1_country_ok_rate`: 0.3004 (delta vs baseline: +0.0041)
  - `precursor_hit_rate`: 0.2716
  - `future_rank_lift_mean`: 0.2716

## Updated interpretation after geo normalization
- Previous country hard fail (`0.0000`) was primarily an evaluation/normalization artifact.
- Under canonicalized country scoring, hotfix is geographically competitive with baseline while retaining large temporal/predictive gains.
- Remaining risk is concentration/attractor instability across seeds, not country-code parsing.
- Governance note: once this quality gate proves stable across production runs, this implementation should be re-implemented as the de facto warehouse-building mechanism.

## Priority plan to raise top1 country rate further (without full rebuild)
1) Evaluation contract hardening (must do first)
- Keep canonicalized country scoring in the audit script.
- Add a report field for `top1_country_inferred` (debug visibility).
- Keep legacy/raw country check as auxiliary telemetry only.

2) Light data normalization patch set (cheap)
- Add a deterministic alias map in warehouse export for known admin labels (`North Sinai`, `West`, etc.) to canonical country.
- Emit both `country_code` and `admin1_name` on rows (non-breaking extension).
- Add ingest assertion: no mixed-format values in canonical `country_code`.

3) Training signal shaping for geo peak
- Introduce country-consistent positive preference at rank-1 via small auxiliary term (low weight), rather than large objective rewrite.
- Upweight negatives that are temporally plausible but wrong-country near misses.
- Keep tuple/fallback weights near current hotfix values (they already support predictive lift).

4) Selection protocol focused on peak country
- Run short 1-epoch seed sweep (>=5 seeds) with fixed objective.
- Promote by constrained score:
  - maximize `top1_country_ok_rate`
  - subject to `top1_time_ok_rate >= hotfix - 0.03`
  - and `precursor_hit_rate >= hotfix - 0.03`
- Reconfirm winner at 2-3 epochs with 3 seeds.

5) Collapse guardrails
- Track `unique_top1`, `dominant_top1_share`, `entropy_top1` in every gate.
- Reject candidates that improve country marginally by collapsing to a single geography attractor.
