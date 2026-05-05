# PIT Retrieval Promotion Gate (v1) + Targeted Ablation Plan

## Promotion gate purpose
Promote checkpoints that improve predictive precursor utility without violating PIT faithfulness or structural grounding.

## Evaluation setup (fixed)
- Same manifest/mmap artifacts for all compared checkpoints.
- Same probe set and k.
- Report must include:
  - global summary
  - summary_by_assumption_emphasis
  - summary_by_geo_bucket
  - delta_vs_baseline
  - delta_by_assumption_emphasis
  - delta_by_geo_bucket
  - concentration stats (unique_top1, dominant_top1_share, entropy_top1)

## Tier-1 hard fails (any one => NO-GO)
1) Temporal integrity fail
- top1_time_ok_rate < baseline - 0.02
OR
- horizon_consistency_rate < baseline - 0.03

2) Geo grounding fail
- top1_country_ok_rate <= 0
OR
- top1_country_ok_rate < baseline - 0.05
OR
- >=2 key geo buckets have delta(top1_country_ok_rate) <= -0.20

3) Collapse fail
- unique_top1 < 0.60 * baseline_unique_top1
OR
- dominant_top1_share > baseline_dominant_share + 0.12
OR
- entropy_top1 < baseline_entropy - 0.20

4) Supervision-path fail (for tuple-aware objectives)
- tuple branch usage is zero/near-zero
OR
- tuple_weight=0 ablation gives indistinguishable probe outcomes

## Tier-2 warning band (review required, not auto-fail)
- Any geo metric in [-0.05, -0.02] vs baseline
- Mild concentration worsening without crossing Tier-1
- Predictive gains concentrated in one assumption bucket only

## Promotion criteria (GO)
All must hold:
1) No Tier-1 hard fail.
2) Predictive utility improves:
- precursor_hit_rate > baseline + 0.03
AND
- future_rank_lift_mean > baseline + 0.03
3) Temporal metrics non-degrading (within tolerance or better).
4) Geo grounding non-degrading beyond tolerance.
5) No significant new concentration/collapse signature.

## Tie-breaker logic (if mixed winners)
Priority order:
1) Fewer/no Tier-2 warnings
2) Better geo-bucket stability
3) Better predictive lift
4) Better temporal metrics
5) Lower concentration risk

## Decision labels
- GO: passes promotion criteria.
- CONDITIONAL GO: passes Tier-1, has Tier-2 warnings; run short confirmatory training before long run.
- NO-GO: any Tier-1 fail.

---

## Ablation objective
Fix the observed failure mode: temporal/predictive gains with geo grounding collapse.

Design principle: smallest possible matrix that isolates causes before any long run.

## Fixed protocol for all ablations
- Keep data artifacts fixed (same warehouse/manifest/mmap).
- Same seed set across runs (at least 2 seeds; 3 preferred for finalists).
- Short runs only (1-2 epochs) for screening.
- Gate every run with the same Phase 4/5 report.
- Compare against baseline and current hotfix smoke.

## Stage A: Supervision-path sanity (cheap, mandatory)
Run these first to verify tuple path is active and influential.

A1) tuple_weight = 0.0
A2) current tuple_weight (hotfix default)
A3) tuple_weight = 2x current

Expected diagnostic:
- If A1/A2/A3 are near-identical on probe metrics, tuple branch is effectively dead/misaligned.
- If predictive metrics move but geo keeps collapsing, tuple signal may be geo-agnostic.

Stop rule:
- Do not proceed to larger grid until tuple-path effect is confirmed non-trivial.

## Stage B: Geo-pressure vs predictive tradeoff (core grid)
Test two orthogonal levers only.

Lever 1: Geo-consistent negatives
- N0: current negative sampling
- N1: +geo-hard negatives (same/near time, wrong country/admin)
- N2: mixed negatives with explicit geo-balance quota per batch

Lever 2: Horizon weighting strength
- H0: current horizon weighting
- H1: reduced horizon emphasis (e.g., 0.75x)
- H2: flatter horizon weighting (less tail amplification)

Matrix: 3 x 3 = 9 short runs
- Prioritize order:
  1) N1H1
  2) N2H1
  3) N1H0
  4) N2H0
  5) N0H1
  6) remaining combinations only if needed

Rationale:
- Separates “too much horizon pressure” from “insufficient geo discrimination”.

## Stage C: Anti-collapse regularization checks (only if needed)
Run only if Stage B improves geo but still shows concentration drift.

C1) increase in-batch diversity regularization (light)
C2) same as C1 + mild top1 geography entropy regularizer
C3) C2 with slightly reduced tuple weight

Keep this stage minimal (max 3 runs).

## Promotion funnel
1) Screen phase:
- Run Stage A + top 4-6 Stage B combos.
- Reject all Tier-1 failures immediately.

2) Confirm phase:
- Take top 2 candidates (by gate + tie-breakers).
- Re-run with 3 seeds and one extra epoch.
- Require stable pass (no seed-specific geo collapse).

3) Long-run eligibility:
- Promote only candidate that passes GO criteria in confirm phase.

## Suggested tracking table (per run)
- run_id
- tuple_weight
- negative_mode (N0/N1/N2)
- horizon_mode (H0/H1/H2)
- top1_time_ok_rate
- top1_country_ok_rate
- precursor_hit_rate
- future_rank_lift_mean
- horizon_consistency_rate
- unique_top1
- dominant_top1_share
- entropy_top1
- tier1_fail_count
- tier2_warning_count
- decision (GO/CONDITIONAL/NO-GO)

## Immediate first 6 runs (recommended)
1) A1 (tuple_weight=0)
2) A2 (current tuple_weight)
3) A3 (2x tuple_weight)
4) N1H1
5) N2H1
6) N1H0

Interpretation target:
- Confirm tuple-path activity,
- recover geo grounding above failure thresholds,
- retain non-trivial predictive lift.
