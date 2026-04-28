# Predictive-Coding Objective Contract (Stage1)

Version date: 2026-04-28
Scope: Calgary Arab Spring Stage1 retriever training + PIT audit diagnostics

## Objective version IDs
- `infonce_v0`
  - Baseline retrieval contrastive objective.
- `predictive_coding_rankchange_v1`
  - Predictive objective using rank-change targets + tuple regularization + fallback negatives.

## Training loss contract
Let:
- q = encoded query vector
- p+ = positive node vector
- N = sampled negatives
- s(q, x) = similarity score
- tau = temperature

Baseline:
- L_infonce = -log( exp(s(q,p+)/tau) / (exp(s(q,p+)/tau) + sum_{n in N} exp(s(q,n)/tau)) )

Predictive:
- L_pred = w_tuple * L_tuple + w_fb * L_fallback
- L_total = lambda_pred * L_pred + (1 - lambda_pred) * L_infonce

where:
- `L_tuple` is tuple-aware predictive ranking loss on semantically disjoint strata.
- `L_fallback` is PIT-safe fallback contrastive loss when tuple supervision is missing/unusable.
- `w_tuple`, `w_fb`, and horizon/strata weights are explicit config knobs and logged in `train_state.json`.

## Horizon alignment contract
- Probe horizons in corpus may differ from tuple horizons.
- Mapping must be explicit and logged under:
  - `predictive_probe_horizon_mapping`
- Tuple branch activation telemetry must be non-zero in any intended tuple run:
  - `tuple_branch_used_count`
  - `fallback_branch_used_count`
  - `both_branch_used_count`
  - `no_branch_used_count`

## Non-collapse / non-overconstraint guardrail
Tuple supervision is a regularizer, not a hard filter.
- Do not enforce tuple-only retrieval.
- Keep fallback path active.
- Gate runs with diversity/collapse diagnostics before long training.

## PIT audit gate metrics (required)
Core:
- `top1_time_ok_rate`
- `topk_time_ok_ratio_mean`
- `top1_country_ok_rate`
- `topk_country_ok_ratio_mean`
- `topk_hint_hit_ratio_mean`

Predictive diagnostics (Phase4):
- `precursor_hit_rate`
  - Fraction of probes with at least one PIT-valid hint-hit in top-k.
- `future_rank_lift_mean`
  - Mean(min_future_rank - min_precursor_rank), treating missing ranks as k+1.
  - Positive means precursor evidence ranks ahead of future-only evidence.
- `horizon_consistency_rate`
  - Fraction of probes where top1 first_seen <= as_of + horizon_days (when horizon present).

Grouped deltas:
- `delta_by_assumption_emphasis`
- `delta_by_geo_bucket`
computed as candidate minus baseline on all summary metrics.

## Go/no-go policy for long runs
Proceed only if all hold:
1) supervision path is active (tuple branch non-zero),
2) time diagnostics improve or hold,
3) no material collapse in country/diversity metrics,
4) grouped deltas do not show major regressions in key geo buckets.

## Section 18 — Rejected Alternatives

### 20.18 Inverse horizon weighting for predictive coding loss
Why rejected: `inverse_horizon` weighting combined with unaligned probe-horizon mapping produced negative `future_rank_lift_mean` in phase3 smoke — the loss landscape pushed the encoder toward narrow temporal hit-rate improvement at the cost of precursor-before-future ordering. `meta` weighting with explicit probe→target horizon mapping corrects this.

Specific failure observed:
- 7/14/21-day probes without explicit mapping to the 30-day predictive target missed intended supervision semantics.
- Branch telemetry was absent, so degenerate tuple/fallback path usage could not be verified during training.

## Section 18 — Locked Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Predictive horizon weight mode | `meta` | `inverse_horizon` caused negative rank lift in phase3 smoke; precursor ordering degraded |
| Probe→horizon mapping | explicit (`7→30`, `14→30`, `21→30`) | Implicit mapping caused supervision misalignment |
| Branch telemetry | required in `train_state.json` | Silent degenerate branch paths are undetectable without it |

Additional scope note:
- For Arab Spring warehouse training, the current hotfix setup is an effective single-horizon (30d) approximation of the full predictive objective, due to limited temporal depth in this corpus.
- Full multi-horizon supervision (30d/180d/2y/10y/30y with meaningful separation) is expected to become canonical on deeper corpora (e.g., Wikipedia time-depth pipeline).
