---
type: concept
tags: [concept, health, surveillance, baseline]
title: "Seasonal Baseline Establishment"
slug: seasonal-baseline
first_observed: 2007
domain: health
related_concepts:
  - outbreak-escalation
  - vaccine-effectiveness
status: active
---
---
---
# Seasonal Baseline

## Definition

A seasonal baseline is the expected level of respiratory disease
activity during periods of low or no circulation — established from
historical data and used as the threshold for declaring elevated
activity, epidemic onset, or pandemic conditions. Baselines are pathogen-
specific, geographically specific, and seasonally calibrated.

## How ILINet Baselines Are Established

The CDC's **ILINet national baseline** (currently 1.9% for 2025-26)
is calculated as the mean ILI percentage during non-influenza weeks
over the previous three seasons, plus two standard deviations.

- **Data source**: ~3,400 outpatient providers reporting weekly ILI
  visit counts to CDC.
- **Recalibration**: Baselines are recalculated annually before each
  flu season using the prior three seasons' data. Major shifts (like
  COVID-era changes in care-seeking behavior) can distort baselines
  for 2-3 years.
- **Regional baselines**: Each of the 10 HHS regions has a separate
  baseline (typically 1.2%–3.5%), used to declare regional epidemics.

## Threshold Hierarchy

| Threshold | Definition | Trigger Condition |
|-----------|-----------|-------------------|
| Baseline | Mean ILI% + 2σ, non-influenza weeks | Always present — reference floor |
| Elevated activity | ILI% above baseline for ≥2 weeks | Signals season onset |
| Epidemic threshold | ILI% above baseline for ≥2 weeks + lab-confirmed influenza in ≥50% of specimens | CDC's official "flu epidemic" declaration |
| Pandemic threshold | Sustained community transmission in ≥2 WHO regions | WHO Phase 6 declaration |

## Key Forecasting Implications

1. **Baseline crossings are binary events**: Questions like "Will ILI%
   exceed the national baseline by week X?" are resolvable with
   weekly FluView data. Monitor regional baselines — some Metaculus
   questions use regional rather than national thresholds.

2. **Baseline distortion post-pandemic**: COVID-19 disrupted care-
   seeking behavior, inflating ILINet baselines for 2020-2023. The
   2025-26 baseline (1.9%) reflects partial normalization. If
   care-seeking patterns shift again (telehealth expansion, urgent
   care growth), baselines may drift downward.

3. **Epidemic vs. pandemic distinction**: Many tournament questions
   ask about "epidemic" thresholds. Confirm whether the question
   resolution criteria reference CDC's epidemic declaration (lab-
   confirmed ≥50%) or a generic "above baseline" standard. These
   resolve at different times.

4. **RSV and COVID-19 baselines**: Unlike influenza, RSV and COVID-19
   lack mature, stable baselines. COVID-19 baselines remain volatile
   5 years post-pandemic; RSV baselines are emerging as CDC expands
   NREVSS surveillance. Composite respiratory baselines (e.g., "ILI +
   COVID-like illness + RSV") are less reliable for forecasting.

## Canonical Example
During the 2022-23 tripledemic, ILI% reached 7.5% nationally (far above
the ~2.5% baseline), driven by simultaneous influenza, RSV, and COVID-19
surges. The baseline was breached in Week 43 — early by historical
standards — but lab confirmation lagged, delaying epidemic declaration
until Week 47.

## Wikilinks
[[domains/health/entities/cdc.md]]
[[domains/health/concepts/outbreak-escalation/_concept]]
