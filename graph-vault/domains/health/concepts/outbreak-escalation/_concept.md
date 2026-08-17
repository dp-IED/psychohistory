---
type: concept
tags: [concept, health, escalation, pandemic]
title: "Outbreak Escalation Stages"
slug: outbreak-escalation
first_observed: 2005
domain: health
related_concepts:
  - seasonal-baseline
  - vaccine-effectiveness
status: active
---
---
---
# Outbreak Escalation

## Definition

A structured framework for classifying respiratory disease outbreaks
from sporadic detection through local, regional, and pandemic stages.
Each stage has defined trigger conditions, typical timelines, and
forecasting implications.

## Escalation Stages

### Stage 1: Sporadic Detection
- **Trigger**: Isolated cases identified, no epidemiological linkage.
- **Typical duration**: Days to weeks.
- **Forecasting implication**: Most novel pathogen detections never
  progress past Stage 1. Default P(escalation) < 0.10 for any single
  detection event. This is the "noise floor" of outbreak surveillance.

### Stage 2: Local Outbreak / Cluster
- **Trigger**: ≥3 epidemiologically linked cases with a common exposure
  (facility, family, event) and no sustained community transmission.
- **Typical duration**: 1–4 weeks if contained; indefinite if not.
- **Forecasting implication**: A cluster with confirmed human-to-human
  transmission (H2H) is the inflection point. Without H2H, clusters
  burn out. With H2H, default P(regional escalation) rises to ~0.30–0.50
  within 3 months.

### Stage 3: Regional Spread
- **Trigger**: Sustained community transmission in a defined geographic
  region (state, province, or multi-county area) with >20 cases and no
  clear single-source exposure chain.
- **Typical duration**: Weeks to months; may stabilize or escalate.
- **Forecasting implication**: This is the most forecast-relevant stage
  for tournament questions. Key variables: R0 in the affected population,
  public health intervention speed, healthcare system absorption
  capacity. At this stage, questions about case-count thresholds become
  tractable.

### Stage 4: Multi-Regional / National Epidemic
- **Trigger**: Sustained community transmission in multiple regions of a
  country, with total cases exceeding seasonal baseline expectations.
- **CDC equivalent**: ILI% above baseline for ≥2 weeks with ≥50% lab
  confirmation of the pathogen.
- **Forecasting implication**: At this stage, peak timing and total
  burden questions dominate. Week-over-week trajectories and hospital
  admission rates become the primary forecast inputs.

### Stage 5: Pandemic
- **Trigger**: WHO Phase 6 declaration — sustained community transmission
  in ≥2 WHO regions.
- **Forecasting implication**: Once pandemic is declared, all
  province/state/country resolution date questions become timing
  problems rather than binary will-it/won't-it problems. The
  structural question shifts to "how fast?"

## Historical Analogues for Calibration

| Pathogen | Sporadic → Local | Local → Regional | Regional → Pandemic | Notes |
|----------|-----------------|------------------|---------------------|-------|
| SARS-CoV-2 (2019) | ~3 weeks | ~2 weeks | ~4 weeks | Unprecedented speed due to asymptomatic transmission |
| H1N1 (2009) | ~4 weeks | ~6 weeks | ~8 weeks | Slower burn; detected in Mexico before global alert |
| H5N1 (2024-25) | ~2 months | NOT REACHED | N/A | Agricultural spillover without H2H — plateau model applies |
| H7N9 (2013) | ~1 month | NOT REACHED | N/A | Poultry-to-human; contained via live market closures |

## Key Forecasting Rule

The stage-to-stage transition probability is multiplicative, not additive:

P(pandemic | sporadic case) ≈ P(local|sporadic) × P(regional|local) × P(pandemic|regional)

Base rate: ~0.10 × 0.30 × 0.15 ≈ **0.0045** for any novel pathogen detection.

This base rate is the structural reason why most tournament questions
about novel outbreak escalation should default strongly toward NO unless
H2H transmission is confirmed.

## Wikilinks
[[domains/health/entities/cdc.md]] [[domains/health/entities/who.md]]
[[domains/health/concepts/seasonal-baseline/_concept]]
