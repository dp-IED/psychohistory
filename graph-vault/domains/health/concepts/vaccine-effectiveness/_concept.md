---
type: concept
tags: [concept, health, vaccine, influenza]
title: "Vaccine Effectiveness"
slug: vaccine-effectiveness
first_observed: 2004
domain: health
related_concepts:
  - seasonal-baseline
  - outbreak-escalation
status: active
---
---
---
# Vaccine Effectiveness

## Definition

Annual influenza vaccine effectiveness (VE) is the estimated reduction
in medically attended, lab-confirmed influenza illness among vaccinated
vs. unvaccinated populations. VE is estimated mid-season and end-of-
season by CDC's US Flu VE Network and similar international networks.
It is a core forecast input for seasonal severity.

## Mechanism

VE depends on three multiplicative factors:

1. **Strain match accuracy**: How closely circulating influenza viruses
   match the WHO-recommended vaccine strains. Historically ranges from
   ~10% (major mismatch, e.g., 2014-15 H3N2 drift) to ~60% (good
   match years). GISRS surveillance determines strain selection — see
   [[domains/health/entities/who.md]].

2. **Vaccine platform**: Egg-based vs. cell-based vs. recombinant.
   Egg-adaptation mutations can reduce VE by ~5-10% for H3N2
   components. The US has been shifting toward cell-based and
   recombinant vaccines.

3. **Population immunity baseline**: Prior-season vaccination and
   natural infection create a complex immunity landscape. Antigenic
   imprinting ("original antigenic sin") can bias immune response
   toward childhood strains rather than current circulating strains.

## VE as Forecast Input

### Mid-season VE estimates
- CDC publishes interim VE estimates in February/March, based on
  ~3,000–5,000 enrolled patients from the US Flu VE Network.
- These are the **earliest quantitative signal** of how well the
  season's vaccine is performing and are critical for late-season
  severity forecasts (Feb–May questions).

### Leading indicators for VE
- **Southern Hemisphere season (Apr–Sep)**: Australia, New Zealand,
  and South American flu seasons preview Northern Hemisphere strain
  dynamics 4–6 months ahead. VE from the SH season is the single best
  predictor of NH VE.
- **Pre-season strain characterization**: WHO's September strain
  composition update and CDC's antigenic characterization reports
  (published Oct–Nov) signal whether drift has occurred.
- **Early-season subtype distribution**: Dominance of H3N2 (historically
  the most vaccine-evasive subtype, VE ~33% avg.) vs. H1N1 (VE ~50%
  avg.) vs. influenza B (VE ~50% avg.) shifts severity expectations.

## Uptake Rates

US influenza vaccination uptake among adults is typically **40-50%**.
Uptake is the multiplier on VE: even perfect VE means little if uptake
is low.

- **2024-25 uptake**: ~44% of adults; ~53% of children 6mo–17yr
  (CDC, March 2025).
- **Declining trend**: Post-COVID vaccine hesitancy and the RFK Jr.
  HHS appointment (February 2025) may depress 2025-26 uptake further.
  Monitor CMS Medicare Part B claims for early signal.
- **Forecasting rule**: Uptake below 40% implies a ceiling on VE's
  population-level impact regardless of strain match. Combine VE ×
  uptake for effective population protection.

## Calibration Reference: VE by Season

| Season | Dominant Strain | VE (all ages) | Match Quality |
|--------|----------------|---------------|---------------|
| 2014-15 | H3N2 (drifted) | 19% | Poor |
| 2017-18 | H3N2 | 38% | Moderate |
| 2019-20 | H1N1 + B/Victoria | 39% | Good |
| 2022-23 | H3N2 + H1N1 | 54% | Good (no B) |
| 2023-24 | H1N1 + B | ~47% | Good |

## Wikilinks
[[domains/health/entities/who.md]] [[domains/health/entities/cdc.md]]
[[domains/health/threads/respiratory-season-2025-26/_thread]]
