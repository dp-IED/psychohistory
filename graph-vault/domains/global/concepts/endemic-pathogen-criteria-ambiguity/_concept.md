---
type: concept
tags: [concept, health, methodology, resolution-criteria]
title: "Endemic Pathogen Criteria Ambiguity"
slug: endemic-pathogen-criteria-ambiguity
domain: global
first_observed: 2026-05-21
canonical_cases:
  - "Hantavirus outbreak >100 cases in 2026 — PIT run p_yes=0.07, criteria unclear: single geographic cluster vs cumulative global surveillance totals"
  - "MV Hondius off-ship cases — PIT run p_yes=0.2, 'linked by a public health authority' vs 'probable/suspected' definitions vary by jurisdiction"
status: active
related_concepts:
  - forecast-resolution-criteria-gotchas
  - question-interpretation-ambiguity
  - zoonotic-outbreak-case-count-forecasting
  - outbreak-escalation
  - seasonal-baseline
---

# Endemic Pathogen Criteria Ambiguity

## Definition

A recurring forecast error pattern: questions about **endemic pathogens** (diseases with established baseline circulation) use language designed for **novel outbreak detection** (WHO "Public Health Emergency of International Concern", "outbreak declaration", "confirmed case threshold"). The resolution criteria do not distinguish between:

1. **Single cluster/spillover event** — Cases traced to a common source (cruise ship, food product, geographic cluster)
2. **Cumulative surveillance totals** — All cases reported nationally or globally in a given period, including endemic background cases

The ambiguity is structurally different from generic resolution-criteria gotchas because it involves **baseline epidemiology** rather than word-level interpretation. The forecaster must estimate not just what the words mean, but what a reasonable resolver would consider the "normal" background count vs a reportable outbreak.

## Why This Is Distinct From Resolution Criteria Gotchas

The existing [[domains/global/concepts/forecast-resolution-criteria-gotchas]] concept covers word-level interpretation errors (e.g., "announces" vs "ratifies", "wins" vs "takes office", "tick" vs "close"). Those are **semantic ambiguities** — the question uses a word that could mean different things.

Endemic pathogen ambiguity is an **epidemiological ambiguity** — the question asks about "an outbreak" of a disease that always circulates at some level, and the criteria don't define the baseline against which "outbreak" is measured.

| Dimension | Resolution Criteria Gotchas | Endemic Pathogen Ambiguity |
|-----------|---------------------------|---------------------------|
| Root cause | Word-level interpretation | Baseline vs detection ambiguity |
| Typical resolution | Oracle/Market decision | WHO/national authority declaration |
| Forecaster action | Audit resolution criteria text | Estimate endemic baseline + case definition |
| Correction method | Read criteria more carefully | Research annual case counts, regional baselines |
| Example error class | "Announces" (mediator) ≠ "Announces" (party) | "100 hantavirus cases" — one outbreak or global annual total? |

## The Two-Meaning Problem

When a question asks about "an outbreak with over 100 confirmed cases" of an endemic pathogen, the resolution criteria can support two incompatible interpretations:

**Interpretation A (Single Cluster):** An outbreak constitutes a discrete, epidemiologically linked set of cases traceable to a common exposure source. This is the standard epidemiological definition — an outbreak is defined by person-place-time clustering, not by cumulative count across unrelated locations.

- If YES: A single geographic cluster (hospital, town, ship) reports 100+ lab-confirmed cases
- If NO: No single cluster reaches 100 cases; multiple small clusters stay under the threshold
- **Forecast anchor**: Historical size of largest single hantavirus cluster (Four Corners 1993: 48 nationwide; Yosemite 2012: ~9; MV Hondius: ~9) → P(100+) < 0.01

**Interpretation B (Cumulative Annual Total):** "Reported cumulatively in 2026" means the sum of all lab-confirmed hantavirus cases globally (or nationally) during calendar year 2026, reported by any official source.

- If YES: Global hantavirus surveillance totals exceed 100 in 2026 (which they almost certainly will — China alone reports thousands of HFRS cases annually)
- If NO: Global totals stay below 100 (extremely unlikely given endemic baseline)
- **Forecast anchor**: Annual global hantavirus case count (10,000-20,000+ in normal years) → P(YES) > 0.99

## Decision Framework

### Step 1: Is the Pathogen Endemic?

- **Endemic**: Circulates at established baseline levels with known seasonality. Examples: hantavirus (Americas, Europe, Asia), dengue (tropical regions), malaria (sub-Saharan Africa), Lyme disease (temperate North America/Europe)
- **Novel/Emerging**: Recently detected or newly spreading in human populations. Examples: COVID-19 (2019), SARS (2003), MERS (2012), H5N1 in dairy cattle (2024)
- **Eliminated/Eradicated**: No sustained transmission. Example: smallpox, polio (most countries), measles (most of Americas)

If the pathogen is endemic → ALWAYS check for criteria ambiguity before forecasting.
If the pathogen is novel → Standard outbreak forecasting framework applies; ambiguity risk is lower.

### Step 2: Identify the Key Ambiguity Signal Words

| Signal Word | Endemic Pathogen Risk |
|-------------|----------------------|
| "an outbreak with over X confirmed cases" | HIGH — "outbreak" vs "cumulative" ambiguity |
| "X confirmed cases reported cumulatively in Y" | HIGH — "cumulatively" favors Interpretation B |
| "X cases linked to [event/location]" | LOW — clear geographic anchor favors Interpretation A |
| "X cases of [disease] in 2026" | MODERATE — baseline-dependent |
| "X cases confirmed by WHO or national authority" | MODERATE — WHO typically declares PHEIC for novel events, but confirms endemic cases as routine surveillance |

### Step 3: Research the Endemic Baseline

| Pathogen | Approximate Annual Baseline | Largest Single Outbreak |
|----------|---------------------------|------------------------|
| Hantavirus (Americas) | 200-400 cases/yr (Pan American WHO data) | Four Corners 1993: 48 (nationwide collection, not single cluster) |
| Hantavirus HFRS (Asia) | 10,000-20,000+ cases/yr (China CDC) | Multiple provincial outbreaks in China |
| Hantavirus (Europe) | 1,000-3,000 cases/yr (ECDC) | 2007 Germany: ~200 Puumala cases (regional outbreak) |
| Dengue (global) | 100-400 million/yr | Multi-country seasonal outbreaks |
| Leptospirosis (global) | 1 million+/yr | Urban flooding clusters |

### Step 4: Estimate the Interpretation Probability

Before forecasting, estimate P(Interpretation B) — the probability that the resolver will use cumulative annual totals rather than single-cluster definition:

| Criteria Text Signal | P(Interp B) | Action |
|--------------------|-------------|--------|
| "cumulatively" explicitly stated | 60-80% | Factor into final probability |
| "outbreak with over X cases" | 10-30% | Likely single cluster interpretation |
| Ambiguous wording | 30-60% | Document assumption explicitly in forecast |

### Step 5: Calibrate

- **If Interpretation A (single cluster) is more likely**: Anchor on historical largest single-cluster size for the pathogen. For hantavirus, no single outbreak has ever reached 100 in the Americas → P(YES) < 0.05.
- **If Interpretation B (cumulative) is more likely**: Anchor on annual surveillance baseline. For most endemic pathogens, 100 is well below annual totals → P(YES) > 0.90.
- **If ambiguous (50/50)**: The probability is the weighted average of both interpretations. E.g., if P(Interp A) = 0.70 and P(Interp A resolves YES) = 0.05 and P(Interp B resolves YES) = 0.95, then:
  `P(YES) = 0.70 × 0.05 + 0.30 × 0.95 = 0.035 + 0.285 = 0.32`

## Canonical Case: Hantavirus Outbreak >100 Cases in 2026 (p_yes = 0.07)

| Parameter | Value |
|-----------|-------|
| Question | "Will a Hantavirus outbreak with over 100 confirmed cases be reported in 2026?" |
| Resolution criteria | "At least 100 laboratory-confirmed cases of Hantavirus reported cumulatively in 2026 by an official source" |
| Pathogen status | **Endemic** — hantavirus circulates at low levels across Americas, Europe, Asia |
| Signal word | "cumulatively" — favors Interpretation B |
| P(Interp B) assessed | ~30% at forecast time (term "an outbreak" suggests single cluster despite "cumulatively") |
| Predicted p_yes | 0.07 |
| Vault coverage | Zero (first hantavirus forecast) — no baseline data available at cutoff |

**Diagnosis**: The forecast correctly identified the ambiguity and estimated that ~70% of the probability mass favored a single-cluster interpretation. However, the "cumulatively" signal in the resolution criteria meant a 30% tail risk that the resolver would use global annual totals — where 100 cases is trivially exceeded. The p_yes=0.07 reflects this weighted average. The vault's lack of baseline epidemiological data at forecast time was a gap (since filled with the hantavirus entity and outbreak event files).

**Post-hoc corrected probability**: If we had known China reports 10,000+ HFRS cases annually, P(Interp B) would be ~50% rather than ~30%, yielding P(YES) = 0.50 × 0.05 + 0.50 × 0.95 = 0.50.

**Vault files**: [[runs/20260521-044449-will-a-hantavirus-outbreak-with-over-100-confirmed-cases-be-]], [[domains/health/entities/hantavirus]], [[events/mv-hondius-hantavirus-outbreak-2026]]

## Canonical Case: MV Hondius Off-Ship Cases (p_yes = 0.2)

| Parameter | Value |
|-----------|-------|
| Question | "Will at least five hantavirus cases linked to the MV Hondius outbreak be reported in people who were not aboard the vessel, before August 1, 2026?" |
| Resolution criteria | Cases "linked to the MV Hondius outbreak by a public health authority"; "probable/suspected" count toward threshold |
| Pathogen status | Andes virus — *known* H2H transmission capability, but limited |
| Ambiguity type | **Definitional** — "probable/suspected" definitions vary by jurisdiction (Scotland vs Netherlands vs Argentina). And "linked by a public health authority" could mean: (a) patient had contact with a confirmed case, (b) patient had contact with a passenger/crew member, or (c) patient was in the same geographic area |
| Predicted p_yes | 0.20 |
| Vault coverage | Zero at forecast time |

**Diagnosis**: This is the **cluster attribution variant** of endemic pathogen ambiguity. The geographical link (off-ship, not aboard the vessel) is clear, but the "probable/suspected" definition varies across the 6+ countries involved. The forecast correctly identified that aggressive contact tracing and 42-day quarantines made 5+ secondary cases unlikely, while the Andes strain's known H2H transmission and 6-week incubation window created residual risk.

**Vault files**: [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius]], [[domains/health/concepts/transport-vector-outbreak]], [[events/mv-hondius-hantavirus-outbreak-2026]]

## Common Error Patterns

| Error | Description | Prevention |
|-------|-------------|------------|
| **Baseline blindness** | Forecaster treats endemic disease as if it were a novel pathogen, ignoring endemic background circulation | Research annual case counts for the pathogen before forecasting |
| **"Outbreak" assumption** | Assumes "outbreak" always means a single cluster, ignoring cumulative-reading possibility | Audit resolution criteria for "cumulative" or "reported in [year]" language |
| **False precision** | Assigning P(YES) based on single-cluster analysis without documenting the interpretation assumption | Always state: "Assuming [Interpretation A/B], P(YES) = X. If resolver uses [alternative], P(YES) = Y." |
| **Novel pathogen transfer** | Applying endemic baseline reasoning to genuinely novel pathogens (e.g., H5N1 in humans, MERS-CoV-2) | Check for established seasonal baseline in WHO/CDC surveillance data |
| **Jurisdiction variance** | Assuming consistent "probable case" definitions across countries | Check whether the question names a single authority (CDC, WHO) or accepts any national authority's report |

## Cross-References

- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — Higher-level concept covering all resolution criteria interpretation errors
- [[domains/global/concepts/question-interpretation-ambiguity]] — Question-scope ambiguity (temporal, entity, geographic)
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept]] — Case count plateau model for spillover events
- [[domains/health/concepts/outbreak-escalation/_concept]] — Outbreak progression stages
- [[domains/health/concepts/seasonal-baseline/_concept]] — Baseline epidemiology framework
- [[domains/health/entities/hantavirus]] — Pathogen entity with endemic baseline data
- [[domains/health/concepts/transport-vector-outbreak]] — Shipborne outbreak dynamics
- [[domains/global/concepts/short-horizon-procedural-certainty/_concept]] — Complementary pre-filter for distinguishing ministerial from discretionary resolution steps
