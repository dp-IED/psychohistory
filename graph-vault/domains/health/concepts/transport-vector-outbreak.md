---
type: concept
tags: [concept, health, outbreak, epidemiology]
title: "Transport-Vector Outbreak Dynamics"
slug: transport-vector-outbreak
domain: health
first_observed: 2026-05
related_concepts: [zoonotic-outbreak-case-count-forecasting, outbreak-escalation, short-horizon-momentum-check]
status: active
---

# Transport-Vector Outbreak Dynamics

## Definition

An outbreak dynamics framework for infectious disease outbreaks that occur **on or via a transport vehicle** — cruise ships, airplanes, trains, buses — where the epidemiological parameters differ fundamentally from both (a) agricultural spillover outbreaks and (b) community-transmitted outbreaks. The key difference is that the exposed population is geographically dispersed at the end of the transport event, creating a **multinational detection and response challenge** that standard single-jurisdiction outbreak models do not account for.

## Why This Differs from Agricultural Spillover

The vault's existing [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting]] concept models agricultural spillover (H5N1, etc.) where:
- Exposed population is geographically stable (agricultural workers in a region)
- Surveillance is centralized under one national health authority
- Transmission is from animal host to human (spillover)
- Case growth is bounded by agricultural operations

Transport-vector outbreaks differ in five structurally important ways:

| Dimension | Agricultural Spillover | Transport-Vector Outbreak |
|-----------|----------------------|--------------------------|
| **Population dispersion** | Geographically stable | Global dispersion within hours/days |
| **Jurisdictions** | Single country | Multi-country (10+) |
| **Surveillance authority** | Single national health agency | WHO-coordinated, multiple national agencies |
| **Transmission type** | Animal→human (spillover) | Human→human (if pathogen permits, or shared surface/food exposure) |
| **Detection lag** | Regional surveillance | Variable by country capacity; lag can extend 2-6 weeks |
| **Case ceiling** | Bounded by agricultural operations | Bounded by contact network on vessel (1000s) |
| **Response coordination** | Single national plan | Multi-country coordination + IHR mechanisms |

## The Two-Phase Pattern

Transport-vector outbreaks follow a characteristic two-phase pattern:

### Phase 1: Index Cluster (Days 1-21)
Cases identified among passengers/crew during or shortly after the voyage. The index cluster is detected because:
- Passengers develop symptoms during the journey or immediately after
- Onboard medical facilities identify unusual symptom patterns
- Port health authorities screen arriving passengers

**Forecasting implication**: The index cluster size is a leading indicator but not a reliable predictor of Phase 2 size. The MV Hondius index cluster of 9 cases is within the expected range for the first Phase 1 detection.

### Phase 2: Secondary Dispersion (Days 14-90)
Secondary cases among close contacts of index cases who have returned to their home countries. The key variables are:

1. **Pathogen characteristics**: Can the pathogen transmit human-to-human (H2H)? Andes virus: yes (limited). Most hantaviruses: no. SARS-CoV-2: yes (efficient).
2. **Incubation period relative to voyage duration**: Longer incubation means more cases detected after disembarkation, creating the appearance of a larger outbreak.
3. **Contact tracing capacity by country**: High-income countries (NL, UK, AU): aggressive contract tracing. Middle/lower-income: variable.
4. **Cruise line cooperation**: Willingness to share passenger manifest and seating/dining assignments with health authorities.

## Canonical Cases

| Outbreak | Year | Vector | Pathogen | Index Cases | Phase 2 Cases | Key Lesson |
|----------|------|--------|----------|-------------|---------------|------------|
| Diamond Princess | 2020 | Cruise ship | SARS-CoV-2 | 10 (index) | ~700+ onboard | Enclosed environment amplification is extreme for efficient H2H pathogens; multi-week quarantine created amplification risk |
| MV Hondius | 2026 | Cruise ship | Andes hantavirus | 9 | 0 (as of cutoff, May 21) | Inefficient H2H + aggressive CT + dispersed contacts = near-zero Phase 2 |
| Grand Princess | 2020 | Cruise ship | SARS-CoV-2 | ~21 | ~100+ | Confirms Diamond Princess pattern for efficient H2H on cruise ships |

## Forecasting Checklist

Before forecasting any transport-vector outbreak milestone question:

1. **Classify the pathogen's H2H capability**: Efficient (COVID-19), Limited (Andes virus), None (most hantaviruses). This is the single most important variable.
2. **Measure the index cluster**: How many confirmed cases at first detection? Larger clusters suggest broader exposure.
3. **Determine incubation relative to voyage**: If incubation > voyage duration, significant post-travel cases are expected even without H2H.
4. **Assess international response**: Are port health authorities, WHO, and cruise line all cooperating? Aggressive multi-jurisdiction response can suppress Phase 2.
5. **Apply the agricultural plateau baseline**: For non-H2H pathogens, transport-vector outbreaks still follow the plateau model — case count stays at tens to low hundreds.
6. **For H2H pathogens on cruise ships**: Enclosed environment amplification risk is real (see Diamond Princess multiple). But for inefficient H2H (Andes virus), the plateau still holds.

## Relationship to Other Vault Concepts

- [[domains/health/concepts/outbreak-escalation]] — situates transport-vector outbreak in the escalation stages framework
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting]] — provides the plateau baseline that non-H2H transport outbreaks follow
- [[domains/global/concepts/short-horizon-momentum-check]] — structural constraint pre-filter applies: if contact tracing is aggressive and no Phase 2 cases detected yet within incubation window, default is NO
- [[domains/health/entities/hantavirus]] — pathogen entity with base rate data for the canonical case
