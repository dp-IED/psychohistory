---
type: entity
kind: pathogen
tags: [health, zoonotic, pathogen, entity]
title: "Hantavirus"
slug: hantavirus
domain: health
status: active
---

# Hantavirus

## Overview

Hantavirus is a genus of rodent-borne viruses in the family Hantaviridae, genus *Orthohantavirus*. Unlike influenza or coronaviruses, hantaviruses are not typically respiratory-transmitted between humans — they are primarily transmitted via aerosolized rodent excreta (urine, feces, saliva). The **Andes virus** (ANDV) is the exception, with documented human-to-human transmission capability. Hantavirus infections in humans present as either hemorrhagic fever with renal syndrome (HFRS, primarily in Eurasia) or hantavirus pulmonary syndrome (HPS, primarily in the Americas).

## Key Strains for Forecasting

| Strain | Region | Disease | H2H Transmission | Case Fatality Rate | Notes |
|--------|--------|---------|-----------------|--------------------|-------|
| Andes virus (ANDV) | South America (Argentina, Chile) | HPS | **YES** — documented limited H2H | ~35-40% | Only hantavirus with confirmed H2H; 1996 El Bolsón outbreak, 2018-19 Epuyén outbreak |
| Sin Nombre virus (SNV) | North America | HPS | No | ~36% | Four Corners outbreak 1993; primary North American strain |
| Seoul virus (SEOV) | Global (urban rats) | HFRS (mild) | No | ~1-2% | Urban rodent reservoir; laboratory outbreaks documented |
| Puumala virus (PUUV) | Europe, Russia | HFRS (NE, mild form) | No | ~0.1-0.4% | Most common in Europe; bank vole reservoir; large outbreaks in Finland, Sweden |
| Dobrava-Belgrade virus (DOBV) | Balkans, Central Europe | HFRS (severe) | No | ~5-12% | Yellow-necked mouse reservoir; sporadic outbreaks |
| Hantaan virus (HTNV) | East Asia | HFRS (severe) | No | ~5-15% | Striped field mouse reservoir; described in 1978; common in Korea, China |

## Forecasting Significance

### MV Hondius Outbreak (2026)

The MV Hondius outbreak (March-April 2026 voyage, confirmed cases reported May 2026) is the first documented shipborne hantavirus outbreak and the most significant hantavirus event since the 2018-19 Epuyén outbreak (Andes virus, 30+ cases with H2H transmission). The outbreak involved Andes virus — the strain with confirmed H2H transmission — creating a qualitatively different risk profile than a rodent-borne-only strain outbreak.

**Why this matters for forecasting:**
- Andes virus on a cruise ship with 1000+ passengers is a novel epidemiological scenario — closed environment, multinational dispersion of close contacts, extended incubation (up to 6 weeks)
- International contact tracing across 10+ countries creates a prolonged detection window (secondary cases may appear weeks after vessel disembarkation)
- The outbreak tests the agricultural-spillover plateau model's applicability to non-agricultural zoonotic outbreaks

### Historical Outbreaks for Base Rate Calibration

| Outbreak | Year | Strain | Cases | Region | Type |
|----------|------|--------|-------|--------|------|
| Four Corners outbreak | 1993 | Sin Nombre | 48 (nationwide) | USA | Rodent spillover (ecological conditions) |
| Epuyén outbreak | 2018-2019 | Andes | 30+ | Argentina | Documented H2H transmission chain |
| Yosemite outbreak | 2012 | Sin Nombre | 9 | USA (Yosemite NP) | Tourist exposure, tent cabins |
| El Bolsón outbreak | 1996 | Andes | 16 | Argentina | First documented H2H transmission |
| Hantavirus << 100 cases | 2026 | Andes | 9 | International (cruise ship) | Shipborne, multinational |

### Key Forecasting Rules

1. **No single hantavirus outbreak has ever reached 100 confirmed cases** in the Americas or Europe. The largest single-outbreak total is the 1993 Four Corners event at 48 cases nationwide (all US states combined, not a single cluster).
2. **Andes virus H2H changes the dynamics** — but even H2H-capable outbreaks (Epuyén 2018-19) peaked at ~30 cases. H2H does not automatically generate exponential growth; close/prolonged contact is required.
3. **Cruise ship context amplifies risk moderately**: enclosed environment + multinational dispersion + long incubation = wider detection window but also more aggressive public health response (42-day quarantine protocols, international contact tracing).
4. **Hantavirus is endemic at low levels globally**: China reports thousands of HFRS cases annually from HTNV/SEOV, but these are routine endemic transmission, not outbreaks. Resolution criteria must be checked carefully — "cumulative cases" vs "single outbreak" language determines whether endemic baseline counts.

## Appears In

- [[domains/health/entities/who]] — WHO coordinated international response, issued risk assessment
- [[domains/health/entities/cdc]] — CDC tracks US hantavirus surveillance, Andes virus expertise
- [[domains/health/concepts/transport-vector-outbreak]] — shipborne outbreak dynamics
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting]] — opposite-case pathogen (non-agricultural reservoir)
- [[events/mv-hondius-hantavirus-outbreak-2026]] — MV Hondius outbreak event
- [[runs/20260521-044449-will-a-hantavirus-outbreak-with-over-100-confirmed-cases-be-]] — p_yes=0.07 relied on hantavirus base rates
- [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius]] — p_yes=0.2 assessed Andes virus H2H risk
