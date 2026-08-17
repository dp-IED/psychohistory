---
type: domain
tags: [domain]
title: "Health & Respiratory Disease"
slug: health
subjects:
  - "[[domains/health/threads/respiratory-season-2025-26/_thread]]"
  - "[[domains/health/concepts/seasonal-baseline/_concept]]"
entities:
  - "[[domains/health/entities/cdc.md]]"
  - "[[domains/health/entities/who.md]]"
  - "[[domains/health/entities/cms.md]]"
  - "[[domains/health/entities/hantavirus.md]]"
  - "[[domains/health/entities/oceanwide-expeditions.md]]"
events:
  - "[[events/mv-hondius-hantavirus-outbreak-2026]]"
procedures:
  - "[[domains/health/procedures/respiratory-forecast.md]]"
threads:
  - "[[domains/health/threads/respiratory-season-2025-26/_thread]]"
concepts:
  - "[[domains/health/concepts/seasonal-baseline/_concept]]"
  - "[[domains/health/concepts/outbreak-escalation/_concept]]"
  - "[[domains/health/concepts/vaccine-effectiveness/_concept]]"
  - "[[domains/health/concepts/transport-vector-outbreak]]"
---
---

# Health & Respiratory Disease

Public health forecasting domain covering seasonal respiratory disease
surveillance, outbreak escalation dynamics, vaccine effectiveness
estimation, hospital capacity monitoring, and zoonotic disease dynamics.
Built for Metaculus respiratory outlook tournament (33 questions) and
expanded to cover zoonotic outbreak forecasting (hantavirus, H5N1).

## Key Threads

- **Respiratory Season 2025-26** — The ongoing season from October 2025
  through May 2026, tracking influenza, RSV, COVID-19, and composite
  respiratory metrics.

## Key Concepts

- **Seasonal baseline** — How ILINet thresholds, epidemic/pandemic
  thresholds, and seasonal baselines are established and used as
  forecast anchors.
- **Outbreak escalation** — The stages from sporadic detection through
  local outbreak, regional spread, to pandemic declaration.
- **Vaccine effectiveness** — Annual VE estimation, strain matching
  accuracy, and uptake rates as leading indicators for seasonal severity.
- **Transport-vector outbreak dynamics** — How outbreaks on cruise ships,
  airplanes, and other transport vehicles differ structurally from both
  agricultural spillover and community-transmitted outbreaks. Two-phase
  pattern (index cluster + secondary dispersion), multinational response
  coordination challenges.

## Key Entities

- **Hantavirus** — Rodent-borne pathogen; strain table including Andes
  virus (H2H-capable), Sin Nombre, Seoul, Puumala; historical outbreak
  base rates for calibration; MV Hondius outbreak (2026) as canonical
  transport-vector case.
- **Oceanwide Expeditions** — Operator of MV Hondius (cruise ship at
  center of 2026 hantavirus outbreak), Dutch expedition cruise company;
  liability and disclosure incentive analysis relevant to outbreak
  reporting timelines.
- **MV Hondius hantavirus outbreak** (event) — First documented
  shipborne hantavirus outbreak; canonical transport-vector case with
  7 confirmed/suspected cases and 3 deaths among 147 passengers/crew;
  ongoing secondary dispersion monitoring through August 2026.

## Active Forecasts

Two Metaculus questions track the MV Hondius outbreak's secondary
dispersion (both with 2026-08-01 deadline):
- [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius\|MV Hondius: 5+ off-ship (cup)]] — p_yes=0.20, metaculus-cup
- [[runs/20260521-184716-will-at-least-5-non-passengers-be-linked-to-the-mv-hondius-h\|MV Hondius: 5+ off-ship (tournament)]] — p_yes=0.24, metaculus-tournament
- [[runs/20260521-044449-will-a-hantavirus-outbreak-with-over-100-confirmed-cases-be-\|Hantavirus >100 cases in 2026]] — p_yes=0.07, metaculus-cup

## Coverage Status

This domain is seeded for the Metaculus respiratory outlook tournament
(May 2026) and expanded for zoonotic outbreak forecasting (May 2026).
Primary coverage is US respiratory surveillance (CDC FluView, CMS
hospital data, NHSN). WHO global influenza program and GISRS coverage
are included for strain surveillance context. The hantavirus entity and
transport-vector outbreak concept extend coverage to non-respiratory
zoonotic pathogens with transport-vector exposure patterns.

### Recent Additions (May 21-22, 2026)
- `entities/hantavirus.md` — Pathogen entity with strain table, base rates
- `concepts/transport-vector-outbreak.md` — Shipborne/plane outbreak dynamics
- `entities/oceanwide-expeditions.md` — MV Hondius operator, liability analysis
- Active forecasts tracked: 3 Metaculus runs cross-referenced (2 MV Hondius off-ship + 1 hantavirus 100+ cases)
