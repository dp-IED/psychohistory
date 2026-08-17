---
type: entity
tags: [entity, health, us-agency, hospital-capacity]
kind: organization
title: "Centers for Medicare & Medicaid Services (CMS)"
slug: cms
date_start: 1965-07-30
pit_cutoff: 2026-05-21
related_threads:
  - "[[domains/health/threads/respiratory-season-2025-26/_thread]]"
---
---
---
# Centers for Medicare & Medicaid Services (CMS)

## Type
US federal agency within HHS; administers Medicare, Medicaid, CHIP, and
the Health Insurance Marketplace. Headquarters in Woodlawn, MD.

## Mandate (Forecasting-Relevant)
CMS is the primary federal authority tracking US hospital capacity,
staffing, and utilization. During respiratory season, CMS data is the
ground truth for questions about hospital strain, ICU occupancy, and
bed capacity.

## Key Data Sources

- **NHSN (National Healthcare Safety Network)**: Although operated by
  CDC, CMS mandates reporting from ~6,000 Medicare/Medicaid-participating
  hospitals. NHSN provides weekly hospital admissions for influenza,
  RSV, and COVID-19 — the definitive source for US hospitalization
  counts.
- **Hospital Capacity Snapshots**: CMS publishes weekly data on inpatient
  bed occupancy, ICU bed occupancy, and ventilator usage. These are
  the resolution sources for "hospital at >X% capacity" questions.
- **CMS QualityNet**: Public-facing dashboards with hospitalization
  rates, ED wait times, and facility-level metrics.

## Structural Position
CMS's regulatory leverage gives it mandatory reporting authority that
CDC lacks. While the CDC publishes FluView, the underlying
hospitalization data flows through CMS's mandatory reporting
infrastructure. **Forecasting rule**: For hospital-capacity questions,
trust CMS/NHSN data over CDC estimates — CMS data is regulatory-grade.

## Forecasting Significance
- Hospital admission rates are the single best severity metric for
  respiratory season. ILI outpatient visits can spike from mild
  illness; hospitalizations filter for severity.
- CMS bed-capacity thresholds (e.g., 85% occupancy) are common
  Metaculus question triggers.
- The ratio of hospitalizations to outpatient ILI visits provides a
  severity index useful for estimating the burden of illness.

## Wikilinks
[[domains/health/entities/cdc.md]] [[domains/health/entities/who.md]]
[[domains/health/threads/respiratory-season-2025-26/_thread]]
[[domains/health/concepts/seasonal-baseline/_concept]]
