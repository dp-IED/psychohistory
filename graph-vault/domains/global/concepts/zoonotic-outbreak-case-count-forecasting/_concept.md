---
type: concept
tags: [concept]
title: "Zoonotic Outbreak Case Count Forecasting"
slug: zoonotic-outbreak-case-count-forecasting
first_observed: 2024
domain: public-health-forecasting
related_concepts: [government-confirmation-requirement, pandemic-preparedness, agricultural-exposure-economics]
status: active
---

# Zoonotic Outbreak Case Count Forecasting

## Definition

A framework for forecasting human case count milestones in zoonotic outbreaks where transmission is primarily from agricultural animal hosts to humans (spillover events) rather than human-to-human spread. These outbreaks follow fundamentally different case count dynamics from respiratory pandemics — the case count plateaus at tens to low hundreds rather than entering exponential growth — and this plateau pattern is the key forecasting insight for questions about whether case counts will breach specific thresholds.

## Core Insight

**Without human-to-human transmission, case counts in agricultural spillover outbreaks plateau at tens to low hundreds regardless of media attention, political urgency, or public concern.** The single event that breaks this ceiling is confirmed human-to-human transmission. Until that event occurs, the default forecast for any threshold above ~200 cases should be NO within the first 12-18 months.

## Structural Dynamics

### 1. Initial Detection Delay (Month 1-3)

The first human cases are detected weeks to months after the initial agricultural spillover because:
- Agricultural workers have limited healthcare access and may not seek testing for mild symptoms
- The initial presenting symptom (conjunctivitis in the H5N1 case) is non-specific and may be misdiagnosed
- Surveillance systems for novel influenza in agricultural populations are under-developed
- Testing requires specialized PCR assays that are not routinely available at point of care

**Forecasting implication**: The observed case count in the first 1-3 months is a significant undercount. The true case count may be 2-5x higher. However, even the upper bound of this undercount is far below 100 in early-stage outbreaks.

### 2. Agricultural Amplification Phase (Month 3-9)

Case growth accelerates as:
- The outbreak spreads through the agricultural host population (dairy herds, poultry flocks)
- More agricultural workers are exposed to higher viral loads in milking parlors, culling operations, and processing facilities
- Surveillance improves as public health agencies deploy targeted testing to exposed populations

**Forecasting implication**: This is the period of fastest case growth, but growth is arithmetic (cases per week increase by ~constant absolute number) rather than exponential (cases per week double). The growth rate is bounded by the number of new human-animal exposure events, which is a function of agricultural operations, not general population contact. For H5N1 in 2024, this phase produced ~3-7 new cases per week at peak.

### 3. Plateau Baseline (Month 9+)

Case counts stabilize as:
- The agricultural infection base saturates (most high-risk herds/flocks have been exposed)
- Mitigation measures take effect (PPE use, testing protocols, culling of infected flocks)
- The virus does NOT acquire human-to-human transmission capability
- Surveillance reaches steady-state detection efficiency

**Forecasting implication**: Once the outbreak reaches the plateau phase, the case count grows very slowly or not at all. For H5N1, the plateau was ~70-71 total cases by mid-2025, essentially flat from December 2024. Questions about future case count thresholds from this phase can default strongly toward NO for any threshold more than ~2x the current plateau value.

### 4. The Pandemic Threshold: Human-to-Human Transmission

The single step-change event that invalidates all prior trajectory models:
- Requires specific viral mutations (e.g., PB2 E627K for mammalian adaptation, HA receptor-binding domain changes for human airway tropism)
- Once confirmed, the case count trajectory shifts from bounded agricultural spillover to uncontrolled community spread
- All prior forecasting assumptions are invalid — the question transforms from "how many spillover cases?" to "how fast does it spread?"

**Forecasting implication**: Until this event is documented by credible public health authorities, assume the plateau model holds. The CDC's risk assessment is the most reliable leading indicator — a risk upgrade from "low" to "moderate" or "high" would be the first signal that the model may be breaking.

## Key Forecasting Variables

| Variable | Signal | Source |
|----------|--------|--------|
| Transmission mode | Zoonotic spillover vs. H2H | CDC/WHO situation reports |
| CDC risk assessment | Low / Moderate / High | CDC public statements |
| Cumulative case count | Baseline PIT number | CDC case counter |
| Weekly new case rate | Acceleration/deceleration | CDC situation summaries |
| Agricultural host spread | Herds/counties affected | USDA APHIS data |
| Mammalian adaptation mutations | PB2 E627K, HA mutations | GenBank sequence data |
| H2H cluster documented | Yes/No | CDC epi investigation reports |
| Case severity distribution | Conjunctivitis vs. respiratory vs. fatal | CDC clinical reports |

## Canonical Example: H5N1 in US Dairy Cattle (2024-2025)

| Date | Cumulative Cases | Phase | Key Event |
|------|-----------------|-------|-----------|
| Apr 2024 | 1 | Detection delay | First human case in Texas dairy worker |
| May 2024 | 2 | Detection delay | Second case in Michigan dairy worker |
| Jun 2024 | 6 | Agricultural amplification | First poultry worker cases in Colorado |
| Jul 2024 | 11 | Agricultural amplification | Ongoing slow accumulation |
| Aug 2024 | 20+ | Agricultural amplification | Colorado poultry cluster |
| Sep 2024 | 25-30 | Agricultural amplification | 200+ dairy herds across 14 states |
| Oct 2024 | ~36 | Agricultural amplification | California emerges as hotspot |
| Nov 2024 | ~53 | Transition to plateau | Rate of new cases slightly increases |
| Dec 2024 | ~61 | Plateau | No H2H transmission documented |
| Jan 2025 | ~67 | Plateau | First US death, risk still "low" |
| Mar 2025 | ~70 | Plateau | Nevada becomes 14th state |
| Jun 2025 | ~71 | Plateau | Cases essentially flat |

## Related Files

- [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]] — Full outbreak history with quarterly snapshots
- [[domains/global/entities/centers-for-disease-control-and-prevention]] — CDC entity stub
- [[domains/global/procedures/outbreak-case-threshold-forecast]] — Step-by-step forecasting procedure
- [[2024-Q3]] — First quarter file with H5N1 mention
- [[2024-Q4]] — Case count snapshot at year-end
