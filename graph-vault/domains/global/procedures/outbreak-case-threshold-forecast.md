---
type: procedure
tags: [procedure]
title: "Outbreak Case Threshold Forecast"
slug: outbreak-case-threshold-forecast
domain: "[[domains/global]]"
concepts:
  - "[[domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept]]"
entities:
  - "[[domains/global/entities/centers-for-disease-control-and-prevention]]"
threads:
  - "[[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]]"
---

# Outbreak Case Threshold Forecast

Estimate whether a disease outbreak will reach a specified number of confirmed human cases in the United States (or globally) by a specific date.

## When

Any question about whether an infectious disease outbreak's case count will breach a threshold (50 cases, 100 cases, 200 cases, etc.) by a deadline date. Use this for zoonotic spillover outbreaks (H5N1, mpox, variants) where transmission is primarily from animal hosts to humans rather than human-to-human.

## Approach

### Step 1: Identify the Outbreak and Its Source

1. **What pathogen?** H5N1 avian influenza, mpox, COVID-19 variant, etc.
2. **What host reservoir?** Dairy cattle, poultry, wild birds, rodents, non-human primates, etc.
3. **What transmission mode?** Zoonotic spillover (animal-to-human), or has human-to-human transmission been documented?
   - If H2H transmission confirmed → exit this procedure; use a pandemic trajectory model instead
   - If only zoonotic spillover → continue with this procedure

### Step 2: Check CDC Risk Assessment

1. **Find the current CDC public health risk level** from the most recent CDC situation summary for this outbreak
2. **Interpret the signal**:
   - **Low**: The agency's internal models do not predict rapid escalation. Case counts are expected to remain bounded. This is the strongest single indicator that the plateau model holds.
   - **Moderate**: The agency has evidence of changing dynamics. This may indicate accelerated case accumulation. Reassess the plateau model.
   - **High**: The agency believes the outbreak trajectory has changed significantly. The plateau model is likely invalid.

### Step 3: Extract the PIT Case Count Baseline

1. **Read the most recent quarter file** that covers the outbreak period
2. **Read the outbreak thread** for the most recent PIT case count snapshot
3. **Record the cumulative case count** as of the last available quarter date
4. **If the quarter file mentions the outbreak** but does not record a specific case count, this is a vault gap — note it for remediation

### Step 4: Assess the Trajectory Phase

Determine which phase the outbreak is in:

| Phase | Timing | Growth Pattern | Case Count Range |
|-------|--------|---------------|------------------|
| Detection delay | Months 1-3 | Sub-arithmetic, cases may be undetected | 1-5 observed (may be 2-5x undercount) |
| Agricultural amplification | Months 3-9 | Arithmetic growth, 3-7 new cases/week | 5-60+ |
| Plateau | Months 9+ | Sub-arithmetic or flat | 60-100+ |
| Pandemic (H2H) | Any | Exponential | Unbounded |

### Step 5: Check for Step-Change Events

If any of the following have occurred, the plateau model may be invalid:
- **Human-to-human transmission cluster confirmed** ← This is THE step-change event
- **Mammalian adaptation mutation detected** (e.g., PB2 E627K in H5N1, or HA receptor-binding domain changes)
- **CDC risk assessment upgraded from low**
- **Case trajectory shows accelerating growth** (weekly case count increasing faster than linearly)
- **Evidence of community transmission** (cases without known animal exposure)

If any step-change event has occurred → probability adjustment needed. If none → the plateau model holds.

### Step 6: Calibrate Probability

#### Zoonotic Spillover Only (Plateau Model)

- **Threshold <= current count**: YES (100%, the threshold has already been met)
- **Threshold <= 2x current count within 0-6 months**: MODERATE (30-50%) — depends on whether the outbreak is still in the amplification phase or has reached plateau
- **Threshold <= 2x current count within 6-12 months**: LOW (10-20%) — plateau dynamics make doubling unlikely
- **Threshold > 3x current count**: VERY LOW (<5%) — requires a step-change event
- **Threshold >= 100 within 12 months of first detection**: LOW (10-20%) — the 2024 H5N1 outbreak template shows ~60 cases at 12 months
- **Threshold >= 200 within 12 months**: VERY LOW (<5%) — requires H2H transmission or massive surveillance expansion

#### Factors That Adjust Probability Upward

| Factor | Adjustment | Notes |
|--------|-----------|-------|
| Outbreak is in amplification phase | +10-15% | Cases are still growing, not yet plateaued |
| Step-change event observed | Rethink model | Plateau model is invalid |
| CDC risk upgraded to moderate | +15-25% | Leading indicator of faster case accumulation |
| New agricultural host species | +5-10% | Expands the exposure base |
| Large culling/processing operation affected | +5-10% | High-exposure event can produce cluster |
| Democratic/reporting administration | +5% | Likelihood of complete reporting |

#### Factors That Adjust Probability Downward

| Factor | Adjustment | Notes |
|--------|-----------|-------|
| Outbreak is in plateau phase | -10-20% | Case growth has largely stopped |
| No H2H transmission for >6 months | -10-15% | Plateau is durable |
| CDC risk remains low | -10-15% | Agency sees no escalation |

### Step 7: Sanity Check

1. **Would this require H2H transmission?** If yes, p_yes should be very low unless H2H is already documented.
2. **Has the case count growth rate been accelerating or decelerating?** Decelerating = plateau confirmation.
3. **Is the deadline date close to the PIT cutoff?** A 2-week lookahead barely changes the case count — the plateau model predicts <5 new cases per week during amplification, fewer during plateau.
4. **What was the case count in the last quarter?** Extrapolate linearly from that baseline and check if the threshold is reachable.

## Example: H5N1 100+ Cases by Jan 31, 2025

Using this procedure in PIT (September 2024):

1. **Outbreak**: H5N1 avian influenza. Source: dairy cattle. Transmission mode: zoonotic spillover only.
2. **CDC risk**: Low (maintained throughout).
3. **Baseline (2024-Q3, Sep 30)**: <20 cases (~25-30 by end of Sep).
4. **Phase**: Agricultural amplification, transitioning to plateau. Weekly case count ~3-5 new cases.
5. **Step-change events**: None. No H2H transmission. No mammalian adaptation mutations. CDC risk still low.
6. **Calibration**: 100 cases from <20 in ~4 months = need ~80 new cases. At 3-5/week, that's 16-27 weeks — 4-7 months. Deadline (Jan 31) is ~4 months away. P(100 cases) = LOW (5-15%).
7. **Sanity check**: No H2H transmission means plateau model holds. <20 cases at Q3 makes 100 by Jan 31 unlikely without amplification acceleration, which the CDC risk assessment rules out.

**Correct forecast: NO.**

## Related Files

- [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]] — Full outbreak history
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept]] — Structural concept
- [[domains/global/entities/centers-for-disease-control-and-prevention]] — CDC entity stub
