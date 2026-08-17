---
type: procedure
tags: [procedure, health, respiratory, forecasting]
title: "Respiratory Disease Forecast"
slug: respiratory-forecast
domain: "[[domains/health]]"
concepts:
  - "[[domains/health/concepts/seasonal-baseline/_concept]]"
  - "[[domains/health/concepts/outbreak-escalation/_concept]]"
  - "[[domains/health/concepts/vaccine-effectiveness/_concept]]"
entities:
  - "[[domains/health/entities/cdc.md]]"
  - "[[domains/health/entities/who.md]]"
  - "[[domains/health/entities/cms.md]]"
---
---
---
# Respiratory Disease Forecast Procedure

Compute P(outcome) for Metaculus respiratory tournament questions,
including ILI thresholds, hospitalization counts, epidemic declarations,
and intervention likelihood.

## When to Use

Any question about US respiratory disease activity during an active
season, including:
- "Will ILI% exceed the national baseline by Week X?"
- "Will total influenza hospitalizations exceed Y by end of season?"
- "Will an influenza epidemic be declared before date Z?"
- "Will [HHS region] trigger regional epidemic threshold?"
- "Will vaccine effectiveness exceed W% this season?"

## Procedure

### Step 1: Establish PIT Baseline

For any question, determine the current PIT values:

1. **Check [[domains/health/threads/respiratory-season-2025-26/_thread]]**
   for current season status and cutoff dates.
2. **Pull latest FluView data** from [[domains/health/entities/cdc.md]]
   — current national ILI%, regional ILI%, lab confirmation rates, and
   hospitalization counts.
3. **Identify the question's resolution standard**: Is it the ILINet
   national baseline (1.9% for 2025-26), a regional baseline, a CMS
   hospitalization threshold, or a WHO/CDC declaration?

**Rationale**: Forecasts must be PIT-grounded. The current trajectory
is the anchor; any deviation from trend requires structural evidence.

### Step 2: Check Baselines and Thresholds

Apply [[domains/health/concepts/seasonal-baseline/_concept]]:

1. Confirm the relevant baseline (national = 1.9%; regional varies).
2. Plot current ILI% against baseline. If currently above baseline:
   calculate weeks-above-baseline streak; determine if lab confirmation
   ≥50% (epidemic condition).
3. If below baseline: assess trajectory — is the weekly change rate
   consistent with an imminent crossing? Use prior-season onset timing
   as a reference.
4. For composite respiratory metrics (ILI+CLI+RSV), apply a discount
   factor (×0.7–0.8) to confidence due to immature baselines.

**Rationale**: Baseline proximity is the strongest single predictor
for binary threshold questions. Most forecasting errors on threshold
questions come from misidentifying which baseline the question
references.

### Step 3: Assess Strain Surveillance and VE

Apply [[domains/health/concepts/vaccine-effectiveness/_concept]]:

1. **Strain match**: Check WHO's September strain update and CDC's
   antigenic characterization reports. Identify dominant circulating
   subtype (H1N1, H3N2, B/Victoria, B/Yamagata).
2. **Southern Hemisphere preview**: If SH season VE data is available
   (Australia/NZ, Apr–Sep), this is the leading indicator. Apply a
   correlation discount (SH VE typically predicts NH VE within ±15%
   for same dominant subtype).
3. **Uptake signal**: Check [[domains/health/entities/cms.md]] for
   early vaccine claims data and CDC for uptake surveys. If uptake
   is trending below 40%, cap effective population protection at a
   lower bound regardless of VE.
4. **Historical VE calibration**: Use the reference table in
   [[domains/health/concepts/vaccine-effectiveness/_concept]] for
   subtype-specific average VE. H3N2-dominant seasons average ~33% VE;
   H1N1 ~50%.

**Rationale**: VE modulates severity but does not determine it.
A poorly matched vaccine in a mild-strain season may still produce
low hospitalization counts. VE is a multiplier on severity, not a
severity metric itself.

### Step 4: Monitor Hospital Capacity

Apply CMS/NHSN data from [[domains/health/entities/cms.md]]:

1. **Hospitalization trajectory**: Plot weekly new influenza admissions
   from NHSN. Is the slope accelerating (+), linear, or decelerating (-)?
   Acceleration signals peak approaching; deceleration signals peak
   passed.
2. **Capacity thresholds**: For questions about "hospital at >X%
   capacity," pull CMS weekly bed occupancy data. National average
   occupancy ~65-75% in non-peak; >85% signals strain. ICU occupancy
   is a more sensitive strain metric.
3. **Pediatric vs. adult split**: Influenza A (especially H1N1)
   disproportionately affects children and young adults. Pediatric
   hospital bed occupancy may be a leading indicator for overall
   surge.
4. **RSV load**: RSV and influenza peaks are often staggered (RSV
   typically peaks earlier, Nov–Jan). If RSV admissions are declining
   while influenza admissions are rising, hospital capacity may
   actually be improving — do not double-count.

**Rationale**: Hospitalization data is the "hard" severity metric.
ILI outpatient visits can spike from mild illness that never reaches
the hospital. CMS data filters for medically significant disease.

### Step 5: Assess Intervention Likelihood

For questions about mask mandates, school closures, or public health
emergency declarations:

1. **Current political environment**: RFK Jr. at HHS (confirmed Feb
   2025) has expressed skepticism toward mandates. Default P(mandatory
   intervention) is lower than pre-2025 baseline.
2. **Intervention threshold precedent**: Historically, US jurisdictions
   have not imposed mask mandates for seasonal influenza alone. The
   threshold for mandates requires: (a) a novel pathogen with pandemic
   potential, OR (b) hospital capacity crisis with ≥95% ICU occupancy
   in a region. Seasonal influenza almost never reaches this threshold.
3. **State-level patchwork**: Some states (CA, NY, WA) have retained
   legal authority for public health emergency powers. Others (FL, TX)
   have restricted them. Intervention likelihood varies by jurisdiction.
4. **Default forecast**: For any question about a US-wide mask mandate
   or school closure due to seasonal influenza, default P < 0.05 unless
   a novel pandemic-potential variant is confirmed. The political and
   legal barriers are structurally prohibitive for seasonal flu.

**Rationale**: Intervention questions are political forecast questions
masquerading as health forecast questions. Analyze the political
incentives, not just the epidemiological conditions.

### Step 6: Check Escalation Risk

Apply [[domains/health/concepts/outbreak-escalation/_concept]]:

1. Classify current outbreak stage (Sporadic → Local → Regional →
   Epidemic → Pandemic).
2. If a novel influenza variant is detected, assess H2H transmission
   status. Without H2H → plateau model; P(escalation) < 0.10.
3. For H5N1 spillover overlay: check [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]]
   for current spillover case counts and the absence/presence of H2H.
   H5N1 tail risk is the primary "black swan" for seasonal respiratory
   forecasting.

**Rationale**: Most tournament questions are about seasonal pathogens
(normal escalation trajectory). A minority ask about novel variants or
spillover events. Correctly classifying the pathogen type determines
which base-rate framework to apply.

### Step 7: Synthesize

Combine findings using this priority order:

| Priority | Factor | Weight in Synthesis |
|----------|--------|---------------------|
| 1 | PIT baseline trajectory (current ILI%, hospitalization rate) | Primary anchor |
| 2 | Strain match and VE estimate | ±20-30% modifier on peak height |
| 3 | Hospital capacity and ICU trends | Severity confirmation signal |
| 4 | Southern Hemisphere analog | Corroboration ± weight depends on subtype match |
| 5 | Intervention likelihood | Binary overlay (only relevant for mandate questions) |
| 6 | Novel variant / spillover tail risk | Low-probability ceiling-buster |

**Output format**: P(outcome) = [probability estimate] with rationale
listing the top 3 evidence factors and their directional impact.

### Step 8: Post-Forecast Audit

After the question resolves:
- Was the ILINet baseline the correct threshold reference?
- Did the Southern Hemisphere VE predict NH VE within expected bounds?
- Was hospitalization trajectory consistent with early-season signals?
- Document the calibration in the relevant concept file's Validated By
  section.

## Key Files to Check (Per Question Type)

| Question Type | Primary File | Secondary File |
|--------------|--------------|----------------|
| ILI threshold | [[domains/health/concepts/seasonal-baseline/_concept]] | CDC FluView weekly |
| Hospitalization count | [[domains/health/entities/cms.md]] | NHSN data dashboard |
| Epidemic declaration | [[domains/health/concepts/seasonal-baseline/_concept]] | CDC MMWR |
| Vaccine effectiveness | [[domains/health/concepts/vaccine-effectiveness/_concept]] | WHO GISRS, AU/NZ data |
| Intervention / mandate | This procedure (Step 5) | State health dept announcements |
| Novel variant risk | [[domains/health/concepts/outbreak-escalation/_concept]] | [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]] |

## Wikilinks
[[domains/health/threads/respiratory-season-2025-26/_thread]]
[[domains/health/entities/cdc.md]] [[domains/health/entities/who.md]] [[domains/health/entities/cms.md]]
