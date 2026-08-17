---
type: forecast
tags: [forecast, usa, legal, trump, correct]
question: "Trump sentenced to between 24 and 35 months prison time?"
market_slug: q15
prediction: "NO"
actual: "NO (unconditional discharge — no jail)"
cutoff_date: "2024-06-15"
date: 2026-05-18
---

# Forecast: Trump Sentenced 24-35 Months (New York Hush Money)

## Result
CORRECT. Predicted NO, actual NO. Trump was sentenced to unconditional discharge on January 10, 2025 — no jail, no probation, no fine.

## Reasoning Trace

At relevant cutoff (pre-conviction, mid-2024):

**Filter A — Base Event (Any prison sentence):**
- Trump was a first-time non-violent offender (age 78 at sentencing) convicted on Class E felonies (lowest NY felony class, max 4 years)
- After conviction, Trump won the November 2024 election and became president-elect by sentencing (Jan 10, 2025)
- President-elect status made ANY incarceration functionally impossible: Secret Service logistics, constitutional novelty, DOJ policy against prosecuting a sitting president
- Manhattan DA Alvin Bragg conceded post-election that incarceration was "no longer a practicable recommendation"
- Result: P(any prison) < 5%

**Filter B — Range Plausibility (24-35 months specifically):**
- Class E maximum sentence: 4 years (48 months)
- Standard first-offender non-violent range: 0-16 months
- 24-35 months corresponds to the range for a **Class C or D felony** (assault, burglary), not a Class E business-records falsification
- Would require aggravating factors (violence, large-scale fraud, criminal history, obstruction) that do not exist in this case
- 24-35 months exceeds the standard Class E range even for non-officeholder defendants
- Result: Range is structurally disproportionate — would need both aggravating factors AND officeholder constraints to be overcome, which is doubly impossible

**Combined assessment:**
- P(any prison) < 5% × P(range plausible) < 5% → P(24-35 months) < 0.25%
- The double filter framework ([[domains/global/concepts/forecast-range-plausibility-filter]]) independently returns NO for each filter
- Even if Trump were a non-officeholder with no special constraints, the 24-35 month range would be structurally disproportionate for a Class E first offender

## Vault Contribution Score: Full (100%)

The vault fully supported this prediction:
- [[domains/usa/concepts/presidential-sentencing-dynamics]] — explicit 24-35 month range coverage (line 143-144): "P(sentence in this range) < 5% even for a non-officeholder Class E defendant. With officeholder status: P < 1%"
- [[domains/global/concepts/forecast-range-plausibility-filter]] — 24-35 months is the canonical example; double filter analysis already documented
- [[domains/usa/procedures/proc-sentencing-range-forecast]] — formalized step-by-step procedure
- [[domains/usa/entities/juan-merchan]] — judge behavior leading indicator analysis
- [[domains/usa/entities/alvin-bragg]] — prosecution posture documentation
- [[domains/usa/threads/trump-criminal-cases/_thread]] — four-case timeline with sentencing resolution

**Vault gaps addressed in this reflection cycle (Q15):**
1. Created forecast record for 24-35 month variant (this file was missing)
2. Created [[domains/global/concepts/sentencing-delay-cascade]] — extracted the "judge delay as leading indicator of leniency" pattern from Merchan entity into a generalizable concept
3. Updated _procedure to add multi-range question handling step

## Wikilinks
[[domains/usa/concepts/presidential-sentencing-dynamics]]
[[domains/global/concepts/forecast-range-plausibility-filter]]
[[domains/global/concepts/sentencing-delay-cascade]]
[[domains/usa/procedures/proc-sentencing-range-forecast]]
[[domains/usa/entities/donald-trump]]
[[domains/usa/entities/juan-merchan]]
[[domains/usa/entities/alvin-bragg]]
[[domains/usa/threads/trump-criminal-cases/_thread]]
