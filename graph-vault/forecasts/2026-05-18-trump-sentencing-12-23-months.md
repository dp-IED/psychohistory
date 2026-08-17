---
type: forecast
tags: [forecast, usa, legal, trump, correct]
question: "Trump sentenced to between 12 and 23 months prison time?"
market_slug: q14
prediction: "NO"
actual: "NO (unconditional discharge — no jail)"
cutoff_date: "2024-06-15"
date: 2026-05-18
---

# Forecast: Trump Sentenced 12-23 Months (New York Hush Money)

## Result
CORRECT. Predicted NO, actual NO. Trump was sentenced to unconditional discharge on January 10, 2025 — no jail, no probation, no fine.

## Reasoning Trace

At relevant cutoff (pre-conviction, mid-2024):
- Trump was a first-time non-violent offender (age 78 at sentencing) convicted on Class E felonies (lowest NY felony class, max 4 years)
- The standard sentencing range for a Class E first-time non-violent offender is 0-16 months, with unconditional discharge or probation being the modal outcomes
- The specified range of 12-23 months EXCEEDS the standard first-offender range even before considering any officeholder or political factors
- For incarceration to reach 12-23 months, aggravating factors (violence, large-scale fraud, criminal history, obstruction) would need to be present — none existed for this case
- Additionally, post-conviction developments (Trump winning the election, becoming president-elect) made ANY incarceration functionally impossible due to Secret Service logistics, constitutional novelty, and prosecution concession

The prediction passed the double-filter test:
- **Filter A (Base Event)** : P(Trump receives any prison sentence) < 5% due to officeholder status, prosecution posture, and practical obstacles
- **Filter B (Range Plausibility)** : Even for a non-officeholder, 12-23 months exceeds the standard Class E first-offender range of 0-16 months — the range itself is structurally improbable for this offense and defendant profile

## Vault Contribution Score: Partial (20%)

The vault contributed limited signal:
- The [[domains/usa/threads/trump-criminal-cases/_thread]] correctly tracked sentencing delays and the post-election status shift
- The [[domains/usa/concepts/judicial-timing-political-deadline]] concept identified that post-election sentencing would face practical obstacles
- **Missing at time of forecast**: No entity stub for Juan Merchan (sentencing judge), no entity stub for Alvin Bragg (prosecutor whose concession was decisive), no sentencing-specific concept, no range-plausibility filter framework, and the Donald Trump entity was a 0-byte empty stub

## Remediation (completed in reflection)
1. Created [[domains/usa/entities/juan-merchan]] — sentencing judge entity with judicial tendency analysis
2. Created [[domains/usa/entities/alvin-bragg]] — Manhattan DA entity with prosecution posture documentation
3. Filled [[domains/usa/entities/donald-trump]] — was 0-byte empty stub, now complete
4. Created [[domains/usa/concepts/presidential-sentencing-dynamics]] — five-stage framework for officeholder sentencing
5. Created [[domains/global/concepts/forecast-range-plausibility-filter]] — double-filter heuristic for range-specified questions
6. Updated [[domains/usa/threads/trump-criminal-cases/_thread]] — added sentencing resolution
7. Updated [[timeline/2025-Q1]] — added Jan 10 sentencing outcome (was missing)
8. Updated [[_procedure]] — added Step 18 (sentencing feasibility assessment)
9. Added Spec Rule (sentencing range double-filter mandate) — ensures future range questions use the structural independence framework
