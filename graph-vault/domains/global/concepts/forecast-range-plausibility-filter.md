---
type: concept
tags: [concept]
title: "Forecast Range Plausibility Filter"
slug: forecast-range-plausibility-filter
first_observed: ~2024
domain: forecasting-methodology
related_concepts: [presidential-sentencing-dynamics, judicial-timing-political-deadline]
---

# Forecast Range Plausibility Filter

## Definition

A forecasting heuristic for questions that specify a numerical range (sentence length, price level, vote share, time window, age, percentage) as the resolution condition. These questions require a **double filter** assessment: (1) does the base event occur at all? AND (2) is the specified numerical range structurally plausible given the underlying phenomenon? If EITHER filter fails, the answer is NO — regardless of the base event's direction.

The most common forecasting error with range-specified questions is **collapsing the two filters into one** — treating "will X happen?" as equivalent to "will X happen at the specified magnitude?" and implicitly assuming the range is plausible. In many prediction market questions, the range itself is the trap: it describes an outcome that is structurally improbable regardless of whether the base event occurs.

This concept is distinct from simple probability multiplication (P(A and B) = P(A) × P(B|A)) because the two filters are often **structurally independent** — the range's implausibility comes from factors (sentencing guidelines, market fundamentals, electoral history) that have nothing to do with whether the base event occurs. The question designer may have selected the range precisely because it seems "in the middle" — making it a tempting prediction — while the structural range analysis shows it is actually a tail outcome.

## Canonical Example

### Trump Sentencing 24-35 Months (2025)

**Question**: Will Donald Trump be sentenced to between 24 and 35 months prison time for the New York hush-money conviction?

**Filter 1 — Base Event**: Does Trump receive any prison sentence at sentencing?
- Trump was president-elect at sentencing (January 10, 2025), 10 days before inauguration
- President-elect status created insurmountable practical obstacles (Secret Service, constitutional novelty)
- Manhattan DA conceded incarceration was "no longer a practicable recommendation"
- Result: NO — Trump received unconditional discharge (0 months prison)

**Filter 2 — Range Plausibility**: Even if the base event were possible, is 24-35 months a structurally plausible sentence range for this conviction?
- Trump was convicted on Class E felonies (NY Penal Law — lowest felony class)
- Class E maximum sentence: 4 years (48 months)
- Standard first-offender non-violent range: 0-16 months
- 24-35 months corresponds to Class C or D felony ranges — structurally disproportionate for a Class E first-time non-violent offender
- Would require aggravating factors (violence, large-scale fraud, criminal history) that do not exist
- Result: NO — the range is structurally implausible regardless of officeholder status

**Combined assessment**: Two independent NOs → NO. The double filter confirms the prediction even if one filter were uncertain.

**Validation**: The [[presidential-sentencing-dynamics]] concept predicted NO independently through Filter 1 analysis. The range plausibility filter (Filter 2) provides an additional, independent NO signal — making the combined prediction over-determined.

### Stock Price Target Questions (Hypothetical)

**Question**: Will Tesla trade above $500/share by June 30?

**Filter 1 — Base Event**: Will Tesla stock rise at all in the period?
- Depends on earnings, macro conditions, sector trends, company-specific news

**Filter 2 — Range Plausibility**: Given Tesla's current price ($X), volatility, market cap, and P/E ratio, is $500 a plausible price within the timeframe?
- $500 requires a specific percentage move from current price
- Compare to Tesla's historical realized volatility, implied volatility, and historical maximum moves
- Compare to the price level implied by analyst targets and fundamental valuation
- If $500 represents a >50% move in 6 months, this is a 2-3 standard deviation event — structurally improbable regardless of direction

### Vote Share Questions (Hypothetical)

**Question**: Will Candidate Y win more than 48% of the vote in a three-way race?

**Filter 1 — Base Event**: Will Candidate Y win? (Does anyone win?)

**Filter 2 — Range Plausibility**: In a three-way race with two other credible candidates, the winner typically wins with 35-45% in single-round plurality systems. 48% is near the theoretical maximum for a three-way split — requiring nearly all the remaining vote to consolidate. Unless one of the other candidates collapses completely before election day, 48% is structurally improbable regardless of who wins.

### Inflation/CPI Questions (Hypothetical)

**Question**: Will US CPI be between 4.0% and 4.5% in the December release?

**Filter 1 — Base Event**: Will CPI be elevated at all?

**Filter 2 — Range Plausibility**: Given the current inflation trajectory, base effects, energy price trajectory, and Fed policy stance, is a 4.0-4.5% reading plausible? Compare to the range of analyst forecasts, implied by the current month-over-month trend, and the historical distribution of CPI readings at this point in the cycle. The narrowness of the range (0.5 percentage points) is itself a structural constraint — CPI rarely lands in such a narrow band unless it is near a plateau.

## Pattern Archetype

### Stage 1: Question Presents a Numerical Range
The question specifies a range (duration, price, percentage, count) as the resolution condition. The range may be symmetric (X-Y) or one-sided (≥X).

### Stage 2: Identify the Two Filters
- **Filter A (Base Event)**: Does the underlying event occur at all?
- **Filter B (Range Plausibility)**: Is the specified range structurally plausible for the phenomenon, independent of the base event's direction?

### Stage 3: Assess Filter B First (Often Faster)
For many questions, Filter B can be assessed more quickly and with more certainty than Filter A:
- Sentence ranges relative to sentencing guidelines
- Price ranges relative to historical volatility
- Vote share ranges relative to electoral system structure
- Percentage ranges relative to base rates

If Filter B returns NO with high confidence, the answer is NO regardless of Filter A. This can save significant analytical effort and prevent forecast errors from misjudging the base event.

### Stage 4: Assess Filter A if Filter B is YES
If the range is structurally plausible, proceed to standard event forecasting — does the base event occur?

### Stage 5: Combine with Structural Independence
If both filters return YES, the combined probability is P(A) × P(B|A). If P(A) and P(B) are independent (the range's plausibility does not depend on whether the event occurs), this multiplication is straightforward.

## Key Variables

### Filter B Intensifiers (Range More Implausible)

| Variable | Effect | Examples |
|----------|--------|----------|
| **Narrow range relative to base distribution** | The narrower the range, the less likely the outcome lands in it | 24-35 months (11-month band) vs. 0-48 months (full range) |
| **Range corresponds to a different class/category** | Even if the base event occurs, the range belongs to a different structural category | Class E felony → Class C/D range |
| **Range is near the theoretical maximum** | Outcomes at the extreme of the distribution require exceptional conditions | 48% in a 3-way race, maximum sentence |
| **Range is below the theoretical minimum** | Floor effects make very low ranges structurally unlikely | "0.5% inflation" when base effects keep it above 2% |
| **Range spans a discontinuity** | The range crosses a structural boundary (threshold, trigger, round number) that creates non-linear probability | 3.9-4.1% CPI (around a psychological round number) |

### Filter B Attenuators (Range More Plausible)

| Variable | Effect | Examples |
|----------|--------|----------|
| **Wide range relative to base distribution** | Higher probability of landing in the band | 0-48 months (full range) |
| **Range corresponds to standard outcome** | Normal outcomes are more likely than tail outcomes | 0-16 months for Class E first offender |
| **Range anchored at a known forecast target** | If analysts/insiders are targeting the range, it becomes more plausible | Earnings guidance range |
| **Range overlaps with historical mode** | Outcomes that have occurred frequently are more likely | 4.5-5.0% unemployment |

## Forecasting Application

### Step 1: Parse the Resolution Text for Numerical Range
Before doing any analysis, extract the numerical range from the question. Write it down explicitly.

### Step 2: Apply the Range Plausibility Pre-Check
For the specific phenomenon type (sentencing, price, vote share, inflation), ask:
- What is the full possible range of outcomes?
- What is the standard/modal outcome?
- What distribution does the specified range fall in? (Normal, tail, off-distribution?)
- What structural constraints bound the range? (Legal guidelines, market mechanics, electoral system)

### Step 3: Document Both Filters Separately
Write two separate assessments:
- **Filter A**: Does the base event occur? (YES/NO/Uncertain with probability)
- **Filter B**: Is the range structurally plausible? (YES/NO/Uncertain with probability)

### Step 4: Combine
- If Filter B = NO with >95% confidence: Answer is NO regardless of Filter A.
- If Filter B = NO with moderate confidence: P(overall YES) = P(Filter A YES) × P(Filter B YES|A) — typically very low.
- If Filter B = YES: P(overall YES) = P(Filter A YES) × P(Filter B YES|A the event occurs).

### Step 5: Check for Question-Design Trap
If the range is the most interesting/reportable outcome (not too extreme, not too boring), consider whether the question designer chose it precisely because it seems plausible to an unsophisticated forecaster. The most valuable prediction market questions often have resolution conditions that sound reasonable but fail the range plausibility filter.

## Validated By

| Forecast | Prediction | Actual | Concept Support |
|----------|-----------|--------|-----------------|
| Trump sentenced to 24-35 months prison? | NO (correct) | Unconditional discharge — 0 months | Double filter: Filter A (no prison for president-elect) + Filter B (Class E conviction makes 24-35 months structurally disproportionate) — both independently returned NO |

## Wikilinks

[[domains/global/concepts/presidential-sentencing-dynamics]]
[[domains/usa/procedures/proc-sentencing-range-forecast]]
[[domains/global/concepts/judicial-timing-political-deadline]]
