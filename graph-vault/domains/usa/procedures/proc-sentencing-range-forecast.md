---
type: procedure
tags: [procedure]
title: "Sentencing Range Forecast Procedure"
slug: proc-sentencing-range-forecast
domain: usa
related_concepts:
  - "[[domains/global/concepts/forecast-range-plausibility-filter]]"
  - "[[domains/usa/concepts/presidential-sentencing-dynamics]]"
  - "[[domains/usa/concepts/judicial-timing-political-deadline]]"
prerequisites:
  - "[[domains/usa/entities/donald-trump]] (or relevant defendant)"
  - "[[domains/usa/concepts/presidential-sentencing-dynamics]]"
---

# Sentencing Range Forecast Procedure

## When to Load

Load this procedure when facing a prediction-market question that specifies:
- A particular sentence range (e.g., "sentenced to 24-35 months")
- A particular sentence type (e.g., "sentenced to prison time")
- A particular sentence duration threshold (e.g., "sentenced to at least 1 year")

The procedure applies to US criminal sentencing questions, with special emphasis on NY state sentencing (the most common venue for political-figure prosecutions) and federal sentencing (for non-political cases).

## Step 1: Identify the Defendant's Status at the Expected Sentencing Date

Before analyzing the sentencing itself, determine the defendant's status at the time sentencing is expected to occur:

### Presidential/Officeholder Status Gate

- **President-elect or sitting president**: Incarceration probability drops below 5% (see [[presidential-sentencing-dynamics]])
- **Former officeholder with no prospect of return**: Standard sentencing factors apply
- **Current officeholder (below presidential)**: Some constraint, but less than presidential — assess case by case
- **No officeholder status**: No constraint from this factor

**Key timing question**: Is the sentencing date BEFORE or AFTER the election (or other status-changing event)? If AFTER, and the defendant won, officeholder constraints dominate regardless of pre-election legal posture.

### Prosecution Posture Check

Check whether the prosecution has changed its sentencing recommendation since any status change:
- Has the prosecution conceded that incarceration is not "practicable" or "appropriate"?
- Is the same DA's office that secured the conviction still advocating for a specific sentence?
- If the prosecution has conceded, the judge nearly always follows — this is the single most informative signal for the base question of "will any prison time occur?"

## Step 2: Apply the Range-Plausibility Double Filter

Use the [[domains/global/concepts/forecast-range-plausibility-filter]] framework:

### Filter A — Base Event: Does the defendant receive ANY prison sentence?

Assess the following factors:

| Factor | Signal | Weight |
|--------|--------|--------|
| Defendant is president-elect/sitting president | P(prison) < 5% | Maximum |
| Prosecution conceded incarceration is impractical | Judge nearly always follows | Very high |
| Defendant is age 70+ / first-time non-violent offender | Standard practice favors probation/discharge | High |
| Felony class severity | Higher class → higher prison probability | Medium-High |
| Judge's demonstrated sentencing severity | Prior sentencing history for similar cases | Medium |
| Jurisdiction (state vs. federal) | State judges have more discretion; federal guidelines constrain | Medium |

### Filter B — Range Plausibility: Is the specified sentence range structurally plausible?

For NY state sentencing (the most common venue for political-figure criminal cases):

**NY Felony Class Sentencing Reference Table:**

| Class | Max Sentence | Standard Range (First Offense, Non-Violent) | Notes |
|-------|-------------|---------------------------------------------|-------|
| A-I | Life, min 15-25 yrs | Not applicable | Murder 1st degree |
| A-II | Life, min 3-8 yrs | Not applicable | Murder 2nd degree |
| B | 25 years | 1-3 to 5-15 yrs depending on sub-class | Violence typically involved |
| C | 15 years | 0-15 yrs; often 3-8 yrs | Assault 1st, Burglary 2nd |
| D | 7 years | 0-7 yrs; often 1-3 yrs | Assault 2nd |
| E | 4 years (48 months) | **0-16 months** | **Falsifying business records** |

**Range Assessment Rules:**

1. **Find the conviction's felony class** — document the specific class and maximum sentence
2. **Calculate the standard first-offender range** — for most non-violent first offenses, the standard sentencing range is 0 to 1/3 of the maximum (rounded to guidelines)
3. **Compare the question's range to the standard range**:
   - If the question's range is WITHIN the standard first-offender range → Filter B = YES (range is plausible for a non-officeholder)
   - If the question's range EXCEEDS the standard range but is within the statutory maximum → Filter B = Conditional (aggravating factors needed — assess if present)
   - If the question's range EXCEEDS the statutory maximum for the conviction class → Filter B = NO (range is structurally impossible regardless of any factor)
   - If the question's range corresponds to a HIGHER felony class → Filter B = NO (the range is structurally disproportionate to the offense class)

4. **Apply officeholder adjustment**: If the defendant faces any officeholder constraint (Step 1), multiply Filter B by the constraint factor — a range that is "conditionally plausible" for a non-officeholder becomes implausible for an officeholder.

**For federal sentencing**: Use the US Sentencing Guidelines range instead of NY class-based ranges. Federal sentencing uses a grid of offense level (severity) × criminal history category. Calculate the guideline range and compare to the question's range.

## Step 3: Assess the Judge

The judge's demonstrated sentencing tendencies are the weakest but still relevant signal:

- Has this judge imposed sentences at the specified range for similar offenses?
- Has the judge shown procedural flexibility (scheduling delays, continuances) toward this defendant? If yes → signals caution, which correlates with lighter sentences. Apply the [[domains/global/concepts/sentencing-delay-cascade]] framework: 2+ delays = moderate-high signal; 3+ delays = near-deterministic leniency.
- Has the judge written opinions suggesting a punitive orientation toward the offense type?
- Does the judge have the institutional confidence to impose a novel sentence in a high-profile case?

**Leading indicator**: A judge who grants repeated sentencing delays is signaling caution. Each delay makes the final sentence more likely to be lenient, because the delay pushes the sentencing closer to or past a status-changing event (election, transition) and because the judge is building a bridge for a safe, defenisible outcome.

## Step 4: Combine into Probability

### Base Probability (Non-Officeholder, No Special Factors)

For a question about a specific sentence range for a non-officeholder:

| Conviction Class | P(Range = Standard 0-1/3 max) | P(Range = Aggravated) | P(Range = Max) |
|-----------------|------------------------------|----------------------|----------------|
| Class E (first offense, non-violent) | >95% | <5% | <1% |
| Class E (with aggravating factors) | 60-80% | 15-30% | 5-10% |
| Class D (first offense) | 70-90% | 8-20% | 2-5% |

### With Officeholder Constraint

If the defendant is a president-elect or sitting president:

| Base Outcome | Adjusted P | Rationale |
|-------------|-----------|-----------|
| Any prison time | <5% | Practical obstacles + prosecution concession |
| Unconditional discharge | 60-80% | Judge defaults to least novel outcome |
| Conditional discharge or probation | 10-25% | Requires supervision infrastructure |
| Fine only | 5-10% | Possible but rare for felony |
| Community service | <5% | Logistically complex for officeholder |

### Range-Specific Calibration

P(SpecificRange) = P(AnySentence) × P(RangeIsPlausibleForThisClass) × P(JudgeSelectsThisRange | RangeIsPlausible)

- P(AnySentence): From Step 1 and Step 2 Filter A
- P(RangeIsPlausible): From Step 2 Filter B
- P(JudgeSelectsThisRange | Plausible): From Step 3; typically low for specific narrow ranges

## Step 5: Check for Multi-Range Questions (Same Event, Different Ranges)

When the same underlying event has multiple range-specified questions (e.g., Q14: "12-23 months" and Q15: "24-35 months" for the same Trump sentencing), apply the following:

1. **Treat each range independently** — do not assume that one range being implausible means the other is plausible. Each range must pass its own double-filter test.

2. **Identify which ranges are within the standard range** vs. which require aggravating factors. For the Trump NY case: 12-23 months exceeded the standard Class E range (0-16 months) but was within the statutory maximum (48 months). 24-35 months corresponded to a HIGHER felony class (Class C/D), making it doubly implausible.

3. **The question's range value provides information about its structural plausibility.** The narrower and more specific the range, the less likely the outcome lands in it. A range that equals or exceeds higher felony class ranges is structurally disproportionate regardless of event outcome.

4. **Document the relationship between ranges**: Are they overlapping? Disjoint? One contained within another? The relationship determines whether the questions are mutually exclusive or overlapping in outcome space.

5. **The combined probability across all ranges for the same event should sum to P(any sentence)**. If the market is offering questions for multiple disjoint ranges, the sum of their probabilities should approximately equal the probability of any sentence at all. This provides a consistency check: if P(12-23 months) + P(24-35 months) + P(36-48 months) > P(any prison), the probabilities are inconsistent.

## Step 6: Document the Reasoning

Write a structured reasoning paragraph containing:

1. **Defendant's status at sentencing**: [Status, any change expected]
2. **Felony class and standard range**: [Class, max sentence, standard first-offender range]
3. **Range implausibility check**: [Does the question's range fall within the standard range? If not, what does it correspond to?]
4. **Prosecution posture**: [What is the prosecution recommending? Has it changed?]
5. **Judge signal**: [Any leading indicators from pre-sentencing behavior?]
6. **Combined assessment**: [P(any prison) × P(range plausible | any prison) × P(judge selects range)]

## References

- [[domains/global/concepts/forecast-range-plausibility-filter]] — general double-filter heuristic
- [[domains/usa/concepts/presidential-sentencing-dynamics]] — full framework for officeholder sentencing
- [[domains/usa/concepts/judicial-timing-political-deadline]] — delay strategy dynamics
- [[domains/global/concepts/sentencing-delay-cascade]] — judge delay cascade as leading indicator of leniency
- [[domains/usa/entities/juan-merchan]] — canonical cautious judge template
- [[domains/usa/entities/alvin-bragg]] — canonical prosecution posture template
- [[domains/usa/threads/trump-criminal-cases/_thread]] — timeline and case interaction
