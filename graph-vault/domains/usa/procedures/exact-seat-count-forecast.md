---
type: procedure
tags: [procedure, usa, elections, house, forecasting]
title: "Exact Seat Count Forecast Procedure"
slug: exact-seat-count-forecast
domain: usa
purpose: "Structure analytical reasoning for questions about exact House seat counts (not ranges)"
applies_to: "Any Polymarket or forecasting question asking whether a party will win exactly N seats in a US House election"
version: 1.0
date: 2026-05-20
---

# Exact Seat Count Forecast Procedure

## When to Load This Procedure

Load this procedure when the question asks about **a single exact integer** of House seats — e.g., "Will Republicans win exactly 223 seats?" This is structurally different from range questions (e.g., "Will Republicans win between 220 and 224 seats?"). Range questions use bin-level distributions; exact-count questions require within-bin distribution estimation.

## Prerequisites

Before using this procedure, ensure these vault assets are loaded:
- [[domains/usa/concepts/exact-count-vs-range-forecast/_concept]] (the exact-count vs. range distinction)
- [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]] (seat-vote distribution, bin-level tables)
- [[domains/usa/concepts/presidential-coattail-variability/_concept]] (coattail adjustment)
- [[domains/usa/threads/us-house-elections/_thread]] (context and recent dynamics)

## Procedure Steps

### Phase 1: First, Classify the Question Type

1. **Determine if the question is exact-count, range, threshold, or binary control**

   | Question Wording | Type | Governing Framework |
   |-----------------|------|-------------------|
   | "Exactly N seats" | Exact-count | This procedure |
   | "Between A and B seats" | Range | house-seat-range-forecast procedure |
   | "At least N seats" | Threshold | Cumulative distribution (P(seats >= N)) |
   | "Will party X control the House?" | Binary control | P(seats >= 218), threshold procedure |
   | "Control N seats" (no "exactly" qualifier) | **Check resolution text** | May be exact-count even if phrasing suggests threshold |

   **Control N Trap (CRITICAL)**: When the question says "control N seats" (e.g., "control 224 seats"), the resolution text may specify "exactly N" rather than "at least N." This transforms a threshold question into an exact-count question. Always read the full resolution text — do not rely on the question title alone. If the resolution text says "exactly N," use this procedure. If missing or ambiguous, forecast both interpretations as a sensitivity check. See [[domains/global/concepts/forecast-resolution-criteria-gotchas]] entry #8.

   **CRITICAL**: If the question is NOT an exact-count question, do NOT use this procedure. Load the appropriate alternative.

2. **Document the question type classification in the reasoning**
   - State explicitly: "This question asks about exactly N seats (exact-count), not a range."
   - Note whether the bin-level distribution table (from generic-ballot-seat-conversion) is the wrong tool — it must be disaggregated to individual seat probabilities.

### Phase 2: Build the Within-Bin Distribution

3. **Establish the generic ballot projection** (same as Phase 1 of house-seat-range-forecast)
   - Find the national House popular vote projection
   - Assess presidential coattail potential
   - Document the expected vote margin

4. **Load the bin-level distribution** from [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]]
   - For a competitive election (GOP popular vote 48-52%), use the default distribution
   - For other vote margins, apply the seat-vote conversion function to shift the distribution

5. **Disaggregate the relevant bin to individual seat count probabilities**

   The within-bin distribution is approximately normal with right-skew from gerrymandering. Use the [[domains/usa/functions/seat_distribution_probability]] Python function for precise calculations (it handles skew, floor effects, and normalization analytically).

   For quick reference, the table below is computed from the function with μ≈219, σ≈3.5, skew=0.5 (competitive regime):

   | Exact Seat Count | Approx. Probability | Confidence Grade |
   |-----------------|-------------------|-----------------|
   | 215 | 5% | Very High |
   | 216 | 7% | High |
   | 217 | 9% | High |
   | 218 | 10% | Moderate |
   | 219 | 11% | Moderate |
   | 220 | 12% | Moderate |
   | 221 | 11% | High |
   | 222 | 9% | High |
   | 223 | 7% | Very High |
   | 224 | 5% | Very High |
   | 225 | 3% | Very High |

   **Formula for estimating within-bin probability**:
   For a bin with probability P_bin and width w=5 seats, and assuming the bin is centered on the distribution mode:
   - Center probability (mode ± 1): ~0.30 × P_bin / 1
   - Adjacent (mode ± 2): ~0.25 × P_bin / 1  
   - Outer (mode ± 3): ~0.15 × P_bin / 1
   - Extremes (edges): ~0.05 × P_bin / 1

   **For bins not containing the mode**:
   - The bin's probability is concentrated at the edge closest to the mode
   - If the queried seat count is on the far side from the mode, probability is very low

### Phase 3: Assess the Specific Exact Count

6. **Look up the probability of the exact seat count from the disaggregated distribution**

7. **Apply boundary and skew corrections**
   - If the gerrymandering advantage creates right-skew (favors GOP), the above-normal seat counts have slightly elevated probability vs. the symmetric normal model
   - For GOP exact-count questions: add 1-2% to probabilities above μ and subtract 1-2% below μ
   - For Democratic exact-count questions: subtract 1-2% above μ and add 1-2% below μ
   - Document the skew adjustment explicitly

8. **Apply the structural floor / ceiling adjustment**
   - GOP structural floor at ~213 seats: below-floor seat counts have near-zero probability regardless of what the normal model suggests
   - GOP structural ceiling high: above-230 seat counts have rapidly diminishing probability
   - These floors/ceilings compress the distribution and shift within-bin probabilities toward the center

### Phase 4: Calibrate the Probability

9. **Calculate p_yes from the within-bin distribution**
   - p_yes = P(exact_count) from the disaggregated model
   - For the competitive regime: p_yes = 3-12% depending on position relative to mode
   - If the exact count is at the mode (e.g., 219 in 2024): p_yes ≈ 10-12%
   - If the exact count is 1 seat from mode: p_yes ≈ 8-10%
   - If the exact count is 2-3 seats from mode: p_yes ≈ 5-7%
   - If the exact count is 4+ seats from mode: p_yes ≈ 3-5%

10. **Apply the exact-count baseline discount**
    - ANY exact-count question in a 435-seat chamber has an inherent low-probability baseline
    - Even the most likely seat count has p_yes < 12-15%
    - No exact-count question in a full House election should ever have p_yes > 20% unless there is extraordinary evidence (e.g., a specific deal that produces exactly that number)
    - This baseline discount is the single most important calibration step — a forecaster who treats an exact-count question as similar to a range question will overestimate p_yes by 5-10x

11. **Apply the mode-distance confidence grading**
    
    Use this look-up to express calibrated confidence based on how far the queried count is from the distribution mode. Values derived from [[domains/usa/functions/seat_distribution_probability]] with μ≈219, σ≈3.5, skew=0.5:
    
    | Mode Distance | Probability | Confidence Grade | Expression |
    |--------------|------------|-----------------|------------|
    | 0 (mode) | 11-12% | Moderate | "Toss-up—even the most likely single outcome is unlikely. NO is favored but not nearly certain." |
    | ±1 seat | 10-12% | Moderate | "Within one standard error of the mode. NO is favored by ~9:1. The actual outcome could plausibly land here." |
    | ±2 seats | 7-9% | High | "Two seats from the mode — in the shoulder of the distribution. NO is favored by ~12:1. Only a wave could produce this." |
    | ±3 seats | 5-7% | High | "Three seats from the mode. NO is favored by ~17:1. Requires a clear shift in the national environment." |
    | ±4 seats | 3-5% | Very High | "Four seats from the mode — well into the distribution tail. NO is strongly favored (~25:1)." |
    | ±5+ seats | 1-3% | Very High | "Five or more seats from the mode — extreme tail. NO is near-certain (~40:1). Only massive polling errors or black swan events would shift the outcome this far." |

    For the 2024 regime (μ≈219, σ≈3.5, skew=0.5):
    - 224 is 5 seats above the mode: Very High confidence in NO. P(224) ≈ 5% (computed from function, up from hand-estimate of 3% due to skew correction).
    - 220 is 1 seat above mode: Moderate confidence. P(220) ≈ 12%. The actual outcome.
    - 215 is 4 seats below mode: Very High confidence in NO. P(215) ≈ 5% (left tail, skew correction reduces it).

    **Note on mode-distance vs. confidence**: The table above applies to the COMPETITIVE regime (generic ballot within 1 point, μ≈219). For other regimes (GOP wave, Democratic wave), the mode shifts and the distribution may widen or narrow. Adjust the mode position and re-reference the distance table.

12. **Check for mid-cycle adjustments**
    - Are there special election results that shift the expected seat count?
    - Has candidate quality in specific races changed?
    - Are there unique district-level factors that could "exactly" produce the queried number?
    - Generally, no — exact-count questions require extraordinary specificity that rarely exists in electoral projections

### Phase 5: Document Reasoning

12. **Write the analytical trace**
    - Question type classification (exact-count, range, threshold, or binary control)
    - Generic ballot margin used
    - Coattail assessment and adjustment
    - Bin-level distribution (from generic-ballot-seat-conversion)
    - Within-bin disaggregated probabilities
    - Probability of the exact count
    - Final p_yes estimate with confidence range
    - Comparison to what the range framework would produce (as a check — note the difference)

13. **Cross-reference vault assets**
    - Link to [[domains/usa/concepts/exact-count-vs-range-forecast/_concept]]
    - Link to [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]]
    - Link to [[domains/usa/concepts/presidential-coattail-variability/_concept]]
    - Link to [[domains/usa/threads/us-house-elections/_thread]]

## Common Pitfalls

### Pitfall 1: Applying range methodology to exact-count questions
- **Wrong**: "220-224 has 35% probability, so 223 is somewhat plausible — I'll predict NO but with doubt."
- **Right**: "The within-bin model shows P(223) ≈ 5%. Even at the mode, P(219) ≈ 11%. NO is a near-certainty at 95%."
- **Fix**: Always check question type first before loading the distribution model.

### Pitfall 2: Assuming uniform within-bin distribution
- **Wrong**: "220-224 has 35% across 5 seats, so each seat has ~7%."
- **Right**: "The within-bin distribution is concentrated near the bin edge closest to the mode. 220 (adjacent to mode at 219) has ~11%; 224 (far edge) has ~3%."
- **Fix**: Apply the within-bin probability formula, not simple division.

### Pitfall 3: Treating exact-count p_yes as comparable to range p_yes
- **Wrong**: "The range question gave 35%, so this exact-count question should be comparable."
- **Right**: "Exact-count p_yes is 5-10x lower than comparable range p_yes because it's asking about a single discrete outcome, not an interval."
- **Fix**: Document the factor difference as a sanity check.

### Pitfall 4: Overconfidence in NO
- **Wrong**: "P(223) is only 5%, so my probability estimate is nearly certain."
- **Right**: "P(223) is 5% individually, but the actual outcome (220) could easily have been 223 with a 2-3 seat swing in competitive districts. The correct p_yes is 5%, but the event is not impossible — just low probability."
- **Fix**: Always note the inherent uncertainty in multi-seat elections and the range of plausible outcomes.

## Validation

| Forecast | Procedure Prediction | Actual | Assessment |
|----------|--------------------|--------|------------|
| Republicans exactly 223 seats in 2024? | NO (p_yes ~5%) | 220 (NO) | Correct prediction. Procedure's within-bin model gives P(223) ≈ 5%, confirming NO as the clear forecast. The correct answer was not close — 223 was never a realistic outcome in a distribution centered at 219-220. |
| Republicans exactly 224 seats in 2024? | NO (p_yes ~5%) | 220 (NO) | Correct prediction. 224 was 5 seats above the distribution mode (μ≈219). The Python function (exact-seat-count-probability) gives P(224) ≈ 5.1% with skew=0.5. The mode-distance confidence table graded this as Very High confidence in NO.
| Republicans exactly 220 seats in 2024? | NO/YES (p_yes ~10%) | 220 (YES) | Correct probability assessment but wrong binary outcome — a 10% event that happened. Within the procedure's expected calibration: the mode has only ~10% probability individually, so the model correctly identifies it as "NO is favored but not certain." The mode-distance table would grade this as Moderate confidence. A 10% event hitting is expected ~1 in 10 forecasts.

## Wikilinks
[[domains/usa/concepts/exact-count-vs-range-forecast/_concept]], [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]], [[domains/usa/concepts/presidential-coattail-variability/_concept]], [[domains/usa/threads/us-house-elections/_thread]], [[domains/usa/procedures/house-seat-range-forecast]], [[domains/global/concepts/forecast-resolution-criteria-gotchas]]
