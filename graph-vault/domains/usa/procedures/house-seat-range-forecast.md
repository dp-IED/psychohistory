---
type: procedure
tags: [procedure, usa, elections, house, forecasting]
title: "House Seat Range Forecast Procedure"
slug: house-seat-range-forecast
domain: usa
purpose: "Structure analytical reasoning for questions about specific House seat count ranges"
applies_to: "Any Polymarket or forecasting question asking whether a party will win a specific seat count or range in a US House election"
version: 1.0
date: 2026-05-20
---

# House Seat Range Forecast Procedure

## When to Load This Procedure

Load this procedure when the question asks about a **specific numerical range** of House seats for a party — e.g., "Will Republicans have between 220 and 224 seats?" or "Will Democrats have at least 235 seats?" This is distinct from questions about which party will control the House; range questions require probability distribution reasoning.

## Prerequisites

Before using this procedure, ensure these vault assets are loaded:
- [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]] (seat-vote distribution)
- [[domains/usa/concepts/presidential-coattail-variability/_concept]] (coattail adjustment)
- [[domains/usa/threads/us-house-elections/_thread]] (context and recent dynamics)

## Procedure Steps

### Phase 1: Establish the Generic Ballot Baseline

1. **Find the national House popular vote projection**
   - Check polling averages (FiveThirtyEight, RealClearPolitics, 270toWin) for generic ballot
   - Record the Democrat-Republican margin in the national House popular vote
   - Note whether the polling is reliable (within 60 days of election) or speculative (earlier)
   - If no polling exists, estimate from presidential polling + typical split-ticket rate (~5%)

2. **Assess presidential coattail potential**
   - Load [[domains/usa/concepts/presidential-coattail-variability/_concept]]
   - Assess whether the presidential candidate is likely to produce coattail effects
   - Key factors: candidate freshness (incumbent/former vs. newcomer), national margin, ticket-splitting trend
   - Apply the coattail adjustment: weak/none = 0 seats, moderate = +2-3 seats, strong = +4-6 seats

### Phase 2: Build the Seat Distribution

3. **Load the seat-vote distribution model**
   - From [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]], get the probability distribution for the current regime (GOP wave, competitive, Democratic wave)
   - For a **competitive** election (GOP popular vote 48-52%), use the primary distribution table:

   | GOP Seat Range | Approx. Probability | Cumulative |
   |---------------|-------------------|------------|
   | 210-214 | 5% | 5% |
   | 215-219 | 30% | 35% |
   | 220-224 | 35% | 70% |
   | 225-229 | 20% | 90% |
   | 230+ | 10% | 100% |

   - Adjust the distribution based on the specific generic ballot margin:
     - Every +1% for GOP → shift distribution right by ~4-5 seats
     - Every +1% for Democrats → shift distribution left by ~4-5 seats
     - The shift is asymmetric: the leftward shift (GOP loss) is shallower due to the GOP structural floor

4. **Apply coattail adjustment**
   - If coattails are active, shift the distribution right by the assessed number of seats
   - Document the adjusted distribution

5. **Check for redistricting shocks**
   - Has a new redistricting cycle occurred since the distribution model was calibrated?
   - Are there court-ordered map changes in key states?
   - Has there been a retirement surge that changes the competitive district count?
   - If yes, apply an ad-hoc adjustment and document uncertainty

### Phase 3: Assess the Specific Range

6. **Map the question's range onto the distribution**
   - Identify the exact range boundaries (e.g., 220-224, meaning seats 220, 221, 222, 223, 224)
   - Sum the probability mass for all seat counts within the range
   - If using 5-seat bins: the probability of the exact 5-seat bin containing the range

7. **Assess boundary sensitivity**
   - Where is the range relative to the central tendency?
   
   | Range Position | Probability | Notes |
   |---------------|------------|-------|
   | Well below central tendency | Very low (~5%) | Only if Democratic wave or major polling miss |
   | Just below central tendency | Low-moderate (~15-20%) | Captures lower tail — plausible but not likely |
   | Covers central tendency | High (~40-50%) | Most likely outcome |
   | Just above central tendency | Moderate (~25-35%) | Right-skewed tail captures significant mass |
   | Well above central tendency | Low (~10%) | Only if GOP wave or coattail surprise |

   - **Boundary trap warning**: When the range's lower bound is AT or near the central tendency (as with 220-224 in 2024 where central tendency was 219), the question is maximally sensitive to small shifts. A 2-3 seat swing in competitive districts can change the answer. The probability of YES may be <50% (justifying a NO prediction) but the actual outcome could easily be YES.

8. **Apply the "peak density" heuristic**
   - If the question's range includes the single most likely seat count (the mode), the probability is higher than the bin probability alone suggests because the outcome is likely to cluster near the mode
   - If the question's range excludes the mode but is adjacent to it, check how steeply probability falls off at that boundary
   - If the mode is at 219 and the range starts at 220, the probability mass at 220 may be nearly as high as at 219 — don't assume a sharp drop-off

### Phase 4: Calibrate the Probability

9. **Calculate p_yes from the distribution**
   - Sum the probability of all seat counts in the range
   - Apply continuity correction: since we're using 5-seat bins, interpolate if the range boundaries don't align with bin edges
   - Example: Range 220-224 in the competitive distribution → ~35%. This gives NO (p_yes < 50%), but with moderate confidence (35% chance of being wrong).

10. **Apply the boundary risk discount**
    - If the question's range is within 1 seat of the central tendency, discount confidence by 10-15 percentage points
    - At a tied popular vote with central tendency ~219, the range 220-224 has a ~35% probability but is sensitive to small shifts → express p_yes as 30-40% (bimodal estimate)

11. **Check for mid-cycle adjustments**
    - Are there special election results that suggest the generic ballot projection is wrong?
    - Has candidate quality in specific races changed the competitive district count?
    - Are there polling surprises in specific districts that affect the seat count?
    - Adjust the distribution if evidence warrants

### Phase 5: Document Reasoning

12. **Write the analytical trace**
    - Generic ballot margin used
    - Coattail assessment and adjustment
    - Seat distribution table (before and after adjustments)
    - Range mapping with probability calculation
    - Boundary sensitivity assessment
    - Final probability estimate with confidence range

13. **Cross-reference vault assets**
    - Link to [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]] for the distribution model
    - Link to [[domains/usa/concepts/presidential-coattail-variability/_concept]] for the coattail assessment
    - Link to [[domains/usa/threads/us-house-elections/_thread]] for recent dynamics

## Common Pitfalls

### Pitfall 1: Using point estimates instead of distributions
- **Wrong**: "The generic ballot is tied, so GOP gets ~217 seats, so 220 is too high for YES"
- **Right**: "The distribution shows 35% of outcomes fall in 220-224, and 220 is at the distribution's mode"
- **Fix**: Always construct the full distribution, not just the point estimate

### Pitfall 2: Ignoring structural bias direction
- **Wrong**: "The range 220-224 seems high, so it's unlikely"
- **Right**: "The GOP structural bias is ~5-8 seats, so in a tied election the GOP seat count is right-skewed, making 220-224 the modal bin"
- **Fix**: Check which direction the structural bias runs and how it shifts the distribution

### Pitfall 3: Treating all 5-seat ranges as equally likely
- **Wrong**: "There are 87 possible 5-seat ranges (219-223, 220-224, etc.), so each has ~1.1% probability"
- **Right**: "The distribution is highly concentrated in 215-229 (90% of outcomes), so the effective range of interest is ~15 seats, and a 5-seat bin covers ~1/3 of the probability mass"
- **Fix**: First compute the effective range (5th to 95th percentile), then assess probability density within it

### Pitfall 4: Underestimating probability at range boundaries
- **Wrong**: "The outcome was 220, which is the bottom boundary of 220-224, so I was almost right"
- **Right**: "The outcome was 220, which falls within the range. The question resolves YES, and the ~35% probability of YES was realized. This is not a near-miss — it's a correct resolution."
- **Fix**: Assign probability to the full range, not just the center of the range. Outcomes at the boundary are still within the range.

## Validation

| Forecast | Procedure Prediction | Actual | Assessment |
|----------|--------------------|--------|------------|
| Republicans 220-224 seats in 2024? | NO (p_yes ~35%) | 220 (YES) | Procedure gave correct p_yes assessment but wrong binary outcome. This is expected — a 35% event happens ~1/3 of the time. The distribution model was sound; the prediction was inherently uncertain. |
| Republicans 215-219 seats in 2024? | NO (p_yes ~30%) | 220 (NO) | Correct prediction. Procedure's distribution model identified this range as capturing the left-of-mode tail, making it unlikely. |

## Wikilinks
[[domains/usa/concepts/generic-ballot-seat-conversion/_concept]], [[domains/usa/concepts/presidential-coattail-variability/_concept]], [[domains/usa/threads/us-house-elections/_thread]], [[domains/usa/procedures/proc-aging-incumbent-early-warning]], [[timeline/2024-Q4]]
