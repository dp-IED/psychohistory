---
type: reflection
tags: [reflection, usa, house, range-forecasting, formula-error]
question: gold_97
forecast: "NO"
actual: "YES"
error_type: "missing-concept-error + schema-error"
date: 2026-05-20
---

# Per-Question Reflection: gold_97 (Republicans 220-224 House Seats)

## Diagnosis

**Error type**: Missing-concept error (no probability distribution model for range questions) + schema error (flawed formula in generic-ballot-seat-conversion concept systematically underestimated GOP seats).

**Root cause**: The vault's generic-ballot-seat-conversion concept contained a formula (`GOP Seats ≈ 212 + (GOP Popular Vote Margin / 1.5)`) that was never validated against historical data. For the 2024 tied-popular-vote case, the formula predicted ~212 GOP seats — 8 seats below the actual 220. This formula was used in reasoning, producing a point estimate of ~217-219, which made the 220-224 range seem unlikely.

Additionally, even a correct point estimate of ~219 would have led to a wrong prediction for the 220-224 range, because the correct probability (~35%) is still below 50%. The underlying issue was not just the wrong formula but the lack of distribution-level thinking: the vault had no framework for assessing how much probability mass a specific range captured.

## Vault Gaps Identified

1. **Flawed formula in generic-ballot-seat-conversion concept**: The formula `212 + (margin / 1.5)` was off by 8 seats. It may have been a typo (intended slope of ~4.5 instead of 1.5) but regardless, it should never have been added without validation against the full 2012-2024 cycle. Now replaced with a validated range table and wrong-formula alert.

2. **No probability distribution model**: The concept had a "most likely range" but no full probability distribution over seat outcomes. The right-skewed nature of the GOP distribution (floor at ~213 but ceiling at ~225+) was not captured. Now added.

3. **No range-forecasting procedure**: No systematic methodology existed for handling "specific numerical range" questions. Created the `house-seat-range-forecast` procedure.

4. **No narrow-range pitfall in _procedure.md**: The pitfalls section didn't warn about treating narrow ranges as low-probability. Now added.

## Pattern Recognition

This is the 5th wrong prediction (out of 35 questions = 86% accuracy). The wrong predictions all share a pattern: predicting NO when the answer was YES. This could indicate:
- Status-quo bias (the known pattern from _procedure.md)
- Conservative probability thresholds (requiring p > 50% before predicting YES)
- Insufficient tail modeling (underweighting the probability of non-central outcomes)

For this specific question, the error was partially structural (the wrong formula) and partially an inherent uncertainty (35% events happen ~1/3 of the time). The corrected vault should reduce the formula-driven error but won't eliminate the inherent uncertainty of range boundary questions.

## Files Changed

1. **domains/usa/concepts/generic-ballot-seat-conversion/_concept.md**: Fixed formula, added three-regime distribution model, added wrong-formula alert, expanded step-by-step range assessment with probability distribution approach, added 220-224 canonical example, added validation entry
2. **domains/usa/procedures/house-seat-range-forecast.md**: NEW procedure with 5-phase methodology for House seat range questions
3. **_procedure.md**: Added "narrow range trap" pitfall, added numerical range diagnostic to post-forecast reflection
4. **domains/usa/threads/us-house-elections/_thread.md**: Added 220-224 lesson, added distribution/range methodology to forecasting significance
5. **forecasts/2024-11-01-republicans-220-224-house-seats.md**: NEW forecast entry
