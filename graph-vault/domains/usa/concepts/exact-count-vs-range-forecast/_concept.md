---
type: concept
tags: [concept, usa, elections, house, forecasting-methodology]
title: "Exact Count vs. Range Forecast in Multi-Seat Elections"
slug: exact-count-vs-range-forecast
first_observed: 2024-11-05
domain: usa
related_concepts:
  - generic-ballot-seat-conversion
  - presidential-coattail-variability
---

# Exact Count vs. Range Forecast in Multi-Seat Elections

## Definition

A methodological distinction between two structurally different question types about numerical outcomes in multi-seat elections (especially the 435-seat US House). **Range questions** ask whether the outcome falls within a specified interval (e.g., "220-224 seats"). **Exact-count questions** ask whether the outcome equals a specific integer (e.g., "exactly 223 seats"). These question types require different probability calibration because the probability mass is distributed across many discrete outcomes.

## The Central Insight

In a 435-seat chamber with ~25 competitive districts that behave like quasi-independent coin flips, the discrete probability distribution of seat outcomes approximates a normal distribution with standard deviation σ ≈ 3-4 seats. This means:

- **No single seat count has high probability**: Even the mode (most likely single outcome) rarely exceeds 10-12%.
- **Within-bin distribution is NOT uniform**: A 5-seat bin like 220-224 (35% total) concentrates ~60% of its probability mass at the lower end (220-221) if the distribution mode is near 219-220.
- **Exact-count questions systematically demand lower YES probabilities than range questions covering the same territory**, because a range aggregates probability across multiple discrete outcomes.

## The Seat Distribution Model (2024 Parameters)

For a competitive US House election (generic ballot within 1 point):

### Bin-Level Distribution (coarse)

| GOP Seat Range | Approx. Probability | Cumulative |
|---------------|-------------------|------------|
| 210-214 | 5% | 5% |
| 215-219 | 30% | 35% |
| 220-224 | 35% | 70% |
| 225-229 | 20% | 90% |
| 230+ | 10% | 100% |

### Within-Bin Distribution (fine)

The bin-level table is sufficient for range questions. For exact-count questions, the within-bin distribution must be estimated. Assuming a normal-like distribution with μ ≈ 219, σ ≈ 3.5, and right-skew from gerrymandering:

| Exact Seat Count | Approx. Probability | Notes |
|-----------------|-------------------|-------|
| 215 | 3% | Left tail |
| 216 | 5% | Lower edge of concentration |
| 217 | 7% | |
| 218 | 9% | |
| 219 | 11% | Mode |
| 220 | 11% | Near-mode (actual 2024 outcome) |
| 221 | 9% | |
| 222 | 7% | |
| 223 | 5% | Upper mode-adjacent |
| 224 | 3% | Upper tail |
| 225 | 2% | Tail |
| 226+ | 3% | Tail |

**Key observation**: The sum of 220-224 across individual probabilities (11+9+7+5+3 = 35%) matches the bin-level table. But the individual probability of any exact count within that bin is much lower: P(223) ≈ 5% vs. P(220-224) ≈ 35%.

### Why This Matters for Forecasting

For a question about "exactly 223 seats":

- **Range framework misapplied**: A forecaster who treats this like a range question might reason "the 220-224 bin has 35% probability, so 223 is plausible" — producing a NO prediction with low confidence. But the correct probability is ~5%, making NO a near-certain prediction.
- **Exact-count framework**: "P(223) ≈ 5%. P(NO) = 95%. The question is a narrow exact-count bet, not a range bet."
- **The vault's existing distribution model (5-seat bins) actively misleads for exact-count questions** because it suggests more probability mass at 223 than actually exists.

## Canonical Example: 223 GOP Seats (2024 House Election)

| Field | Detail |
|-------|--------|
| Question | "Will Republicans control 223 seats in the House after the election?" |
| Question type | Exact-count (not range) |
| Correct framework | Within-bin distribution estimation, not 5-seat bin aggregation |
| P(223) under normal model | ~5% (μ≈219, σ≈3.5, right-skewed) |
| Actual outcome | 220 (NO) |
| Why the vault's 5-seat bin model misleads | The 220-224 bin (35%) appears to contain 223 as a "central enough" value. But within-bin distribution is concentrated at 220-221 (low end). 223 is in the upper half of the bin with ~5% individual probability. |

### What the Forecaster Should Reason

1. Identify that this is an exact-count question, not a range question
2. Note that even the most likely single seat count (219-220) has only ~11% probability
3. 223 is 3-4 seats above the mode — its individual probability is ~5%
4. A YES prediction (P=5%) would need extraordinary evidence of a specific seat configuration producing exactly 223
5. NO is the clear prediction (95% probability) with high confidence

## When to Use This Concept

Load this concept when the question asks about a **single integer** outcome in a multi-seat election — e.g., "exactly 223 seats," "at least 218 seats," "more than 225 seats." The key distinction:

| Question Phrasing | Type | Framework |
|------------------|------|-----------|
| "Will Republicans have between 220 and 224 seats?" | Range | Bin-level distribution (house-seat-range-forecast procedure) |
| "Will Republicans have exactly 223 seats?" | Exact-count | Within-bin distribution (exact-seat-count-forecast procedure) |
| "Will Republicans have at least 225 seats?" | Threshold | Cumulative distribution (threshold probability = sum of tail) |
| "Will Republicans control the House?" | Binary control | Control threshold = 218 seats; P(control) = P(seats >= 218) |

Each type requires different distributional reasoning. Conflating them is a common forecasting methodology error.

### Ambiguity Warning: The "Control N" Trap

A recurrent edge case in House seat forecasting questions: the question phrasing says "control N seats" (suggesting a threshold interpretation — "control enough seats to total N"), but the resolution text specifies "exactly N members." This is a distinct resolution-criteria gotcha documented in [[domains/global/concepts/forecast-resolution-criteria-gotchas]] (entry #8).

**Detection rule**: If a House seat question combines a verb like "control/wins/has" with a specific number and the resolution text contains "exactly," the question is exact-count, not threshold. The verb is irrelevant — the resolution criteria rule.

**Examples**:
- "Will Republicans control 224 seats in the House?" → Resolution: "exactly 224" → Exact-count question. P(exactly 224) ≈ 3%.
- "Will Republicans win 225 seats or more?" → Resolution: "at least 225" → Threshold question. P(>=225) ≈ 20%.
- "Will Republicans have 220-224 seats?" → Resolution: range → Range question. P(220-224) ≈ 35%.

The forecaster must always read the resolution text, not rely on the question title.

## Relationship to Other Vault Concepts

- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — cross-domain catalog of resolution-criteria gotchas; includes entry #8 on "control N" vs. "exactly N" ambiguity
- [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]] — provides the seat-vote conversion function and bin-level distribution tables
- [[domains/usa/concepts/presidential-coattail-variability/_concept]] — provides coattail adjustments that shift the distribution mean
- [[domains/usa/procedures/house-seat-range-forecast]] — the existing range procedure (works for range questions, NOT for exact-count)
- [[domains/usa/procedures/exact-seat-count-forecast]] — the exact-count procedure (load for exact-count questions)

## Wikilinks
[[domains/usa/threads/us-house-elections/_thread]], [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]], [[domains/usa/concepts/presidential-coattail-variability/_concept]], [[domains/usa/procedures/house-seat-range-forecast]], [[domains/usa/procedures/exact-seat-count-forecast]], [[domains/global/concepts/forecast-resolution-criteria-gotchas]]
