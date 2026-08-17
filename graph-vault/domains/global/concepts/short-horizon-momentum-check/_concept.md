---
type: concept
tags: [concept, methodology, pre-filter, momentum]
title: "Short-Horizon Momentum Check"
slug: short-horizon-momentum-check
domain: global
first_observed: 2026-05-20
canonical_cases:
  - "Bitcoin >$72K by Feb 13, 2026 — 7-day window, -38% trend, no catalyst"
  - "Cervia: Marmousez vs Bondioli — impossible draw match at any horizon"
status: active
related_concepts:
  - structural-improbability-check
  - forecast-range-plausibility-filter
  - sports-market-liquidity-signal
  - prior-probability-of-trigger
  - question-interpretation-ambiguity
---

# Short-Horizon Momentum Check

## Definition

A pre-forecast diagnostic for recognizing when a question's YES outcome is **trend-constrained by the forecast window**: the time remaining is too short, the trend is too strong, and no catalyst is known that could reverse it. In these cases, the probability of YES should anchor *far below* the default uncertainty baseline of ~0.50 — typically below 0.10 and often below 0.05.

This is distinct from generic base-rate pessimism ("most things don't happen"). It is a **calendar-agnostic structural constraint**: the forecast window itself bounds the outcome, and the constraint (no catalyst for reversal, no draw entry) is knowable at cutoff.

## Distinction From Related Concepts

| Concept | When to Use | Canonical Example |
|---------|-------------|-------------------|
| **short-horizon-momentum-check** *(this)* | Short window, strong trend, no catalyst known. Checks: can YES physically happen in the time given the trajectory? | Bitcoin >$72K needs +20% in 7 days against -38% trend |
| **structural-improbability-check** | Multi-collapse: YES requires 2+ independent unlikely failures. Checks: is the YES scenario a cascade of independent improbable events? | Far-left party winning Latin American presidency |
| **forecast-range-plausibility-filter** | Numerical range specified. Checks: is the asked-about range structurally plausible for the phenomenon? | Trump sentenced 24-35 months for Class E felony |
| **sports-market-liquidity-signal** | Sub-$1K USDC sports markets. Checks: does the market even signal anything? | Cervia tennis $100 USDC volume |
| **prior-probability-of-trigger** | Rare events with measurable base rates. Checks: what is the historical frequency of this event type? | Nuclear use, pandemic escape |

The four pre-filters are meant to be applied **in order**:

1. **Short-horizon momentum check** — Does YES require reversing a strong trend in a short window?
2. **Structural improbability check** — Does YES require 2+ independent failures?
3. **Range plausibility filter** — Is the specified numerical range plausible?
4. **Liquidity signal** — Is the market even liquid enough to signal?

Each filter that returns "NO" independently rules out the YES outcome or radically lowers its probability. Filters 1 and 4 can be applied in seconds before any substantive analysis begins.

## Canonical Case 1: Bitcoin >$72K by Feb 13, 2026

| Parameter | Value |
|-----------|-------|
| Forecast window | 7 days (cutoff Feb 6, target Feb 13) |
| Trend at cutoff | ~38% drawdown from $97K toward ~$60K |
| Rally required | ~20% in 7 days |
| Known catalyst | None — SBR executive order absent, tariff uncertainty dominant |
| Predicted p_yes | 0.25 |
| Outcome | NO (resolved NO) |
| Brier | 0.0625 |

**Diagnosis**: The forecast correctly called NO but was *underconfident* at p_yes=0.25. With a 7-day window, a -38% trend, and no catalyst, the structural baseline should have been p_yes < 0.10. The 0.25 reflected residual "uncertainty" that was analytically unnecessary — the YES scenario required a price reversal that had no plausible mechanism.

**Post-hoc corrected probability**: p_yes ≤ 0.08 (using the short-horizon momentum check).

**Vault files**: [[runs/20260520-065636-will-the-price-of-bitcoin-be-above-72-000-on-february-13]], [[events/bitcoin-feb-2026-drawdown]], [[domains/global/concepts/policy-expectation-without-delivery]]

## Canonical Case 2: Cervia: Marmousez vs Bondioli Completed Match

| Parameter | Value |
|-----------|-------|
| Forecast window | Open-ended (match must complete) |
| Trend at cutoff | Neither player in same tournament draw → match cannot exist |
| Existence verification | Lilian Marmousez not in Cervia Challenger singles draw (main or qualifying); Federico Bondioli played Petr Nesterov in R1 |
| Market liquidity | ~$100 USDC (nonsense signal) |
| Predicted p_yes | 0.01 |
| Outcome | NO (resolved NO) |
| Brier | 0.0001 |

**Diagnosis**: Correct at near-deterministic confidence. The draw verification immediately established the match was impossible — a stronger constraint than any trend analysis. The liquidity signal (<$1K USDC) corroborated that the market was noise.

**Pattern variant**: This is a **time-independent** short-horizon momentum check case — the "horizon" is the tournament's existence, not a calendar window. If the event cannot exist at any future time given current facts (players in different tournaments), the probability is effectively zero regardless of the time window.

**Vault files**: [[runs/20260520-065750-cervia-completed-match-lilian-marmousez-vs-federico-bondioli]], [[domains/sports/concepts/sports-market-liquidity-signal/_concept]]

## The Combined Structural-NO Pattern

Both recent runs share a deeper structural pattern: **correct NO predictions driven by structural constraints, not deep domain knowledge**.

| Run | Structural Constraint | Information Required | Confidence |
|-----|----------------------|---------------------|------------|
| Bitcoin >$72K | ~20% rally needed in 7 days against -38% trend, no known catalyst | Trend direction + time horizon + catalyst absence → momentum-driven NO | p_yes=0.25 (conservative) |
| Cervia tennis | Players not in same tournament draw → match impossible | Draw verification + liquidity filter → existence-driven NO | p_yes=0.01 (near-deterministic) |

**Key insight**: When a question asks about a specific outcome within a short window and the information needed to confirm YES is absent (no catalyst, no draw entry), the structural baseline is NO. Runfiles should anchor closer to p_yes=0.01-0.10 for such cases rather than treating them as "uncertainty" around 0.50.

## Decision Procedure

Apply this check *before* any substantive analysis:

```
Step 1: Identify the forecast window
  → Calendar window: "by Feb 13" = 7 days from cutoff
  → Existence window: "completed match" = tournament must be ongoing
  → Open-ended: "before 2030" = too long for this check (use improbability check instead)

Step 2: Measure the status-quo trajectory
  → For price/market questions: recent % change over the prior period
  → For existence questions: are the necessary preconditions met?
  → For political questions: is the incumbent ahead/behind? What's the trend?

Step 3: Check for a known catalyst
  → Is there a scheduled event that could reverse the trajectory?
  → Is there a known actor with both motive and means to create the outcome?
  → If no catalyst exists, the status quo trajectory is the best predictor

Step 4: Calculate the required reversal magnitude
  → How large a change is needed in how much time?
  → Compare to historical maximum changes in that timeframe
  → For price: Bitcoin's max 7-day rally in a bear trend → what's the 90th percentile?

Step 5: Anchor the probability
  → If reversal is >2 standard deviations away AND no catalyst exists: p_yes < 0.10
  → If reversal is impossible (existence failure): p_yes < 0.01
  → If reversal is possible but requires a catalyst that doesn't exist: p_yes < 0.25
  → If reversal is within normal variance: proceed with standard forecasting
```

## Forecasting Application

### When to Apply

Trigger on any question that specifies a **short calendar window** (days to ~4 weeks) or an **existence condition** (completed match, signed deal, announced policy) where the underlying preconditions are knowable at cutoff.

### Calibration Table

| Window Length | Trend Strength | Catalyst Known? | Anchor p_yes |
|--------------|----------------|-----------------|-------------|
| 1-7 days | Strong (-20%+) | No | ≤ 0.05 |
| 1-7 days | Moderate (-10-20%) | No | ≤ 0.10 |
| 1-2 weeks | Strong | No | ≤ 0.10 |
| 1-2 weeks | Strong | Yes | 0.15-0.30 |
| 3-4 weeks | Strong | No | 0.15-0.25 |
| Existence fail | N/A | N/A | ≤ 0.01 |
| Any | Flat/neutral | No | Standard forecast (no shortcut) |

### Pitfalls

1. **Overconfidence in trend continuation**: Trends can reverse without warning. Apply the check only when the window is genuinely too short for an unanticipated catalyst to emerge. A "black swan" is always possible in a 7-day window — that's the residual ~5%.

2. **Catalyst blindness**: Just because *you* don't know of a catalyst doesn't mean none exists. The check assumes the forecaster has done reasonable research. If the vault has no signal on a topic, the check is weaker but still applicable — "no known catalyst" is itself a signal in well-covered domains.

3. **Independence from sports liquidity signal**: In sports markets, apply *both* the liquidity signal filter and the momentum check. The Cervia case triggered both — the market was illiquid AND the event couldn't exist. The two filters are independent and additive.

4. **Window-expanding ambiguity**: Some questions have ambiguous time windows (e.g., "before next earnings" when earnings date isn't announced). In these cases, use the most conservative interpretation (latest plausible window) for the check.

## Cross-References

- [[runs/_index]] — Runs index identifying the combined Structural-NO Pattern
- [[runs/20260520-065636-will-the-price-of-bitcoin-be-above-72-000-on-february-13]] — Bitcoin canonical case
- [[runs/20260520-065750-cervia-completed-match-lilian-marmousez-vs-federico-bondioli]] — Cervia canonical case
- [[events/bitcoin-feb-2026-drawdown]] — The drawdown that drove the Bitcoin NO outcome
- [[domains/global/concepts/structural-improbability-check/_concept]] — Related multi-collapse pre-filter
- [[domains/global/concepts/forecast-range-plausibility-filter]] — Related numerical range pre-filter
- [[domains/global/concepts/policy-expectation-without-delivery]] — The Bitcoin drawdown's deeper cause
- [[domains/sports/concepts/sports-market-liquidity-signal/_concept]] — Related liquidity pre-filter for sports
- [[domains/sports/threads/tennis-challenger-forecasting/_thread]] — Sports domain realization of the negative-space pattern
- [[domains/global/concepts/prior-probability-of-trigger]] — Rare-event counterpart
- [[timeline/2026-Q1]] — Quarter context for Bitcoin drawdown
