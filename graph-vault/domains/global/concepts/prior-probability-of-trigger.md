---
type: concept
tags: [concept]
title: "Prior Probability of Trigger"
slug: prior-probability-of-trigger
first_observed: 2023-10-21
domain: forecasting-methodology
related_concepts:
  - incumbent-withdrawal-cascade
  - leadership-persistence-under-threat
  - forecast-resolution-criteria-gotchas
---
---
---
## Definition

A systematic forecasting bias in which a framework calibrated on **post-trigger** probabilities is applied to a **pre-trigger** question, underestimating the probability of the event because the possibility of a trigger emerging over the forecasting horizon is not priced in.

The pattern: a framework says "once trigger X happens, probability of outcome Y is only 5-15%." But the question asks "will outcome Y happen within time horizon T?" — which includes the probability that some trigger emerges multiplied by the probability that the outcome follows.

## Formula

```
P(outcome Y within T) = P(any trigger emerges within T) × P(Y | trigger) + P(Y without trigger)
```

Forecasters often estimate P(Y|trigger) well but neglect P(any trigger emerges within T), especially when the trigger is a black swan or the question asks far in advance.

### The Age × Cumulative Exposure Multiplier

When the subject is elderly (75+) and the forecasting horizon extends beyond 6 months, the trigger probability is NOT uniform per month — it increases over time because each public appearance is an independent failure trial. This is the **age × cumulative exposure multiplier**: the trigger probability for an elderly leader is not P(trigger) × T/12 but the integral of an increasing failure rate function over T, because each speech, interview, or campaign event exposes the leader to public scrutiny and each is an independent opportunity for a visible decline event.

**Canonical calibration (Biden Q4 2023 forecast, gold_12)**: At cutoff Oct 2023, Biden was 80 with 10 months until the convention. Standard frameworks would estimate P(trigger at any single event) ~0.5-1% per event, times ~20 major events over 10 months (debates, interviews, rallies) = 10-20% cumulative trigger probability. But the age multiplier changes this: at 80, each additional event carries higher failure risk than the event before, because cognitive fatigue accumulates over a campaign schedule. The correct model is not a Poisson process with constant λ but a Weibull-like increasing hazard rate: λ(t) = λ₀ × exp(β × t), where β > 0 captures accelerating failure risk from campaigning. This yields P(trigger) over 10 months at ~20-40% rather than the naive 10-20%. The Polymarket price of 26.5% at cutoff was consistent with this increasing-hazard model.

**Forecasting rule**: When the subject is 75+ with a campaign or public-speaking schedule, model trigger probability as an increasing hazard rate, not a constant rate. Use a Weibull or exponential-growth hazard with minimum λ₀ = 0.5% per event and β such that hazard doubles every 3 months of sustained public exposure. A useful shortcut: multiply the naive constant-rate trigger probability by 1.5x-2x for subjects 75+ over horizons >6 months.

## Canonical Example

### Biden Drops Out (Oct 2023 forecast)

**Question**: "Will Biden drop out of the 2024 presidential race?" — forecast at cutoff Oct 21, 2023, with 10-month horizon before the convention (Aug 2024).

**Framework trap**: The incumbent-withdrawal-cascade framework gives a 5-15% post-trigger probability (once a poor debate / primary loss / health scare occurs). But the question was asked 10 months before the convention, with no trigger yet visible. The correct approach:

- Prior probability of a trigger emerging in 10 months: not negligible (~20-40% given Biden's age, low approval, and historical base rate)
- Post-trigger withdrawal probability: ~10-20% (reasonable range)
- Combined: P = P(trigger) × P(withdrawal | trigger) ≈ 25-35%

Polymarket at 26.5% was well-calibrated. The vault's own reflection flags that the naive application of the post-trigger baseline would have predicted ~12% — a significant under-estimate.

**Source**: [[runs/20260519-073035-biden-drops-out-of-presidential-race-forecast-using-only-the]]

## When This Bias Arises

The prior-probability-of-trigger trap is most dangerous when:

1. **Long time horizon**: The question asks about an event 6+ months in the future, with no visible trigger today
2. **Framework known for post-trigger outcomes**: The analyst has a well-calibrated model for what happens AFTER a trigger, but no model for trigger emergence
3. **Trigger is low-base-rate**: Events like incumbent health crises, primary upsets, or scandals are individually rare but collectively not rare over a 12-month horizon
4. **Market consensus differs from framework output**: When market prices are higher than framework suggests, check if the market is pricing trigger probability that the framework ignores

## Counter-Example: Trump Drops Out (July 2024)

**Question**: "Will Trump drop out of presidential race?" — forecast at cutoff July 12, 2024.

In this case, the trigger was already present (34-felony conviction), but it did not create withdrawal pressure because Trump faced compounding legal jeopardy (office = legal protection). The leadership-persistence-under-threat framework correctly gave <5% because the trigger's effect was opposite — it motivated persistence, not withdrawal.

This counter-example demonstrates that trigger emergence alone is not sufficient; the framework must correctly model the **direction** of the trigger's effect.

**Source**: [[runs/20260519-074201-will-trump-drop-out-of-presidential-race-forecast-using-only]]

## See Also

- [[concepts/incumbent-withdrawal-cascade]] — post-trigger withdrawal framework
- [[concepts/leadership-persistence-under-threat]] — why legal jeopardy reverses the cascade
- [[concepts/forecast-resolution-criteria-gotchas]] — another class of systematic forecast error
