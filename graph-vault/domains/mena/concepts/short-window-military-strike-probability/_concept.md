---
type: concept
tags: [concept]
title: "Short-Window Military Strike Probability"
slug: short-window-military-strike-probability
first_observed: 2024-01-25
domain: military-strategy
related_concepts:
  - shadow-war-to-direct-escalation
  - escalation-bargaining-termination
---

## Definition

The probability that a military attack on a state's sovereign territory occurs within a specific short calendar window (days to weeks) follows a characteristic distribution that differs dramatically from longer windows (months to years). Prediction market questions with tight deadlines (e.g., "Will Israel attack Iran by Feb 15?") require a systematic framework because the base rate of a first-ever direct strike within any given 3-week period is structurally low, even when the medium-term probability is substantial.

## Core Principle: The Window-Length Multiplier

Where P(attack occurs within window W) depends primarily on:

1. **Base rate of such attacks per unit time** — how often does entity A strike entity B's territory directly?
2. **Trigger event proximity** — has a tripwire event occurred within or immediately before W?
3. **Decision cycle friction** — how long does it take for a state to plan, authorize, and execute a strike?
4. **US/patron signal** — is the patron signaling green/yellow/red light?
5. **Military bandwidth** — is the attacker's force already committed elsewhere?

## Calibration Scales

### Base Rate: First-Ever Direct Strike

For a state that has never directly struck another state's sovereign territory (as of Jan 2024, Israel had never struck Iranian soil):

| Window Length | Approximate Base Rate |
|---|---|
| 1 week | < 0.01 (first-ever, unprecedented) |
| 3 weeks (this question) | ~0.02 (still very low without trigger) |
| 3 months | ~0.10-0.15 |
| 6 months | ~0.20-0.30 |
| 1 year | ~0.35-0.50 (if structural drivers are present) |

These are rough priors. The base rate is adjusted upward by:
- A recent trigger event (assassination, consulate strike, proxy mass-casualty)
- Explicit Israeli threats with mobilization
- Collapse of diplomatic off-ramps
- US green light signals

And adjusted downward by:
- Active IDF commitment on other fronts
- US red light signals (Biden administration containment efforts)
- No recent trigger event
- Short window (escalation takes time to plan)

### Decision Cycle Friction

Israeli military strikes on sovereign territory of a peer/parity state require:
- **Intelligence preparation**: 2-4 weeks minimum for target vetting (Mossad, Aman, Shin Bet)
- **Cabinet authorization**: Must go through security cabinet — takes 1-7 days
- **US notification**: Israel historically notifies the US 24-72 hours before major strikes
- **Operational planning**: Air force strike packages, refueling, SEAD (suppression of enemy air defenses) — 1-3 weeks
- **Weather/moon phase window**: Precision strikes often require specific conditions — 3-7 day windows

Total minimum from decision to execution: **2-5 weeks**, even in accelerated timelines. For a 3-week window (Jan 25 - Feb 15), a decision would need to be made by approximately Jan 20 at the latest, meaning the trigger event would need to occur before Jan 20.

### Trigger-Based Probability Adjustment

When a trigger event occurs, the probability curve shifts dramatically:

**Trigger Event Types (ranked by escalation probability)**
1. Direct attack on Israeli territory causing mass casualties → P(strike within 2 weeks) ~0.40
2. Israeli casualties from Iranian proxy with Iranian fingerprints → P(strike within 1 month) ~0.20
3. Attack on Israeli diplomatic facility → P(strike within 1 month) ~0.30
4. Iranian nuclear breakout announcement → P(strike within 1 month) ~0.50
5. Houthi/Hezbollah mass-casualty attack on Israel with Iranian weapons → P(strike within 6 weeks) ~0.15

**At the time of the question (Jan 25, 2024):**
- No trigger event had occurred. The war was ongoing in Gaza, Houthis were attacking Red Sea shipping, but no direct Israeli casualties from Iranian proxies that would trigger a strike on Iran proper.
- The Iran-Pakistan exchange (Jan 16-18) was a signal of Iranian willingness to use cross-border force, but against Pakistan (not Israel) — and both sides de-escalated rapidly.
- US was actively containing regional escalation; Biden administration had struck Houthi targets (Jan 12) but repeatedly signaled desire to avoid Israel-Iran war.

**Calibrated P(Israel attacks Iran soil, Jan 25 - Feb 15, 2024): ~0.05-0.10**
- Close to zero because: no trigger event, no Israeli military preparation visible, US containment policy, IDF bandwidth absorbed by Gaza, 3-week window too short
- Non-trivial probability only from: accident/miscalculation scenario (e.g., Israeli strike on Iranian target in Syria triggers spiral)

## Forecasting Application

When asked "Will X attack Y by date Z" where Z is within 6 weeks of the forecast date:

1. **Measure the window**: W = days from forecast to deadline. If W < 30 days, apply the short-window framework automatically.

2. **Check for tripwire**: Has a trigger event occurred in the last 14 days? If no, the base rate drops by ~80% from the 6-month rate.

3. **Assess decision cycle**: Could X have decided and prepared to strike within W days? Look for:
   - Visible military preparations (troop movements, air force readiness)
   - Cabinet meetings on the specific target
   - Changes in rhetorical posture (from "we will respond" to "we will strike")
   - US signals (envoy visits, warnings, force posture changes)

4. **Apply the precedent penalty**: First-ever strikes (X has never struck Y's soil) have a much lower base rate within short windows than repeat strikes. A state that has already crossed the threshold can do so again with less friction. Israel's April 19, 2024 strike on Isfahan (its first on Iranian soil post-April 13) required less decision friction because the April 13 Iranian attack had already normalized the threshold.

5. **Check for parallel bandwidth**: Is X's military already engaged on another major front? If yes, the probability of initiating a second major front within a short window is reduced (not eliminated, but reduced).

## Canonical Examples

| Question | Window | Trigger | Base Rate Applied | Correct? |
|---|---|---|---|---|
| Israel attack Iran by Feb 15, 2024 (gold_03) | 21 days | None | ~0.05 (no trigger, first-ever, 3-week window) | YES (NO outcome) |
| Israel attack Iran after April 1, 2024 | 3 days after consulate strike | Yes (Damascus consulate) | ~0.35 (trigger present, shortened decision cycle) | YES (Israel struck Isfahan April 19) |
| Iran attack Israel after Oct 1, 2024 | ~14 days after Nasrallah assassination | Yes (Nasrallah, Haniyeh kills) | ~0.50 (repeated trigger, ballistic missile precedent) | YES (Iran launched Oct 1) |

## When This Framework Fails

- **Rapid mobilization scenarios**: If an attacker has pre-existing plans ready to execute (e.g., Israel's Operation Rising Lion was planned months in advance), the window constraint relaxes. Look for "plan on the shelf" indicators.
- **Accidental escalation**: A limited engagement that spirals (e.g., Israeli strike on an Iranian target in Syria that kills a senior IRGC commander) can produce rapid escalation that bypasses normal decision cycles.
- **Patron pressure**: If a superpower patron is pressuring for or against action, the decision cycle can compress or expand by an order of magnitude.

## Related Patterns

- [[escalation-bargaining-termination]] — termination dynamics once strikes have occurred
- [[shadow-war-to-direct-escalation]] — the structural path from proxy warfare to direct confrontation
- [[concepts/forecast-resolution-criteria-gotchas]] — watch for resolution criteria that might treat proxy strikes or covert ops as "attack"
