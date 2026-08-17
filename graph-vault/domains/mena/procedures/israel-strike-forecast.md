---
type: procedure
tags: [procedure]
title: "Israel Military Strike Forecast"
slug: israel-strike-forecast
domain: "[[domains/mena]]"
concepts:
  - "[[concepts/shadow-war-to-direct-escalation]]"
  - "[[concepts/short-window-military-strike-probability]]"
  - "[[concepts/escalation-bargaining-termination]]"
entities:
  - "[[entities/israel]]"
  - "[[entities/benjamin-netanyahu]]"
  - "[[entities/iran]]"
  - "[[entities/islamic-revolutionary-guard-corps-irgc]]"
---

# Israel Military Strike Forecast

Compute P(Israel conducts military strike on target T before date D).

## When to Use

Question involves Israeli military action against a state adversary (Iran, Syria, Hezbollah in Lebanon), particularly when the question has a specific deadline (prediction market format).

## Inputs

- Target T (e.g., Iran, Iranian nuclear facility, Hezbollah in Lebanon)
- Deadline D (the date by which action must occur)
- Cutoff date for point-in-time knowledge
- Current IDF operational commitments (which fronts are active?)
- US posture signal (green/yellow/red light from Washington)

## Procedure

### Step 1: Assess the Escalation Stage

Read [[concepts/shadow-war-to-direct-escalation]] and determine current stage (0-8):
- Stage 0-2: P(strike in any 3-month window) < 0.05. Israel is not at the point of direct strikes.
- Stage 3: P(strike within 6 weeks) ~0.30-0.50. The strike on an auxiliary target is itself a potential trigger.
- Stage 4-5: P(strike within 30 days) ~0.50-0.70. Tit-for-tat dynamic is active.
- Stage 6-7: P(strike within 30 days) ~0.60-0.80. Full escalation imminent.
- Stage 8: The strike has already occurred.

### Step 2: Apply the Short-Window Framework

If deadline D is within 6 weeks of the forecast date, apply [[concepts/short-window-military-strike-probability]]:

1. Measure W = days from forecast to deadline
2. If W < 30 and no trigger event in past 14 days: P(strike) reduced by ~80% from medium-term baseline
3. Check for: Israeli military preparations, security cabinet meetings, US signals
4. Apply precedent penalty (first-ever strike vs repeat strike)

### Step 3: Check the Doctrine Trigger

Read [[entities/israel]] — has a Begin Doctrine trigger condition been met?
- Has T (target) crossed a nuclear latency threshold?
- Has IAEA declared T non-compliant?
- Has Netanyahu explicitly invoked the Begin Doctrine?

If IAEA non-compliance finding exists AND target has near-weapons-grade enrichment:
- P(strike within 3 months) ~0.50-0.70 regardless of other constraints

### Step 4: Assess the Patron Signal

US posture is the strongest external moderator of Israeli strike probability:

| US Signal | Effect on P(strike) |
|---|---|
| Explicit green light (envoy visit, weapons delivery, joint planning) | +0.20-0.30 |
| Ambiguous (statements about "Israel's right to self-defense") | +0.05-0.10 |
| Containment (Biden-era statements like "don't escalate," envoy dispatched) | -0.10-0.20 |
| Red line (explicit threat to withhold support) | -0.30-0.40 |

In Jan-Feb 2024, Biden administration posture was containment (striking Houthis but warning Israel against wider war).

### Step 5: Calculate Military Bandwidth

What fronts is Israel currently fighting on?

| Active Fronts | Multiplier on Base Probability |
|---|---|
| 0 (peace) | 1.5x |
| 1 (warning) | 1.0x (baseline) |
| 2 (limited) | 0.6x |
| 3+ (full) | 0.2x |

In Jan-Feb 2024: 3 active fronts (Gaza combat, northern border alert, Red Sea naval). Multiplier ~0.4x.

### Step 6: Synthesize

P(strike) = BaseRate * WindowMultiplier * DoctrineMultiplier * USSignal * BandwidthMultiplier

Where BaseRate is derived from the escalation stage in Step 1.

**Example: Jan 25, 2024 question — Israel attack Iran by Feb 15**
- Stage: 1-2 (proxy + covert, no direct exchange yet)
- BaseRate for 3-month window in Stage 1-2: ~0.10-0.15
- Window length: 21 days (short) → apply short-window penalty: ~0.3x
- No trigger event → further penalty: ~0.5x
- Begin Doctrine: Iran enriching but not yet at breakout → moderate trigger: 1.0x
- US signal: containment → 0.7x
- Bandwidth: 3 active fronts → 0.4x
- P(strike within window) = 0.12 * 0.3 * 0.5 * 1.0 * 0.7 * 0.4 = ~0.005-0.01

Apply soft floor of 0.02-0.03 for tail risk (accident, miscalculation): **~0.03-0.05**

This maps to a clear NO prediction for the 3-week window.

### Step 7: Post-Forecast Audit

After the deadline passes, compare prediction to outcome:
- If correct: was the framework right, or was it luck? Update the base rates.
- If wrong: which factor was misweighted? Update the corresponding multiplier.

## Key Entities to Consult

- [[entities/benjamin-netanyahu]] — leader's risk calculus and thresholds
- [[entities/israel]] — doctrine, multi-front constraints
- [[entities/islamic-revolutionary-guard-corps-irgc]] — Iranian escalation threshold
- [[entities/ali-khamenei]] — Supreme Leader's risk appetite

## Key Concepts to Consult

- [[concepts/shadow-war-to-direct-escalation]] — escalation stage mapping
- [[concepts/short-window-military-strike-probability]] — window-adjusted calibration
- [[concepts/escalation-bargaining-termination]] — post-strike termination dynamics
