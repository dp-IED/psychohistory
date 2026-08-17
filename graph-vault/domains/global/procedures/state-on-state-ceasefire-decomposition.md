---
type: procedure
tags: [procedure]
title: "State-on-State Ceasefire Decomposition"
slug: state-on-state-ceasefire-decomposition
domain: "[[domains/geopolitics]]"
concepts:
  - "[[concepts/ceasefire-pathway-decomposition]]"
  - "[[concepts/escalation-bargaining-termination]]"
  - "[[concepts/short-window-ceasefire-probability]]"
  - "[[concepts/shadow-war-to-direct-escalation]]"
  - "[[concepts/political-deadline-ceasefire]]"
  - "[[concepts/diplomatic-pressure-tipping-point]]"
entities:
  - "[[entities/israel]]", "[[entities/iran]]"
  - "[[entities/israeli-security-cabinet]]"
  - "[[entities/donald-trump]]"
  - "[[entities/iaea]]"
procedures:
  - "[[procedures/inter-state-ceasefire-feasibility]]"
  - "[[procedures/ceasefire-timing]]"
---

# State-on-State Ceasefire Decomposition

## When to Use

Use this procedure when forecasting whether a ceasefire (or ceasefire announcement) will occur between two state actors within a specific date window. This is the primary procedure for questions like "Israel x Iran ceasefire before [date]?" on state-on-state conflicts.

Do NOT use for state-vs-non-state conflicts (Israel-Hamas, Israel-Hezbollah) — use [[procedures/asymmetric-ceasefire-forecast]] instead.

## Step-by-Step

### Step 0: Pre-Analysis — Is This a State-on-State Question?

Confirm both belligerents are sovereign states capable of direct military engagement. If one party is a non-state actor (Hamas, Hezbollah, Houthis), this procedure does not apply. Switch to [[procedures/asymmetric-ceasefire-forecast]].

**Trap:** Some conflicts have dual character. The Iran-Israel conflict is state-on-state at the direct level, but Iran also operates through proxies. The direct state-on-state escalation ladder is the dominant dynamic; proxy activity is a secondary signal. Do not classify a state-on-state conflict as asymmetric just because one side also uses proxies.

### Step 1: Classify the Ceasefire Pathway

Apply [[concepts/ceasefire-pathway-decomposition]] to classify the question:

- **Pathway B (War-Termination)**: State-on-state + escalation ladder + superpower patron + no nuclear deterrent against patron → ceasefire question decomposes into war probability × termination probability.
- **Pathway A (Diplomatic)**: No escalation ladder, or no superpower termination mechanism → ceasefire depends on diplomatic pressure, war exhaustion, mediation.
- **Pathway C (None likely)**: Nuclear-armed adversary, existential war aims, no mediator → P(ceasefire) structurally near-zero.

**Decision rule:** If indicators for Pathway B are present, proceed to Step 2. If Pathway A, proceed to Step 4. If Pathway C, predict NO with high confidence and skip remaining steps.

### Step 2: Estimate P(War in Window | Escalation Ladder)

Using [[concepts/shadow-war-to-direct-escalation]]:

1. **Map the current escalation stage** (0-8). What stage are the belligerents at right now? Each stage has characteristic indicators.
2. **Check the compression principle**: After Stage 3, transition times compress to days/weeks. After Stage 6, war can come within weeks.
3. **Check necessary conditions**:
   - Nuclear latency / WMD threshold approaching? → DRASTICALLY increases war probability
   - Proxy network degradation? → Increases war probability ("use it or lose it")
   - Patron signaling? Is the patron green-lighting or restraining?
   - Political deadline? Is there a known ultimatum or negotiation window expiring?
   - IAEA or international institution finding? → Legal cover for strikes
4. **Calibrate P(war in window)**:
   - 0-1 escalation stages crossed, no nuclear latency: P < 0.05
   - 2-3 stages crossed, nuclear latency present: P ~0.15-0.35
   - 4-6 stages crossed (direct strikes happened), nuclear latency active: P ~0.40-0.65
   - 7+ stages crossed (ballistic exchanges completed), deadline approaching: P ~0.65-0.85
   - Apply adjustments for specific trigger events (IAEA finding, assassination, deadline expiry)

**Validation case (Iran-Israel, PIT 2025-Q1):**
- Escalation stage: 6-7 (ballistic exchange completed Oct 2024, plateau through Q1 2025)
- Nuclear latency: Iran near weapons-grade enrichment; IAEA monitoring
- Proxy degradation: Hezbollah weakened, Hamas degraded, Assad fallen
- Political deadline: Trump's 2-month nuclear ultimatum expiring ~May 2025
- IAEA: Board moving toward non-compliance finding (June 12)
- **Calibrated P(war by June 30)**: ~0.70-0.80

### Step 3: Estimate P(Termination Within Target Window | War Starts)

Using [[concepts/escalation-bargaining-termination]]:

1. **Check the 48-hour rule conditions**:
   - Does the superpower patron have escalation dominance? (Can win any expanded war?)
   - Is the client operationally dependent on the patron? (Intel, resupply, air cover?)
   - Does the adversary have a nuclear deterrent against the patron?
   - Would patron domestic politics favor rapid termination?
   - If YES to first 3 and NO to 3rd: the 48-hour rule applies → P(ceasefire within 72h of war start) > 0.90

2. **Check the target date window**: Given the estimated war start window, is there enough time for the 48-hour rule to produce a ceasefire before the target date?
   - If war starts > 72h before the target date cutoff → P(ceasefire | war) > 0.90
   - If war starts < 72h before cutoff → P(ceasefire | war) drops significantly (war-start timing uncertainty)
   - If the target date is before the most likely war start → P(ceasefire | war) = 0 by definition

3. **Check for the 'damaged mediation' trap**: If the adversary retaliates against the superpower's own forces on the day of entry, do NOT infer that ceasefire probability drops. This is the structurally wrong inference — co-belligerent status accelerates termination because it maximizes the patron's termination imperative.

4. **Calibrate P(termination | war)**:
   - 48-hour rule conditions met: P > 0.90
   - 48-hour rule partially met (patron lacking escalation dominance over nuclear power): P ≈ 0
   - Patron entering as mediator, not combatant: P ~0.30-0.60 (timeline expands to weeks/months)

**Validation case (Iran-Israel, June 2025):**
- US escalation dominance: YES (B-2s strike any target in Iran)
- Israel operational dependency: YES (US intel, air refueling, missile defense)
- Iran nuclear deterrent against US: NO (Iran had no nuclear weapons)
- US domestic politics: YES (Trump needed to avoid protracted Middle East war)
- **48-hour rule applies: P(ceasefire within 72h of war start) > 0.90**

### Step 3b: Calculate Combined Probability

```
P(ceasefire by D) = P(war starts by D - 72h) × P(termination within 72h | war) + P(alternative ceasefire before war)
```

For the June 2025 case:
- P(war starts by June 27) ≈ 0.75 (most likely window June 13-15)
- P(termination | war) ≈ 0.92
- P(alternative diplomatic ceasefire) ≈ 0.02

→ P(ceasefire by June 30) ≈ 0.75 × 0.92 + 0.02 ≈ **0.71**

This is structurally HIGH relative to a diplomatic assessment (which would be < 0.05), because the pathway is different.

**Note on the "War Didn't Happen" tail:** The decomposition produces P(ceasefire) that is bounded by P(war). If the war doesn't happen, the war-termination ceasefire doesn't happen either. The 0.71 probability means there's a ~29% chance the forecaster is wrong because either war doesn't happen (25%) or war happens but isn't terminated within the window (3%). This is a defensible probability for a high-uncertainty event.

### Step 4: Pathway A Decomposition (Diplomatic)

If the question is Pathway A (no escalation ladder to war), use [[procedures/asymmetric-ceasefire-forecast]] or [[procedures/inter-state-ceasefire-feasibility]] instead. Key factors:

- War aims compatibility (most important)
- Mediator leverage structure
- Political deadline proximity
- Leadership decapitation window
- War duration and exhaustion level
- International pressure accumulation

### Step 5: Check Resolution Criteria Pitfalls

Before finalizing, check:

1. **Does "ceasefire" mean announcement, ratification, or effective date?** See [[concepts/ceasefire-announcement-ratification-gap]]. The resolution text determines which date counts. For war-termination ceasefires, all three dates typically compress to 0-2 days (announcement and ratification same day, effective date next day).

2. **Which actor's announcement counts?** If the question says "Israel announces ceasefire," the mediator's announcement of the framework does NOT resolve YES. The war-termination pathway usually produces near-simultaneous announcements from both the mediator and the party, so this is rarely an issue for Pathway B. For Pathway A, track the four-date sequence carefully.

3. **Is the target date measured in the right timezone?** Polymarket uses ET. A ceasefire announced at 11 PM ET on the target date counts. War-termination ceasefires often happen in the middle of the night Israel/Iran local time.

## Validation Table

| Question | Pathway | P(war in window) | P(termination | war) | Combined P | Actual | Correct? |
|----------|---------|-----------------|---------------------|-----------|--------|----------|
| Iran-Israel ceasefire before July 2025 | B | ~0.75 (war June 13-24) | ~0.92 (48h rule applies) | ~0.71 | YES (June 24) | YES — predicted correctly |
| Iran-Israel ceasefire before July 2025 (alternative: diplomatic only) | A | — | — | < 0.05 | YES | NO — would have predicted NO, which was wrong |

The validation shows that applying the wrong pathway decomposition produces 50+ point errors on state-on-state ceasefire questions. Pathway classification is not optional — it is the controlling variable.

## Related Procedures

- [[procedures/inter-state-ceasefire-feasibility]] — General feasibility assessment (complementary; use this decomposition first, then apply feasibility checks)
- [[procedures/ceasefire-timing]] — Timing calibration (use for Pathway A only; Pathway B timing is determined by superpower entry, not diplomatic timelines)
- [[procedures/asymmetric-ceasefire-forecast]] — For state-vs-non-state conflicts (do not use for state-on-state)
- [[procedures/israel-strike-forecast]] — For military strike probability assessment (complements Step 2)
