---
type: procedure
tags: [procedure]
title: "Ceasefire Timing"
slug: ceasefire-timing
domain: "[[domains/geopolitics]]"
concepts:
  - "[[concepts/short-window-ceasefire-probability]]"
  - "[[concepts/war-aims-incompatibility]]"
  - "[[concepts/diplomatic-pressure-tipping-point]]"
  - "[[concepts/escalation-bargaining-termination]]"
  - "[[concepts/political-deadline-ceasefire]]"
  - "[[concepts/leadership-decapitation-negotiation-window]]"
  - "[[concepts/temporary-vs-enduring-ceasefire]]"
entities:
  - "[[entities/hamas]]"
  - "[[entities/israel]]"
  - "[[entities/hezbollah]]"
  - "[[entities/benjamin-netanyahu]]"
  - "[[entities/qatar]]"
  - "[[entities/egypt]]"
functions:
  - "[[functions/run_structured().md|run_structured]]"
  - "[[functions/pit_search().md|pit_search]]"
  - "[[functions/base_rate_check().md|base_rate_check]]"
---

# Ceasefire Timing

Estimate whether a ceasefire will be announced before a specific deadline, or estimate time-to-ceasefire.

## When

Any active conflict where diplomatic pressure is accumulating: Gaza, Ukraine, Iran-Israel.
Use the **asymmetric-ceasefire-forecast** procedure for state-vs-non-state conflicts (Israel-Hamas, Israel-Hezbollah).
This procedure covers the general case and refers to the specific procedure for asymmetric conflicts.

## Approach

0. **Pre-condition: Classify the ceasefire pathway** using [[domains/global/concepts/ceasefire-pathway-decomposition]]:
   - Pathway B (war-termination): Do NOT use this procedure. Use [[domains/global/procedures/state-on-state-ceasefire-decomposition]] instead.
   - Pathway A (diplomatic): Continue with this procedure.
   - Pathway C (none): Predict NO.

1. Read the conflict thread for: peace proposal status, backchannel engagement, international pressure vectors
2. Check [[concepts/diplomatic-pressure-tipping-point]] for accumulation timing
3. Check [[concepts/escalation-bargaining-termination]] if superpower patron is present
4. Run [[functions/run_structured().md|run_structured]] with diplomatic signals + timing calibration

## Calibration

### General Ceasefire Timing

- Diplomatic pressure requires 3-6 months minimum accumulation from conflict start/restart
- Concrete proposal existence: P(ceasefire by 3 months) ~0.6
- Superpower combatant entry: ceasefire within 72h (48-hour rule)
- No peace framework: P(ceasefire) < 0.2 regardless of pressure

### Short-Window Calibration (deadline ≤ 6 weeks from cutoff)

For short-window ceasefire questions, apply the [[concepts/short-window-ceasefire-probability]] framework:

**Mutual-consent penalty**: Two parties must say "yes" for a ceasefire. For any window < 3 months, the base rate is structurally lower than for military strikes because the mutual-consent requirement creates friction that time compression exacerbates.

| Window Length | Base Rate (no pre-existing framework) | Notes |
|---------------|--------------------------------------|-------|
| 1-14 days | < 0.01 | Only possible with superpower imposition |
| 15-30 days | < 0.02 | Requires pre-existing framework AND both sides ready |
| 31-60 days | 0.02-0.05 | Requires active mediation AND no war-aims incompatibility |
| 61-90 days | 0.05-0.12 | Possible with substantial diplomatic pressure |

**Key adjustments**:
- **War aims incompatible (destroy/eliminate)**: Reduce probability by 50-80%, regardless of window
- **Temporary pause exception**: If question resolution criteria allow temporary pauses, probability increases 5-10x (but still requires hostage/prisoner deal mechanism)
- **Leadership decapitation within last 60 days**: Increase probability by 1.5-2x if the 60-120 day window is active
- **Political deadline within next 30 days**: Increase by 2-3x (deadline compression effect)

### Asymmetric Conflict Calibration (State vs Non-State)

For state-vs-non-state conflicts, apply the [[procedures/asymmetric-ceasefire-forecast]] procedure:

The key factors are:
1. **War aims compatibility assessment** (see [[concepts/war-aims-incompatibility]]) — this is the dominant variable
2. **Non-state actor incentives** — does the non-state actor have an attrition strategy? Is its leadership intact?
3. **Mediation structure** — is there a single mediator with leverage on both sides? If no (as in Israel-Hamas where Qatar has leverage on Hamas but not Israel, and US has leverage on Israel but not Hamas), the leverage gap blocks rapid agreements.
4. **Domestic political constraints** — does the state leader face coalition pressure against ceasefire?
5. **Hostage/prisoner exchange mechanism** — is there a tangible exchange mechanism that can unlock negotiations?

**Base rates for asymmetric conflicts (state vs non-state, no prior enduring ceasefire):**
- First 3 months of conflict: P(ceasefire) < 0.01 (unless temporary humanitarian pause)
- 3-6 months: P(ceasefire) ~0.02-0.05
- 6-12 months: P(ceasefire) ~0.05-0.15
- 12-18 months: P(ceasefire) ~0.15-0.30
- 18+ months: P(ceasefire) ~0.25-0.50

**Example: Israel-Hamas ceasefire by Feb 29, 2024 (33-day window, ~4 months into war):**
- Asymmetric conflict (state vs non-state) in early phase: base rate ~0.02-0.05 for a 33-day window
- War aims incompatible (Israel: "destroy Hamas") → major downward adjustment
- Hamas attrition strategy active, Sinwar alive → no decapitation window
- Israel planning Rafah offensive (escalation) → ceasefire improbable
- **Calibrated P: ~0.02-0.05** → clear NO prediction

## Step-by-Step for Deadline Questions

1. **Classify the conflict type**: State-vs-state, state-vs-non-state, or non-state-vs-non-state?
   - State-vs-non-state: use [[procedures/asymmetric-ceasefire-forecast]]
   - State-vs-state with superpower: use [[concepts/escalation-bargaining-termination]]
   - State-vs-state without superpower: use [[concepts/diplomatic-pressure-tipping-point]]

2. **Check for war aims incompatibility** (strongest single factor)
3. **Map mediation structure and leverage**
4. **Check domestic political constraints on the stronger party**
5. **Check non-state actor's strategic calculus** (attrition vs negotiation)
6. **Identify any political deadlines** (inauguration, election) within 3 months
7. **Check if a temporary pause would satisfy resolution criteria** (see [[temporary-vs-enduring-ceasefire]])
8. **Apply the base rate for the specific conflict type and phase**
9. **Calibrate adjustments** based on the factors above
10. **Document the assessment** with explicit multipliers
