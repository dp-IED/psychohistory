---
type: procedure
tags: [procedure]
title: "Asymmetric Ceasefire Forecast"
slug: asymmetric-ceasefire-forecast
domain: "[[domains/mena]]"
concepts:
  - "[[concepts/short-window-ceasefire-probability]]"
  - "[[concepts/war-aims-incompatibility]]"
  - "[[concepts/diplomatic-pressure-tipping-point]]"
  - "[[concepts/political-deadline-ceasefire]]"
  - "[[concepts/leadership-decapitation-negotiation-window]]"
  - "[[concepts/temporary-vs-enduring-ceasefire]]"
  - "[[domains/mena/concepts/multi-front-escalation-ceasefire-barrier/_concept]]"
entities:
  - "[[entities/hamas]]"
  - "[[entities/israel]]"
  - "[[entities/benjamin-netanyahu]]"
  - "[[entities/qatar]]"
  - "[[entities/egypt]]"
  - "[[entities/yahya-sinwar]]"
  - "[[entities/ismail-haniyeh]]"
  - "[[entities/hassan-nasrallah]]"
---

# Asymmetric Ceasefire Forecast

Compute P(ceasefire announced between Israel and non-state actor T before deadline D), specifically for state-vs-non-state conflicts where the state has existential or destruction-oriented war aims.

## When to Use

Question involves a ceasefire between a state and a non-state armed group (Hamas, Hezbollah, Houthis, etc.), particularly when:
- The state has stated a war aim of destroying or eliminating the non-state actor
- A specific deadline is given (prediction market format)
- The conflict is in its early-to-middle phase (first 1-18 months)

## Inputs

- Non-state actor T (Hamas, Hezbollah, etc.)
- Deadline D (the date by which ceasefire must be announced)
- Cutoff date for point-in-time knowledge
- State's stated war aim (official government language)
- Mediation structure (lead mediator, mediator leverage on each party)
- Status of any prior ceasefires in this conflict phase

## Procedure

### Step 0: Multi-Front Escalation Pre-Check (MANDATORY — before any other analysis)

Before assessing war aims or window length, classify the state's current military posture:

1. **Count active fronts**: How many fronts is the state actively fighting on? For each front, assess whether:
   - Active military operations are ongoing
   - The state has NOT yet achieved its publicly-stated objectives on that front
   - The state has NOT yet retaliated for adversary strikes on its territory
   - Ground forces have recently (within 30 days) entered a new theater

2. **If N >= 2 active fronts**: Apply the [[domains/mena/concepts/multi-front-escalation-ceasefire-barrier/_concept]] framework:
   - Default P(ceasefire on any single front within 30 days) < 0.01
   - Check the sequencing constraint: which front is most acute? The ceasefire cannot happen on the least-acute front while the most-acute front is unresolved
   - Check whether a superpower patron can override the barrier (e.g., escalation-bargaining pattern — only for state-on-state conflicts)
   - **If the state is in active multi-front escalation AND the question asks about a ceasefire on one specific front within < 90 days: output NO and stop.** The multi-front barrier is a deterministic veto for asymmetric conflicts (state vs non-state) on any single front during the escalation phase.

3. **If N = 1**: Proceed to Step 1 with the standard framework. The multi-front barrier does not apply.

### Step 1: Assess War Aims Compatibility

Read [[concepts/war-aims-incompatibility]] and classify the stronger party's stated war aim:

| War Aim | Ceasefire Feasibility |
|---------|----------------------|
| "Destroy" / "Eliminate" / "Annihilate" | Ceasefire structurally impossible until aim is either achieved or redefined |
| "Dismantle military capability" | Possible after substantial degradation achieved (6-18 months) |
| "Deter / Degrade" | Possible at any point (base rate for ceasefire in 6-month window ~0.15-0.30) |
| "Remove from power" | Requires political transition mechanism (complex, 12+ months) |

**If the state's war aim is "destroy" or equivalent**: Apply the incompatibility penalty. In the absence of a resolution pathway (leadership decapitation, military achievement, patron imposition), set P(ceasefire in any window < 6 months) < 0.05.

### Step 2: Apply the Short-Window Framework

If deadline D is within 6 weeks of the forecast date, apply [[concepts/short-window-ceasefire-probability]]:

1. Measure W = days from forecast to deadline
2. If W < 30 days: check for pre-existing negotiation framework
   - Framework exists → P(ceasefire) ~0.02-0.05 (mutual consent still needed)
   - No framework → P(ceasefire) < 0.01
3. Check whether a temporary humanitarian pause would satisfy resolution criteria:
   - If yes AND hostage/prisoner exchange talks are active → P increases to ~0.05-0.10
   - If no → apply base rate only
4. Apply mutual-consent penalty: even if mediators are working, both sides must be ready. If one side is clearly not ready, cap P at 0.02 regardless of window length.

### Step 3: Check the Mediation Structure

Mediation in asymmetric conflicts follows a consistent pattern for Israel-Hamas:

**Mediator Triad:** Qatar, Egypt, United States
- **Qatar**: Leverage on Hamas (hosts political leadership, provides funding) but NO leverage on Israel
- **Egypt**: Leverage on Hamas (Rafah crossing, security coordination) but limited leverage on Israel
- **US**: Leverage on Israel (military aid, diplomatic cover) but NO direct access to Hamas

**Leverage gap**: No single mediator has leverage over BOTH parties. This is a structural constraint on ceasefire speed.

**Calibration**:
- All three mediators active and coordinating: P(ceasefire in 3-month window) ~0.15-0.30
- One mediator (usually Qatar) only: P(ceasefire in 3-month window) ~0.05-0.10
- No active mediation: P(ceasefire) < 0.01

**Key signal**: When the US special envoy for Middle East humanitarian issues visits the region (as David Satterfield did in Jan-Feb 2024), this indicates active mediation is occurring but does NOT imply a deal is near. Mediator activity ≠ mediation success.

### Step 4: Check Domestic Political Constraints

For Israel in particular, coalition dynamics determine ceasefire feasibility:

**Assess coalition composition**:
- Does the governing coalition include hardline parties opposed to any ceasefire?
- If the PM relies on far-right votes (as Netanyahu did with Ben-Gvir's Otzma Yehudit and Smotrich's Religious Zionism), a ceasefire that leaves Hamas standing would collapse the government.
- **Netanyahu's decision rule**: Netanyahu will not risk his coalition for a ceasefire unless the alternative (continuing war) poses a greater political threat.

**Calibration**:
- Coalition majority opposed to ceasefire: P(ceasefire in any window) < 0.05 regardless of military factors
- Coalition divided: P(ceasefire) at base rate
- Coalition majority supports ceasefire: P(ceasefire) elevated by 1.5-2x

**For Jan-Feb 2024 specifically**: Netanyahu's coalition included far-right parties that had made opposition to a ceasefire a condition of remaining in government. The alternative to the coalition was an election that polls suggested Netanyahu would lose. This created a near-deterministic veto on any ceasefire that left Hamas standing.

### Step 5a: Assess Trust Erosion After Prior Ceasefire Collapse

If the current conflict phase began AFTER a prior ceasefire collapsed (i.e., the parties were under a ceasefire, one side resumed hostilities, and the question asks about a new ceasefire in the aftermath), apply the trust erosion framework from [[domains/mena/concepts/ceasefire-trust-erosion-after-collapse/_concept]].

**The core insight**: When a ceasefire collapses, the weaker party (non-state actor) suffers a trust shock that makes a new ceasefire 2-10x harder to achieve for 3-6 months, depending on the severity of the collapse and whether the mediation framework has changed.

**Checklist:**
1. Measure time since ceasefire collapse (t_collapse)
2. Apply trust erosion multiplier based on Table 4 in the concept file:
   - < 1 month → P reduced by ~90% (factor ~0.1x)
   - 1-3 months → P reduced by ~50-70% (factor ~0.3-0.5x)
   - 3-6 months → P reduced by ~30-50% (factor ~0.5-0.7x)
   - 6-12 months → P reduced by ~10-30% (factor ~0.7-0.9x)
3. Check for override conditions that can bypass trust erosion:
   - Catastrophic humanitarian pressure (famine, genocide finding) → reduces trust erosion's effect by ~50%
   - New mediation framework with different lead mediator → reduces trust erosion's effect by ~30%
   - Superpower guarantees (US security commitment, UNSCR with enforcement) → reduces trust erosion's effect by ~40%
4. Check domestic coalition dynamics on the stronger party's side: if far-right/hardline coalition members are satisfied with the resumed combat (as Ben-Gvir and Smotrich were after March 18, 2025), the domestic political cost of a ceasefire is very high — adding another independent barrier on top of trust erosion.

**Why this matters**: In the Israel-Hamas case, the March 18 collapse created trust erosion on the Hamas side AND satisfied far-right coalition dynamics on the Israeli side. Both operated simultaneously, making the July 15 ceasefire deadline a near-impossibility — the time since collapse (~4 months) was well within the high-trust-erosion zone, and domestic coalition dynamics added an independent veto.

### Step 5b: Assess Non-State Actor Incentives

**Hamas's strategic calculus in Jan-Feb 2024**:
- **Attrition strategy**: Hamas believed that prolonging the war would increase international pressure on Israel, create legitimacy costs, and eventually force an IDF withdrawal. Ceasefire = abandoning this strategy without having achieved territorial or political gains.
- **Hostage leverage**: The hostages were Hamas's primary bargaining chip. Returning them without a full IDF withdrawal and end-of-war guarantees would waste this leverage.
- **Leadership intact**: Sinwar was alive and in command (he was killed Oct 2024). The military leadership was committed to continued resistance.

**Calibration**:
- Non-state actor pursuing attrition strategy: P(ceasefire) reduced by 50-80% from base rate
- Non-state actor's leadership intact (no decapitation): P(ceasefire) reduced further
- Non-state actor's hostages create a negotiation path: P(ceasefire) slightly increased (but only if hostage deal is credible — it was not in Jan 2024)

### Step 6: Check for Military Escalation vs De-escalation

Is the stronger party preparing a major offensive or winding down operations?

- **Preparing escalation** (e.g., Israel planning Rafah operation for March 2024): P(ceasefire) caps at 0.02. Military momentum is toward escalation, not de-escalation.
- **Active operations without next-phase preparation**: base rate applies
- **Winding down / redeploying**: P(ceasefire) elevated by 1.5-2x

**Indicator**: IDF troop redeployments, new call-ups, cabinet authorization for new operations, statements about "final phase."

### Step 7: Synthesize

P(ceasefire in window) = BaseRate * WarAimsMultiplier * ShortWindowMultiplier * MediationMultiplier * DomesticConstraintMultiplier * TrustErosionMultiplier * NonStateActorMultiplier * EscalationMultiplier

**Where each multiplier is derived from the step above. Add TrustErosionMultiplier (Step 5a) when the question follows a ceasefire collapse; otherwise set to 1.0.**

**Example 1: Jan 27, 2024 cutoff — Israel-Hamas ceasefire by Feb 29 (33-day window, no prior collapse)**

| Factor | Assessment | Multiplier |
|--------|-----------|------------|
| Base rate for 1-month window (asymmetric conflict, no prior enduring ceasefire) | ~0.04 | — |
| War aims incompatibility (Israel: "destroy Hamas") | Incompatible | 0.2x |
| Short window (33 days) | Mutual-consent penalty active | 0.5x |
| Mediation (Qatar/Egypt active, US engaging) | Leverage gap on both sides | 0.7x |
| Domestic constraint (Netanyahu coalition — far-right veto) | Near-deterministic block | 0.3x |
| Non-state actor incentive (Sinwar alive, attrition strategy) | Delay preferred | 0.5x |
| Escalation trajectory (Rafah operation planned) | Escalation, not de-escalation | 0.3x |

P(ceasefire) = 0.04 * 0.2 * 0.5 * 0.7 * 0.3 * 0.5 * 0.3 = **~0.0013**

Apply soft floor of 0.02 for tail risk (accidental mediator breakthrough): **~0.02-0.05**

This maps to a clear NO prediction for the 33-day window.

**Example 2: July 8, 2025 cutoff — Israel-Hamas ceasefire by July 15 (7-day window, post-collapse)**

| Factor | Assessment | Multiplier |
|--------|-----------|------------|
| Base rate for 1-week window (asymmetric conflict, post-15-month war) | ~0.01 | — |
| War aims incompatibility (Israel: "destroy Hamas," resumed combat Mar 18) | Incompatible | 0.2x |
| Short window (7 days) | Mutual-consent penalty active | 0.3x |
| Mediation (Iran war just ended, diplomatic infrastructure not yet rebuilt) | No active mediation framework for Gaza | 0.3x |
| Domestic constraint (far-right satisfied with resumed combat, coalition crisis risk) | Near-deterministic block | 0.2x |
| **Trust erosion (Mar 18 collapse ~4 months prior, trust deficit still high)** | **High erosion (~4 months post-collapse)** | **0.4x** |
| Non-state actor incentive (Hamas "burned" by Jan deal collapse, hardened stance) | Delay preferred | 0.4x |
| Escalation trajectory (IDF in full campaign mode post-collapse) | Escalation, not de-escalation | 0.3x |

P(ceasefire) = 0.01 * 0.2 * 0.3 * 0.3 * 0.2 * 0.4 * 0.4 * 0.3 = **~0.000017**

Apply soft floor of 0.02 for tail risk (Trump surprise announcement, last-minute shuttle diplomacy): **~0.02-0.05**

This maps to a clear NO prediction for the 7-day window. The trust erosion multiplier (0.4x) and the domestic coalition satisfaction (0.2x) are the two new factors compared to Example 1 — both operate independently on different sides of the conflict, making the July 15 deadline a near-impossibility. The market price at ~0.45-0.50 was significantly overpriced relative to this structural analysis.

### Step 8: Post-Forecast Audit

After the deadline passes, compare prediction to outcome:
- If correct: which factors were strongest? Update the multiplier estimates. For the post-collapse case, specifically validate the trust erosion multiplier: was the actual time-to-new-ceasefire consistent with the trust erosion decay schedule?
- If wrong: which multiplier was misweighted? Was the trust erosion estimate too strong or too weak? Was the mutual-consent penalty too strong or too weak?

## Key Entities to Consult

- [[entities/hamas]] — organizational structure, leadership status, decision-making
- [[entities/benjamin-netanyahu]] — coalition calculus, risk tolerance
- [[entities/israel]] — military doctrine, multi-front constraints, war aims
- [[entities/qatar]] — mediator role, leverage on Hamas
- [[entities/egypt]] — mediator role, Rafah crossing leverage
- [[entities/yahya-sinwar]] — hardline leader status (pre-decapitation)
- [[entities/ismail-haniyeh]] — political leadership in Doha

## Key Concepts to Consult

- [[concepts/war-aims-incompatibility]] — the dominant blocking factor
- [[concepts/short-window-ceasefire-probability]] — window-adjusted calibration
- [[concepts/temporary-vs-enduring-ceasefire]] — the temporary pause exception
- [[concepts/political-deadline-ceasefire]] — deadline compression effects
- [[concepts/leadership-decapitation-negotiation-window]] — decapitation as resolution pathway
- [[concepts/diplomatic-pressure-tipping-point]] — medium-term pressure accumulation
