---
type: procedure
tags: [procedure, conflict-termination]
slug: inter-state-ceasefire-feasibility
applies_to: "Russia-Ukraine, India-Pakistan, Iran-Israel, Taiwan Strait, Korean Peninsula, Ethiopia-Eritrea"
---
# Inter-State Ceasefire Feasibility Assessment

## When to Use
Use this procedure when forecasting whether a ceasefire will occur in an ongoing inter-state war within a specified time window (e.g., "by end of year," "within 6 months," "by deadline X").

## Required Inputs
- Identity of both warring states
- Date range of the forecast window
- Any known mediation attempts or diplomatic frameworks

## Procedure

### Step 1: Territorial Incompatibility Check
Determine whether the war aims of both parties are mutually exclusive:
- [ ] Does one side demand territorial concessions the other side's domestic politics cannot accept?
- [ ] Has one side formally annexed or declared sovereignty over disputed territory?
- [ ] Is the conflict framed as existential (regime survival / national existence) rather than limited?
- **If YES to 2+ of these → territorial incompatibility is HIGH → ceasefire probability baseline LOW (P < 0.15 within 12 months)**

### Step 2: Military Trajectory Check
Determine whether the military situation is a positional stalemate:
- [ ] Has front-line movement been <5km/month on primary axes for 6+ consecutive months?
- [ ] Can both sides conduct defensive operations effectively?
- [ ] Does neither side have air superiority enabling decisive ground maneuver?
- [ ] Are offensive operations suffering disproportionate casualties relative to gains?
- **If YES to 3+ of these → military stalemate → ceasefire probability baseline LOW**

### Step 3: Mutually Hurting Stalemate Check
Determine whether the war costs are unbearable for either party:
- [ ] Is either side's regime at risk of collapse from war costs? (Popular unrest, elite defections, coup risk)
- [ ] Are external patrons threatening or implementing aid cutoffs?
- [ ] Are domestic publics demanding withdrawal or settlement?
- [ ] Is the wartime economy showing signs of terminal stress?
- **If NO to all of these → no mutually hurting stalemate → ceasefire probability baseline LOW**

### Step 4: Credible Mediator Check
Determine whether an external actor can effectively enforce a ceasefire:
- [ ] Does a mediator have military or economic leverage over BOTH parties?
- [ ] Is the mediator WILLING to use that leverage (not just offer good offices)?
- [ ] Do BOTH sides trust or fear the mediator sufficiently to accept terms?
- [ ] Does the mediator have capacity to enforce/guaranty terms?
- **If NO to 2+ of these → no credible mediator → ceasefire probability baseline LOW**

### Step 5: External Sustainment Check
Determine whether both sides can continue fighting:
- [ ] Is the weaker party receiving sufficient arms/munitions from patrons?
- [ ] Can the stronger party sustain its war effort from domestic production or imports?
- [ ] Are economic sanctions against the aggressor porous or insufficient?
- [ ] Do energy/commodity export revenues fund the war effort?
- **If YES to 3+ of these → external sustainment is STABLE → war can continue → ceasefire probability baseline LOW**

### Step 6: Political Deadline Check
Determine whether an approaching political transition could force a breakthrough:
- [ ] Is there an election, inauguration, or leadership transition within 3 months?
- [ ] Would the incoming administration change aid posture, sanctions policy, or diplomatic engagement?
- [ ] Does the deadline create a narrow window for both sides to accept terms?
- **Apply [[domains/global/concepts/political-deadline-ceasefire]] framework if any YES**

### Step 7: Synthesis
Use [[domains/global/concepts/protracted-war-stalemate]]:
- **If 4+ steps above point toward continuation → P < 0.15 for ceasefire within 12 months**
- **If 5+ steps point toward continuation → P < 0.05 for ceasefire within 12 months**

### Step 8: Context-Specific Adjustments
- **Nuclear power conflict**: Ceasefire via patron entry ([[domains/global/concepts/escalation-bargaining-termination]]) does NOT apply because the nuclear power cannot be militarily coerced by a superpower patron. The conflict termination mechanism must be diplomatic or internal.
- **Asymmetric (state vs non-state)**: Use [[domains/global/concepts/diplomatic-pressure-tipping-point]] pattern instead — accumulated international pressure and patron leverage over the non-state actor, typically 4-8 months timeline.
- **Territorial conquest vs regime change**: If the war aim is regime change rather than territory, territorial incompatibility (Step 1) is lower but the conflict may be harder to end because the target state cannot surrender without dissolving.

## Validation
- **Russia x Ukraine ceasefire in 2024?** (PIT: 2024-12-31) — All 5 primary factors pointed toward continuation. Steps 1-5 all confirmed structural barriers. No credible mediator existed. Neither side was hurting enough to accept unfavorable terms. Outcome: NO. ✓
- **Iran-Israel ceasefire June 2025?** — Different pattern (escalation-bargaining-termination). Superpower entry as co-belligerent created fast termination pathway not available in nuclear-power conflicts. **Classified as Pathway B (war-termination) under [[domains/global/concepts/ceasefire-pathway-decomposition]]** — the decomposition formula P(ceasefire) = P(war) × P(termination | war) correctly predicted YES.
- **Israel-Hamas ceasefire Oct 2025?** — Different pattern (diplomatic-pressure-tipping-point). Accumulated international pressure + Trump mediation created conditions that took ~7 months from March 2025 collapse to October 2025 ceasefire. **Classified as Pathway A (diplomatic) under the ceasefire-pathway-decomposition framework.**

## Pre-Assessment Required

**Before using this procedure, classify the ceasefire pathway using [[domains/global/concepts/ceasefire-pathway-decomposition]]:**
- If Pathway B (war-termination): Use [[domains/global/procedures/state-on-state-ceasefire-decomposition]] instead. This procedure's feasibility factors (territorial incompatibility, military trajectory, mutually hurting stalemate) do not apply to war-termination ceasefires.
- If Pathway A (diplomatic): This procedure's feasibility factors apply as documented.
- If Pathway C (none likely): Skip to NO prediction.

## Related Concepts
- [[domains/global/concepts/protracted-war-stalemate]] — Theoretical framework
- [[domains/global/concepts/escalation-bargaining-termination]] — Fast ceasefire pattern (does NOT apply to nuclear-power conflicts)
- [[domains/global/concepts/diplomatic-pressure-tipping-point]] — Medium-term ceasefire pattern
- [[domains/global/concepts/political-deadline-ceasefire]] — Political transition as deadline catalyst

## Related Procedures
- [[domains/global/procedures/ceasefire-timing]] — Timing calibration for date-specific questions

## References
- [[domains/global/threads/russia-ukraine-war/_thread]] — Full war timeline and structural analysis
- [[domains/global/concepts/protracted-war-stalemate/_concept]] — Theoretical foundations
