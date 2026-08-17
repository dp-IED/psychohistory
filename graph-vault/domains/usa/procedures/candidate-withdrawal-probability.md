---
type: procedure
tags: [procedure]
title: "Candidate Withdrawal Probability Assessment"
slug: candidate-withdrawal-probability
domain: risk-assessment
related_concepts:
  - post-nomination-persistence-baseline
  - leadership-persistence-under-threat
  - incumbent-withdrawal-cascade
related_procedures:
  - proc-aging-incumbent-early-warning
---

# Candidate Withdrawal Probability Assessment

## When to Use

Apply this procedure whenever a question asks whether a US presidential candidate will withdraw from the race, drop out, or otherwise cease their candidacy before the general election. This procedure synthesizes three related concepts into a single structured workflow:

- [[concepts/post-nomination-persistence-baseline]] — structural baseline for non-incumbent nominees
- [[concepts/leadership-persistence-under-threat]] — threat-hardening dynamics (legal jeopardy, assassination attempts)
- [[concepts/incumbent-withdrawal-cascade]] — internal-pressure-driven withdrawal dynamics

## Prerequisites

Before starting, ensure the following vault files exist:
- [[entities/donald-trump]] or the entity file for the subject candidate
- [[threads/2024-us-presidential-election]] or equivalent thread for the current cycle
- [[concepts/leadership-persistence-under-threat]]
- [[concepts/incumbent-withdrawal-cascade]]
- [[concepts/post-nomination-persistence-baseline]]

If any are missing, create the missing stubs before proceeding.

## Procedure Steps

### Step 1: Classify the Candidate Type

Determine which framework governs this candidate:

| Candidate Type | Governing Framework | Default Baseline |
|----------------|-------------------|-----------------|
| **Non-incumbent, nomination clinched** | [[concepts/post-nomination-persistence-baseline]] | <1% withdrawal probability |
| **Non-incumbent, likely nominee (>80% delegates)** | [[concepts/post-nomination-persistence-baseline]] | <5% withdrawal probability |
| **Non-incumbent, primary phase** | [[concepts/leadership-persistence-under-threat]] | Variable — proceed to Steps 2-7 |
| **Incumbent, nomination clinched** | [[concepts/incumbent-withdrawal-cascade]] | Variable — apply 5-condition framework |
| **Incumbent, primary challenge ongoing** | [[concepts/incumbent-withdrawal-cascade]] | Variable — check trigger events |
| **VP nominee** | [[concepts/post-nomination-persistence-baseline]] | <5% (only 1 VP withdrawal since 1972) |

**Gate question**: Has the candidate secured the nomination AND is the candidate a non-incumbent?
- If YES → baseline is <1%. Proceed to Step 7 for documentation, but the forecast is effectively deterministic.
- If NO → proceed to Steps 2-7.

**Document the classification explicitly in the reasoning.**

### Step 2: Assess Legal Jeopardy

Does the candidate face pending criminal charges that would be enforced upon loss of office?

- **Legal jeopardy present**: Creates existential motivation to persist (office = legal protection). This is the strongest single predictor of persistence.
  - Reference: [[concepts/leadership-persistence-under-threat]] — "the combination of legal jeopardy + nomination lock-in is nearly deterministic for persistence"
  - Baseline adjustment: add +40-50% to persistence probability (or subtract equivalent from withdrawal probability)

- **Legal jeopardy absent**: The existential motivation is removed, making withdrawal possible if other conditions (internal pressure, trigger event, successor available) converge.
  - Baseline adjustment: removal of legal jeopardy is a NECESSARY condition for the withdrawal cascade to operate.
  - If legal jeopardy is present, withdrawal probability remains <5% regardless of other factors (barring total incapacitation).

**Document the legal jeopardy status explicitly.**

### Step 3: Check Nomination Status

Where does this candidate stand in the nomination process?

- **Nomination clinched (delegate majority secured)**: For non-incumbents, structural lock-in makes withdrawal effectively impossible (<1%). For incumbents, the lock-in is weaker — incumbency creates a different pathway (see Step 4).
- **Presumptive nominee (but not yet clinched)**: The baseline is slightly higher (<5%) for non-incumbents but still very low. The psychological barrier to withdrawal from a commanding lead is immense.
- **Primary phase, leading**: Withdrawal is possible but unlikely (5-10%) unless a specific cascade develops.
- **Primary phase, trailing badly**: Withdrawal probability varies by candidate strategy (10-50%). Some candidates (Kasich 2016, Gabbard 2020) stay to the convention for leverage; others (Biden 1988, Dean 2004) withdraw early after poor showings.

For all cases: **A leader's stated intention to stay is NOT a reliable signal.** See [[concepts/incumbent-withdrawal-cascade]] Stage 0 — leaders who will ultimately withdraw deny intention up to the moment of withdrawal.

**Document the nomination status explicitly.**

### Step 4: Map Internal Pressure

Is the pressure to withdraw coming from inside the candidate's own party, or from external sources?

- **Internal pressure present (party leaders, donors, elected officials calling for withdrawal)**: This is the SINGLE PATHWAY to withdrawal WITHOUT legal jeopardy. Document who the defectors are — first defectors from the safe-seat/retiring-member category are a leading indicator; institutional leadership defection (party elders, committee chairs) makes withdrawal probable within 7-14 days.
- **Internal pressure absent**: Withdrawal is unlikely for any candidate who has secured the nomination, regardless of external pressure (media, opponents, poll numbers). External pressure typically produces the backlash effect described in [[concepts/leadership-persistence-under-threat]].
- **Mixed**: Some individual defectors but no institutional leadership engagement. The cascade is still containable — see Cascade Velocity section of [[concepts/incumbent-withdrawal-cascade]].

**Document the pressure source and intensity explicitly.**

### Step 5: Check Trigger Event

Has a specific, public, undeniable failure occurred that converts latent vulnerability into a cascade?

- **Trigger event present AND internal pressure present**: Withdrawal cascade may be active. Estimate timing using velocity benchmarks from [[concepts/incumbent-withdrawal-cascade]].
- **Trigger event present but internal pressure absent**: The trigger alone is insufficient. The leader may be weakened but will not withdraw unless internal pressure converges.
- **Trigger event absent**: Default to the post-nomination baseline or leadership-persistence framework. Without a trigger, even a deeply vulnerable candidate will not withdraw.
- **Trigger event type matters**: External threats (assassination attempt, indictment) produce the persistence-hardening effect. Performance failures (debate disaster, primary loss) produce the cascade trigger effect. Categorize the trigger type before assessing its impact.

**Document the trigger type and presence explicitly.**

### Step 6: Assess Successor Availability

Is there a natural successor who can absorb the campaign infrastructure?

- **Successor ready (VP, popular cabinet member, party leader)**: Withdrawal is structurally easier. The cost of replacement (infrastructure lost) is lower. The Biden→Harris 72-hour absorption is the benchmark.
- **Successor unclear or contested**: Withdrawal is structurally harder. The party faces a costly scramble and potential contested convention. This raises the barrier to withdrawal.
- **No credible successor (party restructured around the candidate)**: Withdrawal is effectively impossible. The campaign infrastructure, donor network, and party loyalty are all personal to the candidate. There is no one who can absorb them. This was the case for Trump 2024.

**Document successor availability explicitly.**

### Step 7: Combine and Calibrate

Integrate all six assessments into a single probability estimate.

#### For Non-Incumbent Nominees (Nomination Clinched)

Default baseline: **<1% withdrawal probability**. This overrides all other factors (legal jeopardy, internal pressure, trigger events) because the structural lock-in is the dominant variable.

| Compounding Factors | Withdrawal Probability | Reasoning |
|---------------------|----------------------|-----------|
| Baseline only (nomination clinched) | <1% | Zero historical cases since 1972 |
| + Legal jeopardy | <0.5% | Compounding — legal jeopardy makes persistence existential |
| + Assassination attempt | <0.5% | Threat creates rally effect, further hardens persistence |
| + Internal pressure | <5% | Internal pressure alone cannot overcome structural lock-in for a non-incumbent. The party CANNOT replace a non-incumbent nominee after ballot-access deadlines without severe legal and logistical consequences. |
| + All of the above | <1% | Over-determined — persistence is near-100% regardless |

**Key insight for non-incumbents**: The candidate's polling, debate performance, donor confidence, and media narrative are essentially irrelevant to withdrawal probability once the nomination is clinched. The structural lock-in overrides all performance-based variables. Only total incapacitation (death, coma, catastrophic health event) can cause withdrawal.

#### For Incumbent Nominees

Use the 5-condition framework from [[concepts/incumbent-withdrawal-cascade]]:

| Conditions Met | Withdrawal Probability | Reasoning |
|---------------|----------------------|-----------|
| 0 | <5% | Incumbent is fully secure |
| 1-2 | 10-20% | Latent vulnerability present but no trigger/cascade |
| 3 (incl. legal jeopardy absent) | 30-50% | Trigger event present + one other condition |
| 4 | 50-70% | Internal pressure + trigger + successor ready |
| 5 (all) | >70% | Full cascade: no legal jeopardy + internal pressure + trigger + successor + weak position |

**If legal jeopardy is present for an incumbent**: The persistence-under-threat framework overrides the withdrawal-cascade framework. Withdrawal probability drops to <5% even if other conditions are met. An incumbent facing charges cannot afford to leave office.

#### For Primary-Phase Candidates

| Scenario | Withdrawal Probability | Notes |
|----------|----------------------|-------|
| Leading big (>20 point lead), soon to clinch | <5% | Transitioning to post-nomination baseline |
| Competitive (within 5-10 points), path to nomination | 15-25% | Depends on financial health, trajectory |
| Trailing badly, no path to nomination | 40-80% | Depends on strategic calculation: stay for VP/future leverage vs. fold early |
| Trailing badly, and legal jeopardy exists | 5-15% | Legal jeopardy creates motivation to stay in race for legal protection (if victory is remotely plausible) |
| Pressure from party donors to clear the field | 30-60% | If party donors coordinate behind a single alternative |

### Step 7b: Check for Stated-Intention Discount

**Mandatory**: The candidate's public statements denying any intention to withdraw are Stage 0 behavior (see [[concepts/incumbent-withdrawal-cascade]] Stage 0). They carry ZERO evidentiary weight.

- The leader genuinely intends to stay at the moment of speaking — the denial is not deceptive, it reflects their current intention.
- But structural vulnerability conditions override stated intentions. Truman (1952), LBJ (1968), and Biden (2024) all denied intention to withdraw up to the moment of withdrawal.
- **Overweight structural conditions** (legal jeopardy, nomination status, internal pressure, trigger events, successor availability).
- **Underweight stated intentions** — they are a structural feature of the Stage 0 phase, not a signal.

### Cascade Velocity Check (Post-Trigger Only)

If a trigger event has occurred AND internal pressure is present AND the candidate is an incumbent, estimate time-to-withdrawal using the velocity benchmarks from [[concepts/incumbent-withdrawal-cascade]]:

| Trigger Type | Historical Time-to-Withdrawal | Denial Phase Duration |
|-------------|------------------------------|----------------------|
| Primary loss (unambiguous) | ~18 days (Truman 1952) | ~0 days |
| Primary near-loss | ~19 days (LBJ 1968) | ~3-5 days |
| Debate/performance failure | ~24 days (Biden 2024) | ~11 days |
| Health scare | No clear benchmark | Estimate 14-21 days |

**Document the cascade phase and expected timing explicitly.**

## Verification Checklist

After completing the assessment, verify:

- [ ] Candidate type is classified and governing framework selected
- [ ] Legal jeopardy status documented (the binary gate)
- [ ] Nomination status documented
- [ ] Internal pressure mapped (source, intensity, cascade phase)
- [ ] Trigger event documented (type, date, effect direction)
- [ ] Successor availability assessed
- [ ] Stated intentions explicitly discounted in the reasoning
- [ ] Combined probability estimate documented with rationale
- [ ] Both the persistence AND withdrawal cases assessed (dual-frame requirement)
- [ ] Cross-reference to the [[concepts/post-nomination-persistence-baseline]] applied for non-incumbent nominees

## Validated By

| Forecast | Prediction | Actual | Procedure Support |
|----------|-----------|--------|------------------|
| Will Trump drop out of 2024 race? | NO | NO (correct) | Step 1 → Non-incumbent, nomination clinched → <1% baseline. Step 2 → Legal jeopardy compounding → <0.5%. Correctly predicted NO. |
| Will Biden drop out of 2024 race? | NO | YES (wrong) | Step 1 → Incumbent, nomination clinched → should have triggered the incumbent-withdrawal-cascade framework. Early version of this procedure did not exist at forecast time. The error was using a persistence frame for an incumbent with no legal jeopardy. |

## Wikilinks

[[concepts/post-nomination-persistence-baseline]], [[concepts/leadership-persistence-under-threat]], [[concepts/incumbent-withdrawal-cascade]]
[[entities/donald-trump]], [[entities/joe-biden]]
[[threads/2024-us-presidential-election]]
[[procedures/proc-aging-incumbent-early-warning]], [[procedures/proc-incumbent-withdrawal]]
