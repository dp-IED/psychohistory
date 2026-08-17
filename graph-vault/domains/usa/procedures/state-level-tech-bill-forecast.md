---
type: procedure
tags: [procedure]
title: "State-Level Tech Bill Forecasting"
slug: state-level-tech-bill-forecast
created: 2026-05-20
---

# State-Level Tech Bill Forecasting Procedure

## Applicability

Use this procedure when forecasting whether a state-level technology regulation bill will pass (be signed into law) by a given deadline. This covers AI safety bills, privacy legislation, content moderation requirements, algorithmic accountability bills, and similar tech policy legislation at the state level — with primary applicability to California as the canonical jurisdiction.

## Prerequisites

Before starting, ensure the following vault assets are loaded:
- [[domains/usa/concepts/state-level-tech-regulation-bellwether/_concept]] — bellwether dynamics
- [[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]] — veto assessment
- [[domains/usa/threads/state-level-ai-regulation/_thread]] — ongoing thread
- Entity stubs for: governor [[domains/usa/entities/gavin-newsom]], bill author [[domains/usa/entities/scott-wiener]], legislature [[domains/usa/entities/california-state-legislature]]

## Step 1: Bill Stage Assessment

Determine the bill's current stage in the legislative process:

| Stage | Description | Baseline passage probability |
|-------|-------------|------------------------------|
| Introduced | Bill filed, assigned to committee | 10-20% |
| Committee passed | Approved in policy committee | 20-40% |
| Passed one chamber | Floor vote in one chamber | 30-60% |
| Passed both chambers | Approved by both, on governor's desk | 40-80% (governor veto risk becomes dominant) |
| On governor's desk | Delivered for signature or veto | 10-80% depending on governor posture |

For "will [bill] pass?" questions, passage means the bill becomes law (signed or veto overridden). For California, the August 31 legislative deadline defines the deadline — a bill that hasn't passed both chambers by Aug 31 dies.

**Key check**: If the question asks about passage by a specific date that falls BEFORE the legislative deadline, the bill must be further along in the process. A bill introduced in June must pass through committee(s), floor votes in both chambers, and get signed — in 2-3 months. This is a compressed timeline that significantly reduces passage probability.

## Step 2: Legislative Calendar Analysis

California's legislative calendar creates structural constraints:

- **Even-numbered years**: Bills must pass by Aug 31. Governor signs/vetoes by Sep 30.
- **Odd-numbered years**: First year of session. Bills can carry over to the next year. Aug 31 deadline still applies but bills introduced late in the year carry over.
- **End-of-session crunch**: The final 2-3 weeks before Aug 31 see massive bill volume. Controversial bills face procedural obstacles (last-minute amendments, gut-and-amend tactics, floor time shortage).

**Calendar-driven probability adjustments**:
- Bill introduced within 3 months of Aug 31 → baseline passage probability reduced by 30-50%
- Bill has not passed one chamber by July 1 → probability reduced by 50-70%
- Bill on governor's desk with >14 days before Sep 30 → standard assessment
- Bill on governor's desk in final week before Sep 30 → increased probability of pocket veto or inaction

## Step 3: Veto Point Analysis

Use the [[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]] framework:

1. **Governor national ambition**: Is the governor positioning for higher office?
   - If yes → 50-70% veto probability for broad, novel, industry-opposed bills
   - If no → 30-50% veto probability

2. **Alternative pathways**: Can the governor achieve similar policy through executive action?
   - If yes → 60-80% veto probability (substitutes executive order for legislation)
   - If no → 30-50% veto probability

3. **Industry opposition intensity**: Are major tech companies united against the bill?
   - Unified opposition → 50-70% veto probability
   - Divided (some support, some oppose) → 30-50% veto probability
   - Tech industry neutral or supportive → 10-30% veto probability

4. **Bill novelty**: First-of-its-kind?
   - Yes → 60-80% veto probability
   - No, similar bill exists in another state or prior CA session → 30-50%

5. **Override assessment**: Can the legislature override a veto?
   - Party holds <2/3 in either chamber → override impossible → governor's veto is final → veto probability remains high
   - Party holds 2/3+ in both chambers → override theoretically possible → governor may be more cautious in vetoing
   - For California: Democratic supermajority exists in both chambers but does NOT reach 2/3 for override in practice (ideological diversity reduces unity). Treat override as practically impossible for controversial bills.

**Composite adjustment**: Combine the above signals. If 4+ factors point toward veto and override is impossible, P(signed) = 20-40% even if the bill has passed both chambers. If 2 or fewer factors point toward veto, P(signed) = 60-80%.

## Step 4: Governor National Ambition Signal

The governor's national ambition is the most dynamic variable. Key signals:

- **Convening national expert panels** on the topic → high national ambition signal
- **Issuing executive orders** on the topic while a bill is pending → negative signal for bill passage (governor prefers alternative pathway)
- **Making public statements** about the need for "balanced" or "careful" regulation → cautious posture, increases veto probability
- **Avoiding public position** on the specific bill → neutral-to-cautious, watch for other signals
- **Explicitly endorsing the bill** → strong positive signal (but rare for controversial first-of-its-kind bills)

## Step 5: Political Landscape Mapping

Map the key political actors and their positions:

- **Governor**: Position, national ambition, relationship with tech industry
- **Bill author**: Influence, coalition-building history, willingness to amend
- **Assembly Speaker / Senate President pro Tem**: Whether leadership is actively advancing or blocking the bill
- **Key committee chairs**: Position on the bill, gatekeeping power
- **Notable opposition figures**: Influential party members (like Nancy Pelosi on SB 1047) who split the party caucus
- **Tech industry lobbying**: Which companies are active, their spending, their California presence/operations

## Step 6: Probability Calibration

Synthesize steps 1-5 into a calibrated probability:

1. **Baseline**: Start from the stage-based baseline (Step 1)
2. **Calendar adjustment**: Reduce if timeline is compressed (Step 2)
3. **Veto adjustment**: Apply veto probability from Step 3 — if veto is predicted, multiply passage probability by (1 - veto_prob)
4. **Governor ambition check**: Downward adjustment if governor has ambition signals and alternative pathways (Step 4)
5. **Final calibration**: P(pass) = P(stage baseline) × P(no veto) × calendar factor

## Step 7: Document Reasoning

Write the forecast reasoning documenting:
- The bill's current stage and legislative history
- The calendar deadline and its implications
- The governor's posture and national ambition assessment
- The veto override assessment
- The industry opposition landscape
- The key uncertainty variable (most often the governor's final decision)
- The final calibrated probability with justification

## Wikilinks
- [[domains/usa/concepts/state-level-tech-regulation-bellwether/_concept]]
- [[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]]
- [[domains/usa/threads/state-level-ai-regulation/_thread]]
- [[domains/usa/entities/gavin-newsom]]
- [[domains/usa/entities/scott-wiener]]
- [[domains/usa/entities/california-state-legislature]]
