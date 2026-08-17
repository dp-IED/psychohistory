---
type: procedure
tags: [procedure, usa, forecasting, policy-analysis]
title: "Executive Action Feasibility Assessment"
slug: executive-action-feasibility-assessment
version: 1.0
date: 2026-05-20
author: hermes-agent
---

# Executive Action Feasibility Assessment

## Purpose

Assess whether a proposed executive action (EO, agency memo, regulatory change, or legislative push) is likely to succeed within a given time horizon. This procedure is designed for forecasting questions about presidential "will X happen" questions, particularly where the action involves program elimination, restriction, or creation.

## Procedure

### Step 1: Classify the Action Type

| Action Type | Description | Examples |
|-------------|-------------|----------|
| **Executive Order** | Directive to executive branch agencies; must cite statutory or constitutional authority | Travel ban, Buy American EO |
| **Agency Guidance/Memo** | Interpretation of existing law or regulation; no public comment required | USCIS policy memo narrowing specialty occupation definition |
| **Rulemaking (APA)** | New regulation requiring notice-and-comment; 6-18 month timeline | H-1B wage rule, public charge rule |
| **Legislation** | Bill passed by Congress and signed or vetoed | RAISE Act (never passed), ACA repeal (failed) |
| **Appointment** | Nominate or recess appoint an agency head who redirects policy | Mulvaney at CFPB, Miller as policy chief |

### Step 2: Identify Statutory Basis

- **If program is statutory** (created by Congress via law): elimination requires legislation or a legally dubious executive action. Probability of durable elimination: <10% in first year.
- **If program is executive-created** (established by EO or agency action): elimination is easier. Probability: 50-80% in first year.
- **If action is within agency discretion** (e.g., adjudication guidance, enforcement prioritization): probability: 70-90% in first 100 days.

### Step 3: Assess Time Horizon

Use [[concepts/first-100-days-action-horizon]] to determine what fits in the available window. General heuristics:

| Action | First 100 days | First year | Full term |
|--------|---------------|------------|-----------|
| Executive order | High (70-90%) | Very high (90%+) | Near-certain |
| Agency guidance | Medium (40-70%) | High (70-90%) | Near-certain |
| APA rulemaking | Very low (<5%) | Medium (30-50%) | High (70-90%) |
| Legislation | Very low (<1%) | Low (5-20%) | Medium (20-50%) |
| Program elimination via EO | Very low (<2%) | Low (5-15%) | Low-Medium (10-30%) |
| Program restriction via EO | Medium (30-60%) | High (70-90%) | Near-certain |

### Step 4: Evaluate Coalition Constraints

**Key question**: Would the action unite the president's own coalition in opposition?

Assess:
- Is there a faction within the governing coalition that benefits from the current program?
- Can that faction credibly threaten defection on higher-priority items?
- Does the faction have independent political power (media reach, donor leverage, constituency)?

**Heuristic**: If the president's largest donor, a powerful media ally, or a critical electoral constituency would oppose elimination, the action is unlikely even if the legal path exists.

Document the specific factions:

| Faction | Position | Power Metric | Constraint Strength |
|---------|----------|-------------|-------------------|
| Tech/Libertarian (Musk, Ramaswamy) | Pro-H-1B | Donor leverage, X platform | High for H-1B elimination |
| Nativist/Restrictionist (Miller) | Anti-H-1B | Policy proximity, base messaging | High for all immigration restriction |
| (Add as applicable per question) | | | |

### Step 5: Precedent and Pattern Check

- Has this president taken this action before? (If yes and it survived legal challenge, probability is higher.)
- Has this president taken related but milder action? (If restriction precedent exists, elimination is unlikely — see [[concepts/program-restriction-vs-elimination]].)
- What happened to similar actions by other presidents? (Litigation outcomes, congressional response.)

### Step 6: Legal Vulnerability Assessment

| Legal Risk Factor | Low Risk | High Risk |
|-------------------|----------|-----------|
| Authority cited | Clear statutory delegation | Novel constitutional claim |
| Precedent | Well-established | No or negative precedent |
| Circuit split | No circuit conflict | 2+ circuits would rule differently |
| Standing parties | None | Major corporations, states, unions |
| Congressional action | None/delegation | Flat prohibition in statute |

**Heuristic**: If the action would be challenged by entities with legal standing, clear injury, and a friendly circuit court, litigation risk is high and the action is unlikely to survive long-term.

### Step 7: Apply Bayesian Updating from Market Prices

Compare your assessed probability (from steps 1-6) against prediction market prices. If market prices deviate from structural assessment:

- Market is too high (market pricing in elimination at 20%+ when structural assessment says <5%): market may be reacting to campaign rhetoric or recent news; structural factors typically dominate over time.
- Market is too low (market pricing elimination at <5% when structural assessment says 10-20%): market may be overcorrecting for perceived political constraints.

### Step 8: Combine and Output

For YES/NO binary questions:
- Combine factors above into a probability estimate.
- Output with explicit reasoning that references the specific structural factors identified.

## Example Application: H-1B Elimination in Trump's First 100 Days

| Factor | Assessment | Probability Impact |
|--------|-----------|-------------------|
| Action type | Legislation or legally aggressive EO | Very low base-rate |
| Statutory basis | INA § 1101(a)(15)(H) — statutory program | Very high bar for elimination |
| Time horizon | 100 days — no APAnearulemaking possible | Very low |
| Coalition constraint | Tech donors would oppose directly | High constraint |
| Precedent | Trump restricted H-1B in first term, never eliminated | Restriction pattern |
| Legal vulnerability | Immediate suit under Youngstown and APA | High vulnerability |
| **Combined** | | **<5% (effectively NO)** |

## Related Concepts
- [[concepts/program-restriction-vs-elimination]]
- [[concepts/first-100-days-action-horizon]]
- [[domains/global/concepts/forecast-resolution-criteria-gotchas]]

## Related Procedures
- [[domains/usa/procedures/candidate-withdrawal-probability]] — analogous structural baseline approach
