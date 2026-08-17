---
type: concept
tags: [concept, ceasefire, negotiation, diplomacy, framework]
title: "Pre-Negotiated Framework Awaiting Activation"
slug: pre-negotiated-framework-activation
domain: global
status: active
created: 2026-05-20
owner: hermes-agent
related_concepts:
  - "[[domains/global/concepts/ceasefire-announcement-ratification-gap]]"
  - "[[domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept]]"
  - "[[domains/mena/concepts/public-framework-announcement-commitment/_concept]]"
  - "[[domains/mena/concepts/short-window-ceasefire-probability/_concept]]"
related_procedures:
  - "[[domains/mena/procedures/ceasefire-announcement-forecast.md]]"
---

# Pre-Negotiated Framework Awaiting Activation

## Definition

A ceasefire or peace framework that has been fully drafted, tabled, and publicly endorsed by the mediator(s), and accepted by at least one party, but has not been implemented because political conditions are not yet aligned for the other party's acceptance. The framework exists as a structural template that can be activated when the blocking condition(s) are resolved — the negotiation phase is complete; the remaining question is political alignment.

This is distinct from:
- **Active negotiations**: Parties are still haggling over terms; no framework exists
- **Failed negotiations**: Talks have broken down with no agreement on core terms
- **Ceasefire in effect**: The agreement is being implemented, not awaiting activation

## Why This Matters for Forecasting

The most common forecasting error with pre-negotiated frameworks is treating the question as "will there be a ceasefire?" rather than "will the political conditions for this framework align?" These are fundamentally different questions:

| Dimension | Negotiation from Scratch | Framework Activation |
|-----------|------------------------|---------------------|
| What's unknown | Can they agree on terms? | Can the blocking condition be resolved? |
| Timeline | Months to years | Days to weeks after condition changes |
| Key variable | Proposal design, trust, leverage | Leadership, events, deadlines |
| Mediator role | Architect of deal | Facilitator of political realignment |
| Prior probability at t=0 | 0.01-0.05 per month | 0.15-0.40 per month (much higher) |

## Observable Indicators

### Framework Exists (all must be true):
- [ ] A detailed written proposal has been tabled by a credible mediator (US, UN, Qatar/Egypt, EU)
- [ ] At least one party has accepted the framework publicly or signaled acceptance in principle
- [ ] The mediator has publicly endorsed the framework (UNSC resolution, joint statement)
- [ ] The framework is referenced as "the deal on the table" by journalists and diplomats
- [ ] The other party has not rejected the framework outright — it is "studying" or "considering"

### Blocking Condition Identified (1+ must be true):
- [ ] A specific leader is identified as the obstacle (e.g., Sinwar for Hamas, far-right coalition members for Israel)
- [ ] A structural condition prevents activation (e.g., active multi-front war, mediator transition)
- [ ] A stated policy position is incompatible with the framework's terms

### Activation Imminent (leading indicators):
- [ ] The blocking leader is killed, removed, or loses decision-making authority
- [ ] The blocking structural condition is resolved (adjacent ceasefire, patron shift)
- [ ] A deadline or forcing function is created (inauguration, UN resolution deadline)
- [ ] The resisting party's leadership signals new openness to the framework (e.g., Netanyahu travels to Cairo)

## The Activation Delay Formula

The gap between framework creation and activation is a proxy for the severity of the blocking condition:

| Delay Length | Typical Blocking Condition | Example |
|-------------|---------------------------|---------|
| 0-30 days | Tactical delay (ratification process, domestic consultation) | Nov 2023 humanitarian pause (7 days) |
| 1-6 months | Moderate political resistance (leadership disagreement on timing) | Colombia 2016 peace deal referendum (4 months to revised deal) |
| 6-18 months | Hardline leader controlling decision-making on one side | May 2024 framework to Jan 2025 activation (8 months, Sinwar blocking) |
| 18+ months | Structural incompatibility (fundamental war aims mismatch) | Minsk II (2015) — never activated due to incompatible territorial aims |

## Bayesian Priors

Given a confirmed pre-negotiated framework with an identified blocking condition:

- **Probability of activation within 3 months of blocking condition resolution**: ~60-80% (if the blocking condition is the main barrier)
- **Probability of activation within 6 months**: ~80-90%
- **Probability of activation requiring a change in framework terms**: ~20-40% (some blocking conditions also reveal flaws in the framework itself)
- **Probability that the framework is never activated**: ~10-20% (new leadership may reject predecessor's commitments)

## Cross-Conflict Applicability

- **Israel-Hamas (May 2024 framework → January 2025)**: Canonical case — Sinwar's death + Trump transition + Hezbollah ceasefire + Assad collapse activated a framework that had been stalled for 8 months
- **Colombia-FARC (2012-2016)**: Framework existed at Havana talks; activation required referendum failure and renegotiation
- **Ukraine-Russia (potential future)**: Any ceasefire framework that Ukraine accepts but Russia rejects (or vice versa) pending a US election or battlefield shift
- **Sudan-Jeddah (2023-present)**: Framework exists for SAF-RSF ceasefire; activation requires battlefield resolution or external pressure shift
- **Yemen (2022 UN-brokered truce)**: Framework existed for Houthi-government ceasefire; activation required Saudi-Iran normalization

## Canonical Case: May 2024 Israel-Hamas Framework

The May 2024 ceasefire framework (see [[domains/mena/threads/gaza-ceasefire-negotiations-2025/events/may-2024-ceasefire-framework]]) is the canonical case of the activation pattern:

- **May 5, 2024**: Hamas accepts the three-phase framework — framework exists
- **May-Jul 2024**: Israel rejects; Sinwar blocks Hamas flexibility; negotiations stall
- **Aug-Oct 2024**: Framework remains "on the table" but politically deadlocked
- **Oct 16, 2024**: Sinwar killed — primary blocking condition removed
- **Nov 2024**: Hezbollah ceasefire removes secondary blocking condition (northern front)
- **Nov 2024-Jan 2025**: Biden-Trump transition creates forcing function
- **Jan 15, 2025**: Framework activated — Israel accepts a variation of the same deal it had rejected in May

**Key insight**: The framework terms did NOT change substantially between May 2024 and January 2025. What changed was the political conditions surrounding the parties' decision-making. This is the essence of the activation pattern — the deal was always there; the question was whether the conditions for acceptance existed.

## Wikilinks

- [[domains/mena/threads/gaza-ceasefire-negotiations-2025/_thread]] — full Gaza ceasefire arc
- [[domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept]] — US transition as activation mechanism
- [[domains/mena/concepts/leadership-decapitation-negotiation-window]] — leadership removal as activation
- [[domains/mena/entities/yahya-sinwar]] — the blocking actor
- [[domains/mena/entities/benjamin-netanyahu]] — the resisting decision-maker
