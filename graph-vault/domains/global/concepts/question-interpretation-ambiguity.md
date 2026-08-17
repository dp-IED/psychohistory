---
type: concept
tags: [concept]
title: "Question Interpretation Ambiguity"
slug: question-interpretation-ambiguity
first_observed: 2026-05-18
domain: forecasting-methodology
related_concepts:
  - forecast-resolution-criteria-gotchas
  - sports-forecasting-liquidity-signal
---
---
---
# Question Interpretation Ambiguity

## Definition

The class of forecasting errors that arise when a prediction market question can be interpreted in multiple valid ways — leading agents to produce correct answers for the wrong question, or wrong answers for the right one.

This concept generalizes and complements [[forecast-resolution-criteria-gotchas]]. Where that concept focuses on *resolution criteria* (the formal rules that determine payout), this one focuses on *question scope ambiguity* (what time period, entity, or context the question refers to).

## Primary Pattern: Temporal Ambiguity

The most common subclass — a question that could refer to multiple time windows.

### Case: Kilmarnock 0-0 Dundee (Polymarket)

**Question:** "Exact Score: Kilmarnock FC 0-0 Dundee FC?"

A 0-0 *had already occurred* between these teams on August 23, 2025. But the market's cutoff date (2026-05-09) strongly implied it referred to the upcoming Round 37 post-split fixture on May 12, 2026.

| Interpretation | Evidence | Correct? |
|---------------|----------|---------|
| Seasonal (any 0-0 in 2025-26) | A 0-0 DID occur on Aug 23, 2025 | p_yes = 1.0 |
| Match-specific (May 12, 2026) | Cutoff suggests upcoming match; match ended 3-1 | p_yes = 0.05 |

Two of three sub-agents correctly interpreted match-specific; one (europe-regional-specialist) hit the seasonal interpretation. The contrarian debater correctly identified this as the pivotal uncertainty.

**Key insight**: The question didn't specify *which* match. Polymarket "Exact Score" markets for specific fixtures typically name the date or round. The absence of a date qualifier created the ambiguity.

### Detection Heuristic

When a question involves a recurring event (sports season, fiscal quarter, calendar year), check whether:
1. A previous instance of the same event type already resolved the question
2. The market cutoff date resolves the ambiguity (cutoff after the event = historical lookup, cutoff before = future prediction)
3. The volume/age of the market suggests when it was created relative to the event cycle

## Secondary Pattern: Entity Ambiguity

The question references an entity that could be interpreted at different levels of granularity.

### Case: Schull/Cavataio PPA Finals

**Question:** "Will Alexa Schull / Ava Cavataio win the 2026 PPA: PPA Finals?"

Ambiguity: Does the question refer to the *existing paired team* (which doesn't exist — they've never played doubles together) or the *possibility* they form a partnership and qualify? If paired identity is a precondition, p_yes drops to near-zero. If it's an open possibility over a season, residual probability exists.

### Detection Template
- **"Will [Person A] / [Person B] win..."**: Check if they are a established partnership
- **"Will [Country] / [Organization] do X by [Date]"**: Check if the entity has the institutional capacity

## Resolution Criteria vs. Interpretation Ambiguity: Relationship

| Dimension | Resolution Criteria Gotchas | Question Interpretation Ambiguity |
|-----------|---------------------------|-----------------------------------|
| Error source | Misreading formal resolution text | Misreading question scope |
| Detection | Read market details carefully | Consider all temporal/entity interpretations |
| Fix | Structured 5-step audit (see concept) | Interpretive range check: ask "what are the 2-3 plausible readings?" |
| Frequency | ~60% of observed errors | ~20% of observed errors |

## Pre-Forecast Checklist for Ambiguity

Before forecasting, ask:

1. **Time window**: Does this question unambiguously refer to one specific time period? Could it refer to past OR future?
2. **Entity scope**: Is the named entity a specific individual/team, or could it refer to a class/type/role?
3. **Conditional framing**: Does the question assume a precondition that is itself uncertain (e.g., two players partnering)?
4. **Cutoff test**: Does the market's PIT cutoff date resolve the temporal ambiguity?
5. **Base rate match**: Does the question text match standard market patterns for this domain?

## Cross-References
- [[forecast-resolution-criteria-gotchas]] — sibling concept for formal resolution errors
- [[sports-forecasting-liquidity-signal]] — sports domain context for these errors
- [[entities/kilmarnock-fc]]
- [[entities/dundee-fc]]
