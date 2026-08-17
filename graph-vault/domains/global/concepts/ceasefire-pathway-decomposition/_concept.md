---
type: concept
tags: [concept]
title: "Ceasefire Pathway Decomposition"
slug: ceasefire-pathway-decomposition
first_observed: 2025-06-23
domain: forecasting-methodology
related_concepts:
  - escalation-bargaining-termination
  - short-window-ceasefire-probability
  - shadow-war-to-direct-escalation
  - diplomatic-pressure-tipping-point
  - political-deadline-ceasefire
status: active
---

# Ceasefire Pathway Decomposition

## Definition

A forecasting framework that classifies ceasefire questions by **how the ceasefire would be achieved** — not by the identity of the belligerents or the nature of the conflict. The core insight: "ceasefire" as a Polymarket resolution term masks fundamentally different causal mechanisms whose probabilities and timelines vary by orders of magnitude. **Before estimating probability, classify the likely pathway.**

The framework distinguishes two primary pathways:

| Pathway | Mechanism | Typical Timeline | Example |
|---------|-----------|-----------------|---------|
| **Type A: Diplomatic/Negotiated** | Accumulated international pressure, mediation, political deadlines | Months to years | Israel-Hamas Jan 2025, Gaza Oct 2025 |
| **Type B: War-Termination via Superpower Entry** | State-on-state escalation ladder → war → superpower entry → rapid termination | Hours to days (after war starts) | Iran-Israel June 2025 |

A single conflict can produce ceasefires via different pathways at different times. The Israel-Hamas Jan 2025 ceasefire was Type A (diplomatic); the Iran-Israel June 2025 ceasefire was Type B (war-termination).

## The Core Forecasting Rule

**For any ceasefire question on a state-on-state conflict where a superpower patron relationship exists, decompose the probability as:**

```
P(ceasefire by date D) ≈ P(war starts by D - T) × P(superpower terminates within T | war starts)
```

Where `T` = the superpower's known termination speed (typically 48-72 hours for escalation-bargaining, or effectively infinite — years — for conflicts where the superpower cannot terminate, e.g., Russia-Ukraine).

**This decomposition is the single most important step for state-on-state ceasefire questions.** Applying a diplomatic-ceasefire base rate to a war-termination scenario produces forecast errors of 50+ percentage points.

## Why This Matters: The Gold_01 Error

The gold_01 question ("Israel x Iran ceasefire before July?") had PIT evidence including a clear escalation ladder, a superpower patron with escalation dominance, and a known termination mechanism. A Type-A decomposition (diplomatic) would yield P < 0.05: no negotiations in progress, no mediator framework, no war aims compatibility. A Type-B decomposition yields P > 0.70:

- **P(war in June window | escalation ladder at Stage 6-7)**: High (~0.7-0.8). Trump's 2-month nuclear deadline to Iran was expiring; IAEA was moving toward a non-compliance finding; the escalation ladder compression principle predicted accelerating transitions.
- **P(US terminates within 72h | war starts)**: High (~0.9-0.95). The escalation-bargaining concept's 48-hour rule applies: superpower combatant entry produces ceasefire within 72h conditional on escalation dominance, client dependency, no nuclear deterrent.
- **Combined**: P(ceasefire) ≈ 0.7 × 0.9 = **0.63**, before any upward adjustments for the June-30 window being 7 days after the most likely war-start date (June 13-15).

A forecaster who applied Type-A reasoning would see "no ceasefire negotiations" and predict NO. A forecaster who decomposed via Type-B would see "escalation ladder approaching war" and predict YES.

## Pathway Diagnostic

### Pathway A: Diplomatic/Negotiated Ceasefire

**Indicators:**
- Active mediation framework exists (Qatar/Egypt/US shuttling proposals)
- International pressure accumulating (UN resolutions, sanctions, diplomatic isolation)
- Political deadline approaching (inauguration, election, summit)
- Conflict duration > 6 months with war exhaustion visible
- War aims NOT existential (degrade, not destroy)
- Hostage/prisoner exchange mechanism exists as negotiation lever
- Asymmetric (state vs non-state) or symmetric with no superpower termination mechanism

**Application example:** Israel-Hamas ceasefire in any window. Gaza conflicts rarely have a war-termination pathway because Israel has no state-on-state adversary with escalation capacity — Hamas is a non-state actor. The only pathway is diplomatic pressure accumulation.

**Base rates for diplomatic pathway:**
- < 3 months since conflict start/restart (or ceasefire collapse): P < 0.05
- 3-6 months: P ~0.05-0.15 (but apply trust erosion multiplier if post-collapse: 0.3-0.5x)
- 6-12 months: P ~0.15-0.30
- 12+ months with active pressure: P ~0.25-0.50
- With political deadline < 30 days: multiply by 2-3x

**Post-collapse adjustment**: When the conflict entered its current phase AFTER a ceasefire collapsed (see [[domains/mena/concepts/ceasefire-trust-erosion-after-collapse/_concept]]), apply the trust erosion multiplier to the base rates above. For the first 3-6 months after a collapse, the trust deficit reduces feasible ceasefire probability by 50-90% below the standard base rate for time-since-restart.

### Pathway B: War-Termination Ceasefire (Superpower Entry)

**Indicators:**
1. **State-on-state conflict** — both belligerents are sovereign states with military capacity to strike each other's territory directly
2. **Escalation ladder present** — the states have crossed multiple escalation thresholds (proxy → covert → direct strike → ballistic exchange)
3. **Superpower patron relationship** — one belligerent has a patron with escalation dominance who can credibly threaten to win any expanded war
4. **No nuclear deterrent against the superpower** — the adversary cannot deter the superpower from entering
5. **Nuclear latency/WMD threshold approaching** — the adversary is near a "point of no return" that creates a preemption window
6. **Patron with mediation leverage over both sides** — can offer terms to both client and adversary

**Application example:** Iran-Israel June 2025. All 6 indicators present. The escalation ladder went through 8 stages; the US had escalation dominance and could mediate; Iran had no nuclear deterrent against the US; Iran's nuclear program was near-weapons grade.

**Base rates for war-termination pathway:**
- Superpower enters as combatant → ceasefire within 72h: P > 0.90
- Superpower enters as combatant, no prior war in this dyad → ceasefire within 1 week: P > 0.80
- War starts with superpower patron present but not entering → war may continue months to years (transition to Type A)

### Pathway C: None (No Ceasefire Likely)

**Indicators:**
- Nuclear-armed adversaries (MAD prevents superpower entry)
- No credible mediator with leverage on both sides
- Existential war aims
- No escalation ladder (hot war started without threshold-crossing)
- Superpower patron lacks escalation dominance

**Example:** Russia-Ukraine war (2022-ongoing). Nuclear deterrent prevents NATO entry; no superpower has leverage on both sides; existential stakes on both sides. See [[escalation-bargaining-termination]] for the full failure case analysis.

## The Escalation Ladder → Ceasefire Coupling

The strongest forecasting insight from this framework: **when a state-on-state conflict has an identifiable escalation ladder approaching the war threshold, a ceasefire question about that conflict should be treated as conditionally coupled to the war probability.**

This couples two frameworks that are often treated independently:
- [[shadow-war-to-direct-escalation]] → P(war in window)
- [[escalation-bargaining-termination]] → P(ceasefire | war)

The coupling means: the same structural factors that increase the probability of war also increase the probability of ceasefire (because they make the war-termination mechanism more likely). This is counterintuitive — one might think escalating tensions make ceasefire LESS likely — but it's the correct inference for conflicts with superpower termination capacity.

**Forecasting rule:** If indicators for Pathway B are present (ladder + patron + no-nuclear-deterrent), treat the ceasefire question as a **derivative** of the war question. Do not assess it independently.

## When to Use Each Pathway

When you receive a "ceasefire by date X?" question:

1. **Is it state-on-state?** If no → Pathway A (diplomatic only). If yes → proceed to step 2.
2. **Are the escalation conditions present?** Are there at least 4 escalation thresholds crossed? Is superpower entry plausible? If yes → Pathway B dominates. If no → Pathway A.
3. **Is the adversary nuclear-armed against the superpower?** If yes → Pathway C (no superpower entry possible) or A at best.
4. **Calculate pathway-specific probability.**

## Canonical Validation

| Question | Pathway Classified | P(ceasefire) | Actual | Framework Validates? |
|----------|-------------------|-------------|--------|---------------------|
| Iran-Israel ceasefire before July 2025 | B (war-termination) | ~0.60-0.75 | YES (ceasefire June 24) | ✓ — Type-B decomposition gave correct high probability |
| Israel-Hamas ceasefire by July 15, 2025 | A (diplomatic) | ~0.10-0.20 | NO (no ceasefire) | ✓ — Type-A base rate correctly low for short window |
| Israel-Hamas ceasefire Jan 2025 | A (diplomatic, with political deadline) | ~0.60-0.80 | YES (Jan 15 announcement) | ✓ — Type-A with deadline modifier gave correct high probability |
| Russia-Ukraine ceasefire 2024 | C (none — nuclear adversary) | < 0.05 | NO | ✓ — Type-C correctly predicted no ceasefire |
| Gaza ceasefire Oct 2025 | A (diplomatic, accumulated pressure) | ~0.40-0.55 | YES (Oct 8 announcement) | ✓ — Type-A with multi-month pressure accumulation |

## Related Concepts

- [[escalation-bargaining-termination]] — The specific termination mechanism for Pathway B, including the 48-hour rule and the 'damaged mediation' counter-inference.
- [[short-window-ceasefire-probability]] — Base rates for Pathway A, with explicit exception noted for Pathway B.
- [[shadow-war-to-direct-escalation]] — The escalation ladder framework that provides P(war | escalation stage) for Pathway B decomposition.
- [[diplomatic-pressure-tipping-point]] — The international pressure accumulation mechanism for Pathway A.
- [[political-deadline-ceasefire]] — How known deadlines compress Pathway A timelines.
- [[inter-state-ceasefire-feasibility]] — The existing procedure for assessing state-on-state ceasefire feasibility, which this concept supplements with the pathway decomposition step.

## Wikilinks

- [[events/iran-israel-twelve-day-war]]
- [[events/gaza-january-ceasefire-2025]]
- [[events/gaza-october-ceasefire-2025]]
- [[threads/iran-israel-escalation]]
- [[threads/gaza-ceasefire-negotiations-2025]]
- [[domains/mena/entities/israel]], [[domains/global/entities/iran]]
- [[domains/global/entities/iran-israel-conflict]]
- [[domains/global/entities/iaea]]
- [[domains/usa/entities/donald-trump]]
- [[domains/usa/entities/steve-witkoff]]
- [[entities/israeli-security-cabinet]]
- [[concepts/escalation-bargaining-termination]]
- [[concepts/short-window-ceasefire-probability]]
- [[concepts/shadow-war-to-direct-escalation]]
