---
type: concept
tags: [concept, meta, methodology, forecasting-competitions]
title: "Question Battery Saturation"
slug: question-battery-saturation
first_observed: 2026-05-20
domain: global
related_concepts:
  - midterm-referendum-dynamics
  - regional-third-way-squeeze
  - abstraction-gap
---

# Question Battery Saturation

## Definition

A **question battery** is a set of forecasting questions that all test knowledge of the same underlying event or domain. In prediction market datasets and forecasting competitions, batteries arise naturally: multiple resolutions of the same event (different specific outcomes, different parties, different candidates) create multiple questions. A domain is **saturated** when additional questions about the same event produce zero marginal learning — the vault already has all the analytical structure needed.

## Canonical Example: Argentina 2025 Legislative Election

The Argentina 2025 Chamber of Deputies election generated the following battery:

| Question # | Question Text | Correct | Vault Contribution |
|------------|--------------|---------|-------------------|
| C8 (FIT-U) | Will FIT-U hold the most seats in the Chamber of Deputies following the 2025 Argentina election? | YES (NO) | 0% — freebie, no vault content |
| Q6 (HNP-1) | Will HNP win the most seats in the Chamber of Deputies following the 2025 Argentina election? | YES (NO) | ~40% — partial, thread created |
| Q55 (HNP-2) | Will HNP hold the most seats in the Chamber of Deputies following the 2025 Argentina election? | YES (NO) | ~70% — strong partial, entities+concepts added |
| Q57 (LLA) | Will LLA win the most seats in the Chamber of Deputies following the 2025 Argentina election? | YES | ~75% — strong, cross-national concepts extracted |
| Q58 (HNP-3) | Will HNP win the most seats in the Chamber of Deputies following the 2025 Argentina election? | YES (NO) | ~90%+ — saturated domain, lookup from existing thread |

The battery reveals a **saturation cadence**: ~3-4 questions per event domain before marginal learning approaches zero. After saturation, the vault should shift effort from covering the domain to abstracting cross-domain patterns.

## Identifying a Question Battery

When a new question arrives, check:

1. **Same event, different angle**: Is this question about the same election, conflict, or legislative process as previous questions? Compare named entities (country, election year, institution).
2. **Same domain, consecutive questions**: Do prior questions in the recent sequence share the same domain tag (e.g., `latin-america`, `argentina`, `elections`)?
3. **Vault saturation test**: Does the vault already have:
   - A complete thread documenting the event's causal chain?
   - Entity stubs for ALL named actors in the question?
   - At least one concept capturing WHY the outcome occurred?
   - If all three are true → domain is saturated.

## Saturation Cadence (Estimated)

| Pass | Coverage Level | Effort Allocation |
|------|---------------|------------------|
| 1st question in battery | 0-20% coverage | Create thread, stub entities, do basic research |
| 2nd question in battery | 20-60% coverage | Fill entity gaps, extract domain-specific concepts |
| 3rd question in battery | 60-90% coverage | Fill residual named entities, extract cross-national patterns |
| 4th+ question in battery | 90%+ coverage | Shift effort to abstraction; what general pattern was missed? |

## What to Do in a Saturated Domain

When the saturation test returns true (ALL three criteria met):

1. **Do NOT re-cover the domain** — no new threads, no new entity stubs, no new domain-specific concepts. The existing structure supports the forecast.
2. **Shift effort to abstraction**: Ask "What cross-national or cross-domain pattern does this question type expose?" Create concepts and procedures in `domains/global/` rather than `domains/[region]/`.
3. **Check for question-specific gotchas**: Does the question's wording differ from previous questions in the battery? Any subtle change in resolution criteria (e.g., "hold" vs "win" vs "secure") could matter.
4. **Validate existing concepts**: Add this forecast to the "Validated By" table of any concept that supported the reasoning. This builds the concept's track record.
5. **Document saturation in the reflection**: Note that this was the Nth question in the battery and the domain is saturated. This helps future reflections recognize the pattern.

## Relationship to Other Concepts

- **[[domain-saturation-signal]]**: The quantitative threshold for saturation (completeness of coverage across thread+entity+concept layers).
- **[[abstraction-gap]]**: The complementary concept — saturated domains create the opportunity for abstraction that covers the gap.
- **[[midterm-referendum-dynamics]]**: A cross-national concept that was extracted only after the Argentina battery reached saturation (Q57). The extraction was triggered by the "abstraction gap" realization, which itself required the vault to recognize that Argentina-specific learning was complete.

## Wikilinks

- [[argentina-milei-realignment]]
- [[midterm-referendum-dynamics]]
- [[legislative-plurality-forecast]]
- [[hacemos-por-nuestro-pais]]
- [[la-libertad-avanza]]
