---
type: concept
tags: [concept, forecasting-methodology]
title: "Comprehensive Exclusion List Forecast"
slug: comprehensive-exclusion-list-forecast
first_observed: 2024-01-01
domain: forecasting-methodology
description: "How to forecast Polymarket-style questions that list N specific entities and ask whether an unlisted entity will achieve a specified outcome — the 'another X' question archetype"
---
# Comprehensive Exclusion List Forecast

## Concept

A recurring Polymarket question archetype asks: "Will [another/an unlisted] [person/party/country] achieve [outcome]?" — where the market lists N specific entities (the "exclusion list") and the question resolves YES only if someone/something NOT on the list achieves the outcome. These questions rely on the market creator's implicit claim that the plausible universe is finite and captured by the list.

**Core insight**: The exclusion list itself encodes the most powerful forecasting signal. A long exclusion list (10+ entries) in a transparent, well-reported selection process nearly always resolves NO, because:
1. The plausible candidate pool for any major US political selection (VP, Cabinet, SCOTUS) is typically 10-20 people
2. A market that lists 12-15 names has already captured most or all of that pool
3. The question becomes: "Will a dark horse emerge?" — which is structurally unlikely in high-transparency processes
4. US political selections are heavily media-covered; there are no "unknowns" who could be VP without appearing in press speculation first

## Exclusion List Composition Types

Not all exclusion list entries are equal. The list typically contains three categories of entries:

### Category A: Genuine Contenders (50-100% plausible)
People who were actually under consideration or had a realistic path to the role. These are the **effective exclusion list** — they define the "canonical candidate pool."

Examples: Mark Kelly, Josh Shapiro, Tim Walz, Roy Cooper, Andy Beshear, Pete Buttigieg for the 2024 Democratic VP

### Category B: Plausible Mentions (10-50% plausible)
People who were mentioned in media speculation but never formally vetted. These add length to the list but are less informative because they were never truly in contention.

Examples: J.B. Pritzker, Gavin Newsom, Wes Moore for the 2024 Democratic VP

### Category C: Padding (0-5% plausible)
People included to expand the list who are constitutionally or practically ineligible, or who have zero realistic path to the role. These artificially inflate the list length.

Examples: Barack Obama (ineligible under 12th Amendment), Mark Cuban (no political experience) for the 2024 Democratic VP

**Forecasting implication**: If Category C entries make up 20%+ of the exclusion list, the effective exclusion list is shorter than the nominal list. The question is harder to answer with "NO" because some of the nominal exclusion list entries were never real options.

## Calibration Framework

### Step 1: Classify the Selection Process
- **High-transparency processes** (US VP selection, Cabinet appointments, SCOTUS nominations): The media covers the vetting process in near-real-time. At most 20 people could plausibly be selected. An exclusion list of 12+ names covering all categories captures >90% of the plausible universe. → Strong NO baseline.
- **Low-transparency processes** (authoritarian succession, private corporate selection, closed-list nominations): The plausible universe is unknown. An exclusion list may be aspirational rather than comprehensive. → Weak or no baseline.
- **Competitive processes** (primary or general elections with many candidates): The plausible universe is large. Exclusion lists of 5-10 names capture only the frontrunners, not the field. → Little signal from list length alone.

### Step 2: Count and Categorize the Exclusion List
- Count the total entries
- Classify each entry as Category A, B, or C
- Effective exclusion list = Category A + (0.3 × Category B)
- If effective exclusion list > 10 AND process is high-transparency: P(YES) < 5% — the universe is saturated
- If effective exclusion list < 5: P(YES) could be 15-40% — the market has not captured the full candidate pool

### Step 3: Check for Process Changes
- Did a major event (candidate withdrawal, new nominee, scandal) occur after the exclusion list was created?
- If yes: the exclusion list may be outdated. The 2024 Democratic VP "another man" question changed fundamentally on July 21 when Biden withdrew and Harris became the nominee.
- An outdated exclusion list that still captures the new-phase universe supports NO; an outdated list that misses new-phase candidates supports YES.

### Step 4: Assess the Plausible Unlisted Candidate Pool
- Are there any plausible candidates NOT on the list? List them explicitly.
- If none → NO (structural)
- If 1-2 plausible names → check why they weren't included: oversight, lesser-known, or surfaced late
- If 3+ → YES becomes plausible

### Step 5: Calibrate Probability

| Condition | P(YES) | Reasoning |
|-----------|--------|-----------|
| High-transparency, effective exclusion > 10, no plausible unlisted candidates | <3% | Universe saturated |
| High-transparency, effective exclusion > 10, 1-2 plausible borderline candidates | 5-10% | A surprise pick at the margin |
| High-transparency, effective exclusion 5-10 | 10-20% | Process may still surprise |
| High-transparency, effective exclusion < 5 | 20-40% | Market didn't capture full pool |
| Low-transparency, any list length | 20-50% | Unknown unknowns dominate |
| Process changed after list creation, new list captures new universe | <5% | Structural NO |
| Process changed after list creation, new list is wider | 10-30% | New uncertainty |

## Canonical Examples

### 1. "Will another man be the 2024 Democratic VP nominee?" — NO (correct)
- Exclusion list: 13+ men (Kelly, Shapiro, Cooper, Beshear, Walz, Buttigieg, Pritzker, Cuban, Moore, Obama, Newsom, Ryan, Biden)
- Category A: Kelly, Shapiro, Walz, Cooper, Beshear, Buttigieg (~6 genuine contenders)
- Category B: Pritzker, Newsom, Moore (~3 plausible mentions)
- Category C: Obama (ineligible), Cuban (no experience), Ryan (lost office) (~3 padding)
- Effective exclusion list: 6 + (0.3 × 3) = ~7
- Process: High-transparency. Walz was the pick and was on the exclusion list.
- **Key insight**: The actual pick (Walz) WAS on the list, so the question wasn't about predicting the pick but about predicting a dark horse. NO was structurally correct before the Harris campaign even announced.

### 2. "Will another woman be the 2024 Democratic VP nominee?" — NO (correct)
|- Exclusion list: 9 women ([[domains/usa/entities/gretchen-whitmer|Whitmer]], [[domains/usa/entities/michelle-obama|Obama]], [[domains/usa/entities/hillary-clinton|Clinton]], [[domains/usa/entities/kamala-harris|Harris]], [[domains/usa/entities/elizabeth-warren|Warren]], [[domains/usa/entities/tammy-duckworth|Duckworth]], [[domains/usa/entities/alexandria-ocasio-cortez|Ocasio-Cortez]], [[domains/usa/entities/tammy-baldwin|Baldwin]], [[domains/usa/entities/amy-klobuchar|Klobuchar]])
- Category A: Whitmer, Warren, Klobuchar (~3 genuine contenders)
- Category B: Baldwin, Duckworth, Ocasio-Cortez (~3 plausible mentions)
- Category C: Clinton (not in office), Harris (was the nominee herself), Obama (ineligible) (~3 padding)
- Effective exclusion list: 3 + (0.3 × 3) = ~4
- Process: High-transparency, plus structural gender-balancing dynamic (female nominee → P(woman VP) <5%)
- NO was driven by two structural dynamics: the [[domains/usa/concepts/incumbent-vp-renomination]] baseline (Harris was the sitting VP, making any "another woman" scenario require the near-impossible dropping of an incumbent VP) and the [[domains/usa/concepts/gender-balancing-ticket-composition/_concept|gender-balancing]] dynamic (if Harris became presidential nominee, she would pick a male VP). The exclusion-list framework was tertiary reinforcement — all plausible women were captured, but the structural dynamics made their selection impossible regardless of the list.

### 3. "Will another country be Ukraine's largest military aid donor?" — potential future question type
- This is the international-relations extension: exclusion lists naming specific countries
- Same framework applies: if 10+ countries on the list, the remaining universe is small

## Relationship to Other Concepts

- [[domains/usa/concepts/veepstakes-electoral-signal/_concept]] — narrower concept about VP selection dynamics specifically
- [[domains/usa/concepts/gender-balancing-ticket-composition/_concept]] — structural dynamic that can override exclusion-list analysis
- [[domains/global/concepts/forecast-range-plausibility-filter]] — similar meta-framework for filtering implausible outcomes

## Sources
- Polymarket question structure analysis (multiple VP, Cabinet, SCOTUS questions)
- 2024 US vice presidential selection process
- Media veepstakes coverage (CNN, NBC, NYT, July-August 2024)
