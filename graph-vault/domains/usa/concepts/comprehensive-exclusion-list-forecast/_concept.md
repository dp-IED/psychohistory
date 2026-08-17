---
type: concept
tags: [concept]
title: "Comprehensive Exclusion List in Polymarket Questions"
slug: comprehensive-exclusion-list-forecast
first_observed: 2024-07-22
related_concepts: [veepstakes-electoral-signal, gender-balancing-ticket-composition]
---

# Comprehensive Exclusion List in Polymarket Questions

## Concept

Certain prediction market questions use a **comprehensive exclusion list** pattern: the question asks whether "another" person/candidate/outcome will occur, and the resolution text lists a large set of named entities that are explicitly excluded from "another."

Examples:
- "Will **another** man be the 2024 Democratic VP nominee?" — excluding 13+ named men
- "Will **another** woman be the 2024 Democratic VP nominee?" — excluding 9+ named women
- "Will any candidate OTHER THAN [Biden/Trump] win [state]?" — excluding the major-party nominees

### Structural Features

1. **The list signals what the question-writer considers plausible.** The exclusion list is not random — it represents the question creator's estimate of the credible candidate pool. A list of 13 men for "another man" means the creator believes these 13 cover essentially all plausible male contenders.

2. **The list length correlates inversely with "another" probability.** The more names on the exclusion list, the lower the probability that someone outside the list will be selected. This is because:
   - Longer lists exhaust the viable candidate pipeline
   - Each additional name captures another plausible scenario
   - The marginal candidate (outside the list) is progressively less viable

3. **The list implicitly encodes tier analysis.** The excluded names are almost always Tier 1 (most plausible) candidates. "Another" = Tier 2 or lower. The probability that a Tier 2+ candidate is chosen when Tier 1 candidates are available and not independently disqualified is structurally low.

### Diagnostic Test: Is the List Exhaustive?

To determine whether the exclusion list is genuinely exhaustive of plausible outcomes, run this diagnostic:

| Test | Question | Answer Signals |
|------|----------|----------------|
| **Coverage test** | Does the list include every candidate who had a plausible path to selection? | YES → list is likely exhaustive; NO → gap exists |
| **Surprise test** | Would the named "another" candidate be a genuine surprise to informed observers? | YES → low probability; NO → the list missed someone plausible |
| **Pipeline test** | What is the next-tier pool (candidates outside the list)? Are they materially less qualified/prominent than the listed candidates? | Yes, materially worse → low "another" probability |
| **Context test** | Does the selection context (electoral system, party rules, nominee's constraints) restrict the pipeline further? | YES → even if non-exhaustive, the pipeline is narrow |

### How the "Another Man" Question Passed These Tests

| Test | Assessment | Implication |
|------|-----------|-------------|
| Coverage | The 13 excluded men included every plausible Democratic VP contender: sitting senators (Kelly, Ryan), governors (Shapiro, Walz, Beshear, Moore, Newsom, Pritzker), business figures (Cuban), former presidents (Obama), and the sitting president (Biden). No Tier 1 candidate was omitted. | List is exhaustive |
| Surprise | Any male VP pick outside the list would be an extraordinary surprise — someone who had not been discussed as a contender, vetted by the campaign, or speculated about by media. | P("another" = surprise) < 5% |
| Pipeline | The next-tier pool outside the list consists of lower-prominence figures: House members without national profiles, second-tier governors (Hobbs, Polis), Cabinet secretaries (Mayorkas, Blinken), none of whom had the national profile, swing-state value, or private-sector credibility of the listed names. | Sharp drop-off in quality |
| Context | Biden's pre-July 21 woman VP pledge PLUS Harris's post-July 21 nomination created dual constraints: (a) Biden → woman, so no man; (b) Harris → would pick a man, but it would be from the listed pool (Walz was on the list) | Dual paths to NO |

### Calibration Rules

| Condition | P("another" = YES) | Rationale |
|-----------|---------------------|-----------|
| List covers ALL plausible candidates + selection context constrains pipeline | <5% | Any outcome outside list requires both a list-candidate being disqualified AND a non-list candidate being picked, which is rare in a constrained selection process |
| List covers most plausible candidates but 1-2 are missing | 10-20% | The missing candidate(s) could plausibly be selected, but the list's breadth still constrains the space |
| List covers only a few candidates | 30-50%+ | The question is closer to open-ended; the list may be incomplete or strategically narrow |
| List exists but selection context changed after question creation | Variable | Events (candidate withdrawal, nominee change) may create new pathways. Re-apply the full diagnostic in the new context. |

### The "Another Man" Question: Multi-Pathway Resolution

This question is the canonical case of a question whose resolution depends on TWO independent pathways, each of which individually guarantees a NO outcome:

**Pathway 1: Biden remains nominee (pre-July 21)**
- Biden had pledged to pick a woman as VP in 2020, a pledge he reaffirmed repeatedly
- P(any man as VP | Biden remains) ≈ 0% due to the binding pledge
- P(another man | Biden remains) = 0% (zero men, so zero "another" men)

**Pathway 2: Harris becomes nominee (post-July 21)**
- Harris (woman) would pick a man as VP per the gender balancing dynamic
- But ALL plausible male candidates were on the exclusion list
- Tim Walz, her actual pick, was on the list
- Therefore P(another man | Harris) = 0% (the actual pick was excluded)

**Combined**: P(YES) = P(Pathway 1) × 0% + P(Pathway 2) × 0% = 0%
Any weighting of the two pathways yields zero. This is a rare case where the probability is genuinely near-zero, not just structurally low.

### Why This Pattern Matters for Forecasting

1. **Exclusion-list questions reward thorough list analysis, not domain expertise.** The critical question is not "who will be VP?" but "is the exclusion list exhaustive of plausible candidates?" This requires systematic comparison of the list against the known candidate pool.

2. **The absence of a name does NOT make them plausible.** Just because a candidate is NOT on the exclusion list does not make them a viable contender. The list captures Tier 1; exclusion means the candidate is in Tier 2+ and would require extraordinary circumstances to be selected.

3. **Context changes can invalidate the exclusion list.** If the selection rules change (the nominee changes, a pledge is broken, a candidate drops out), the exclusion list may no longer be comprehensive. Always check for context changes before relying on the exclusion list.

4. **The longer the list, the stronger the NO signal.** A question with 13 excluded names is qualitatively different from a question with 3 excluded names. The list length itself conveys information about the question writer's assessment of the candidate space.

### Validated By

| Forecast Question | Prediction | Actual | Date |
|-------------------|-----------|--------|------|
| "Will another man be the 2024 Democratic VP nominee?" | NO | NO | 2026-05-20 |
| "Will another woman be the 2024 Democratic VP nominee?" | NO | NO | 2026-05-18 |

### Related Concepts
- [[concepts/veepstakes-electoral-signal]] — broader VP selection framework
- [[concepts/gender-balancing-ticket-composition]] — gender dynamics in VP selection
- [[concepts/campaign-pledge-binding-constraint]] — campaign promises as selection constraints

### Sources
- Polymarket resolution texts for 2024 Democratic VP nominee questions
- Democratic VP vetting process media coverage (July-August 2024)
