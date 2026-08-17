---
type: concept
tags: [concept, methodology, calibration, meta]
title: "Paired-Questions Calibration"
slug: paired-questions-calibration
domain: global
status: active
created: 2026-05-22
canonical_cases:
  - "Venezuela 2024: Gonzalez wins (0.37) + Maduro wins (0.12) — complement pair revealing resolution-criteria ambiguity"
  - "Taiwan 2024: Lai wins (0.82) + Ko wins (0.02) — third-party FPTP ceiling analysis"
  - "Israel ceasefire: By Jan 17 (0.91) + By Sunday (0.90) — same event, different windows"
  - "Argentina seats: HNP holds (0.005) + HNP wins (0.006) — near-zero semantic distinction"
  - "Fed policy: Decrease 25bps (0.07) + Increase 25+bps (0.007) — asymmetric guidance"
  - "US VP: Another man (0.001) + Another woman (0.0005) — structural near-zero, different NO mechanisms"
  - "MV Hondius: Cup (0.20) + Tournament (0.24) — sibling-run evidence weighting divergence"
related_concepts:
  - sibling-run-calibration-divergence
  - market-vault-structural-divergence
  - short-horizon-momentum-check
  - structural-improbability-check
  - forecast-resolution-criteria-gotchas
---

# Paired-Questions Calibration

## Definition

A calibration framework for analyzing **pairs of forecasting questions** that share the same topic, event, or deadline but differ in framing, time window, candidate, or platform. The relationship between the two probabilities — their sum, ratio, gap, and overlap — reveals information that neither question provides alone.

Paired-question analysis is distinct from sibling-run analysis (same question, different agent sessions). Pairs are intentionally different questions that share a common structure, making their joint probability distribution informative.

## Pair Types

### Type 1: Complementary Pairs

Two questions about mutually exclusive but not collectively exhaustive outcomes (e.g., "Candidate X wins" + "Candidate Y wins" in a multi-candidate election). P(YES for X) + P(YES for Y) should NOT sum to 1 due to third candidates and resolution tail risks.

**Canonical Case: Venezuela 2024**

| Question | p_yes | Market |
|----------|-------|--------|
| Gonzalez wins | 0.37 | Polymarket |
| Maduro wins | 0.12 | Polymarket |
| Sum | 0.49 | — |

**Insight**: The 0.51 gap (1.0 - 0.49) reflects the market's priced-in resolution-criteria ambiguity (win-vote vs take-office) plus election-day tail risks (cancellation, fraud) that neither candidate overcomes. A naive forecaster might set P(Gonzalez) + P(Maduro) ≈ 1.0; the market's 0.49 reveals that 51% of the probability mass is in resolution-criteria ambiguity, not in either candidate winning.

### Type 2: Multi-Way Split Pairs

Three or more candidates in a single election where pairwise comparisons reveal structural dynamics (e.g., FPTP system, divided opposition).

**Canonical Case: Taiwan 2024**

| Question | p_yes | Note |
|----------|-------|------|
| Lai Ching-te wins | 0.82 | DPP frontrunner, structural advantage in 3-way FPTP |
| Ko Wen-je wins | 0.02 | Third-party FPTP ceiling |
| Hou Yu-ih (implicit) | ~0.16 | Residual: KMT candidate |

**Insight**: P(Lai) + P(Ko) + P(Hou) ≈ 1.0 (approximately exhaustive). The 0.82 reflects DPP's structural advantage in a three-way FPTP race where the opposition is divided. Ko's 0.02 demonstrates the third-party FPTP ceiling: even a popular third-party candidate with strong youth support cannot overcome the structural disadvantage of FPTP vote-splitting. The pair reveals that the question is really "Will Lai win against a divided opposition?" rather than "Who wins Taiwan 2024?"

### Type 3: Same-Event, Different-Window Pairs

The same binary outcome asked over different time horizons — isolates temporal decay from structural probability.

**Canonical Case: Israel Ceasefire January 2025**

| Question | p_yes | Window |
|----------|-------|--------|
| Ceasefire by Jan 17 | 0.91 | ~1 day |
| Ceasefire by Sunday (Jan 19) | 0.90 | ~3 days |

**Insight**: Both rely on the same procedural-certainty mechanism (commitment trap after Jan 15 dual-presidential framework announcement). The slightly LOWER probability on the LONGER window (0.90 vs 0.91) is a subtle over-caution — the mechanism was strongest on the immediate window, and extending the window introduced uncertainty about whether the framework would hold. The pair reveals: when a mechanism creates a near-certain outcome, the time horizon shouldn't reduce probability — it should increase it (more time = more opportunities for the mechanism to operate). The reversed ordering is a reasoning artifact.

### Type 4: Different-Metric Pairs

Same entity/party, different framing of the outcome metric — tests whether semantic distinctions matter.

**Canonical Case: Argentina HNP Seats**

| Question | p_yes | Framing |
|----------|-------|---------|
| HNP holds most seats | 0.005 | Maintains existing control |
| HNP wins most seats | 0.006 | Achieves control |
| Ratio | 0.83 | — |

**Insight**: Both correctly near-zero. The semantic distinction ("holds" = maintains vs "wins" = achieves) is irrelevant when the baseline seat count is near-zero. The 0.83 ratio (0.005/0.006) reflects the mild asymmetry: "holds" requires not losing what you barely have, while "wins" requires gaining from an even lower base. Both are structurally determined NO outcomes. The pair's value is in confirming that the semantic framing didn't matter — the structural improbability dominated both formulations.

### Type 5: Opposite-Direction Pairs

The same policy lever asked in both directions — reveals guidance asymmetry.

**Canonical Case: Fed Interest Rate July 2024**

| Question | p_yes | Direction |
|----------|-------|-----------|
| Rate decrease 25bps after July 2 | 0.07 | Cut (signaled direction) |
| Rate increase 25+ bps after July 2 | 0.007 | Hike (counterfactual) |
| Ratio | 10:1 | — |

**Insight**: The 10:1 ratio reflects guidance asymmetry: the Fed had signaled cuts but prioritized data dependency (7% near-term cut probability), while hikes were a counterfactual tail risk (<1%) against all forward guidance. The pair reveals the market's asymmetric confidence: the Fed was 10x more likely to cut than hike, but the base rate for any action in a 2-week window was low (<8% combined). The ratio encodes the consensus on policy direction; the sum encodes the consensus on near-term action probability.

### Type 6: Same-Question, Different-Demographic Pairs

Same structural context (VP nomination), different demographic category — reveals which mechanism dominates.

**Canonical Case: 2024 Democratic VP Nominee**

| Question | p_yes | Mechanism |
|----------|-------|-----------|
| Another man as VP nominee | 0.001 | Exclusion-list capture (Walz on 13-man list) |
| Another woman as VP nominee | 0.0005 | Gender-balancing dynamic (female nominee picks male VP) |
| Ratio | 2:1 | — |

**Insight**: Both structural near-zero, but via DIFFERENT mechanisms. The man question resolved via exclusion-list capture (Walz was the most likely male pick, already confirmed). The woman question resolved via gender-balancing dynamic (Harris, as a female nominee, was structurally more likely to pick a male running mate). Two different NO mechanisms producing the same result. The pair's value: near-zero outcomes can arise from different structural paths, and asking both directions reveals which mechanism dominates. Formalized in [[domains/usa/concepts/comprehensive-exclusion-list-forecast]].

### Type 7: Sibling-Platform Pairs

Same question, same deadline, different forecasting platform — reveals evidence-weighting differences.

**Canonical Case: MV Hondius Off-Ship Cases (Metaculus Cup vs Tournament)**

| Question | p_yes | Platform |
|----------|-------|----------|
| 5+ off-ship cases by Aug 1 (cup) | 0.20 | Metaculus Cup |
| 5+ off-ship cases by Aug 1 (tournament) | 0.24 | Metaculus Tournament |
| Gap | 0.04 | — |

**Insight**: The 0.04 gap reveals reasoning divergence: the cup run leaned more on the "zero confirmed cases after 17+ days" surveillance signal, while the tournament run gave more weight to the historical base rate (36% category for "will it reach N cases" questions). The gap itself (~20% relative difference) is a useful uncertainty estimate — the true probability lies somewhere in [0.20, 0.24]. This is the canonical sibling-run calibration divergence case, formalized in [[domains/global/concepts/sibling-run-calibration-divergence/_concept]].

### Type 8: Sequential-Phase Pairs

Two or more questions about different phases of the same multi-stage event (e.g., first round → runoff of an election, or preliminary vote → final ratification). The phase-1 outcome is a structural precondition for phase-2 pricing, creating a **cascading error risk** when phase-1 is mispriced by the market.

**Canonical Case: Colombia Presidential Election 2026 (First Round → Runoff)**

| Question | Pre-Phase Market Price | Actual Outcome | Post-Phase Corrected Price |
|----------|----------------------|---------------|---------------------------|
| Who wins first round? | Cepeda 82.5% | De la Espriella 43.7% | — |
| Who wins runoff? | Cepeda 50.5%, de la Espriella 44.0% | Pending (Jun 21) | De la Espriella ~65-70%, Cepeda ~30-35% |

**Insight**: This is the largest documented polling-error cascade in prediction market history. The first-round market error (Cepeda 82.5% → actual second place) created a mispricing cascade into the runoff market. Pre-first-round, the runoff was priced as a toss-up; post-first-round, the entire market repriced around the new structural reality. The pair documents two distinct failure modes:

1. **Phase-1 pricing error** (polling-error cascade): The market treated polling averages as ground truth despite structural conditions (populist candidate, multi-candidate field, social desirability bias) that should have raised red flags
2. **Phase-2 pricing inherits phase-1 assumptions**: The runoff market inherited flawed assumptions about relative candidate strength from the first-round market's mispricing

**Forecasting rule**: In multi-phase events, never treat phase-2 pricing as independent of phase-1. A large phase-1 market error creates a correction opportunity in the phase-2 market — if you can identify the phase-1 error before it's resolved. Formalized in [[domains/global/concepts/election-market-polling-error-cascade/_concept]].

## Calibration Signal: Pairs Insight

| Pair Type | What the gap/ratio reveals |
|-----------|---------------------------|
| Complementary (Type 1) | Resolution-criteria ambiguity mass |
| Multi-way split (Type 2) | Structural advantage of frontrunner |
| Same-event, different-window (Type 3) | Mechanism decay vs accumulation over time |
| Different-metric (Type 4) | Whether semantic framing changes outcome |
| Opposite-direction (Type 5) | Guidance asymmetry ratio |
| Same-question, different-demographic (Type 6) | Dominant mechanism identification |
| Sibling-platform (Type 7) | Evidence-weighting divergence |
| Sequential-phase (Type 8) | Phase-1 error propagation into phase-2 pricing |

## Decision Procedure

When encountering a question that appears to be part of a pair:

### Step 1: Detect the Pair

Check if any of the following are true:
- Same topic has another question asking about a different candidate/side (Type 1, 2)
- Same event has another question with a different time window (Type 3)
- Same entity appears with different metric framing (Type 4)
- Same policy lever asked in opposite direction (Type 5)
- Same structural context with different demographic/eligibility (Type 6)
- Same question on a different forecasting platform (Type 7)
- Same multi-stage event with sequential phases (Type 8)

### Step 2: Check Consistency

Before forecasting, check whether the two probabilities are internally consistent:
- **For Type 1 (complementary)**: P(A) + P(B) should be << 1.0 if resolution-criteria ambiguity exists. If P(A) + P(B) ≈ 1.0, the market is pricing out tail risks — consider whether tail risks matter.
- **For Type 3 (different window)**: P(longer window) should be >= P(shorter window). If reversed (like Israel case), check for reasoning artifact.
- **For Type 4 (different metric)**: P(holds) should ≈ P(wins) when both are near-zero. If diverging significantly, check whether the semantic distinction actually matters.
- **For Type 5 (opposite direction)**: The ratio encodes asymmetric confidence. A >10:1 ratio suggests strong guidance asymmetry. A near-1:1 ratio suggests the market is equally uncertain about both directions.
- **For Type 6 (different demographic)**: If one is >10x the other, the dominant mechanism is identifiable. If both are similarly near-zero, multiple mechanisms are at work.
- **For Type 7 (sibling platform)**: A gap <0.05 suggests convergence on the same mechanism. A gap >0.15 suggests high structural uncertainty — re-forecast.
- **For Type 8 (sequential-phase)**: P(phase 2) after phase 1 resolves should differ significantly from pre-phase-1 pricing if phase 1 was informative. If the phase-1 result contradicts pre-phase-1 market pricing (Colombia case), the phase-2 correction reveals the magnitude of the polling-error cascade. A small correction (<0.05) suggests phase 1 was accurately priced; a large correction (>0.15) signals a cascading error in the market's phase-1 pricing.

### Step 3: Extract the Pair Signal

After both forecasts resolve, the pair provides a calibration point:
- **Which type was it?** (Determines what signal to extract)
- **Was the internal relationship consistent?** (If P(A) > P(B) but resolution was B, the reasoning artifact identified)
- **What rate does the pair reveal?** (E.g., Venezuela: 0.49 total = resolution ambiguity is ~51% of probability mass)

## Cross-References
- [[runs/_index]] — Runs index with full paired-questions analysis (7 pairs documented in "Cross-Domain Complementary Pattern" section)
- [[domains/global/concepts/sibling-run-calibration-divergence/_concept]] — Sibling run analysis (Type 7: same question, different agent sessions)
- [[domains/global/concepts/market-vault-structural-divergence]] — Related calibration divergence framework
- [[domains/global/concepts/election-market-polling-error-cascade/_concept]] — Specialized subtype for Type 8 (sequential-phase) where phase-1 mispricing propagates into phase-2 markets
- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — Resolution criteria interpretation framework (powers Type 1 and Type 5 analysis)
- [[domains/global/concepts/structural-improbability-check/_concept]] — Structural check that powers Type 4 and Type 6 analysis
- [[domains/usa/concepts/comprehensive-exclusion-list-forecast]] — Specific framework for Type 6 (US VP nomination)
