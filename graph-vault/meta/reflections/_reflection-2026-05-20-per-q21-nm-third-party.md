---
type: reflection
tags: [reflection]
date: 2026-05-20
cycle: per-q21-nm
question: "Will a candidate from another party win New Mexico Presidential Election?"
prediction: "NO"
actual: "NO"
vault_contribution: "30% (partial — correct via general knowledge, not vault content)"
---

# Per-Question Reflection Q21: New Mexico — Third-Party Candidate

## Diagnosis

**Prediction was correct but vault contributed ~30% of the signal.**

The vault had:
- State-electoral-reliability concept (classified NM as Likely D, D+5.7 in 2024)
- New Mexico entity with voting history
- 2024 US election thread with electoral map section
- Third-party-ceiling-fptp concept in east-asia domain (validated by Q27)

**What helped**: The state-electoral-reliability concept correctly classified NM as Likely D and the New Mexico entity contained the Gary Johnson 9.3% ceiling data. These provided the raw factual basis for the forecast.

**What hindered**: Three critical gaps:

1. **Cross-domain concept isolation (Gap A)**: The third-party-ceiling-fptp concept lived in `domains/east-asia/` with no path from the US domain. A forecaster answering a US question would not naturally find a Taiwan-centric concept file. The concept was general (FPTP systems) but filed under a specific domain, violating the principle that cross-domain concepts should have explicit bridges.

2. **No US-specific third-party ceiling analysis (Gap B)**: The vault had no analysis of why US third-party/independent candidates cannot win states. Questions like "Will a candidate from another party win State X?" are a specific class that requires combining:
   - The state's partisan floor (state-electoral-reliability)
   - The general third-party ceiling (third-party-ceiling-fptp)
   - US-specific barriers (Electoral College, ballot access per state, debate threshold, campaign finance)
   - Key historical evidence (Perot 18.9% → 0 states in 1992; Johnson's 9.3% ceiling as former NM governor)

3. **Missing entity stubs for 2024 third-party candidates (Gap C)**: The vault had detailed entities for Trump, Biden, Harris, Walz, Vance, Haley, DeSantis — but nothing for RFK Jr., Jill Stein, or Chase Oliver. The question was *about* these third-party candidates; without stubs, the vault provided no signal about their viability.

## Changes Made

| File | Action | Purpose |
|------|--------|---------|
| `domains/usa/concepts/us-third-party-ceiling/_concept.md` | **Created** | US-specific third-party ceiling concept: no third-party candidate has won a US state since 1968; US-specific barriers (Electoral College, ballot access variation, debate commission threshold, campaign finance); New Mexico-specific history (Johnson 9.3% as high-water mark); cross-referenced to east-asia parent concept. |
| `domains/usa/concepts/us-third-party-ceiling.md` | **Created** | Redirect stub from `concepts/us-third-party-ceiling` to canonical `concepts/us-third-party-ceiling/_concept` (matching pattern used by divided-opposition-plurality-win). |
| `domains/usa/entities/robert-f-kennedy-jr.md` | **Created** | Entity stub for RFK Jr.: 2024 independent campaign, peak polling ~15%, withdrawal Aug 2024, structural analysis of why high-name-recognition independent still couldn't win states. |
| `domains/usa/entities/jill-stein.md` | **Created** | Entity stub for Jill Stein (Green): 2024 campaign, 1-2% national ceiling pattern, entrenched-minor-party category analysis. |
| `domains/usa/entities/chase-oliver.md` | **Created** | Entity stub for Chase Oliver (Libertarian): 2024 campaign, <1% national, ballot-access consistency ≠ breakthrough analysis. |
| `domains/usa/procedures/us-third-party-state-win-forecast.md` | **Created** | 8-step procedure combining state-electoral-reliability + third-party-ceiling for state-level US questions. Step 1: identify candidate type. Step 2: classify state floor. Step 3: check historical state ceiling. Step 4: organizational depth assessment. Step 5: withdrawal risk check. Step 6: probability calibration. Step 7: combined third-party caveat. Step 8: documentation template. |
| `domains/usa/threads/2024-us-presidential-election/_thread.md` | **Updated** | Added "Third-Party/Independent Landscape in 2024" section with candidate table, key finding (no third-party candidate >5% in any state), forecasting significance (even RFK's favorable conditions confirmed the ceiling), and New Mexico-specific analysis (Johnson 9.3% as ceiling proxy). Added Related Concepts section linking to both third-party-ceiling concepts. |
| `domains/east-asia/concepts/third-party-ceiling-fptp/_concept.md` | **Updated** | Added US cross-references: new row in relationship table for US-specific concept, validated-by entry for NM question, related procedure entry for US procedure, wikilinks to US concept and thread. |
| `domains/usa/concepts/state-electoral-reliability/_concept.md` | **Updated** | Added relationship to new US third-party ceiling concept: the state's partisan floor is also the third-party ceiling. |
| `domains/usa/_domain.md` | **Updated** | Added new concept, entities, and procedure to domain index. |

## Cumulative Insight

This question reveals a **cross-domain bridge gap** that the previous reflection (Q27, Taiwan) did not catch. The Q27 reflection created the third-party-ceiling-fptp concept as a reusable framework but filed it under `domains/east-asia/`. When a US question arose, the concept was not findable from the US domain.

**The fix**: Create domain-specific instances (concept + procedure + entities) that explicitly inherit from and cross-reference the cross-domain parent. The US-specific concept is not a duplicate — it adds US-specific structural barriers, historical evidence, and procedure steps that the general concept does not cover. A forecaster in the US domain finds the US-specific concept first, which in turn links to the cross-domain parent.

**Recurring pattern for future reflections**: When a prediction is correct but the vault contributed <100% signal:
1. Check if a cross-domain concept exists but is siloed under a different domain → create a domain-specific instance with bidirectional cross-references
2. Check if the question references entities that lack stubs → create them (third-party candidates, minor parties)
3. Check if there's a procedure gap for the specific question type → create a targeted procedure

This is the first US third-party question to appear in the 84-question set, so the vault's east-asia coverage naturally anticipated the concept but not the domain application. The pattern will repeat when other domains (Latin America, Europe) encounter third-party questions — each will need domain-specific instances of the cross-domain ceiling framework.
