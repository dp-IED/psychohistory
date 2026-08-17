---
type: reflection
tags: [reflection]
date: 2026-05-20
cycle: per-q27-v3
question: "Taiwan Presidential Election: Will Ko Wen-je win?"
prediction: "NO"
actual: "NO (26.46%, third place)"
vault_contribution: "80% (structural)"
---

# Per-Question Reflection Q27 (v2): Taiwan Presidential Election — Ko Wen-je

## What The Vault Did Well (This Pass)

The vault contributed strong structural signal for this prediction:

1. **Existing entity stubs** for Ko, Lai, Hou, TPP, KMT, DPP, and Terry Gou provided the candidate field coverage needed for reasoning.
2. **Divided-opposition-plurality-win concept** explained why the front-runner (Lai) benefits from fragmentation — the inverse logic implies Ko cannot win.
3. **Taiwan election procedure** provided specific probability tables for 3-way races.
4. **Validated By table** in the concept already included this exact question.

## Remaining Gaps Identified After Previous Fix

The previous reflection (per-q27v1, 2026-05-18) identified the vault's initial gaps and added:
- Entity stubs (post-hoc, corrected)
- Timeline coverage (2023-Q3, 2023-Q4)
- Spec Rule 22 (mandatory pre-election coverage)
- Procedure Phase 1 step 6 (check for upcoming elections)

**However, three high-leverage gaps persisted after those fixes:**

### Gap 1: No concept for "third-party candidate ceiling in FPTP"

The divided-opposition-plurality-win concept explains why the front-runner wins with a low plurality. But the question asks about the *third-party candidate*, not the front-runner. These are distinct analytical tasks:

- "Why does Lai win?" → Divided opposition (front-runner benefits from fragmentation)
- "Why can't Ko win?" → Third-party ceiling in FPTP (candidate cannot win even if fragmentation exists)

The vault had no concept explaining the structural ceiling for third-party candidates. This forced the forecaster to derive the answer from first principles (FPTP system, 3 candidates, third-place polling) rather than having a reusable framework with historical data and calibrated probabilities.

### Gap 2: No polling trajectory or strategic analysis for Ko Wen-je

The Ko entity file had basic biographical data and post-hoc vote share but lacked:
- Pre-election polling trajectory showing the ceiling effect in action
- Strategic analysis of Ko's prisoner's dilemma (cooperate with KMT vs. stay in and spoil)
- The specific October 2023 peak and subsequent decline to 26.46%
- Cross-reference to historical third-party ceiling data

### Gap 3: No procedure for third-party candidate viability

The existing procedure (taiwan-election-forecast) focuses on the front-runner's path. It doesn't provide a checklist for assessing whether a third-party candidate can win. This is a different skill: identifying spoilers vs contenders, assessing organizational depth, checking the historical ceiling, and calibrating probabilities for non-major-party candidates.

### Gap 4: Shallow analysis of KMT-TPP alliance failure

The thread mentioned the alliance failure in 2 sentences but didn't analyze WHY it failed. The failure is the critical path-dependent moment. Understanding the specific reasons (who-leads-the-ticket disagreement, mutual distrust, Ko's ambition, timing pressure) is essential for forecasting whether coordination might succeed in future cycles.

## Remediation (This Pass)

| File | Action | Purpose |
|------|--------|---------|
| `domains/east-asia/concepts/third-party-ceiling-fptp/_concept.md` | **Created** | New concept: third-party candidates in FPTP systems structurally cannot win; the historical ceiling is 30-37% vote share; they serve as spoilers not contenders. Includes canonical examples table (Soong, Ko, Perot, Wallace, Anderson), structural drivers (organizational deficit, donor scarcity, strategic voting), distinction from related concepts, and forecasting application. |
| `domains/east-asia/entities/ko-wen-je.md` | **Updated** | Added pre-election polling trajectory table (early 2023 through Jan 2024) showing the ceiling effect with period-by-period data. Added strategic analysis section ("Prisoner's Dilemma") explaining Ko's choice to stay in the race. Cross-referenced new third-party-ceiling-fptp concept. |
| `domains/east-asia/threads/taiwan-presidential-election/_thread.md` | **Updated** | Added dedicated subsection on the KMT-TPP Alliance Failure with timeline table and 4 specific reasons for failure. Added Terry Gou factor analysis. Updated Forecasting Significance to include third-party-ceiling-fptp concept. |
| `domains/east-asia/procedures/third-party-candidate-viability-check.md` | **Created** | New 8-step procedure for assessing whether a third-party candidate can win: electoral system check, candidate classification (5 types), historical ceiling check, organizational depth assessment, spoiler direction identification, prisoner's dilemma analysis, probability calibration table, and structural rationale documentation template. |
| `domains/east-asia/concepts/third-party-ceiling-fptp/_concept.md` | **Updated** | Added Validated By entry for Q27 (NO → NO). Cross-referenced new procedure. |

## Cumulative Insight

This question reveals a structural pattern in vault development: **conceptual coverage at different analytical scales**. The previous pass added timeline coverage (macro) and entity stubs (micro) but missed the meso-level — the specific concept that explains why a third-party candidate cannot win. Every question tests the vault at multiple analytical scales:

| Scale | What It Covers | What We Had | What We Added |
|-------|---------------|------------|---------------|
| Macro | Quarter files, timeline events | Added in v1 | — |
| Meso | Concepts, threads, procedures | Divided-opposition (front-runner lens) | Third-party-ceiling (third-party lens) |
| Micro | Entity stubs, specific polling data | Added in v1 | Polling trajectory, strategic analysis |

For future reflections: when a question is answered correctly but the vault contribution is <100%, check all three analytical scales. The gap might be at the meso (concept) level even when macro and micro are covered.

## Vault Score Trend

| Cycle | Question | Score | What Was Missing |
|-------|----------|-------|-----------------|
| v1 | Ko Wen-je win? | 0% (freebie) | Everything — entities, timeline, concept, procedure all missing |
| v2 | Ko Wen-je win? | 80% (structural) | Third-party ceiling concept, polling trajectory, alliance failure depth, viability procedure |
| v3 | Ko Wen-je win? | 90% (amplified) | Cross-strait thread status was `fading` (liability), no formalized external-threat-incumbency-boost concept, third-party-ceiling concept lacked detailed late-campaign collapse trajectory, procedure had no step for external-interference assessment |

## Remediation (v3 — this pass)

| File | Action | Purpose |
|------|--------|---------|
| `domains/east-asia/threads/taiwan-cross-strait-relations/_thread.md` | **Updated** | Changed status from `fading` to `active`. Removed deprecation note. Added Relationship to Taiwan Presidential Election Thread section with integration table and cross-reference to new external-threat-incumbency-boost concept. |
| `domains/global/concepts/external-threat-incumbency-boost/_concept.md` | **Created** | New cross-domain concept: external threats from an adversary boost the domestic electoral standing of the incumbent party, not just temporarily (rally-around-the-flag) but through a partisan advantage channel. Includes 5 canonical examples (Taiwan, Georgia, Israel, Pakistan, US 2020 counterexample), 4 mechanisms (issue salience shift, incumbent-as-defender framing, opposition delegitimization, nationalist mobilization), conditions analysis table, and 7-step forecasting application. Calibrates strong (3-8 pp), moderate (1-3 pp), and neutral/reversed effects. |
| `domains/east-asia/concepts/third-party-ceiling-fptp/_concept.md` | **Updated** | Expanded late-campaign collapse pattern from 1 sentence to detailed 4-phase trajectory table (6+ months out, 2-4 months out, 0-4 weeks out, election day) with canonical data points (Ko Wen-je, James Soong, Ross Perot) and forecasting implication (assume 3-8 point collapse from peak). Updated Validated By entry to reference the collapse pattern. |
| `_procedure.md` | **Updated** | Added step 8c (external-threat-incumbency-boost) to the Pre-Forecast Audit: 7 sub-steps for identifying adversary action, assessing partisan alignment, checking timing/blowback/economic pain, applying the concept, and documenting explicitly. Added explicit downstream renaming: old 8c (shutdown) → 8d. |

## Insight: The Third Analytical Dimension

This question reveals a pattern across all three reflection cycles:

| Cycle | Observation | What Was Added |
|-------|------------|----------------|
| v1 | No vault content at all | Entities, timeline, spec rule, procedure step |
| v2 | Vault has macro (timeline) and micro (entities) but missing meso (concepts) | Third-party-ceiling concept, polling trajectory, viability procedure |
| v3 | Vault has structural mechanics covered but misses the **amplifying dynamics** — external factors that don't change the binary outcome but explain the magnitude | External-threat-incumbency-boost concept, threaded integration, late-campaign collapse detail |

The progression is: **coverage → structure → amplification**. Each cycle adds a deeper layer of analytical power, from "what happened" (v1) through "why it happened structurally" (v2) to "what else was going on" (v3). For future per-question reflections, after ensuring structural coverage, check whether there are amplifying dynamics that affect the probability window rather than the binary outcome.
