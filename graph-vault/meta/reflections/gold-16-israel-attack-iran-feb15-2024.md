# Reflection: Question 16 — Israel Attack Iran by February 15, 2024

## Result
- Prediction: NO (correct)
- Ground truth: NO

## Diagnosis

The prediction was correct because the structural improbability of a first-ever direct Israeli strike on Iranian soil within a 3-week window (Jan 25 - Feb 15, 2024) was high. The shadow war was at Stage 1-2 (proxy + covert ops), no tripwire event had occurred, IDF bandwidth was absorbed by Gaza, and the Biden administration was actively containing escalation.

However, the vault contributed near-zero analytical signal to this prediction. The correct answer was reached through general knowledge of the period (no major Israel-Iran escalation had occurred by mid-Feb 2024). Per spec principle #8, this is as much a vault gap as a wrong prediction.

## Specific Gaps Found

1. **No short-window attack probability concept**: The vault had no framework for calibrating P(military strike) within a specific narrow calendar window. Without this, it's impossible to distinguish "this is unlikely within 3 weeks" from "this is unlikely at all" — a critical distinction for prediction market questions with tight deadlines.

2. **No shadow-war-to-direct-escalation concept**: While the `escalation-bargaining-termination` concept covers the post-strike termination phase, no concept documents the structural path from proxy warfare through covert ops to direct confrontation. This is the pre-strike escalation ladder concept.

3. **Missing IRGC entity stub**: Iran's most important military institution (the IRGC) had no stub. Its decision-making patterns, retaliation history, and internal factionalism are critical for forecasting Iranian responses.

4. **Netanyahu entity stub focused only on 2025 war**: The Benjamin Netanyahu stub described his actions during the June 2025 Twelve-Day War but had no coverage of his shadow war phase decision-making (Oct 2023 - Mar 2024), no Begin Doctrine connection, and no threshold analysis.

5. **Israel entity lacked military doctrine section**: The Israel stub had no coverage of the Begin Doctrine, preemptive strike doctrine, multi-front operational constraints, or decision-making structure for strike authorization.

6. **2024-Q1 timeline had no Israel-Iran section**: The quarter file covered Israel-Gaza war but had zero analysis of Israel-Iran shadow war dynamics, despite major regional escalation events (Iran-Pakistan exchange Jan 16-18, US strikes on Iranian proxies Feb 2).

7. **No Israel military strike forecast procedure**: No structured methodology existed for forecasting Israeli attack probabilities.

## Improvements Made

| File | Type | Change |
|---|---|---|
| `domains/mena/concepts/short-window-military-strike-probability/_concept.md` | Concept (NEW) | Framework for estimating P(military strike within N-day window) with base rates by window length, tripwire adjustment, decision-cycle friction, and precedent penalty. Includes calibration tables from Israel-Iran cases. |
| `domains/mena/concepts/shadow-war-to-direct-escalation/_concept.md` | Concept (NEW) | 9-stage escalation ladder from cold rivalry through proxy warfare to full conventional war. Documents the compression principle (transition times decrease as escalation progresses). Includes necessary conditions, failure modes, and forecasting application. |
| `domains/iran/entities/islamic-revolutionary-guard-corps-irgc.md` | Entity (NEW) | IRGC institutional entity with retaliation history table, forecasting patterns, internal factionalism, and decision-making opacity. |
| `domains/global/entities/benjamin-netanyahu.md` | Entity (UPDATED) | Added shadow war phase decision-making (Oct 2023 - Mar 2024), Begin Doctrine connection, threshold-based decision rules, and risk calculus under multi-front constraints. |
| `domains/global/entities/israel.md` | Entity (UPDATED) | Added Military Doctrine section covering the Begin Doctrine, preemption chain, multi-front operational constraints, and strike authorization structure. Added escalation timeline (Apr 2024+ events). |
| `domains/mena/procedures/israel-strike-forecast.md` | Procedure (NEW) | Step-by-step procedure computing P(Israel strikes target T before deadline D) using escalation stage, window length, Begin Doctrine trigger, US signal, and bandwidth multiplier. |
| `timeline/2024-Q1.md` | Timeline (UPDATED) | Added "Israel-Iran Shadow War (Regional Escalation Risk)" subsection with Jan-Feb-Mar chronicle, key signal events, and PIT forecasting significance section explicitly calibrating P(Israel strike on Iran during Q1 2024) at <0.05. |
| `domains/mena/threads/israel-iran-shadow-war-gaza-2023-2024/_thread.md` | Thread (UPDATED) | Added "Narrow-Window Calibration" subsection with 5-factor analysis (no tripwire, short window constraints, bandwidth exhaustion, US containment, shadow war stage). Explicitly computes P~0.03-0.05 for the Feb 15 window. |
| `_forecast_instructions.md` | Instructions (UPDATED) | Added Rule 3: Short-window military strike calibration with 9-step checklist, tripwire taxonomy, documentation template, and reflection requirements. |

## Next Question Impact

For future questions about military strikes within narrow windows (e.g., "Will X attack Y by date Z?"), the vault now has:
- Base rate data by window length (1 week, 3 weeks, 3 months, 6 months, 1 year)
- Tripwire event taxonomy with probability adjustment per type
- Decision cycle friction estimates for Israel (2-5 weeks minimum)
- Escalation stage mapping (Stage 0-8) to contextualize current position
- Structured procedure (israel-strike-forecast) with composable multipliers
- Explicit Rule 3 in forecast instructions that triggers for short-window military questions
- Entity stubs for key actors (Netanyahu, IRGC, Israel doctrine)
