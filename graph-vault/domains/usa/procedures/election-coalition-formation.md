---
type: procedure
tags: [procedure]
title: "Election Coalition Formation"
domain: "[[domains/elections]]"
concept: "[[concepts/divided-opposition-plurality-win]]"
functions:
  - "[[functions/run_structured().md|run_structured]]"
  - "[[functions/build_forecast_prompt().md|build_forecast_prompt]]"
  - "[[functions/pit_search().md|pit_search]]"
---
---
---
# Election Coalition Formation

Estimate whether parties will form a coalition, whether a specific coalition will win, or whether party unity holds before an election.

## When

Questions about coalition formation (French left unity), primary coordination, or party unity ahead of an election.

## Approach

1. Decompose into YES requirements and NO blockers
2. YES: unique candidate, united programme, hegemonic coordination
3. NO: internal division, too many candidates, failure to expand beyond base
4. P(YES) = P(req1) * P(req2|req1) * P(req3|req1,req2) — each multiplicative

## Calibration

- 3+ viable candidates + single-round plurality → front-runner wins at 30-45% (p > 0.85)
- Coalition without a hegemon → failure rate > 60%
- Party split mid-campaign → re-evaluate all conditions
