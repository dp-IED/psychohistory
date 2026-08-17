---
type: procedure
tags: [procedure]
title: "Incumbent Withdrawal Forecast"
slug: proc-incumbent-withdrawal
domain: "[[domains/elections]]"
concept: "[[concepts/incumbent-withdrawal-cascade]]"
functions:
  - [[functions/run_structured().md|run_structured]]
  - [[functions/pit_search().md|pit_search]]
  - [[functions/build_forecast_prompt().md|build_forecast_prompt]]
gold_case: gold-18 (Biden DNC dropout 2024, vault MAE 0.005)
---
---
---
# Incumbent Withdrawal Forecast

Compute P(withdrawal) for a leader using the 5-condition cascade framework and velocity benchmarks.

## When to use

Question asks whether an incumbent leader, presumptive nominee, or sitting officeholder will withdraw, resign, or drop out.

## Composition

```
1. pit_search("<leader> thread", cutoff)
   → returns entity timeline: approval ratings, trigger events, defection signals

2. build_forecast_prompt(question, cutoff, concept="incumbent-withdrawal-cascade")
   → constructs prompt with 5-condition framework, cascade velocity benchmarks

3. run_structured(question, cutoff, vault_dir, concept="incumbent-withdrawal-cascade")
   → evaluates: legal jeopardy, internal pressure, trigger event, successor, electoral position
   → returns p_yes, condition_assessment, reasoning
```

## Calibration from concept

| Conditions met | P(withdrawal) |
|---|---|
| 0 | <5% |
| 1-2 | 5-15% |
| 3 (incl. no legal jeopardy) | 30-50% |
| 4 | 50-70% |
| 5 | >70% |

**Cascade velocity**: trigger → withdrawal = 18-24 days depending on trigger clarity.

## Output schema

```json
{
  "p_yes": 0.XX,
  "conditions_met": N,
  "legal_jeopardy": true/false,
  "trigger_type": "debate|primary|health|none",
  "estimated_days_to_withdrawal": N,
  "reasoning": "..."
}
```
