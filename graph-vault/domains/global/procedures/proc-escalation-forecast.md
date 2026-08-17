---
type: procedure
tags: [procedure]
title: "Escalation Forecast"
slug: proc-escalation-forecast
domain: "[[domains/geopolitics]]"
concept: "[[concepts/escalation-bargaining-termination]]"
functions:
  - [[functions/run_structured().md|run_structured]]
  - [[functions/pit_search().md|pit_search]]
  - [[functions/base_rate_check().md|base_rate_check]]
---
---
---
# Escalation Forecast

Compute P(escalation) for state-on-state conflict questions using ladder dynamics and the 48-hour rule.

## When to use

Question involves conflict escalation thresholds, a known escalation ladder, or a superpower patron relationship.

## Composition

```
1. pit_search("escalation thread for <entity>", cutoff)
   → returns current ladder position, recent threshold events

2. base_rate_check("escalation_ladder_completion", entity)
   → returns historical completion rate for similar ladders

3. run_structured(question, cutoff, vault_dir, concept="escalation-bargaining-termination")
   → LLM synthesizes with vault context, 48-hour rule, ladder calibration
   → returns p_yes, reasoning
```

## Calibration from concept

- **48-hour rule**: superpower combatant entry → ceasefire within 72h (p > 0.9)
- **Ladder dynamics**: each threshold crossed increases next-step probability
- **Asymmetric limitation**: pattern applies to state-on-state wars, not asymmetric conflicts

## Output schema

```json
{
  "p_yes": 0.XX,
  "ladder_position": {"current_step": "...", "target_step": "...", "steps_remaining": N},
  "superpower_present": true/false,
  "48hr_rule_applied": true/false,
  "reasoning": "..."
}
```
