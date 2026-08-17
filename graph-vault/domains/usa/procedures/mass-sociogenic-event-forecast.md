---
type: procedure
tags: [procedure, usa, domestic-security, forecasting]
title: "Mass Sociogenic Event Forecast"
slug: mass-sociogenic-event-forecast
domain: usa
related_concepts:
  - mass-sociogenic-event
  - government-confirmation-requirement
---

# Mass Sociogenic Event Forecast Procedure

## When to Activate

Trigger when a question asks about the nature, cause, or origin of a mysterious event involving widespread public reports of anomalous sightings — drone waves, UFO flaps, unexplained lights, strange sounds, or similar phenomena.

## Step 1: Classify the Question Type

Three distinct question types require different methodologies:

**Type A: Government Confirmation Required** — Resolution requires an official government statement confirming a specific explanation (e.g., "searching for nuclear weapons," "foreign surveillance," "extraterrestrial origin").
- Apply the full government-confirmation bar: default P(yes) < 10%
- Load [[domains/global/concepts/government-confirmation-requirement/_concept]]
- This was the type for Q44 (Mystery Drones)

**Type B: Event Existence** — Resolution asks whether an anomalous event is happening, regardless of government confirmation (e.g., "are drones operating over military bases?").
- Standard probability assessment using official incident data
- Distinguish between civilian reports and verified military incursions
- Load [[domains/usa/threads/us-drone-security-incidents/_thread]] for base incursion data

**Type C: Attribution** — Resolution asks whether a specific actor is responsible (e.g., "is China behind the drone sightings?").
- Apply the same government confirmation bar as Type A
- Check for foreign-adversary attribution patterns (Type B in the government-confirmation concept)
- For mass sociogenic events, attribution questions have very low resolution probability because investigations conclude nothing anomalous to attribute

## Step 2: Identify the Event Age

Days since initial reports determine the expected stage of official response:

| Days | Stage | Typical Official Posture | P(confirmation) |
|------|-------|--------------------------|-----------------|
| 1-7  | Emergence | No official statement yet | 10-25% (max) |
| 8-14 | Investigation Opening | "We are looking into it" | 5-15% |
| 15-21 | Preliminary Findings | "No evidence of threat" | 2-5% |
| 22-60 | Investigation Conclusion | "Nothing anomalous; routine explanations" | <2% |
| 61+ | Residual | Agencies move on; legislation may follow | <1% |

## Step 3: Check for Official Denials

Search for agency-specific denials of the theory. The following sources are dispositive:

- **NNSA**: Denial that emergency response unit uses drones for radiological detection (nuclear search theories)
- **FBI**: Joint statements concluding nothing anomalous found (general mystery events)
- **DHS**: Secretary statements attributing sightings to misidentified aircraft
- **DoD**: Pentagon spokesperson statements on foreign origin or military involvement
- **White House**: Press secretary statements on FAA authorization or routine nature

A prior denial from any of these sources reduces P(confirmation) to near zero — the agency would need to publicly reverse itself.

## Step 4: Map the Theory's Origin

Credibility hierarchy for theories about anomalous events:

1. **Federal investigator statement** (highest) — If a federal agency states the theory, it has evidentiary basis
2. **Congressional statement based on briefings** — May reflect classified briefings but also political positioning
3. **Local official statement** — Low credibility for national security claims; local officials lack clearance and may be responding to constituent concern
4. **Social media / online speculation** (lowest) — Does not correlate with confirmation probability; may reflect AI-generated or deceptively edited content

## Step 5: Check Entity Stub Coverage

Before forecasting, ensure entity stubs exist for:
- Lead investigating agency (typically FBI)
- Any agency named in the question or resolution text
- Any agency whose denial/disclosure would change the probability
- Agency head if they have confirmation authority

## Step 6: Calibrate and Document

Produce the final probability estimate with this reasoning template:

```
Question type: [A/B/C]
Event age: [X days since initial reports]
Event stage: [Emergence / Investigation / Preliminary / Conclusion / Residual]
Official denials: [List any pre-existing denials with agency and date]
Theory origin: [Federal investigator / Local official / Social media]
Government confirmation type: [Type A/B/C/D from government-confirmation concept]
External compulsion: [Any lawsuits, subpoenas, leaks, or bipartisan pressure that could force confirmation]
P(yes): [Calibrated probability]
P(yes) would be without government confirmation bar: [Counterfactual to show the adjustment]
```

## Canonical Case Scorecard (Q44)

| Variable | Value |
|----------|-------|
| Question Type | A (government confirmation required) |
| Event Age at Prediction | ~30 days (mid-December 2024) |
| Official Denials | NNSA, NJ DEP, FBI/DHS/DoD joint statement, White House |
| Theory Origin | Local officials (Belleville mayor, Congresswoman Malliotakis) |
| Government confirmation type | Type A (vulnerability admission) |
| External compulsion | None |
| P(yes) predicted | 0% (NO) |
| Actual outcome | NO |
| Correct? | Yes |
