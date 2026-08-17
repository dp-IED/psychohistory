---
type: concept
tags: [concept]
title: "Temporary vs. Enduring Ceasefire Definition Gap"
slug: temporary-vs-enduring-ceasefire
first_observed: 2023-11-22
domain: forecasting-methodology
related_concepts:
  - ceasefire-announcement-ratification-gap
  - forecast-resolution-criteria-gotchas
  - political-deadline-ceasefire
status: active
---

# Temporary vs. Enduring Ceasefire Definition Gap

## Definition

A recurring source of forecasting error in which the everyday meaning of "ceasefire" (an enduring end to hostilities, typically intended to be permanent or at least long-term) differs from the market-resolution definition (any "publicly announced and mutually agreed halt in military engagement," including temporary humanitarian pauses of days or weeks). The resolution criteria nearly always use the broad definition unless explicitly qualified. The forecaster who defaults to the narrow everyday meaning will predict NO on questions that should be YES.

This is distinct from the [[concepts/ceasefire-announcement-ratification-gap]] (which concerns date-type confusion) and the [[concepts/forecast-resolution-criteria-gotchas]] meta-pattern (which this concept feeds). It is a **definition-of-term** error rather than a date-mapping error.

## The Two Meanings

| Feature | Everyday meaning | Market-resolution meaning |
|---------|-----------------|--------------------------|
| Duration | Implicitly enduring or long-term | Any duration, including hours/days |
| Intent | Parties intend to stop the war | Parties agree to stop shooting, possibly temporarily |
| Collapse | If fighting resumes, the ceasefire "failed" | Ceasefire occurred even if it later collapses |
| Synonyms | Armistice, truce, peace agreement | Humanitarian pause, cessation of hostilities, lull |
| Example | "The 1918 armistice ended WWI" | "Israel and Hamas agree to 4-day humanitarian pause" |

## Canonical Error

### Israel-Hamas November 2023 Humanitarian Pause

| Field | Detail |
|-------|--------|
| Question | "Israel and Hamas ceasefire in 2023?" |
| Forecast | NO (incorrect) |
| Actual | YES |
| Resolution criteria | "publicly announced and mutually agreed halt in military engagement" |
| What happened | Nov 22: Israel and Hamas agreed to a 4-day humanitarian ceasefire (later extended to 7 days). 105 hostages released, 240 Palestinian prisoners exchanged. Fighting paused Nov 24-30, then resumed. |
| Why NO was wrong | The forecaster interpreted "ceasefire" as meaning an enduring end to the war, not a temporary humanitarian pause. The resolution criteria did not require permanence — any mutually agreed halt qualified. |
| Lesson | "Ceasefire" in Polymarket resolution criteria means ANY publicly announced halt in military engagement, regardless of duration or whether fighting later resumes. Temporary humanitarian pauses (even multi-day ones) are ceasefires. |

## Pattern Recognition

### Questions Where This Matters

The temporary vs. enduring distinction is decisive for forecast accuracy whenever:

1. **The question uses unqualified "ceasefire"**: If the resolution criteria say "ceasefire agreement" without "permanent," "lasting," or "comprehensive," a temporary pause qualifies. The resolution text matters more than the question title.

2. **The historical analog involves failed ceasefires**: When a war includes a temporary halt that was negotiated and publicly announced but later failed (e.g., 2014 Gaza ceasefire that collapsed, 2023 November pause that expired), the event still counts as a ceasefire under Polymarket criteria. A ceasefire is not invalidated by its subsequent collapse — it happened and then it ended.

3. **The pause involves hostage/prisoner exchanges**: Humanitarian pauses tied to hostage releases or prisoner swaps are structurally likely to have explicit start and end dates, making them seem "less than" a ceasefire. But they are still mutually agreed halts in military engagement — the core definition.

4. **The fighting continues before and after**: If fighting resumed immediately after the pause ended, the forecaster may dismiss the pause as "not a real ceasefire." But the pause itself was a ceasefire — the resumption does not retroactively erase it.

### When the Distinction Doesn't Matter

- When the resolution criteria explicitly say "permanent ceasefire," "comprehensive ceasefire," or "end to hostilities" — in those cases, temporary pauses are excluded.
- When the question asks about "ceasefire that holds for X days" — the market is calibrated on durability, not just the announcement.
- When the question specifies "ceasefire agreement ending the war" — the permanent meaning is explicit.

## Key Indicators for the Forecaster

Before finalizing a "NO" on any ceasefire question, check:

1. **Read the FULL resolution text, not just the title.** Look for qualifiers like "permanent," "comprehensive," "lasting," "ending the war." If none appear, the broad definition applies.

2. **Is there a temporary pause or humanitarian truce in the relevant period?** Even a multi-day halt with specific exchange mechanisms counts. The November 2023 pause was 4-7 days and included hostage releases — it counted.

3. **Would a reasonable observer say "fighting stopped"?** If the answer is yes (even temporarily and with a known end date), the event likely satisfies "halt in military engagement."

4. **Does the pause have the structural form of a ceasefire?** Public announcement, mutual agreement by named parties, defined start time, defined terms (e.g., exchanges, zones) — if these are present, it's a ceasefire under standard Polymarket criteria.

5. **Did media outlets, governments, or international organizations describe it as a "ceasefire"?** Media reports are evidence for resolution. If BBC, Reuters, and the UN called the November 2023 event a "humanitarian ceasefire," Polymarket would resolve on that characterization.

## Resolution Text Examples

| Resolution text wording | Includes temporary pause? | Example question |
|------------------------|-------------------------|------------------|
| "publicly announced and mutually agreed halt in military engagement" | YES — any halt qualifies | Israel and Hamas ceasefire in 2023? |
| "ceasefire agreement" (bare) | YES — default broad meaning | various Polymarket questions |
| "permanent ceasefire agreement" | NO — explicitly qualified | various |
| "ceasefire that holds for at least 30 days" | PARTIAL — must endure beyond the pause | various |
| "end to hostilities" | AMBIGUOUS — depends on context; likely means permanent | various |

## Related Concepts

- [[ceasefire-announcement-ratification-gap]] — Different gotcha: date-type confusion for announcement vs. ratification
- [[forecast-resolution-criteria-gotchas]] — Meta-pattern of everyday-word vs. market-resolution-meaning errors
- [[political-deadline-ceasefire]] — Temporal compression of ceasefire negotiations by known deadlines
- [[diplomatic-pressure-tipping-point]] — Accumulated pressure that produces ceasefires

## Wikilinks

- [[events/november-2023-humanitarian-pause]]
- [[threads/gaza-ceasefire-negotiations-2025]]
- [[timeline/2023-Q4]]
- [[entities/hamas]], [[entities/israel]]
- [[entities/qatar]], [[entities/egypt]]
