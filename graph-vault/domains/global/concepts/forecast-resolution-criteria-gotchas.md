---
type: concept
tags: [concept]
title: "Forecast Resolution Criteria Gotchas"
slug: forecast-resolution-criteria-gotchas
first_observed: 2024-07-28
domain: forecasting-methodology
related_concepts:
  - diplomatic-pressure-tipping-point
  - presidential-sentencing-dynamics
  - incumbent-withdrawal-cascade
  - late-candidate-substitution
---
---
---
# Forecast Resolution Criteria Gotchas

## Definition

A recurring class of forecasting error where the resolution criteria of a prediction market question is misinterpreted, causing the forecaster to predict the wrong event outcome despite correctly assessing the underlying situation. The error is not in understanding the world, but in mapping real-world events to market resolution rules.

## Canonical Vault Examples

### 1. "Wins" = Vote Count vs. Assumes Office (Venezuela 2024)

| Field | Detail |
|-------|--------|
| Question | "Will Edmundo González win the 2024 Venezuela presidential election?" |
| Forecast | NO (incorrect) |
| Error | Interpreted "wins" as "assumes power" rather than "receives most votes" |
| Resolution | YES — González received ~67% of the vote; Maduro's CNE fraud did not change resolution |
| Lesson | In authoritarian elections, "winning" (plurality of votes) and "assuming office" frequently diverge. Prediction markets resolve on the vote outcome, not the power transition, unless explicitly stated otherwise. [[2026-05-18-venezuela-election-gonzalez]] |

**Why it's a gotcha:** The everyday meaning of "winning an election" conflates electoral outcome with assumption of office. In democratic contexts they're the same. In authoritarian contexts they're not. The forecaster must check: does the resolution source (election authority, independent audit, vote count) exist independent of who holds power?

### 2. "Announces" vs. "Ratifies" (Gaza Ceasefire Oct 2025)

| Field | Detail |
|-------|--------|
| Question | "Will Israel first announce ceasefire on October 9?" |
| Forecast | YES (incorrect) |
| Error | Conflated "cabinet ratification" (Oct 9) with "first official announcement" (Oct 8) |
| Resolution | NO — Israel's PMO confirmed the ceasefire on Oct 8, cabinet ratified on Oct 9 |
| Lesson | A multi-step process (agreement → announcement → ratification → effective) has distinct dates for each step. The question specifies one step. The forecaster must map the resolution criteria to the exact step, not just "something happened on that date." [[2026-05-18-gaza-ceasefire-october-9]] |

**Why it's a gotcha:** Political events are reported as chains ("ceasefire approved", "deal struck", "cabinet okays"). The vault timeline naturally records the most salient date. But the market question picks one precise step in the chain. Without explicit annotation of which date corresponds to which step, the timeline entry misleads.

### 3. "Dips to $1.20" — Tick vs. OHLC Close (XRP March 2025)

| Field | Detail |
|-------|--------|
| Question | "Will XRP dip to $1.20 on March 21?" |
| Forecast | NO (correct — but tail risk identified) |
| Error type | Near-miss that revealed a resolution edge case |
| Artifact | Binance OHLC showed low of $1.4003. But a tick-level touch on a low-liquidity exchange *could* satisfy the condition without appearing in standard OHLC data. |
| Lesson | Price-based questions with precise targets depend on data source and sampling methodology. A tick on a low-liquidity exchange resolves differently than an OHLC close on a major exchange. [[2026-05-18-xrp-dip-march21-synthesis]] |

**Why it's a gotcha:** The question doesn't specify which exchange or data sampling method. The forecaster assumes a standard reference (Binance, Coinbase) but the resolution source may be Polymarket's oracle which could use any CMC/CryptoCompare feed. The edge case was correctly identified as tail risk but wasn't the primary forecast error.

### 4. "Sentenced to 24-35 Months" — Sentence Range vs. Actual Incarceration (Trump 2025)

| Field | Detail |
|-------|--------|
| Question | "Will Trump be sentenced to 24-35 months prison?" |
| Forecast | NO (correct) |
| Error type | Correct prediction, but the reasoning had to navigate nested conditionals |
| Artifact | The question specified a sentence *range*, not a binary incarceration question. A sentence of 24-35 months could theoretically be imposed and then immediately suspended — does that count? |
| Lesson | Prison sentencing questions have nested gotchas: (a) does the range correspond to the felony class? (b) does "sentenced to X months" mean imposed or served? (c) can a sentence be appealed before it's served? [[2026-05-18-trump-sentencing-24-35-months]] |

### 5. "Drop Out" — Withdraawl vs. Suspended Campaign (Biden 2024)

| Field | Detail |
|-------|--------|
| Question | "Will Biden drop out of presidential race?" |
| Forecast | NO (incorrect) |
| Error | Interpreted "drop out" as "withdraw under pressure" vs. "announce non-candidacy" |
| Resolution | YES — Biden announced he would not seek re-election on July 21, 2024 |
| Lesson | "Drop out" in prediction markets means "cease to be a candidate" regardless of the precipitating cause. Resistance-to-change signals (incumbent advantage, party unity) are not equivalent to impossibility of withdrawal. [[2026-05-18-biden-dropout]] |

### 6. "Ceasefire" = Temporary Pause vs. Enduring Agreement (Israel-Hamas Nov 2023)

| Field | Detail |
|-------|--------|
| Question | "Israel and Hamas ceasefire in 2023?" |
| Forecast | NO (incorrect) |
| Error | Interpreted "ceasefire" as "permanent end to hostilities" rather than "any mutually agreed halt" |
| Resolution | YES — November 22 humanitarian pause (4-day, extended to 7) with hostage/prisoner exchanges satisfied "publicly announced and mutually agreed halt in military engagement" |
| Lesson | "Ceasefire" on Polymarket means ANY mutually agreed halt, regardless of duration or whether fighting resumes afterward. Temporary humanitarian pauses that expire are still ceasefires. Only check for "permanent," "comprehensive," or "lasting" qualifiers in the resolution text — if absent, the broad definition applies. [[2026-05-20-israel-hamas-ceasefire-nov2023]] |

**Why it's a gotcha:** The everyday meaning of "ceasefire" implies an enduring end to fighting — a ceasefire that collapses is a "failed ceasefire," and a temporary pause is a "humanitarian pause," not a "ceasefire." But the market resolution criteria use the literal definition: any publicly announced and mutually agreed halt counts. The forecaster who applies the narrower everyday meaning will miss temporary ceasefires that expire before the resolution deadline. This is formally defined in [[concepts/temporary-vs-enduring-ceasefire]].

### 7. "Banned" = Legal Status vs. Practical Enforcement (TikTok 2025)

| Field | Detail |
|-------|--------|
| Question | "TikTok banned in the US before May 2025?" |
| Forecast | YES (correct) |
| Error type | Correct prediction — but the legal-vs-enforcement ambiguity made this a hidden trap |
| Resolution | YES — the Protecting Americans from Foreign Adversary Controlled Applications Act took effect Jan 19, 2025; TikTok was removed from Apple/Google app stores and went dark for ~14 hours, satisfying "banned for download and/or use by the majority of Americans" |
| Lesson | "Banned" resolves on LEGAL STATUS (did the law take effect? was the app removed from stores?) not on ENFORCEMENT PERSISTENCE (is the ban still in effect?). Executive enforcement delay (Trump's Jan 20 executive order suspending enforcement) does NOT retroactively negate the ban. The app was banned for resolution purposes even though service was restored. [[2026-05-18-tiktok-ban-us]] |

**Why it's a gotcha:** The everyday meaning of "banned" implies an ongoing prohibition — if the app is available again, it wasn't "really banned." But the market resolution criteria focus on whether the ban legally took effect and enforcement actions occurred, not whether the ban persists. This mirrors the ceasefire gotcha (example 6): a temporary cessation is still a ceasefire, and a temporary ban is still a ban.

**Key insight for future forecasts**: When a question asks "is X banned by [date]?", the resolution depends on:
1. Did the law/order take legal effect before the deadline?
2. Did enforcement actions occur (app store removal, service shutdown, fines)?
3. NOT on whether enforcement continues past the resolution date.

This maps to the [[concepts/executive-enforcement-delay]] framework, which distinguishes legal implementation from practical enforcement.

## Meta-Patterns

These errors share a common structure:

1. **Everyday-word fallacy**: The question uses a word ("wins", "announces", "dips", "drops out") whose everyday meaning differs from its market-resolution meaning.

2. **Single-point anchoring**: The vault records an event on a single date, but the resolution criteria pick a specific sub-event within a multi-step process.

3. **Source ambiguity**: The question doesn't specify what data source resolves it, forcing the forecaster to guess. The resolution source (Polymarket oracle, API feed, manual judge) may use different data or standards.

## Pre-Forecast Audit Steps

Before finalizing any forecast, check:

1. **What exactly resolves this market?** Is the resolution text available? Read it literally — words like "wins", "announces", "approves" have specific meanings.

2. **Can the outcome diverge from the underlying reality?** Examples: authoritarian elections (votes vs. power), ceasefire announcements (declaration vs. implementation), price events (tick vs. close).

3. **What data source resolves the market?** Polymarket oracles, API feeds, manual judges — each has different standards. A price sourced from CoinMarketCap's volume-weighted average differs from a single exchange's last price.

4. **Does the question specify a date?** Verify that the event matching the resolution criteria occurred on that exact date — not just a related event on that date (announcement ≠ ratification ≠ implementation).

5. **Apply dual-frame analysis**: For every question, forecast both the resolution outcome AND the real-world outcome. If they can diverge, note the divergence risk.

### 8. "Control N Seats" — Threshold vs. Exact Count (US House 2024)

| Field | Detail |
|-------|--------|
| Question | "Will Republicans control 224 seats in the House after the election?" |
| Forecast | NO (correct) |
| Error type | Correct prediction — but the question wording was a hidden resolution-criteria trap |
| Resolution | NO — Republicans won exactly 220 seats, not 224. The resolution text specifies "exactly 224 voting House members are Republican." |
| Lesson | The word "control" in the question implies a **threshold** meaning ("control enough seats to have 224" or "control the chamber with at least 224"), but the resolution text specifies an **exact count** ("exactly 224 members"). These are completely different questions: P(at least 224) ≈ 20-25% vs. P(exactly 224) ≈ 3%. A forecaster who read the question literally ("control 224") without checking the resolution text would overestimate p_yes by 5-10x. |

**Why it's a gotcha**: In everyday political language, "control 224 seats" means "control enough seats to total 224" — the number modifies the scope of control. But the resolution criteria use "exactly 224," making this an exact-count question masquerading as a threshold question. The pattern generalizes: any question combining "control/wins/has" with a specific number should be checked against the resolution text for exact-count vs. threshold semantics. See [[domains/usa/concepts/exact-count-vs-range-forecast/_concept]] for the exact-count framework and [[domains/usa/procedures/exact-seat-count-forecast]] for the procedure.

**General rule for "control N" questions**: When a question says "control N seats" or "win N seats," the resolution text may specify:
1. **Exact count** (as in this case) — requires the exact integer
2. **Threshold / at least** — requires count >= N
3. **Range** — requires N-1 to N+1 or similar
Always resolve by reading the resolution text literally, not the question title. If no resolution text is available, flag the ambiguity explicitly and forecast both interpretations before reading market prices.

### 9. "First announce" — Initiation vs. Chronological Priority in Multi-Ceasefire Conflicts (Gaza Oct 2025)

| Field | Detail |
|-------|--------|
| Question | "Will Israel first announce ceasefire on October 8?" |
| Forecast | YES (correct) |
| Error type | Potential gotcha — correct prediction, but a forecaster misreading the title could have been misled |
| Resolution | YES — Israel's PMO confirmed agreement on Oct 8; this was the initiating announcement of the October 2025 deal (the first/initial announcement as distinct from the Oct 9 cabinet ratification) |
| What happened | By October 2025, Israel had already announced ceasefires twice in this conflict (Nov 2023 humanitarian pause, Jan 2025 ceasefire). The word "first" in the title could be misread as "first time ever" → automatic NO because prior ceasefires existed. But the resolution text specified "the next date that Israel officially announces" — temporal next, not first-ever. The "first" in the title meant the initiating/first announcement of THIS specific deal, not the first announcement ever in the conflict. |
| Why it's a gotcha | The everyday meaning of "first announces" implies "the first time" a party does something. When prior ceasefires exist in the same conflict, a forecaster who reads the title without checking the resolution text will: (a) think "Israel has already announced ceasefires → first-ever is impossible" → predict NO, or (b) confuse "first announcement of this deal" with "ratification" → predict the wrong date. The resolution text's "next date" wording disambiguates — it refers to the chronological next announcement in sequence, not the historical first. Parallel questions for Oct 8 and Oct 9 existed on Polymarket, testing whether the forecaster could distinguish announcement (Oct 8) from ratification (Oct 9). The correct YES on Oct 8 and correct NO on Oct 9 required understanding that "first announce" = initiating announcement of this deal, "ratification" = subsequent internal process. |
| Lesson | In multi-ceasefire conflicts, the word "first" in a question title is ambiguous — it can mean (1) historical first-ever, (2) initiating announcement of this deal (vs. ratification), or (3) chronological next per resolution text. Before predicting: (a) read the FULL resolution text to find what wording resolves the market — "next date," "first announces," "finally agrees," etc. all have distinct meanings; (b) check how many prior ceasefires exist in the conflict — if multiple, "first" cannot mean "first-ever"; (c) check for parallel questions with adjacent dates (Oct 8 vs Oct 9) — these signal the market is testing date-type precision, not first-ever status; (d) apply the announcement-vs-ratification distinction from [[concepts/ceasefire-announcement-ratification-gap]] regardless of whether the title says "first." |

**Why it's a gotcha (generalized)**: This pattern extends beyond ceasefires to any multi-event sequence where "first announce / first to [action]" appears in the title but the resolution text uses different wording ("next," "subsequent," "within period"). The forecaster's default reading of "first" as "first ever" leads to the wrong frame. The correct approach is: treat the question title as a hint, the resolution text as law. If they diverge, the resolution text controls. Always check for parallel questions testing adjacent dates — their existence confirms the date-type distinction is the actual question being asked.

**Key insight for future forecasts**: When a question uses "first announce [action] on [date]":
1. Read the resolution text — does it say "next," "first," "initial," or something else?
2. Count prior occurrences of the action in the same conflict. If >0, "first" cannot mean "first-ever."
3. Check for parallel questions with adjacent dates — their presence indicates date-type precision is the test.
4. If the conflict has had prior ceasefires, the question is almost certainly about the initiating announcement of the current deal vs. its ratification, not historical first-ever status.

## Cross-References

- [[concepts/diplomatic-pressure-tipping-point]] — the ceasefire announcement concept that needed announcement-vs-ratification annotation
- [[concepts/incumbent-withdrawal-cascade]] — the withdrawal pattern that was missed for Biden
- [[forecasts/2026-05-18-venezuela-election-gonzalez]] — the canonical resolution-criteria error
- [[forecasts/2026-05-18-gaza-ceasefire-october-9]] — the announcement-vs-ratification error
- [[forecasts/2026-05-18-gaza-ceasefire-october-8]] — the "first announce" wording ambiguity in multi-ceasefire conflicts (entry #9)
- [[forecasts/2026-05-18-biden-dropout]] — the "drop out" ambiguity
- [[forecasts/2026-05-18-trump-sentencing-24-35-months]] — the sentencing-range gotcha
- [[forecasts/2026-05-18-xrp-dip-march21-synthesis]] — the tick-vs-close gotcha
- [[concepts/executive-enforcement-delay]] — the legal-vs-practical enforcement gap
- [[domains/global/procedures/ban-resolution-checklist]] — the "banned" resolution checklist procedure
- [[forecasts/2026-05-18-tiktok-ban-us]] — the TikTok ban forecast that established the legal-vs-enforcement gotcha
- [[domains/usa/concepts/exact-count-vs-range-forecast/_concept]] — the exact-count vs. range framework that supports the "control N" gotcha analysis
- [[domains/usa/procedures/exact-seat-count-forecast]] — the procedure for forecasting on exact-count questions like "control 224 seats"
