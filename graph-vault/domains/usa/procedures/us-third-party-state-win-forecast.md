---
type: procedure
tags: [procedure, usa, elections, third-party]
title: "US Third-Party State-Win Forecast"
slug: us-third-party-state-win-forecast
version: 1.0
created: 2026-05-20
---

# US Third-Party State-Win Forecast

## When to Use

Apply this procedure when a Polymarket or forecasting competition question asks whether a third-party or independent candidate will win a **specific state's popular vote** in a US presidential election. Examples: "Will RFK Jr. win New Mexico?", "Will a candidate from another party win Maine?"

## Pre-Forecast Audit

### Step 1: Identify the Third-Party Candidate

Who is the "candidate from another party"? Possible categories:

| Type | Definition | Example (2024) |
|------|-----------|----------------|
| **Major independent** | High-name-recognition figure running without party label | Robert F. Kennedy Jr. |
| **Third-party nominee** | Candidate of a minor party with consistent ballot access | Jill Stein (Green), Chase Oliver (Libertarian), Cornel West (Independent) |
| **Write-in candidate** | No formal ballot status, negligible vote share | Various |
| **Multiple candidates** | Question may aggregate "any non-D/R candidate" | Combined vote share of all third-party candidates |

### Step 2: Classify the State's Partisan Floor

Use [[domains/usa/concepts/state-electoral-reliability/_concept]] to classify the state:

- **Safe D** (D+15+) or **Safe R** (R+15+): Third-party candidate would need to overcome a 30+ point gap — structural impossibility.
- **Likely D** (D+5 to D+15) or **Likely R** (R+5 to R+15): Third-party candidate would need to capture almost all of one party's base AND the other party's base — structural impossibility because the two-party floor (45%+ or 45%+ combined) exceeds the third-party ceiling.
- **Lean D/R** (D+0 to D+5 or R+0 to R+5): More competitive in theory but still requires the third-party candidate to overcome a 45%+ party floor.
- **Tossup** (<5 points either way): Most plausible for third-party success, but still ~0% because both parties have competitive ground operations and turnout machinery that squeeze third-party support on election day.

### Step 3: Check the Historical Third-Party Ceiling for the State

Using [[domains/usa/concepts/us-third-party-ceiling/_concept]]:

- Has any third-party candidate won this state since 1968? (If yes, identify conditions — usually a sectional base or unique local circumstances.)
- What is the highest third-party vote share in this state's history?
- For New Mexico: Gary Johnson (former governor, 2016) reached 9.3% — the best-case local favorite son scenario. No third-party candidate has exceeded 20% in NM since statehood.

### Step 4: Assess the Candidate's Organizational Depth

Key indicators of whether the candidate could theoretically outperform the historical ceiling:

| Indicator | Strong Signal | New Mexico (2024) Assessment |
|-----------|--------------|------------------------------|
| Ballot access in state | On ballot | RFK Jr. was on ballot; Stein was on ballot; Oliver on ballot |
| Local campaign infrastructure | Field offices, paid staff | None of the candidates had NM field offices |
| State-level endorsements | Local politicians, media | None of significance |
| In-state polling | >15% in state-level polls | RFK Jr. peaked ~7-10% in NM early 2024 |
| Paid advertising in state | TV, digital ads | No significant ad buys |
| Previous in-state electoral success | Party has won local offices | Green Party has no NM state-level elected officials |

If 0-2 strong indicators: ceiling holds. If 3-4: marginal case — still unlikely but worth monitoring. If 5-6: ceiling breach theoretically possible.

### Step 5: Check for Withdrawal Risk

Third-party candidates frequently withdraw late in the cycle:

- **RFK Jr. 2024**: Withdrew August 23, polled ~5-7% at time of withdrawal in NM
- **Gary Johnson 2016**: Did not withdraw, polled 9.3% in NM (best in nation)
- **Historical pattern**: 2/4 major third-party candidates (2000-2024) who polled >5% at any point withdrew or significantly downscaled before election day

If the candidate has a plausible path to withdrawal (endorsement deal, financial exhaustion, personal factors), adjust probability downward.

### Step 6: Calibrate Probability

| Scenario | P(Third-party wins state) | Rationale |
|----------|--------------------------|-----------|
| Genuine third-party nominee, any state | <0.5% | Structural ceiling; no state won since 1968 |
| Major independent polling 10-15%, Likely D/R state | <0.5% | Partisan floor alone exceeds what 3rd party can reach |
| Former governor of state runs as independent | 1-3% | Best-case local scenario; Johnson 2016 (9.3%) is ceiling proof |
| Tossup state + major independent polling 15-20% | 2-5% | Theoretically possible but historically never happened |
| Sectional/regional candidate with deep base (Wallace 1968 pattern) | 5-25% for states in region | Only replicable by a candidate with strong regional identity |
| Multiple third-party candidates dividing non-major vote | <0.1% | Fragmentation makes it even harder for any one to reach a plurality |

For New Mexico specifically: **P(third-party wins) ≈ 0%**. Even with a favorite-son former governor reaching 9.3% (Johnson 2016), the state's D+8 PVI and combined major-party floor of ~85-90% leave no realistic path to a third-party plurality.

### Step 7: Consider the "Combined Third-Party" Caveat

Some questions aggregate "any candidate from another party" — meaning the combined vote of all third-party candidates matters, not just one. This makes the outcome slightly more likely:

- The combined third-party vote in New Mexico (RFK Jr. + Stein + Oliver + others) was estimated at 3-5% in final polls.
- Even combined, this is a structural impossibility for winning a plurality.
- The aggregate third-party question resolves NO unless one candidate specifically pulls ahead of both major parties.

**Rule of thumb**: Combined third-party vote share correlates with individual candidate viability but is not a meaningful path to winning. The two major-party candidates always account for 90%+ of the vote in any state unless a major-party candidate collapses entirely.

### Step 8: Document the Structural Rationale

Example:

> New Mexico is a Likely D state (D+5.7 in 2024, D+8 PVI) with a strong Democratic floor (~45%) and a substantial Republican floor (~40-45%). No third-party candidate has won a US state since George Wallace in 1968. Even Gary Johnson — a former two-term NM governor running in his home state — only reached 9.3% in 2016. The 2024 third-party candidates (RFK Jr., Stein, Oliver) lack the organizational infrastructure, ballot-access footprint, and on-the-ground campaigning to approach even 15% in any state. Combined third-party vote share in New Mexico was estimated at 3-5%. For a third-party candidate to win New Mexico, they would need to simultaneously win over ~20% of Democrats AND ~20% of Republicans — a coalition that has never existed in US electoral history. P(third-party wins NM) < 0.5%.

## Post-Forecast Reflection

After outcome known, verify:

- What was the combined third-party vote share in the state?
- Did any candidate approach the historical ceiling?
- Was the withdrawal risk correctly assessed?
- Add the forecast to [[domains/usa/concepts/us-third-party-ceiling/_concept]] Validated By table.
- Add the forecast to [[domains/east-asia/concepts/third-party-ceiling-fptp/_concept]] Validated By table (the east-asia concept is the cross-domain parent and should accumulate global FPTP confirmations).

## Wikilinks

[[domains/usa/concepts/us-third-party-ceiling/_concept]] [[domains/east-asia/concepts/third-party-ceiling-fptp/_concept]] [[domains/usa/concepts/state-electoral-reliability/_concept]] [[domains/usa/threads/2024-us-presidential-election/_thread]] [[domains/usa/entities/new-mexico]] [[domains/usa/entities/robert-f-kennedy-jr]] [[domains/usa/entities/jill-stein]] [[domains/usa/entities/chase-oliver]]
