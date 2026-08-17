---
type: concept
tags: [concept, elections, vp-selection, incumbency]
title: "Incumbent VP Renomination Baseline"
slug: incumbent-vp-renomination
first_observed: 1952-01-01
domain: usa
description: "The structural baseline that sitting vice presidents are almost always renominated — with documented historical exceptions and the conditions under which an incumbent VP might be dropped from the ticket."
---

# Incumbent VP Renomination Baseline

## Concept

In modern US presidential politics, a sitting vice president whose president is seeking re-election is almost certain to be renominated as the VP candidate. The baseline probability is >95%. This is one of the strongest structural regularities in US electoral politics.

### Historical Data

| Cycle | President | Sitting VP | Renominated? | Notes |
|-------|-----------|------------|------|-------|
| 1952 | Truman (declined re-election) | Alben Barkley | Not renominated | President didn't run; Barkley sought nomination but lost to Stevenson (who picked Sparkman) |
| 1956 | Eisenhower | Nixon | Yes | |
| 1960 | Nixon (as nominee, not incumbent prez) | N/A | Not applicable | |
| 1964 | Johnson | Humphrey | Yes | |
| 1968 | Johnson (withdrew) | Humphrey (won nomination) | N/A | Humphrey was president's VP but won nomination himself |
| 1972 | Nixon | Agnew | Yes | Agnew later resigned in 1973 |
| 1976 | Ford (appointed VP) | Rockefeller | Dropped | Ford dropped Rockefeller for Dole in 1976 — but Ford was never elected VP himself |
| 1980 | Carter | Mondale | Yes | |
| 1984 | Reagan | Bush | Yes | |
| 1992 | Bush | Quayle | Yes | |
| 1996 | Clinton | Gore | Yes | |
| 2004 | Bush | Cheney | Yes | |
| 2012 | Obama | Biden | Yes | |
| 2020 | Trump | Pence | Yes | |
| 2024 | Biden | Harris | Yes* | Biden withdrew before general election; Harris then became presidential nominee and chose Walz |

**Pattern**: No sitting VP has been dropped from a re-election ticket since Rockefeller (1976), who was an appointed VP, not elected. The last elected VP dropped from a re-election ticket was John Nance Garner (1940, FDR's third term), a distinct historical context.

### Exceptions and Their Conditions

**Exception 1: Rockefeller (1976)** — Appointed VP (25th Amendment), president Ford was also appointed (not elected), facing primary challenge from Reagan. Dropping Rockefeller was a base-consolidation move in a fractured electoral context.

**Exception 2: Barkley (1952)** — Truman declined to run; Barkley was seeking the presidential nomination himself. Not a case of a sitting president dropping their VP — Truman didn't run at all.

**Exception 3: Hannibal Hamlin (1864)** — Lincoln dropped Hamlin for Andrew Johnson (a War Democrat) in a unity ticket during Civil War. Pre-modern party convention era.

### Conditions for VP Replacement

A sitting VP is at meaningful risk of being dropped only under these conditions:
1. **President declines re-election** → VP may seek presidency themselves (Barkley 1952, Humphrey 1968)
2. **Appointed VP** (25th Amendment) → lower retention probability, as the VP wasn't on the original ticket
3. **VP is a political liability** → scandal, incompetence, factional hostility (required Agnew 1973 resignation, but he was still on 1972 ticket)
4. **Fundamental strategic reorientation** → Lincoln 1864 (Civil War unity government)

## Forecasting Application

### For "Will [person] be the VP nominee?" style questions

When the VP in question could be the sitting VP:
1. **Check if the sitting VP is running for re-election with same president**: If yes → P(renomination) >95%
2. **Exception check**: Is the VP appointed (25th Amendment)? Did the president decline re-election? Is the VP in a major scandal?
3. **If the president withdraws**: The VP-picks-a-VP case falls under [[domains/usa/concepts/gender-balancing-ticket-composition/_concept]] — not this concept

### For "Will another [demographic] be VP?" style questions

When the exclusion list includes the sitting VP:
- The sitting VP being on the exclusion list makes the question structurally harder to resolve YES: the default, high-probability outcome is already on the list
- To get a YES, the president would need to either (a) drop their sitting VP (extremely rare) or (b) withdraw and have the new nominee pick a different person in that demographic
- Path (a) has <5% probability. Path (b) requires a presidential withdrawal first, then gender-balancing constraints.

### Specific Application: "Another woman" (2024 Democratic VP)

- The exclusion list included Kamala Harris, the sitting VP
- Under the incumbent VP renomination baseline, Biden retaining Harris was >95% probable
- Therefore, the "another woman" scenario required Biden to either: (1) drop Harris (violating the baseline) OR (2) withdraw and have the new nominee's VP be a woman not on the list
- Path (1): <5% — no historical precedent for dropping an elected sitting VP
- Path (2): Biden withdrawal was a real possibility (the [[incumbent-withdrawal-cascade]] was building), but even then, gender-balancing made a woman VP from a female nominee <5% likely
- Net: P(another woman) < 5%

## Relationship to Other Concepts

- [[domains/usa/concepts/gender-balancing-ticket-composition/_concept]] — applies when sitting VP becomes presidential nominee
- [[domains/usa/concepts/veepstakes-electoral-signal/_concept]] — framework for understanding why a sitting VP might or might not be replaced
- [[domains/usa/concepts/comprehensive-exclusion-list-forecast]] — the sitting VP's presence on the exclusion list is the strongest single factor predicting NO

## Sources
- Historical VP selection data (1940-2024)
- Congressional research reports on vice presidential succession
- Election cycle analysis (Pew, 270towin)
