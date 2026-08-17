---
type: procedure
tags: [procedure, mena, ceasefire]
title: "Ceasefire Announcement Forecast Procedure"
slug: ceasefire-announcement-forecast
domain: mena
status: active
owner: hermes-agent
date: 2026-05-20  # Updated: Added Phase 0.3 (multi-ceasefire "first announce" wording ambiguity)
related_concepts:
  - "[[domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept]]"
  - "[[domains/mena/concepts/public-framework-announcement-commitment/_concept]]"
  - "[[domains/global/concepts/ceasefire-announcement-ratification-gap]]"
  - "[[domains/global/concepts/forecast-resolution-criteria-gotchas]]"
related_threads:
  - "[[domains/mena/threads/israel-hamas-war-ceasefire/_thread]]"
---

# Ceasefire Announcement Forecast Procedure

## When to Use

When a forecasting question asks whether a party (typically Israel, but
generalizable to any conflict party) will announce a ceasefire agreement with
an adversary within a specific date window.

## Phase 0: Resolution Criteria Analysis

### 0.1 Identify WHAT the Question Asks

- [ ] Does the question ask about "ceasefire" (effective date) or "ceasefire
      announcement" (announcement date)? These are DIFFERENT triggers.
- [ ] Does the question specify WHICH party must announce? ("Israel announces",
      "Hamas agrees", "ceasefire agreement between...")
- [ ] Does the question's window have a specific start and end time?

### 0.2 Identify WHO Has Already Acted

⚠️ **CRITICAL: CHECK FOR PRIOR ANNOUNCEMENTS** — This is the most commonly
missed step and the leading source of error on ceasefire-announcement questions.

Before making any prediction, determine:

- [ ] Has a ceasefire FRAMEWORK been accepted in principle by both parties?
- [ ] Has a MEDIATOR (US, Qatar, Egypt, UN) publicly announced that a deal
      has been reached?
- [ ] Has the PARTY IN QUESTION made a formal announcement, or only confirmed
      a mediator's framework announcement?
- [ ] If the mediator announced but the party did not: the question is about
      the party's FORMAL ANNOUNCEMENT/RATIFICATION, not about a new negotiation.

**Decision rule**: If the mediator has already publicly announced the framework
but the party has not formally ratified/announced, load
[[domains/mena/concepts/public-framework-announcement-commitment/_concept]]
and assign P(YES) ~0.90-0.95 for the party's follow-through within 1-3 days.
The public framework announcement activates a commitment trap: the party cannot
reject without damaging its relationship with the superpower patron.

**If the mediator has NOT announced**: Proceed with Phase 1 (standard ceasefire
forecasting), which has MUCH lower probability (0.01-0.05 for short windows).

### 0.3 Interpret "First" / "Next" / "Initial" Wording

⚠️ **CRITICAL: RESOLUTION-CRITERIA GOTCHA** — When the question title uses
"first announce" but PRIOR ceasefires have already occurred in the same
conflict, the wording is ambiguous and must be resolved before forecasting.

By October 2025, Israel had already announced ceasefires in Nov 2023 and
Jan 2025. A question title "Will Israel first announce ceasefire on
October 8?" could be misinterpreted as "first ever" → automatic NO.

**Resolution guidance**:

- [ ] Read the resolution text. Does it use "next date," "first," or "initial"?
      "Next date" means chronological next in sequence; "first" in the title
      usually means "initiating announcement of this deal" (vs. ratification).
- [ ] Count prior ceasefire announcements by the same party in the same
      conflict. If >0, "first" cannot mean "first-ever."
- [ ] Check for parallel Polymarket questions with adjacent dates (e.g., Oct 8
      and Oct 9). Their existence confirms the market is testing date-type
      precision (announcement vs. ratification), not historical-first status.
- [ ] Apply the announcement-vs-ratification distinction from
      [[domains/global/concepts/ceasefire-announcement-ratification-gap]]
      regardless of title wording. The "first announce" date is almost always
      the party's initial public confirmation of agreement, NOT the cabinet
      ratification date.
- [ ] See [[domains/global/concepts/forecast-resolution-criteria-gotchas]]
      entry #9 for the canonical example.

**Rule**: Treat the question title as a hint, the resolution text as law.
If they diverge, the resolution text controls. When prior ceasefires exist,
"first announce [date]" means "the initiating official announcement of the
current deal occurs on [date]" — not "the first time ever the party
announces a ceasefire."

## Phase 1: Pre-Flight Check

### 1.1 Identify the Parties

- [ ] Who are the two parties to the ceasefire?
- [ ] Is one party a US ally or client state?
- [ ] Is one party a non-state militant group?
- [ ] Are there mediating parties (Qatar, Egypt, UN, Turkey)? Are they
      actively engaged?

### 1.2 Check for a US Presidential Transition

- [ ] Is a US presidential transition in progress (between election day and
      inauguration day)?
- [ ] If YES: Load [[domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept]]
      and score the leading/mid/late indicators. Apply probability multipliers.
- [ ] If NO: Baseline ceasefire probability is lower. Proceed with standard
      framework below.

### 1.3 Check for Pre-Negotiated Frameworks

- [ ] Does a ceasefire framework already exist that has been accepted by one
      party and rejected by the other?
- [ ] If YES: The question is about political activation, not negotiation from
      scratch. The key variable is: what changed to make the holdout party
      accept?

### 1.4 Assess Leadership Decapitation Status

- [ ] Has either party suffered a recent leadership decapitation (killing or
      removal of the key decision-maker)?
- [ ] If YES: Decapitation typically increases ceasefire probability because
      the new leadership may be more pragmatic, but can also decrease it if the
      successor is more hardline.

## Phase 2: Structural Analysis

### 2.1 Each Party's Incentive Matrix

For each party, assess:

| Factor | Weight | Scoring |
|--------|--------|---------|
| Military objectives achieved | High | 0=no, +1=mostly, +2=fully |
| War exhaustion (domestic) | Medium | +1 for each: public opinion turning, economic pressure, casualty sensitivity |
| Hostage/ POW leverage dynamic | Medium | 0=no hostages, +1=one side has hostages, +2=hostages held by adversary creating domestic pressure for deal |
| External pressure on ally | High | +1 per major external actor (US, EU, regional) applying public pressure |
| Coalition/ domestic political constraint on leader | Medium-High | -1 per identified constraint (far-right coalition partner, parliamentary opposition) |

### 2.2 The Ceasefire Equation

```
P(ceasefire) = baseline * transition_multiplier * decapitation_multiplier *
                alignment_score
```

Where:
- **baseline**: 0.05 for a conflict >6 months old with no ceasefire in the
  question window
- **transition_multiplier**: 4-10x if transition-window indicators are positive
  (use the concept file to score)
- **decapitation_multiplier**: 1.5-3x if the main obstacle to a deal has been
  removed through leadership decapitation
- **alignment_score**: 0.5-2.0 based on how many parties have strong incentives
  to reach a deal NOW (1.0 = neutral, >1 = aligned toward deal, <1 = misaligned)

### 2.3 Countervailing Case (Dual-Frame Analysis)

Before finalizing, articulate the strongest argument AGAINST the ceasefire:

- [ ] What prevents Party A from accepting?
- [ ] What prevents Party B from accepting?
- [ ] What has changed since the last failed attempt?
- [ ] Is the holdout party's objection structural (existential threat to
      regime) or tactical (terms, sequencing)?

## Phase 3: Calibration

### 3.1 Historical Analog Calibration

Check analogous ceasefire announcements:

- Previous ceasefires in the same conflict (e.g., Nov 2023 temporary ceasefire
  for Israel-Hamas) — what timelines and conditions applied?
- Ceasefires during US presidential transitions (load the concept for
  canonical case)
- Ceasefires following leadership decapitation

### 3.2 Market-Adjacent Information

- [ ] What are Polymarket / other prediction market prices for this question?
- [ ] Does market price diverge from your estimate? If so, what information
      might the market have that you don't?
- [ ] Check the `_macro_gaps.md` for domain-specific biases identified in
      prior questions.

### 3.3 Final Probability Statement

Produce a single probability estimate with:
- Base rate reference
- Key assumptions and their sensitivity
- The countervailing case (what would need to be true for the NO outcome)

## Phase 4: Post-Forecast Reflection

After the question resolves:

1. Was your forecast correct or incorrect?
2. What vault content helped or was missing?
3. Were there structural factors (transition, decapitation, mediator role,
   public framework announcement commitment trap) that were under- or
   over-weighted?
4. Create/update files: thread entries, entity stubs, concept refinements.
