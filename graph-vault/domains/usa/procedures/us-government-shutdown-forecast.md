---
type: procedure
tags: [procedure, usa, shutdown]
title: "US Government Shutdown Forecast"
slug: us-government-shutdown-forecast
domain: usa
related_concepts:
  - cr-governance-shutdown-dynamics
---

# US Government Shutdown Forecast

## When to Use

Apply this procedure whenever a forecast question asks whether a US government shutdown will occur (or whether funding will be passed before a deadline). The procedure covers the structural dynamics of CR-governance in the 118th-119th Congresses and is calibrated for the 2024-2026 period.

## Pre-Forecast Checklist

Before making any US government shutdown forecast, complete these steps:

### 1. Identify the Exact Deadline

- [ ] What is the funding deadline? (Typically Sep 30, Dec 20, or Mar 14)
- [ ] Is this a fiscal-year transition (Sep 30) or a CR extension?
- [ ] Does the deadline fall in a lame-duck session (post-election)?

### 2. Map the Institutional Variables

Apply the [[domains/usa/concepts/cr-governance-shutdown-dynamics/_concept]] framework:

- [ ] **F (Freedom Caucus defections)**: 20-30 baseline. Is the CR clean or partisan?
- [ ] **D (Jeffries posture)**: Has Jeffries indicated support or opposition? (Most important variable.)
- [ ] **E (External intervention)**: Has Musk or Trump weighed in? If Musk opposes, 20-30% higher shutdown probability.
- [ ] **J (Johnson procedure)**: Will the bill come under suspension-of-rules? If yes, bipartisan deal locked in.
- [ ] **T (Time-bound policy clocks)**: Are any policies expiring with the funding deadline? (ACA subsidies, debt ceiling, farm bill.)

### 3. Check Entity Coverage

- [ ] Does an entity stub exist for Mike Johnson? ([[domains/usa/entities/mike-johnson]])
- [ ] Does an entity stub exist for Hakeem Jeffries? ([[domains/usa/entities/hakeem-jeffries]])
- [ ] Does an entity stub exist for the House Freedom Caucus? ([[domains/usa/entities/house-freedom-caucus]])
- [ ] If Musk/Trump are mentioned in the question context, do entity stubs exist? ([[domains/usa/entities/elon-musk]])

### 4. Read the Thread

- [ ] Read [[domains/usa/threads/us-government-shutdown-crises/_thread]] — check if the current deadline's CR cycle is already documented.
- [ ] Read the most recent quarter file in timeline/ for budget developments.

### 5. Check the Calendar

- [ ] Is this CR deadline after a presidential election? Lame-duck sessions have different dynamics (Trump post-election leverage, expiring administration).
- [ ] Is there a new Congress starting soon? (New Congress on Jan 3 changes committee composition and leadership.)
- [ ] **Does the question have a broad time window (multi-month, spanning multiple CR deadlines)?** If the question asks "by end of year" or covers 90+ days, it likely spans 2+ deadlines. THIS CHANGES THE BASELINE fundamentally (see Step 6b).

## Forecasting Algorithm

### Step 1: Baseline Probability

Start with the structural baseline: **NO shutdown = 65-75%** (the default outcome in most funding cycles).

Reason: In the 2023-2025 period, funding always passed eventually, even if late. The system has a strong bias toward resolution because:
- Both parties prefer funding to shutdown.
- OMB discretion allows technical lapses without shutdown activation.
- Bipartisan coalitions form reliably at the last minute.

### Step 2: Adjust for Jeffries Posture

- **Jeffries supports the CR**: Add 25-30% to "no shutdown." Passage probability >95%.
- **Jeffries opposes the CR**: Subtract 30-40% from "no shutdown." Shutdown probability 40-60%.
- **Jeffries is silent/neutral**: No adjustment. Default structural baseline applies.

### Step 3: Adjust for Policy Clock Conflicts

- **Time-bound policy expiration coincides with deadline AND parties diverge on it**: Subtract 25-35% from "no shutdown." Doubles shutdown risk.
- **No policy clock conflict**: No adjustment needed.

### Step 4: Adjust for External Intervention

- **Musk actively opposes the CR**: Subtract 15-25% from "no shutdown." His intervention killed a deal in Dec 2024.
- **Trump publicly opposes the CR**: Subtract 10-20% from "no shutdown." He influences House Republican votes.
- **Neither engaged**: No adjustment.

### Step 5: Adjust for Freedom Caucus Unity

- **HFC unified in opposition (>25 defections)**: Subtract 10-15% from "no shutdown" (reduces Speaker's margin, increases drama, may delay passage past deadline).
- **HFC split (<15 defections)**: No adjustment.

### Step 6: Check Resolution Text

**Critical**: The Polymarket resolution text may define "shutdown" differently from government practice.

- [ ] Does the resolution require OMB to activate shutdown procedures? If yes, a clean CR passing late but before OMB action is NOT a shutdown.
- [ ] Does the resolution consider any funding lapse (even hours) as a shutdown? If yes, a late-passing bill IS a shutdown.
- [ ] Does the resolution reference a specific trigger (executive order, presidential action)? If yes, the shutdown definition is narrower.

**Adjustment**: If the resolution defines shutdown broadly (funding lapse = shutdown) AND the deadline is during the 2024-2025 pattern of late-passing bills: add 15-25% to "yes shutdown."

### Step 6b: Adjust for Broad Time Windows (Multi-Deadline Questions)

**Critical distinction**: Questions with a narrow window (1-45 days, centered on a specific deadline) differ fundamentally from questions with a broad window (90+ days, multiple CR deadlines, "by end of year").

**Mechanism**: A broad window captures multiple independent failure points. If there are two CR deadlines during the window (e.g., Sep 30 AND Dec 20), the probability of at least one funding-lapse event is:

p(shutdown in window) = 1 - (1 - p(shutdown at deadline 1)) * (1 - p(shutdown at deadline 2))

Even if each individual deadline has only a 30% chance of producing a shutdown event, a two-deadline window has 1 - (0.7 * 0.7) = 51% chance — substantially higher than either individual deadline.

**Additionally**, broad windows capture technical funding lapses that a narrow window around a single deadline might miss. The Dec 2024 episode involved a funding lapse of ~hours (between CR expiration and bill signing). A narrow question ending Nov 30 would have missed this entirely.

**Checklist**:

- [ ] Does the question window span 90+ days or multiple CR deadlines? (Sep 30, Dec 20, Mar 14 are the canonical deadlines.)
- [ ] How many CR deadlines fall within the window? Count them — each is an independent failure point.
- [ ] Does the window include a post-election lame-duck period? If yes, the post-election disruption multiplier applies to each deadline within the window.
- [ ] Does the resolution text define "shutdown" broadly? If yes, even a technical funding lapse at any deadline counts.

**Adjustment**: For broad windows (90+ days or 2+ CR deadlines):

- **2 deadlines in window**: Subtract 15-25% from "no shutdown" (relative to single-deadline baseline). This accounts for the compound probability of at least one lapse.
- **3+ deadlines in window**: Subtract 25-35% from "no shutdown."
- **Broad window + broad resolution definition + post-election period**: Baseline shifts to 50-60% YES.
- **Narrow window (single deadline, <45 days)**: No adjustment beyond standard algorithm.

### Step 7: Aggregate

```
p(NO_shutdown) = 0.70 (baseline)
p(NO_shutdown) += 0.25 * I(Jeffries supports)  # if yes
p(NO_shutdown) -= 0.35 * I(Jeffries opposes)   # if yes
p(NO_shutdown) -= 0.30 * I(Policy clock conflict)
p(NO_shutdown) -= 0.20 * I(Musk opposes)
p(NO_shutdown) -= 0.15 * I(Trump opposes)
p(NO_shutdown) -= 0.15 * I(HFC unified >25)
p(NO_shutdown) += 0.20 * I(Broad resolution definition)
p(NO_shutdown) -= 0.20 * I(Broad time window, 2 deadlines)
p(NO_shutdown) -= 0.30 * I(Broad time window, 3+ deadlines)
```

Clamp to [0.05, 0.95].

**Final conversion**: p(YES_shutdown) = 1 - p(NO_shutdown).

### Step 8: Document Reasoning

Write the forecast with explicit reference to each variable:

- F count and composition
- Jeffries posture (with source)
- External actor positions
- Procedure track (suspension vs. regular)
- Policy clock check
- Resolution text analysis
- The specific cascade stage the process is currently in

## Common Pitfalls

1. **Confusing "funding lapse" with "shutdown"**: OMB can continue operations when passage is imminent. Check the resolution text carefully.

2. **Assuming last CR's pattern repeats mechanically**: Each deadline has different variables. The Dec 2024 Musk intervention was unprecedented. The 2025 shutdown was triggered by a policy clock conflict. Avoid extrapolating from a single case.

3. **Counting Freedom Caucus members who vote "no" as shutdown-instigating**: 15-25 "no" votes is structural and does NOT prevent passage if Jeffries supports. The vote count only matters in the context of the procedure track.

4. **Assuming Johnson will replicate McCarthy's fate**: Johnson has survived multiple bipartisan CRs. The motion-to-vacate threat is real but the threshold for actually deposing a Speaker is higher after the 22-day chaos of Oct 2023.

5. **Overweighting drama, underweighting structure**: The 24-hour news cycle amplifies each CR crisis. The structural fact is that funding has always passed eventually in the 2023-2025 period. The drama does not change the underlying incentives for resolution.

## Post-Forecast Remediation

After each shutdown forecast (whether correct or wrong):

1. Update [[domains/usa/threads/us-government-shutdown-crises/_thread]] with the current deadline's events.
2. Update entity files for any actors whose role changed.
3. If a new variable emerged (e.g., new external actor, new procedural tactic), update the concept file.
4. Document the forecast outcome and what variables the forecast got right/wrong.

## Wikilinks

- [[domains/usa/concepts/cr-governance-shutdown-dynamics/_concept]]
- [[domains/usa/threads/us-government-shutdown-crises/_thread]]
- [[domains/usa/entities/mike-johnson]]
- [[domains/usa/entities/hakeem-jeffries]]
- [[domains/usa/entities/house-freedom-caucus]]
- [[domains/usa/entities/office-of-management-and-budget]]
- [[domains/usa/entities/shalanda-young]]
- [[domains/usa/entities/elon-musk]]
