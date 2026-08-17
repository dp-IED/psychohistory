---
type: procedure
tags: [procedure]
title: "Legal Timeline Estimation Procedure"
slug: proc-legal-timeline-estimation
domain: usa
---

# Legal Timeline Estimation Procedure

## When to Use

When the forecast question asks whether a legal proceeding (trial start, verdict, sentencing, ruling) will reach a milestone before a political deadline (election, inauguration, transition, end of term).

## Steps

### Step 1: Establish the Baseline Timeline

1. Record the **original trial date** set by the court. If no date was set, record the expected timeline based on normal case progression for that jurisdiction.
2. Record the **date the trial date was vacated or postponed** (if applicable). A vacated date with no new date set is the strongest possible signal that the trial will not start before the political deadline.
3. Record the **political deadline date** (election, inauguration, end of term).

### Step 2: Identify All Active Delay Mechanisms

Check the docket for:

- **Interlocutory appeals** — especially on immunity, qualified immunity, or jurisdictional questions
- **Motions to dismiss** — especially raising novel constitutional or procedural questions
- **Discovery disputes** — document requests, privilege claims requiring court intervention
- **Recusal motions** — challenges to the judge's impartiality
- **Venue change motions** — requesting transfer to a different district
- **Continuance requests** — routine requests for more trial preparation time

For each mechanism found, record whether it carries an **automatic stay** of proceedings. The presence or absence of automatic stays is the single most consequential variable.

### Step 3: Map the Appellate Structure

For federal cases:

- **Which appellate path?** Direct to SCOTUS (if novel constitutional question) or Circuit Court first?
- **Does an automatic stay apply?** Most interlocutory appeals on immunity questions carry automatic stays.
- **What is the expected timeline?**
  - SCOTUS cert grant to oral argument: 60-90 days
  - SCOTUS oral argument to decision: 60-90 days
  - SCOTUS mandate issuance (after decision): 25 days minimum
  - Circuit Court appeal (full cycle): 6-12 months
- **Remand time**: After appellate ruling, 60-120 days for district court to apply the new framework.

For state cases:

- **Appellate path**: State supreme court or intermediate appellate court
- **Expected timeline**: 3-6 months (typically faster than federal)
- **Automatic stays**: Rare for state appeals in criminal cases — most state appeals do NOT automatically stay proceedings.

### Step 4: Calculate Total Minimum Delay

Use the [[functions/estimate-legal-timeline]] function with the parameters gathered in Steps 1-3.

```
trial_start_estimate = estimate_legal_timeline(
    trial_date_initial=original_date,
    date_vacated=vacated_date,
    appeal_filed=appeal_date,
    political_deadline=deadline_date,
    case_type="federal" or "state",
    delay_incentive=true,
    has_automatic_stay=true,
    has_novel_constitutional_question=true
)
```

### Step 5: Assess Mooting Risk

Two distinct mooting mechanisms:

1. **OLC Doctrine (federal cases only)**: If the defendant wins the presidency, DOJ internal policy prohibits federal prosecution of a sitting president. This moots ALL federal cases regardless of their pre-election status. The probability of mooting is equal to the defendant's probability of winning the election.

2. **SCOTUS Constitutional Immunity**: Even if the case survives to trial, the SCOTUS immunity ruling in Trump v. United States may bar prosecution of certain official-act charges. This is narrower than OLC doctrine and only protects against charges involving official presidential conduct.

Record both assessments.

### Step 6: Distinguish State vs. Federal

| Dimension | Federal | State |
|-----------|---------|-------|
| OLC doctrine applies | YES | NO |
| SCOTUS delay available | YES | NO |
| Automatic stays common | YES (especially immunity appeals) | NO (rare) |
| Typical appellate timeline | 6-12 months | 3-6 months |
| Baseline P(trial before political deadline if defendant wants delay) | <10% | 40-60% |

### Step 7: Document the Causal Chain

Write the reasoning as a step-by-step chain:
1. Defendant has X incentive to delay (electoral mooting possible? Y/N)
2. Available delay mechanisms: [list]
3. Appellate timeline covers: [start] to [end]
4. Total delay: [X] days vs [Y] days to deadline
5. Verdict: Trial [will/will not] start before deadline
6. Key uncertainties: [list]

### Step 8: Cross-Reference with Existing Vault

Before finalizing, check:
- Does the [[concepts/judicial-timing-political-deadline]] concept exist and is it applicable?
- Does the [[trump-criminal-cases]] thread have a precedential event that matches?
- Are entity stubs for all named legal actors (prosecutor, judge, defendant) created?
- Does the [[entities/us-department-of-justice]] entity document the applicable DOJ policies?
- Does the [[entities/doj-office-of-legal-counsel]] entity have the correct OLC doctrine?

## Common Errors

1. **Assuming state and federal cases follow the same timeline**: They do not. State cases are structurally faster, lack SCOTUS delay, lack OLC mooting, and rarely have automatic stays. The NY hush-money case (pre-election conviction) and DC case (no trial before election) are the canonical contrast.

2. **Forgetting the remand phase**: Even after an appellate ruling on immunity, the district court typically needs 2-4 months to apply the ruling to the specific charges. This is not "delay" in the legal sense — it is necessary judicial process — but it consumes calendar time.

3. **Underestimating cumulative delay**: Each procedural step consumes 2-4 months. With 3-4 steps in sequence, 8-14 months can be consumed. The election window (typically 10-12 months from early primary to November) can be fully consumed by a determined delay campaign.

4. **Ignoring the 60-day rule**: Even without a formal appeal, the DOJ's internal policy against election-influencing investigative steps in the final 60 days before an election creates a hard institutional deadline. Any trial or proceeding that falls within 60 days of a federal election will be opposed by the DOJ itself.

## Wikilinks

[[functions/estimate-legal-timeline]]
[[concepts/judicial-timing-political-deadline]]
[[entities/us-department-of-justice]], [[entities/doj-office-of-legal-counsel]]
[[entities/jack-smith]], [[entities/tanya-chutkan]], [[entities/us-supreme-court]]
[[trump-criminal-cases]]
[[:procedure-root]]
