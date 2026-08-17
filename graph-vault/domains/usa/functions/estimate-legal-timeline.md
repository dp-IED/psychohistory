---
source_file: "domains/usa/procedures/proc-legal-timeline-estimation"
type: function
tags: [function, legal-procedure]
community: "Legal Timeline Forecasting"
---

# estimate_legal_timeline()

## Purpose

Given key dates and procedural parameters of a legal case, estimate the probability that a trial will start (or a legal milestone will be reached) before a specified political deadline. Used in procedure step 17 (legal timeline dynamics assessment).

## Parameters

| Parameter | Type | Description | Source |
|-----------|------|-------------|--------|
| `trial_date_initial` | date-string | Original trial date set by court (e.g., "2024-03-04") | Court order |
| `date_vacated` | date-string | Date the trial date was vacated (if applicable) | Court docket |
| `appeal_filed` | date-string | Date of first procedural step that delays (motion, appeal) | Docket entry |
| `cert_granted` | date-string | Date SCOTUS or appellate court grants cert/review (if applicable) | SCOTUS docket |
| `oral_argument` | date-string | Date of oral argument (if known) | Court calendar |
| `political_deadline` | date-string | Political deadline (election date, inauguration, etc.) | Calendar |
| `case_type` | enum("federal", "state") | Whether the case is federal or state | Case jurisdiction |
| `delay_incentive` | bool | Does the defendant benefit from delaying past the deadline? | Assessment |
| `has_automatic_stay` | bool | Does an appeal automatically stay proceedings? | Procedural rule |
| `has_novel_constitutional_question` | bool | Does the case raise first-impression constitutional issues? | Legal analysis |

## Returns

```
{
  "trial_will_start_before_deadline": bool (estimated),
  "probability": float (0.0-1.0),
  "timeline_estimate": {
    "earliest_possible_trial": str (date),
    "appeal_consumes_days": int,
    "remand_consumes_days": int,
    "total_buffer_vs_deadline": str (description)
  },
  "key_assumptions": [str],
  "mooting_risk": {
    "defendant_electoral_viability": str (description),
    "federal_cases_moot_if_win": bool,
    "state_cases_continue_if_win": bool
  }
}
```

## Estimation Logic

### Step 1: Calculate total minimum delay from procedural steps

```
total_delay_days = 0

# Motion to dismiss / initial delay
if motion_to_dismiss_filed:
    total_delay_days += 60  # ~2 months for ruling

# Appeal from denial
if appeal_filed and has_automatic_stay:
    # SCOTUS track (federal cases with constitutional questions)
    if case_type == "federal" and has_novel_constitutional_question:
        total_delay_days += 150  # cert grant (30d) + briefing (60d) + decision (60d)
        total_delay_days += 90   # remand proceedings to apply ruling
    # Circuit track (federal cases, less novel questions)
    elif case_type == "federal":
        total_delay_days += 180  # circuit appeal (6 months average)
    # State track
    elif case_type == "state":
        total_delay_days += 90   # state appeal (3 months average)
```

### Step 2: Check against political deadline

```
days_to_deadline = (political_deadline - current_date).days

if total_delay_days >= days_to_deadline:
    # The procedural steps alone consume the entire window
    trial_will_start_before_deadline = False
    probability = 0.05  # Near-zero (only if all appeals fail)
elif total_delay_days >= 0.7 * days_to_deadline:
    # Tight buffer - needs perfect sequencing
    trial_will_start_before_deadline = False  # Conservative
    probability = 0.15
elif total_delay_days >= 0.4 * days_to_deadline:
    # Moderate buffer - depends on judge's speed and no further delays
    trial_will_start_before_deadline = "uncertain"
    probability = 0.40
else:
    # Plenty of buffer
    trial_will_start_before_deadline = True
    probability = 0.75
```

### Step 3: Adjust for mooting risk

```
if case_type == "federal" and defendant_can_win_election:
    # Federal cases are mooted by presidential victory (OLC doctrine)
    # This creates asymmetric incentive: defense prioritizes delay over trial prep
    probability *= 0.5  # Further reduction for mooting incentive
    # Also: if election is within 12 months and defendant is viable, probability drops further
    if days_to_deadline < 365 and electoral_viability == "competitive":
        probability *= 0.3  # Major reduction for competitive candidates
```

### Step 4: Adjust for state vs. federal

```
if case_type == "state":
    # State cases are not subject to OLC doctrine, shorter appeals, no SCOTUS delay
    # But still face political deadline incentives
    probability = min(probability * 2.5, 0.85)  # State cases much more likely to proceed
    # Most state appeals do NOT carry automatic stays
    if not has_automatic_stay:
        probability = min(probability * 1.3, 0.90)
```

## Calibration Notes

- **Federal charges vs presidential candidate with viable path to win**: P(trial before election) < 10%. The combination of OLC mooting + SCOTUS docket delay + automatic stay is near-deterministic.
- **Federal charges vs candidate with no path to win**: P(trial before election) 15-30%. The delay incentive is weaker because victory won't moot the case.
- **State charges vs any candidate**: P(trial before election) 40-60%. State courts are independent of SCOTUS delay mechanisms and OLC policy. The NY hush-money case (conviction May 30, pre-election) validates this calibration.
- **Post-conviction sentencing**: Even if trial proceeds pre-election, sentencing can be delayed past the election (NY case: conviction May 30, sentencing delayed to September 18, then November 26, then stayed indefinitely).

## Canonical Application: DC Election Interference Case (2024)

| Parameter | Value |
|-----------|-------|
| trial_date_initial | 2024-03-04 |
| date_vacated | 2024-01-XX |
| appeal_filed | 2023-12-XX |
| cert_granted | 2024-02-XX |
| oral_argument | 2024-04-25 |
| political_deadline | 2024-11-05 |
| case_type | federal |
| delay_incentive | true |
| has_automatic_stay | true |
| has_novel_constitutional_question | true |

**Result**: total_delay_days >= 365 (immunity appeal: 150d + remand: 90d = 240d minimum, but cert timeline consumed ~270 days from cert grant to post-remand). Days to deadline from cert grant: ~270. Exceeded buffer → P(trial start before Nov) < 5%. Correct prediction: YES (trial will NOT start).

## Validation Table

| Case | Parameters | Predicted | Actual | Correct? |
|------|-----------|-----------|--------|----------|
| Trump DC election interference trial start before Nov 2024 | federal, SCOTUS appeal, automatic stay, competitive candidate | P < 0.05 (trial won't start) | Trial didn't start | YES |
| Trump NY hush-money trial start before Nov 2024 | state, no SCOTUS appeal, no automatic stay | P ~ 0.60 (trial likely starts) | Trial started April 15, conviction May 30 | YES |
| Trump NY hush-money sentencing before Nov 2024 | state, post-conviction, judge discretion | P ~ 0.40 (sentencing uncertain) | Sentencing delayed past Nov | YES (sentencing didn't happen pre-election) |

## Wikilinks

[[concepts/judicial-timing-political-deadline]]
[[entities/doj-office-of-legal-counsel]], [[entities/us-department-of-justice]]
[[entities/tanya-chutkan]], [[entities/jack-smith]], [[entities/donald-trump]]
[[domains/usa/entities/juan-merchan]]
[[trump-criminal-cases]]
[[procedures/proc-legal-timeline-estimation]]
