## Report: What Changed and Why

### Diagnosis

gold_28 is a **validation case** — the vault previously answered the SAME question incorrectly (gold_50: predicted NO for Israel ceasefire by Sunday, actual YES). The reflection loop after gold_50 added the WHO-announces distinction, the commitment trap concept, and the ceasefire-announcement-forecast procedure. On gold_28, the vault's infrastructure worked: the prediction was YES at 0.93 confidence, and the reasoning directly referenced vault concepts.

This proves the reflection methodology functions. But two novel gaps were identified even in this correct prediction.

### Gap 1: Dual-Presidential Endorsement (most important)

The raw output notes "Biden and Trump jointly announced" — this was a **historically unprecedented** phenomenon (no previous US transition had the outgoing AND incoming president co-announce a ceasefire). The standard commitment trap (P ~0.90-0.95) understates the Jan 2025 case because the dual endorsement:
- Binds BOTH administrations to the deal
- Eliminates the party's hope of better terms under the next president
- Makes rejection mean opposing both presidents simultaneously

**Created**: `domains/mena/concepts/dual-presidential-endorsement-ceasefire/_concept.md` (11,972 bytes) — full documentation of the mechanism, Bayesian priors (P ~0.97-0.99 vs standard ~0.90-0.95), canonical case analysis, observable indicators, and application to forecasting.

### Gap 2: Day-of-Week Deadline Metonym

The question says "by Sunday" but the real deadline was the Jan 20 inauguration. Sunday (Jan 19) was the day before inauguration and coincidentally the ceasefire effective date — making "Sunday" a **metonym** for the political deadline. The vault didn't document this dissociation pattern.

**Updated**: `domains/global/concepts/political-deadline-ceasefire.md` — added "Deadline Metonym Extension" section with: a dissociation pattern table, 5-step pre-forecast check (map date, check ±3 day political deadlines, determine what "by" means, separate announce vs effective dates, check day's independent significance), and gold_28/gold_50 contrast.

### Gap 3: Missing Validation Entry

The concept files listed gold_50 as an error but not gold_28 as evidence the fixes worked.

**Updated**: `domains/global/concepts/ceasefire-announcement-ratification-gap.md` — added gold_28 validation entry to the Validated By table documenting that the WHO-announces framework produced a correct prediction on the same question structure.

### Spec Update

**Updated**: `_spec.md` Rule 46 — added step 7 (check for dual-presidential endorsement) as a mandatory pre-forecast step for transition-window ceasefire questions.

### Reflection File

**Created**: `meta/reflections/_reflection-2026-05-21-per-q28-same-ceasefire.md` — full diagnosis documenting the validation result, gaps found, and vault health assessment.