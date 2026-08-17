---
type: reflection
tags: [reflection]
date: 2026-05-18
cycle: per-q27
question: "Taiwan Presidential Election: Will Ko Wen-je win?"
prediction: "NO"
actual: "NO"
vault_contribution: "0% (freebie)"
---
---
---
# Per-Question Reflection Q27: Taiwan Presidential Election — Ko Wen-je

## Diagnosis

**Prediction**: NO (correct)

**Ground truth**: NO — Ko Wen-je received 26.46%, finishing third behind Lai Ching-te (40.05%) and Hou Yu-ih (33.49%).

**Vault contribution**: 0% (freebie). The vault had NO pre-election coverage of the Taiwan campaign in the relevant quarter files (2023-Q3, 2023-Q4). The correct prediction relied entirely on general knowledge of the election dynamics: three-way race in single-round plurality system, Ko polling in third place at ~22-24%, front-runner Lai at ~35%. The vault's existing post-hoc entities (Ko Wen-je, Lai Ching-te, TPP) and concept (divided-opposition-plurality-win) were created AFTER the question was answered, so they didn't contribute to the forecast.

**Lesson**: This is the clearest example yet of the "no freebie" principle (spec rule 8). A correct prediction that the vault cannot support is a vault gap. The gap was structural: the procedure had no step requiring the writer to check for upcoming elections and include pre-election campaign coverage. Contemporary quarter files covered dozens of global events but systematically missed the Taiwan election — held just 2 weeks after the Q4 2023 cutoff.

## Remediation Performed

### Files Created
- **entities/terry-gou.md** — Entity stub for Foxconn founder who ran as independent and withdrew on registration deadline
- **forecasts/2026-05-18-taiwan-ko-wen-je.md** — Full forecast entry documenting reasoning and vault gaps

### Files Updated
- **timeline/2023-Q3.md** — Added "Taiwan 2024 Presidential Election Campaign" section covering candidate nominations (March-May 2023), early polling trends (Ko briefly competitive June-July, then declining), and Terry Gou's independent entry in August 2023
- **timeline/2023-Q4.md** — Added "Taiwan 2024 Presidential Election Campaign" section covering the KMT-TPP alliance negotiations and collapse (November 13-24, 2023), Gou's qualification and withdrawal, final polling data showing Ko at 17-24% in third place, and the December presidential debate. Also added emerging pattern #8 noting the election's structural dynamic.
- **entities/ko-wen-je.md** — Added pre-election context section with specific polling data and structural analysis explaining why his ~24% ceiling made victory impossible
- **concepts/divided-opposition-plurality-win.md** — Added Validated By entry for this forecast (NO prediction validated by the concept's inverse logic)
- **_spec.md** — Added Rule 22: "Pre-election coverage is mandatory in contemporary quarter files" with 6 required fields (candidate field, electoral system, opposition coordination status, polling data, key campaign events, forecasting significance)
- **_procedure.md** — Added step 6 to Phase 1 (Research Protocol): "Check for upcoming forecast-relevant elections" before writing contemporary quarter files. Added "Pre-Election Coverage Is the Most Common Omission" to Lessons from Cycle 4 section.

## Key Insight

The Taiwan 2024 election question revealed that the vault suffers from a systematic **post-hoc bias in contemporary coverage**: quarter files extensively document what happened but rarely document what WAS GOING TO HAPPEN (imminent events). The 2023-Q4 file covered elections in the Netherlands, Argentina, New Zealand, Poland, Portugal, and Ecuador — but not the Taiwan election happening 2 weeks after the cutoff. This is because the vault's writing methodology looks backward, not forward.

The fix is structural: before writing any contemporary quarter file, check the calendar for major elections in the next 2 quarters and proactively document the campaign dynamics. This forward-looking approach is now encoded in both _spec.md (Rule 22) and _procedure.md (Phase 1 step 6).

## Cumulative Stats

Questions completed: 27 (22 correct, 5 wrong — correcting from earlier count based on this data)
