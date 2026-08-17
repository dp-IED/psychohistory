---
type: reflection
tags: [meta, curation]
date: 2026-05-22
trigger: "Graph vault reflection after recent forecast runs — curation pass 9"
---

# Graph Vault Curation — May 22, 2026 (Pass 9)

## Trigger

Standard curation sweep after recent forecast runs (Raúl Castro, MV Hondius hantavirus, Colombia, Tereza Cristina, AI safety bill) and review of gold set catalog.

## Vault State Before

- 22 run files, 19 event files, 306 entities, 147 concepts, 21 timeline quarters
- 3 broken wikilinks in timeline files: `[[french-presidential-election-2026]]`, `[[switzerland-population-initiative-2026]]`, `[[israel-gaza-ceasefire-2025]]`
- Ivan Cepeda cross-reference in us-cuba-relations thread (noise — no relation to Raúl Castro)
- `_macro_gaps.md` had stale "current" section but all gaps actually filled

## Actions Taken

### 1. Created 3 Event Stubs (Broken Wikilink Fix)

| File | References From | Content |
|------|----------------|---------|
| `events/french-presidential-election-2026.md` | 2026-Q2 timeline, slovenia thread | Structural context (2-round system, cordon sanitaire, RN ceiling), key actors (Le Pen, Philippe, Attal), EU/NATO implications |
| `events/switzerland-population-initiative-2026.md` | 2026-Q2 timeline | Direct democracy structural context, historical restrictive immigration initiative track record, EU treaty "guillotine clause" |
| `events/israel-gaza-ceasefire-2025.md` | 2026-Q1, Q2 timelines | Four-date cascade, gold set cross-reference (gold_21, gold_28), structural preconditions (Assad collapse, Hezbollah ceasefire, Trump win) |

All three stubs include forecasting-significant structural context, not just descriptive fillers. Each connects to existing vault concepts.

### 2. Pruned Noise from us-cuba-relations Thread

Removed `[[ivan-cepeda-castro]]` reference labeled "No relation; same surname is coincidental" — this was a false link that connected an irrelevant Colombian politician to the Cuba thread for no analytical reason. Replaced with structured Vault Cross-References section linking to the forecast, run record, structural-improbability-check, and market-vault-structural-divergence concepts.

### 3. Enriched Raúl Castro Entity

Added "Calibration Cross-Reference" section showing Raúl Castro (p_yes=0.005) as part of the cross-domain structural-NO cluster alongside Bitcoin (0.25), Cervia tennis (0.01), and HNP holds (0.005). Explicitly connects to short-horizon-momentum-check concept.

### 4. Updated _index.md

- Bumped version 2.1 → 2.2
- Added curation pass table documenting all changes
- Fixed event count: 19 → 22
- Fixed run count: 22 → 23

### 5. Swept _macro_gaps.md

- Confirmed all P1-3 gaps filled
- Added 2 new identified gaps: Colombian candidate entity stubs (P2), Colombian election-dynamics concept (P3)

### 6. Updated events/_index.md

- Added Europe Domain section (French election, Switzerland initiative)
- Added Israel-Gaza ceasefire 2025 to Gaza Ceasefire Cycle
- Updated curation pass documentation

## Vault State After

- **23** run files (was 22)
- **22** event files (was 19; +3 stubs)
- **306** entity files (unchanged; enriched raul-castro)
- **147** concept files (unchanged)
- **67** thread files (unchanged; pruned us-cuba noise)
- **21** timeline quarters (unchanged)
- **0** broken wikilinks among checked files

## Decisions

### Why event stubs instead of pruning the broken wikilinks?

Could have removed the broken wikilinks from timeline files instead of creating stubs. Decided to create stubs because:
1. The events are real and forecasting-relevant (French presidential election, Swiss referendum, Israel ceasefire)
2. The broken links signal a genuine vault gap — the timeline references events that should have vault representation
3. Event stubs provide structural analysis that improves downstream forecasts

### Why not create Colombian entity stubs now?

The new gaps in _macro_gaps.md identify that Abelardo de la Espriella and Paloma Valencia lack entity files despite being active candidates in a $6.1M market. These are P2 (medium priority) because the Colombia election is already vaulted at the thread level and the forecast is complete. Entity stubs would help cross-run calibration but don't block any active forecast.

## Lessons for Future Curation

1. **Check timeline wikilinks systematically**: Timeline files are the most likely source of broken links since they reference many events that may not have been vaulted yet. A pre-flight check of timeline wikilinks before creating new timeline entries would catch this earlier.
2. **Prune as you create**: The Ivan Cepeda reference was created when the Cuba entity was first written — it was a convenience cross-link that added no value. Cross-domain links should only connect genuinely related entities.
