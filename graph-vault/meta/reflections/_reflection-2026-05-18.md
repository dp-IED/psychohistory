---
type: reflection
tags: [reflection]
date: 2026-05-18
author: hermes-agent
cycle: 3
focus: "Structural consolidation, frontmatter standardization, thread/concept/topic expansion"
---
---
---
# Vault Reflection — 2026-05-18 (Cycle 3)

## Problems Identified and Fixed

### 1. Dual Directory Structure (CRITICAL)
The vault had TWO directories containing quarter files: `quarters/` (newer refined versions of Q1-Q3) and `timeline/` (original versions plus Q4 and annual summary). This was confusing and unsustainable.

**Fix**: Copied the refined `quarters/` versions into `timeline/` (replacing the originals), and removed the `quarters/` directory. `timeline/` is now the canonical quarter-file location.

### 2. Frontmatter Inconsistency (widespread)
Entity files used three different frontmatter patterns (`name:`+`slug:`, `label:`, `title:`+`slug:`). Thread files used `label:`, `name:`+`slug:`, `start_date:` etc. Concept files used `label:` or `name:`+`slug:`.

**Fix**: Standardized all frontmatter to `title:` + `slug:` pattern for threads, entities, and concepts. Added `tags:`, `inception:`, `conclusion:`, `status:` fields to threads. Added `domain:`, `related_concepts:` to concepts.

### 3. Redundant Threads
`threads/second-boer-war-guerrilla-phase.md` was a subset of `threads/second-boer-war.md` — they covered the same content with high overlap.

**Fix**: Absorbed the guerrilla-phase analysis (patterns section, scorched-earth dynamics) into the main Boer War thread. Deleted the redundant file.

### 4. Missing Entities
Heavily linked entities lacked files: Lord Salisbury, Galveston Hurricane, Concentration Camps, Anarchism, Franz Ferdinand.

**Fix**: Created all five entity files with proper frontmatter, summaries, significance, and wikilinks.

### 5. Missing Concepts
Several recurring patterns embedded in the 1900 material lacked formal concept files.

**Fix**: Created `counterinsurgency-tactical-radicalization.md` (progressive brutality in COIN campaigns) and `multiple-scientific-discovery.md` (simultaneous independent discoveries).

### 6. Empty Topics Directory
`topics/` existed but had zero files.

**Fix**: Created `science-tech-1900.md` and `geopolitics-1900.md` — thematic syntheses of the year.

### 7. Stale Index
`_index.md` had incomplete thread listings and didn't reflect the vault's actual content.

**Fix**: Rewrote with current directory structure, all threads (with status labels), all concepts, all topics, and quick links to newly added entities.

## Spec & Procedure Upgrades

### _spec.md v2.0
- Added `type: thread` schema with full frontmatter conventions
- Added `type: concept` schema with `domain:` and `related_concepts:`
- Added thread lifecycle rules (update each quarter, cumulative, status tracking)
- Added entity priority tiers (multi-quarter > thread-central > forecasting-relevant)
- Added wikilink hygiene rules (avoid pipe syntax, resolve orphaned links)
- Added backlink conventions (entity files should list their appearance quarters)

### _procedure.md v2.0
- Added explicit Phase 3: Update Thread Files (after writing quarter)
- Added thread status guidelines (nascent > active > climaxing > fading > resolved)
- Added Phase 5: Review Concept Files (check for new examples)
- Added writing checklist items for thread and concept maintenance
- Added "thread neglect" and "entity bloat" to pitfalls
- Organized recurring tasks into a table

## Lessons Learned

1. **Dual directories are a landmine**: The `quarters/` vs `timeline/` split went unnoticed until a full audit. Single canonical locations prevent confusion.
2. **Frontmatter drift is inevitable without enforcement**: Even with a spec, files drift. Periodic grep-based audits should be part of the maintenance cycle.
3. **Threads are the most valuable files**: The thread + concept + topic pattern provides three levels of abstraction over raw chronology. This layered approach should be the standard.
4. **Concept files need canonical examples**: A concept without concrete examples from the vault is just philosophy. Every concept should reference specific events, threads, and entities.
5. **Entity creation should happen in batches**: Creating entities one-at-a-time is inefficient. Better to batch-create all entities from a quarter's wikilinks after the quarter is written.

## Open Questions

- Should we add an `## Appears In` backlink section to entity files? (Spec says yes; none have it yet. Batch backfill needed.)
- Should thread files have a "Forecasting Significance" section in addition to Key Dynamics? (Experiment: yes — main Boer War thread has one. Evaluate after more threads adopt it.)
- When to create topic files: per-year synthesis vs per-theme multi-year? (Current: per-year for science and geopolitics. Consider per-theme when the vault expands to 1905.)
