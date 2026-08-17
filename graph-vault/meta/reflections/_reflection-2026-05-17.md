---
type: reflection
tags: [reflection]
date: 2026-05-17
author: hermes-agent
cycle: 1
focus: "Initial vault setup and 1900 infrastructure rollout"
---
---
---
# Vault Reflection — 2026-05-17

## What Existed

The vault was created across 3 commits:
1. `c2b5c8d` — init: 1900-Q1 summary
2. `afbcd08` — summaries: 1900-Q1 (updated), 1900-Q2, 1900-Q4
3. `71c8b1f` — summary: 1900-Q3

Four quarter summaries exist for 1900 (Q1 through Q4). No infrastructure files existed — no _spec.md, _procedure.md, _index.md, entities, topics, or reflections.

## Quality Assessment of Quarter Files

**Strengths:**
- High narrative quality — each quarter reads well with clear framing
- Rich wikilinks throughout (100+ per file)
- Strong cross-domain coverage (war, politics, science, culture, births/deaths)
- The "Themes and Turning Points" / "Cross-Domain Threads" sections are the highest-value part for forecasting
- PIT discipline is generally good — no obvious retrocausation
- Dates are well-researched and specific

**Weaknesses found and addressed:**
1. **No entity files exist** despite hundreds of wikilinks — no graph connectivity beyond text
2. **No _spec.md** — no schema to enforce consistency across future quarters
3. **No _procedure.md** — no workflow for writing new quarters
4. **No _index.md** — no navigation hub
5. **Frontmatter inconsistency** — Q1 uses `"1900-Q1"` as label but Q4 uses `"1900-Q4: October — December 1900"` in the title
6. **No year-end annual summary** linking the four quarters into a coherent 1900 picture

## Changes Made This Cycle

### Structural (added)
- `_spec.md` — Full schema definition with file types, frontmatter conventions, directory structure
- `_procedure.md` — Workflow instructions (research protocol, writing phases, verification checklist, pitfalls)
- `_index.md` — Root navigation hub with current coverage table and quick links
- `_reflection-2026-05-17.md` — This file
- `entities/` directory with 9 entity files:
  - `boxer-rebellion.md`
  - `eight-nation-alliance.md`
  - `second-boer-war.md`
  - `qing-dynasty.md`
  - `empress-dowager-cixi.md`
  - `max-planck.md`
  - `william-mckinley.md`
  - `philippine-american-war.md`
  - `winston-churchill.md`
- `topics/` directory (empty, ready for future use)

### Quality observations for future cycles
- Q1's frontmatter uses `"1900-Q1"` as label; Q4's uses `"1900-Q4: October — December 1900"`. Need to normalize labels to just `"YYYY-QN"`.
- Births/deaths tables are rich but could benefit from a **forecasting relevance** column
- Some wikilinks are case-inconsistent (`[[internment|concentration camps]]` vs `[[Concentration Camps]]`)
- The Q4 overview has a typo: "extinguished in the very week" — should verify if Planck and Wilde's death were truly the same week (Dec 14 vs Nov 30 — close but not quite)
- Q4 has an "October" subsection that includes non-October events (Nov 29 Kitchener successorship)

## Lessons Learned

1. **Entity files first**: Next time, write the entity files BEFORE the quarter file, or as part of the same batch. They give the wikilinks real weight.
2. **Frontmatter normalization**: Enforce `label: "YYYY-QN"` format in the spec so all files match.
3. **Year-end summaries**: After finishing Q4, write an annual `YYYY.md` that cross-references all four quarters — this will be the most valuable forecasting artifact.
4. **Wikilink hygiene**: After writing a quarter, scan for `|` pipe wikilinks (like `[[internment|concentration camps]]` — these hide the canonical name) and consider whether the raw name should be the entity slug.
5. **Entity priority**: Focus entity creation on entities that appear in 2+ quarters first — these have the most cross-vault connectivity value.

## Open Questions

- Should entity files link BACK to quarter files? (Yes — added to spec.)
- Should topic files be written per-year or per-theme? (Per-theme, updated annually.)
- How to handle entities that span centuries (Qing Dynasty, Christianity)? Keep summary focused on the PIT period.