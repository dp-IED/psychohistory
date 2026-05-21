All files verified. Here's the report.

---

## Report: What changed and why

### Diagnosis

The prediction was **correct (NO) with ~80% vault contribution**. The vault's mature Argentina coverage from prior cycles provided the seat counts (LLA 64 vs FIT-U 3), the structural ceiling framework (far-left-marginalization-polarization concept), and the historical baseline (FIT-U entity with 2-5% decade-long ceiling). However, three residual gaps remained even after 3+ prior reflection cycles:

### Changes Made

**1. Removed stale duplicate entity** (`domains/europe/entities/fit-u.md`)
- FIT-U is an Argentine party, not European. The duplicate was inferior (37 lines vs 59) and mislabeled as "Slovenian political party" in the Europe domain index. Deleted the file and removed its cross-reference from `europe/_domain.md`. This eliminates a data-integrity risk where vault domain loads could retrieve the wrong entity.

**2. Created 3 missing entity stubs** (`domains/latin-america/entities/`)
- **myriam-bregman.md** — 2023 FIT-U presidential candidate, PTS faction leader, most nationally visible FIT-U figure after del Cano. Her absence was the highest-risk gap: a question about "FIT-U's next presidential candidate" would have found no vault content.
- **romina-del-pla.md** — Partido Obrero labor organizer and deputy. Relevant for questions about FIT-U's union strategy or internal faction balance.
- **christian-castillo.md** — PTS intellectual leader, sociologist. Relevant for questions about FIT-U strategic direction or doctrinal evolution.

**3. Updated FIT-U entity** with wikilinks for all three new stubs (both in Key Figures section and Wikilinks section at bottom).

**4. Updated `_spec.md` Rule 9** with a "Recursive completeness" subsection requiring that named individuals within entity files ("Key Figures", "Leadership") also get entity stubs if they could plausibly appear in a forecast question.

**5. Updated `_procedure.md`** post-forecast reflection with Step 4: "Scan for recursive entity completeness" — a procedural forcing function to catch second-tier entity gaps before they become information cascades.

**6. Wrote reflection file** at `meta/reflections/_reflection-2026-05-20-per-q4-fit-u-argentina-seats.md` documenting the diagnosis, gaps, and fixes.

**7. Updated `_index.md`** with the per-question reflection entry documenting all changes.

### Key lesson

Mature domains still have residual gaps after multiple reflection passes. The most persistent gap type is **second-tier entity completeness** — the primary subject (FIT-U) has a stub, but the named individuals within it (Bregman, Del Pla, Castillo) don't. The cost of stub creation (~2 min each) is negligible compared to the risk of future misses. The new recursive completeness check in the procedure prevents this cascade going forward.