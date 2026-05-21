All tasks complete. Here's the summary:

## Report: Changes Made (Q5/30 Reflection — HNP Argentina Seats)

### Diagnosis

The prediction was correct (NO) and the vault contributed ~90%+ signal. The Argentina coverage is mature, built over 3+ prior learning cycles. The NO outcome was structurally overdetermined — HNP won 8 seats vs. LLA's 64 — so the question didn't expose a content gap in the traditional sense. But it revealed three structural issues:

### Changes Made

**1. Relocated misplaced concept: regional-third-way-squeeze**
- Was at: `domains/usa/concepts/regional-third-way-squeeze.md` (flat file, wrong domain)
- Now at: `domains/latin-america/concepts/regional-third-way-squeeze/_concept.md` (directory structure, correct domain)
- Its canonical case is Argentina (Schiaretti/HNP) with secondary cases in Spain and France. The USA has no canonical example of this pattern.
- Updated both `domains/usa/_domain.md` (removed reference) and `domains/latin-america/_domain.md` (added reference, plus two more missing concepts)

**2. Created new concept: structural-improbability-check**
- Path: `domains/global/concepts/structural-improbability-check/_concept.md`
- 5-step decision tree for recognizing when a question's YES scenario is structurally impossible (requires 2+ independent failures of larger actors, joint p < 0.01)
- Fills the abstraction gap between domain-specific ceiling concepts (regional-third-way-squeeze, far-left-marginalization-polarization) and provides a reusable diagnostic for any future "Will minor party X win?" question
- Canonical cases from vault experience: HNP (this Q), FIT-U (Q4), US third-party state win, far-left presidential win

**3. Updated _procedure.md with blind-test battery detection**
- Added Step 12a: Blind-test battery detection — vault saturation test, structural improbability pre-check, trace-the-chain heuristic, four-level effort allocation
- Added Step 12b: Pre-forecast structural improbability check — explicitly run the structural-improbability-check before building domain-specific content for minor-party questions

**4. Wrote reflection file**: `meta/reflections/_reflection-2026-05-20-per-q5-hnp-argentina-seats.md`

### Key lesson

When a domain is saturated and the next question asks about a structurally-impossible outcome, the vault's marginal value is zero for domain-specific content. The correct response is to extract a cross-domain pattern — the structural-improbability-check concept is the abstraction from four separate hard-NO patterns into one reusable framework.