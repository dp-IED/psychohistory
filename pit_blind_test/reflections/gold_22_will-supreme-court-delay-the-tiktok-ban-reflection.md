All edits verified. Here's my full report:

---

## REPORT: Q22 Reflection — What Changed and Why

### Diagnosis

This question ("Will Supreme Court delay the Tiktok ban?") was **correct** (predicted NO, actual NO). The vault already had strong analytical infrastructure: the SCOTUS Procedural Signals concept, the Executive Enforcement Delay concept, procedure step 20, timeline coverage, and entity stubs — all built from earlier reflections.

**The critical gap identified**: These two concepts existed in separate files with **zero cross-references** between them. The TikTok saga involves BOTH types of delay operating independently in sequence — SCOTUS did NOT delay (judicial), but Trump DID delay enforcement afterward (executive). A future question testing either type independently would not benefit from the comparison. The vault had no decision tree for disambiguating ambiguous "will X be delayed?" questions.

### Files Changed (5 files + 1 new)

**1. `domains/usa/concepts/scotus-procedural-signals.md`** — Added **Stage 6: Judicial Delay vs Executive Delay** (~60 lines)
- Comparison table (actor, mechanism, timing, resolution impact, TikTok outcome)
- 5-step decision tree for "will X be delayed?" questions: identify delay type, check resolution text, map timeline relative to effective date, check for sequential operation, apply the appropriate framework
- Two-delay sequence timeline showing both types operating independently on the same legal timeline
- Added `executive-enforcement-delay` to related_concepts and wikilinks

**2. `_procedure.md`** (Step 20) — Added delay-type disambiguation sub-bullet (~6 lines)
- Explicitly distinguishes "will Court delay?" (judicial, SCOTUS signals framework) from "will the ban persist?" (executive, executive-enforcement-delay concept)
- Documents the "no" on judicial delay does NOT imply "no" on executive delay
- Decision rule for ambiguous passive-voice questions
- Preserved the existing trajectory principle sub-bullet

**3. `domains/global/concepts/executive-enforcement-delay/_concept.md`** — Added **"Relationship to Judicial Delay"** section (~15 lines)
- Comparison table mirroring the one in the SCOTUS concept
- Forecasting rule: which framework applies to which question type
- Cross-reference to the SCOTUS concept's Stage 6 section
- Added `scotus-procedural-signals` to related_concepts

**4. `domains/usa/entities/us-supreme-court.md`** — Added Pattern Summary row to TikTok timeline table
- Documents the "no delay" track record
- Cross-references the judicial-vs-executive delay distinction in the SCOTUS signals concept

**5. `_index.md`** — Index updated with all 5 changes listed under Recent Additions

**6. `meta/reflections/_reflection-2026-05-20-per-q22-scotus-tiktok-delay.md`** — **NEW** per-question reflection file
- Full diagnosis, causal chain, all improvements documented, vault health metrics

### Why These Changes Matter

The vault previously had the right content but was **disconnected**. A forecaster loading the SCOTUS signals concept for a "will Court delay?" question would not be prompted to check whether the question was actually about executive delay instead. The decision tree forces disambiguation before analysis begins — preventing the category error of applying the wrong framework. This is the next-forecast-relevant improvement: the next "will X be delayed?" question (about any regulatory action in any jurisdiction) will trigger the decision tree and route the forecaster to the correct analytical framework.