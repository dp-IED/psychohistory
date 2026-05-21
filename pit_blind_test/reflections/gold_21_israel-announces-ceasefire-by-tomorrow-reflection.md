---

## Q21/30 Reflection Report

**Diagnosis**: Q21 was correct (YES) with significant vault support (~70%), a dramatic improvement over Q52/84's 0% vault contribution for the same underlying question. The prior reflections had already created MENA domain infrastructure (threads, concepts, entities, procedures, spec rules). The remaining gaps were precision issues — structural elements that existed as references but lacked dedicated documentation.

**What was changed and why:**

### New files (5):

1. **`domains/mena/threads/gaza-ceasefire-negotiations-2025/events/may-2024-ceasefire-framework.md`** — The May 2024 framework was cited across 7+ files as the "pre-existing framework" but had no standalone event file. A forecaster could not understand WHY the deal existed in May 2024 but couldn't be activated until Jan 2025 without reading scattered references. This event file documents the three-phase proposal, timeline of the 8-month stall, why it failed in May vs. succeeded in Jan, and five forecasting lessons.

2. **`domains/global/concepts/pre-negotiated-framework-activation/_concept.md`** — The pattern of a framework existing but awaiting political activation is distinct from negotiation-from-scratch. The concept formalizes this with an activation delay formula (0-30d to 18mo+) calibrated by blocking condition severity, Bayesian priors (60-80% P within 3 months of blocking condition resolution), and cross-conflict applicability (Israel-Hamas, Colombia, Sudan, Yemen, Ukraine).

3. **Three entity stubs**: `khalil-al-hayya.md` (Hamas lead negotiator post-Sinwar), `yoav-gallant.md` (Israeli defense minister, ICC warrant, internal ceasefire advocate), `mohammed-deif.md` (Hamas military commander, Oct 7 architect, decapitation cascade target). All were key actors in the ceasefire dynamics but lacked vault presence.

### Updated files (3):

4. **`timeline/2025-Q1.md`** — The one-line entry ("Jan 15 — Israel and Hamas approve ceasefire") risked gold_50-type confusion between mediator and party announcements. Expanded to include four-date tracking, explicit WHO-announces distinction, and links to the pre-negotiated framework activation concept.

5. **`domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept.md`** — Added a cross-conflict comparison section contrasting transition-window diplomacy (Israel-Hamas Jan 2025) with superpower combatant entry (Iran-Israel June 2025). These are mutually exclusive mechanisms with different timelines (10 weeks vs 48 hours), probability ranges (40-60% vs 70-90%), and leading indicators. Conflating them produces wrong forecasts.

6. **`_index.md`** and **`_macro_gaps.md`** — Both updated with new reflection entries and gap logs.

### Key lesson for the vault

Even well-covered topics can have residual precision gaps. The May 2024 framework was a ghost reference — cited constantly but never documented as a standalone entity. Creating the event file and activation concept closes the gap between "the vault mentions this" and "the vault can support predictive reasoning about this." The pattern of "framework existence != framework activation" is generalizable to ANY protracted conflict where a mediator has tabled a proposal.