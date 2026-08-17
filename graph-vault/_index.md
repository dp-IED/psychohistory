---
type: index
tags: [meta, index]
version: 2.46
date: 2026-06-21
purpose: "Graph vault reflection — curation pass 52: Temporal refresh for Colombia Election Day. Updated Q3 timeline date from Jun 20→Jun 21. Updated peace-broker-diplomatic-playbook concept pit_cutoff and cascade tracking heading from T+6→T+7. All 55 runs remain fully cataloged. Entity count: 374 (stable)."
---

# Graph Vault — Navigation

## Vault Structure

- `timeline/` — Live PIT quarters (2022-Q1 through 2026-Q2, 21 files)
- `runs/` — Forecast run records (55 vault-managed + 30 harness gold set)
- `forecasts/` — Forecast reasoning files
- `events/` — Event files (32 entries, see [[events/_index]])
- `agent-roles/` — Sub-agent role definitions (20 roles + 1 orchestrator)
- `procedures/` — Cross-domain forecasting procedures
- `domains/` — Domain-organized content (15 domains):
  - [[domains/africa/_domain|Africa]] (Nigeria, South Africa, Kenya, Sudan, state fragility)
  - [[domains/usa/_domain|USA]] (politics, regulation, immigration, trade)
  - [[domains/economics/_domain|Economics]] (monetary policy, crypto, macro indicators)
  - [[domains/latin-america/_domain|Latin America]] (Brazil 2026, Argentina, Venezuela, Mexico)
  - [[domains/iran/_domain|Iran]] (post-war diplomacy, Supreme Leader succession, nuclear negotiations)
  - [[domains/mena/_domain|MENA]] (Israel-Hamas, Iran, Syria)
  - [[domains/europe/_domain|Europe]] (UK, EU, France, Germany)
  - [[domains/east-asia/_domain|East Asia]] (Taiwan, China, Japan)
  - [[domains/southeast-asia/_domain|Southeast Asia]] (Thailand, Philippines, Myanmar)
  - [[domains/health/_domain|Health]] (respiratory, hantavirus, outbreaks)
  - [[domains/sports/_domain|Sports]] (tennis, pickleball, NBA props)
  - [[domains/france/_domain|France]] (2027 presidential election)
  - [[domains/global/_domain|Global]] (cross-domain concepts, patterns)
  - [[domains/religion/_domain|Religion]] (Papal succession, elderly leader mortality)
  - [[domains/global/concepts|Global Concepts]] (~60 files)
- `meta/` — Session reflections, thread audits, changelog archives
- `_macro_gaps.md` — Current open vault gaps (tracked here)
- `_forecast_instructions.md` — Behavioral rules for forecasting agent
- `_procedure.md` — Cross-domain structural reasoning procedure

## Changelogs

Historical changelogs from May 18-21, 2026 (per-question reflections, curation passes, vault reorganization) are archived at:
- [[meta/changelog/_changelog-2026-05-18-21]]

Current open gaps are tracked in [[_macro_gaps]].

## Vault Health (June 21, 2026 — After Graph Vault Reflection 52: Temporal Refresh, Colombia Election Day)

| Metric | Count |
|--------|-------|
| Domains | 16 |
||| Entity files | **374** |
|||||| Concept files | **168** |
|| Thread files | **71** |
|| Timeline quarters | **23** (2022-Q1 through 2026-Q4 + 3 annual summaries) |
|| Event files | **32** (+1: Colombia runoff) |
|| Agent roles | 20 + 1 orchestrator |
||||| Run files (vault-managed) | **55** |
||||| Forecast files | **25** |
||| Redirect stubs | **45** |

### This Curation Pass (Graph Vault Reflection 49 — June 21, 2026: Saudi Arabia Entity, Iran Rewrite, Temporal Refresh, Hijri Fix)

|| Action | Detail |
||--------|--------|
|| **Created Saudi Arabia entity** | Filled critical structural gap — no Saudi entity existed despite coverage of Hormuz traffic (run #49), Iran enrichment (run #52), peace-broker playbook, oil markets, and regional MENA dynamics. Covers regional realignment post-Iran peace deal, Hormuz/energy dependency, OPEC+ dynamics, US-Saudi/Israel normalization track, and China triangulation. Forecasting significance: Saudi appears as a key variable in 5+ active forecasts. |
|| **Rewrote Iran global entity** | Repaired triple `---` frontmatter closure, stale `pit_cutoff: 2025-12-31`, missing Khamenei assassination (Feb 28, 2026) and peace deal (Jun 14, 2026), 7 broken flat-path wikilinks (`[[threads/...]]`, `[[ali-khamenei]]` etc.). Rebuilt with current post-war context, active forecast table (5 forecasts), key events timeline, and domain-organized wikilinks. |
|| **Fixed Hijri stale data** | Corrected 2 references treating Ali Khamenei as alive — entity said "87yo Supreme Leader" and "de facto head of state" when Khamenei was assassinated Feb 28, 2026. Replaced with post-assassination succession context. Removed duplicate paragraph. |
|| **Temporal refresh: cluster T-10→T-9** | Updated [[domains/global/concepts/short-window-expiration-cluster/_concept]] for June 21 — T-9 from T-10. Added Colombia resolution day context to summary. Updated Díaz-Canel base rate (10-day 0.14%→9-day 0.13%), countdown table, and window references. |
|| **Temporal refresh: enrichment concept T-10→T-9** | Updated [[domains/iran/concepts/iran-nuclear-enrichment-post-war/_concept]] status check heading and updated date. |
|| **Added Saudi Arabia to MENA domain** | Inserted [[domains/mena/entities/saudi-arabia]] into MENA domain entities list. Updated remaining coverage gaps to mark Saudi-Israeli normalization as partially addressed. |
|| **Bumped vault version 2.42→2.43** | Updated index frontmatter, purpose field, vault health header, entity count (373→374), changelog entry. |

### This Curation Pass (Graph Vault Reflection 50 — June 21, 2026: New Runs #53-55, GCC/Cuba Threads, UK Labour Concept)

|| Action | Detail |
||---|--------|
|| **Cataloged 3 new forecast runs** | Added runs #53 (Andy Burnham UK PM 0.35, $991K), #54 (Cuba regime falls 0.18, $987K), #55 (UAE leave GCC 0.10, $98.5K) to [[runs/_index]]. Updated unresolved count to 52 pending. All 3 forecasts were agent-generated on June 21 from the latest Polymarket scan — now fully vaulted with run files and cross-references. |
|| **Created GCC/Gulf politics thread** | Filled Gap #30: Created [[domains/mena/threads/gcc-gulf-politics/_thread]] covering 43-year GCC durability, zero-exit base rate, Saudi-UAE rivalry dynamics, collective security framework (Peninsula Shield Force), OPEC+ coordination, Qatar blockade precedent, and regional security architecture post-Iran peace deal. Cross-referenced to UAE entity, Saudi Arabia entity, Iran entity, and runs #55 and #49. |
|| **Created US-Cuba relations thread** | Filled Gap #31: Created [[domains/latin-america/threads/us-cuba-relations/_thread]] covering Helms-Burton Act framework, Obama thaw/Trump reversal timeline, economic pressure indicators (GDP decline, tourism collapse, remittance fall, Venezuelan oil subsidy reduction), migration diplomacy analysis (emigration as pressure valve), and cross-references to Cuba, Díaz-Canel entities and runs #54 and #38. |
|| **Created UK Labour leadership contest concept** | Filled Gap #32: Created [[domains/europe/concepts/uk-labour-leadership-contest/_concept]] with full contest rules (nomination threshold, MP elimination rounds, OMOV final ballot), timeline constraints (8-12 weeks typical), MP-filter analysis (no non-MP has ever won), historical precedents table (1994-2020), and cross-domain comparison with US VP selection and Israeli PM succession patterns. Directly relevant to the anomalous Burnham PM market (95.3% YES vs vault 0.35). |
|| **Bumped vault version 2.43→2.44** | Updated index frontmatter, purpose field, vault health header (runs 52→55, concepts 167→168, threads 69→71), changelog entry. |

### This Curation Pass (Graph Vault Reflection 51 — June 21, 2026: UK Entity Repair, Redirect Fix, Q4 Timeline Enrichment, Temporal Refresh)

|| Action | Detail |
||---|--------|
|| **Fixed UK entity structural defects** | UK entity (`domains/europe/entities/united-kingdom`) had triple `---` frontmatter closure, was a climate stub with zero political forecasting content, and was invisible to the `[[domains/europe/entities/uk]]` wikilink used by run #53. Rewrote as active political entity covering Westminster system, party structure, next-PM forecasting relevance, Labour leadership contest rules, Burnham cross-refs, plus retained climate content. Added 7 domain-organized wikilinks. |
|| **Created redirect stub for broken uk/United Kingdom link** | Run #53 (Andy Burnham) linked to `[[domains/europe/entities/uk]]` which did not exist. Created redirect stub → `[[domains/europe/entities/united-kingdom]]`. This fixes the only broken wikilink in run #53. |
|| **Fixed Labour Party entity** | Entity had triple `---` frontmatter closure, 7 flat-path wikilinks (`[[uk-domestic-politics]]`, `[[rachel-reeves]]`, etc.), and zero cross-references to the active Burnham forecast or the UK Labour leadership contest concept. Rewrote with clean frontmatter, domain-organized wikilinks (7 paths), leadership contest rules summary, and cross-references to run #53 and the concept. |
|| **Enriched Q4 timeline with runs #53-55** | Added 3 new geopolitical sections to [[timeline/2026-Q4]]: UK Labour leadership path (Burnham 0.35 divergence), Cuba regime stability (0.18 consensus), GCC/UAE exit risk (0.10 consensus). Added all 3 runs to Forecast Files table. Added 7 wikilinks covering new entities/concepts. Updated source date to 2026-06-21. |
|| **Temporal refresh: short-window cluster** | Updated [[domains/global/concepts/short-window-expiration-cluster/_concept]] pit_cutoff from 2026-06-18→2026-06-21 and updated field from 2026-06-20→2026-06-21. T-9 heading and Díaz-Canel base rate (9-day 0.13%) were already correct from Pass 50. No content changes needed — all 6 forecasts remain structurally NO with no emerging mechanisms. |
||| **Bumped vault version 2.44→2.45** | Updated index frontmatter, purpose field, vault health (redirect stubs: 44→45), changelog entry. Entity count stable at 374 (2 enhanced: united-kingdom, labour-party-uk). |

### This Curation Pass (Graph Vault Reflection 52 — June 21, 2026: Temporal Refresh, Colombia Election Day, Version Bump)

|| Action | Detail |
|---|--------|--------|
|| **Temporal refresh: Q3 timeline date** | Updated [[timeline/2026-Q3]] compiled-date from 2026-06-20 → 2026-06-21 (Colombia Election Day). No content changes needed — Colombia section already correctly reflects Election Day status. |
|| **Temporal refresh: peace-broker concept** | Updated [[domains/global/concepts/peace-broker-diplomatic-playbook/_concept]] pit_cutoff 2026-06-20 → 2026-06-21 and cascade tracking heading from "T+6 Days" to "T+7 Days" post-Iran deal. Confirmed all other date-dependent concepts (short-window-expiration-cluster, Iran enrichment) were already current at 2026-06-21 from Pass 50-51. |
|| **Run catalog verification** | Verified all 9 [?] forecast entries in the prompt correspond to already-cataloged runs #46-55 with full source paths in their run files. No new runs needed — the [?] entries reflect unresolved Brier status which is expected pre-resolution. |
|| **Vault health stable** | Entity count: 374 (stable). All 55 runs fully cataloged. No new gaps identified beyond open Gap #18 (Colombia runoff postmortem, expected Jun 22+). |
|| **Bumped vault version 2.45→2.46** | Updated index frontmatter, purpose field, changelog entry. No changes to entity/run/concept counts — maintenance and temporal refresh pass. |

### This Curation Pass (Graph Vault Reflection 37 — June 17, 2026: Entity & Concept Curation, Wikilink Cleanup)

| Action | Detail |
|--------|--------|
| **Created Switzerland entity** | Missing global entity for US protecting power in Iran since 1980. Referenced by US-Iran Switzerland run (#44, p=0.68) and Swiss direct democracy concept. Covers protecting power role, diplomatic infrastructure, Swiss political system for forecasting. |
| **Enriched leader-succession-compounding-probability concept** | Expanded from 2 Israeli PM cases → 5-case cross-domain "lottery ticket" leadership cluster: added Shaked (0.002), Haley (0.01), Mekonnen (0.01). Added 2 new subtypes (collapsed-base resurgence, ex-faction outsider). Added pooled calibration value analysis. |
| **Enriched short-window-expiration-cluster concept** | Added Kharg Island as 5th canonical case in the June 30, 2026 expiration cluster (was listed as "4" in original concept). Updated table, description, and wikilinks. |
| **Enriched Ayelet Shaked entity** | Added Israeli PM "lottery ticket" cluster section linking Golan, Ohana, and Shaked as 3 structurally-impossible PM bets. Added cross-references to run file, concept, Israeli domestic politics thread. |
| **Enriched Peru entity** | Removed placeholder text (broken `[[César Sánchez]]`, "verify exact name" comment, "TBD" candidates). Added proper election context, run #40 cross-reference, vote-margin structural analysis. |
|| **Fixed 3 entities with flat-path wikilinks** | Demeke Mekonnen (`[[ethiopia]]` → `[[domains/africa/entities/ethiopia]]`), Ethiopia entity (`[[demeke-mekonnen]]` → `[[domains/africa/entities/demeke-mekonnen]]`), Kharg Island (`[[strait-of-hormuz]]` → domain-organized paths). |
|| **Bumped vault version 2.30→2.31** | Updated index frontmatter, vault health (entities: 331→332), changelog entry. |

### Curation Pass 38 — June 18, 2026: US Bank Failure Probability Concept

|| Action | Detail |
||--------|--------|
|| **Created banking stability concept** | Filled structural gap in economics domain: zero prior coverage of bank failure probability, FDIC resolution, CRE stress, or systemic-risk exception. Created [[domains/economics/concepts/us-bank-failure-probability/_concept]] with base rate table (annual failures 2008-2026), conditioning multipliers (rate cycle ×5-10×, CRE stress ×3-5×), structural indicator framework (HTM losses, CRE exposure, deposit beta, problem bank list, discount window, FHLB advances), and 12-day Polymarket assessment (vault p_yes=0.015 vs market 0.075). Flagged by $9.9M "US bank failure by June 30" market. |
|| **Bumped vault version 2.31→2.32** | Updated index frontmatter, vault health (concepts: 164→165), changelog entry. |

### Curation Pass 39 — June 18, 2026: Event Enrichment, Wikilink Repair, Vault Health Verification

||| Action | Detail |
|||--------|--------|
||| **Enriched US-Iran peace deal event** | Added war-termination-velocity and peace-broker-diplomatic-playbook concept cross-references to significance frontmatter, immediate consequences section, and wikilinks section. Strengthens event→concept feedback loop. |
||| **Fixed broken wikilink** | Replaced nonexistent [[domains/global/concepts/elderly-leader-mortality-risk/_concept]] with correct [[domains/global/concepts/leadership-decapitation-negotiation-window]] in US-Iran peace deal event. Original concept was never created; correct concept existed as flat file at different path. |
||| **Verified June 30 expiration cluster readiness** | Confirmed 5 vault forecasts expiring June 30 (Israel-Lebanon withdrawal 0.07, Díaz-Canel 0.07, Raúl Castro 0.005, Trump-al-Sharaa 0.15, Kharg Island 0.01) all have run files, forecast files, and domain entity cross-references in place. |
||| **Bumped vault version 2.32→2.33** | Updated index frontmatter, vault health section date, changelog entry. |

### Curation Pass 40 — June 18, 2026: Timeline Count Fix, Colombia Runoff Event, June 30 Cluster Enrichment

||| Action | Detail |
|||--------|--------|
||| **Fixed timeline count in vault health** | Updated vault health from "22" to "23" timeline quarters to account for 2026-Q4 timeline file (created June 15). Also corrected entity count (332→364) and concept count (165→166) to match actual file system state. |
||| **Created Colombia runoff outcome monitoring event** | New [[events/colombia-runoff-2026-outcome]] event file positioned 3 days before June 21 runoff. Includes pre-resolution scenario table, outcome-tracking sections for each candidate, and post-resolution analysis framework for the largest active Polymarket political event ($29.9M). |
||| **Enriched short-window-expiration-cluster concept** | Added June 30 resolution monitoring section with post-resolution analysis framework tracking all 5 cluster forecasts. Added calibration cohort utility (5 forecasts as pooled calibration signal). Cross-referenced Díaz-Canel and Trump-al-Sharaa run files. |
||| **Added short-window-expiration-cluster cross-references** | Enriched Díaz-Canel run and Trump-al-Sharaa run files with wikilinks to [[domains/global/concepts/short-window-expiration-cluster/_concept]]. Both were part of the June 30 expiration cluster but lacked explicit concept cross-reference in their run file wikilinks. |
||| **Bumped vault version 2.33→2.34** | Updated index frontmatter, vault health section, changelog entry, events count (31→32). |

### Curation Pass 41 — June 18, 2026: Q2 Timeline Fixes, Cross-Reference Health

|||| Action | Detail |
||||---|--------|
|||| **Fixed Q2 timeline June 30 cluster count** | Corrected "Four vault forecasts" → "Five vault forecasts" in [[timeline/2026-Q2]] — Kharg Island (0.01) was missing from the cluster listing. Added domain diversity note (MENA, Latin America, US foreign policy, Iran energy) and direct run file link. |
|||| **Added Colombia runoff event link to Q2 timeline** | Linked [[events/colombia-runoff-2026-outcome]] from [[timeline/2026-Q2]] Colombia section — establishing bidirectional link between the timeline and the pre-resolution event monitoring file for tomorrow's $29.9M outcome. |
|||| **Bumped vault version 2.34→2.35** | Updated index frontmatter, purpose field, changelog entry. |

### Curation Pass 43 — June 19, 2026: Russia-Ukraine Entity Cluster, Run #48 Cataloging, Peace-Broker Enrichment

||| Action | Detail |
||---|--------|--------|
|| **Cataloged run #48** | Added [[runs/20260618-russia-ukraine-ceasefire]] (Russia-Ukraine ceasefire by Dec 31, p_yes=0.48, $99.3K Polymarket) to [[runs/_index]]. Vault estimate at 48% vs market 32.5% — above-market divergence driven by peace-broker diplomatic bandwidth cascade thesis. |
|| **Created Russia/Ukraine entity cluster** | Created 5 entity files filling a structural gap: [[domains/global/entities/russia]] (Putin decision calculus, ceasefire pathway analysis, structural position), [[domains/global/entities/ukraine]] (Zelenskyy decision calculus, armistice acceptability terms), [[domains/global/entities/vladimir-putin]] (leadership stability, negotiation style, succession risk), [[domains/global/entities/volodymyr-zelenskyy]] (wartime political survival, post-war trajectory), [[domains/global/entities/nato]] (alliance cohesion, Ukraine membership dynamics, transatlantic strain under Trump). Each entity links to run #48, the Russia-Ukraine war thread, and US-Russia diplomatic relations thread. |
|| **Enriched peace-broker-diplomatic-playbook concept** | Added [[domains/global/concepts/peace-broker-diplomatic-playbook/_concept#the-diplomatic-bandwidth-cascade|diplomatic bandwidth cascade]] sub-pattern — formalizes the insight that each completed deal frees NSC capacity for the next conflict. Includes cascade table (Alaska Summit → Armenia-Azerbaijan → Gaza → Iran → Ukraine), bandwidth mechanics, forecasting implications, and falsifiability condition (if Trump does NOT pivot to Ukraine in Q3 2026, the model is falsified). |
|| **Filled Gap #9 (Sweden election thread)** | Marked Priority-3 gap #9 as filled in [[_macro_gaps]] — [[domains/europe/threads/sweden-2026-election/_thread]] was created with full electoral mechanics, polling tracker framework, SD kingmaker paradox analysis, and campaign timeline. |
|| **Updated Q3 timeline** | Replaced generic "Alaska Framework Implementation" section with "Russia-Ukraine: Ceasefire Push (Phase 4 of Peace-Broker Cascade)" — adds specific Q3 watchpoints (bandwidth cascade, security guarantees, Zelenskyy domestic politics). Added 5 new entity wikilinks and run #48 reference. |
|| **Bumped vault version 2.36→2.37** | Updated index frontmatter, vault health (entities: 368→373, runs: 47→48), changelog entry. |

| ### Curation Pass 44 — June 20, 2026: Temporal Refresh for Colombia Runoff (T-24h) & June 30 Cluster (T-10)

||| Action | Detail |
||||---|--------|
|||| **Elevated Colombia runoff to election eve (T-24h)** | Updated [[events/colombia-runoff-2026-outcome]] with full Election Eve Watch section — market status ($29.9M Polymarket ~63-66%), campaign activity, silencing period, election-day risks, turnout projection, watchpoints (first tranche results, Bogotá margin). Largest single-market test of vault's structural methodology resolves tomorrow. |
|||| **Updated Q2 timeline status** | Refreshed [[timeline/2026-Q2]] Colombia section from T-48h → Election Eve (June 20) status. |
|||| **Refreshed June 30 cluster countdown** | Updated [[domains/global/concepts/short-window-expiration-cluster/_concept]] for T-10 (from T-11). All 5 forecasts remain structurally NO with no emerging mechanisms. Díaz-Canel base rate corrected: 11-day 0.15% → 10-day 0.14%. |
|||| **Pre-positioned Gap #18 postmortem** | Confirmed post-resolution analysis framework in place in Colombia event file (Sections 1-4) and _macro_gaps entry already open for Jun 22+ fill. No action needed now — structure is ready for tomorrow's resolution. |
|||| **No new runs to catalog** | All 48 forecast runs fully cataloged. The 9 [?] recent forecasts (0.89, 0.22, 0.48, 0.004, 0.15, 0.07, 0.005, 0.68, 0.002) all correspond to existing vault entries — no gaps found. |
|||| **Bumped vault version 2.37→2.38** | Updated index frontmatter, date, purpose, vault health date. No changes to entity/run/concept counts. |

|||||### Curation Pass 46 — June 20, 2026: Midterm Thread Consolidation & Duplicate Merge

||||| Action | Detail |
|||||--------|--------|
||||| **Merged duplicate midterm election threads** | Duplicate threads `2026-us-midterm-elections` (54 lines, stale May 21) and `us-2026-midterm-elections` (87 lines, better framework) merged into single canonical `us-2026-midterm-elections` thread (now 169 lines, enriched June 20). Old thread converted to redirect stub. Added: CA governor race ($24M market, Becerra vs Hilton), AI safety bill as midterm issue (p=0.15 post-crash), Iran peace deal as midterm X-factor, updated Senate battleground data, current market prices (House control, Senate control), expanded cross-cutting forces from 3 to 4 factors (added AI safety axis as cross-cutting concern). |
||||| **Promoted Pakistan entity** | Changed `domains/global/entities/pakistan` status from `seed` to `active` — entity has full mediator history, US-Iran shuttle diplomacy timeline, and direct forecasting relevance for July 31 diplomatic meeting (p=0.89). |
||||| **Fixed wikilink references** | Updated old duplicate path references in Q4 timeline wikilinks and _index.md active threads table to canonical `us-2026-midterm-elections` path. Redirect stub preserves backward compatibility for remaining reference files. |
||||| **Bumped vault version 2.39→2.40** | Updated index frontmatter, purpose field, changelog entry. Thread count stable (merged threads, net zero due to canonical enrichment). |

### Curation Pass 47 — June 20, 2026: Pre-Election Prep, Stale Run-Gap Cleanup, Entity Structural Repair

| Action | Detail |
|--------|--------|
| **Fixed Alaska Summit entity structural defects** | Entity had duplicate `---` frontmatter closure (3 separators → 1) and all 7 wikilinks used flat paths (`[[donald-trump]]`, `[[vladimir-putin]]`, `[[nato]]`, `[[russia-ukraine-war]]`) instead of domain-organized paths (`[[domains/usa/entities/donald-trump]]`, etc.). Rewrote entire entity: clean frontmatter, 14 domain-organized wikilinks, enhanced peace-broker/ceasefire significance framing with direct links to [[runs/20260618-russia-ukraine-ceasefire]] and [[domains/global/concepts/peace-broker-diplomatic-playbook/_concept]]. Also fixed 2 downstream flat-path `[[alaska-summit]]` references in US-Russia diplomatic relations and US-NATO relations thread wikilink sections — all 3 occurrences now use domain-organized paths.
| **Resolved stale gap notes in run #46 and #47** | Run #46 (US-Iran meeting, p=0.89) had a vault gap note requesting Pakistan mediator entity — filled in Pass 42. Run #47 (Sweden Kristersson, p=0.22) requested Sweden entity stubs and election thread — filled in Pass 42-43. Both gap notes updated with `(RESOLVED)` markers, direct wikilinks to the filled resources, and pass attribution. Removes confusion for future vault readers who would see gaps that were actually closed. |
| **Added Election Day (T-0) monitoring to Colombia event** | Inserted new `Election Day Watch (June 21, 2026 — T-0)` section between the T-24h section and the Significance paragraph. Includes: polls open/close times (8 AM COT / 4 PM COT), first-tranche signal analysis (de la Espriella >5pp at 30% as reliable indicator), Bogotá threshold math (Cepeda needs >15pp Bogotá margin), Polymarket resolution triggers, market payout impact ($18-20M shift), and explicit vault significance framing for tonight's resolution. Post-resolution postmortem sections (1-4) remain pre-positioned for Jun 22+ fill. |
| **Refreshed Q3 timeline Colombia section** | Changed heading from `Colombia Presidential Runoff (June 21 — T-48h as of June 19)` to `Colombia Presidential Runoff (June 21 — Election Day)`. Updated content with T-0 timing data, Polymarket price context (~63-66%), and redirect to event file for full monitoring framework. Removes stale T-48h timestamps. |
|| **Bumped vault version 2.40→2.41** | Updated index frontmatter, purpose field, Vault Health header, changelog entry. No changes to entity/run/concept counts — structural cleanup and pass enrichment pass.
|
|### Curation Pass 48 — June 20, 2026: New Run Vaulting (#49-52), Iran Enrichment Concept, Bandwidth Cascade Tracking, June 30 Cluster Expanded
|
|| Action | Detail |
||--------|--------|
|| **Cataloged 4 new forecast runs** | Added runs #49 (Hormuz traffic 0.58, $996K), #50 (Hijri Iran head of state 0.002, $98.9K), #51 (US-Iran meeting in Iran 0.008, $994K), #52 (Iran enrichment June 30 0.09, $9.9M) to [[runs/_index]]. Updated unresolved count to 49 pending. All 4 forecasts were agent-generated but un-vaulted — now fully cross-referenced. |
|| **Expanded June 30 cluster to 6 members** | Added Iran enrichment ($9.9M, p_yes=0.09) to [[domains/global/concepts/short-window-expiration-cluster/_concept]] — adds nuclear nonproliferation domain to what was previously a security/military cluster. Updated all 3 tables (overview, pre-resolution status, calibration cohort), pooled Brier calculation (0.0068 for all-NO resolution), and wikilinks. Cluster now spans 6 domains across 6 vault-managed forecasts expiring June 30. |
|| **Enriched Iran nuclear enrichment post-war concept** | Added run #52 reference, T-10 status tracking, June 30 cluster cross-reference, and peace-broker bandwidth cascade connection to [[domains/iran/concepts/iran-nuclear-enrichment-post-war/_concept]]. Concept now fully cross-referenced to its active forecast, cluster context, and the broader post-war diplomatic framework. |
|| **Added bandwidth cascade tracking section** | Inserted Cascade Tracking table into [[domains/global/concepts/peace-broker-diplomatic-playbook/_concept#the-diplomatic-bandwidth-cascade]] with T+6 status on Trump's Ukraine pivot (✅ confirmed), NSC bandwidth, Zelenskyy/Putin postures, European reaction, and market pricing. Added Q3 watchpoints table (Jul 1–Nov 3) with falsification thresholds: if no high-level US-Russia ceasefire talks by Oct 1, the cascade model is falsified and Ukraine forecast (0.48) should be revised to 0.15-0.25. Updated pit_cutoff to 2026-06-20. |
|| **Added Hormuz normalization, Iran enrichment, June 30 cluster to Q3 timeline** | Three new sections added to [[timeline/2026-Q3]]: Hormuz traffic normalization (July 15 deadline, vault 0.58), Iran enrichment status (June 30, vault 0.09), and the six-member June 30 expiration cluster. Updated compiled-from date to 2026-06-20. Added 7 new wikilinks including runs #49, #52, and the enrichment/short-window concepts. |
|| **Bumped vault version 2.41→2.42** | Updated index frontmatter, purpose field, Vault Health header (runs: 48→52, forecasts: 19→25, concepts: 166→167), changelog entry. No new entities created — all 4 forecasts reference existing entities. |
|
|||||### Curation Pass 45 — June 20, 2026: Concept Cross-Reference Enrichment & Structural Repair

||||| Action | Detail |
|||||--------|--------|
||||| **Fixed attention-scarcity concept structural defects** | Repaired duplicate frontmatter closure (3 `---` separators → 1). Replaced 5 broken flat-path wikilinks (`[[concepts/...]]`, `[[threads/...]]`) with domain-organized paths. Added bidirectional cross-references to peace-broker-diplomatic-playbook bandwidth cascade — they are complementary inverses (crises consume bandwidth; resolved deals free it). Added relationship comparison table. Updated `updated` field to 2026-06-20. |
||||| **Linked peace-broker concept with attention-scarcity** | Added attention-scarcity to related_concepts frontmatter. Added explicit cross-reference paragraph in bandwidth cascade section explaining the inverse relationship. Both concepts now mutually reinforce: attention-scarcity explains why Iran war (Feb-Jun 2026) consumed full bandwidth; bandwidth cascade explains why Iran peace deal freed it for Ukraine push. |
||||| **Updated Russia-Ukraine war thread** | Bumped `updated` from 2026-05-20 → 2026-06-20. Added related_forecasts frontmatter with run #48 (ceasefire, p_yes=0.48). Added ceasefire forecast reference to overview paragraph. Added peace-broker and attention-scarcity concepts to Related Concepts section with explanatory notes. Thread now fully cross-referenced to its active forecast and both supporting conceptual frameworks. |
|
|### Curation Pass 42 — June 18, 2026: New Run Cataloging, Sweden Entity Cluster, Pakistan Mediator Entity

||| Action | Detail |
|||--------|--------|
||| **Cataloged 2 new forecast runs** | Added runs #46 (US-Iran diplomatic meeting July 31, p_yes=0.89, $994K) and #47 (Kristersson next PM of Sweden, p_yes=0.22, $99.6K) to [[runs/_index]]. Updated unresolved count to 44 pending. |
||| **Fixed flat-path wikilinks in Sweden entity cluster** | Sweden entity had `[[ulf-kristersson]]`, `[[magdalena-andersson]]`, `[[jimmie-akesson]]` using flat paths. Fixed all 3 → domain-organized paths. Also fixed the same pattern in the child entity files (Kristersson, Andersson, Åkesson). Updated Kristersson PM market price from 17.5%→22.0% to match current Polymarket. |
||| **Created Pakistan mediator entity** | New [[domains/global/entities/pakistan]] entity covering Pakistan's unique mediator role in the US-Iran peace deal (June 14, 2026) — flagged by the US-Iran July 31 meeting run as a coverage gap. Covers mediator credentials, timeline of shuttle diplomacy, and forecasting relevance for follow-up meetings. |
||| **Updated Europe domain index** | Added Sweden entities to [[domains/europe/_domain]] (sweden, ulf-kristersson, magdalena-andersson, jimmie-akesson) as "Key Entities - Other" section — previously absent despite active $99.6K market. |
||| **Bumped vault version 2.35→2.36** | Updated index frontmatter, vault health (entities: 364→368, runs: 45→47), changelog entry. |

## Active Threads by Domain

| Domain | Thread | Status |
|--------|--------|--------|
|| USA | [[domains/usa/threads/2024-us-presidential-election/_thread]] | Active |
|| USA | [[domains/usa/threads/us-ai-safety-regulation-federal/_thread]] | Active |
|| USA | [[domains/usa/threads/trump-immigration-policy/_thread]] | Active |
|| USA | [[domains/usa/threads/us-2026-midterm-elections/_thread]] | Active (redirect consolidated) |
|| USA | [[domains/usa/threads/2026-la-mayoral-election/_thread]] | Active |
|| USA | [[domains/usa/threads/clarity-act/_thread]] | Active |
|| Latin America | [[domains/latin-america/threads/brazil-2026-presidential-election/_thread]] | Active |
|| Latin America | [[domains/latin-america/threads/venezuela-authoritarian-resilience/_thread]] | Active |
|| Latin America | [[domains/latin-america/threads/mexican-politics/_thread]] | Active |
|| Latin America | [[domains/latin-america/threads/us-cuba-relations/_thread]] | Active |
||| Latin America | [[domains/latin-america/threads/us-colombia-relations/_thread]] | Active |
||| MENA | [[domains/mena/threads/israel-hamas-war-ceasefire/_thread]] | Active |
||| MENA | [[domains/mena/threads/iran-israel-escalation/_thread]] | Active |
||| MENA | [[domains/mena/threads/israeli-domestic-politics/_thread]] | Active | (Knesset elections, coalition dynamics, right-wing dominance, center-left decline, PM succession forecasting) |
||| MENA | [[domains/mena/threads/syria-post-assad-transition/_thread]] | Active | (HTS governance under al-Sharaa, terrorist designation barrier, international engagement) |
||| Health | [[domains/health/threads/respiratory-season-2025-26/_thread]] | Active |
||| Europe | [[domains/europe/threads/uk-domestic-politics/_thread]] | Active |
|| Sports | [[domains/sports/threads/tennis-challenger-forecasting/_thread]] | Active |
|| Sports | [[domains/sports/threads/ppa-pickleball-tour/_thread]] | Active |
|| France | [[domains/france/threads/2027-french-presidential-election/_thread]] | Active | (Le Pen eligibility, Bardella succession, centrist fragmentation). See [[domains/global/concepts/two-round-runoff-dynamics/_concept|two-round runoff dynamics]] concept for structural framework. |
| Technology | [[domains/usa/threads/us-ai-safety-regulation-federal/_thread]] | Active | US federal AI safety bill (Polymarket $98.8K). Vault p_yes=0.38. Market converging toward vault estimate (↓8.5pp in 2 days). |

## Key Concepts by Domain

| Domain | Key Concepts |
|--------|-------------|
|| Africa | african-multi-domain-state-fragility (three-dimension model: electoral, conflict, governance) |
||| Global | short-horizon-momentum-check, short-horizon-procedural-certainty, market-vault-structural-divergence, theater-specific-strike-base-rates, sibling-run-calibration-divergence, endemic-pathogen-criteria-ambiguity, ceasefire-pathway-decomposition, pre-negotiated-framework-activation, structural-improbability-check, policy-expectation-without-delivery, midterm-legislative-bandwidth, two-round-runoff-dynamics, short-window-expiration-cluster, leader-succession-compounding-probability |
| USA | comprehensive-tech-regulation-gridlock, program-restriction-vs-elimination, first-100-days-action-horizon, trump-tariff-escalation-bargaining, second-term-cabinet-formation, presidential-coattail-variability |
| Latin America | authoritarian-electoral-facade, late-candidate-substitution, peronist-fragmentation-reconstitution, populist-coattail-legislative-wave, radical-reformer-political-survival, presidential-removal-risk, far-left-marginalization-polarization, argentina-milei-realignment, regional-third-way-squeeze |
|| MENA | transition-window-ceasefire-diplomacy, dual-presidential-endorsement-ceasefire, public-framework-announcement-commitment, ceasefire-trust-erosion-after-collapse, short-window-ceasefire-probability |
|| Iran | iran-nuclear-enrichment-post-war (post-war enrichment capacity, verification framework, Khamenei succession policy uncertainty) |
|| Health | transport-vector-outbreak, endemic-pathogen-criteria-ambiguity, seasonal-baseline, outbreak-escalation, vaccine-effectiveness |
| Economics | regulatory-precedent-cascade, monetary-policy-cycle-phases, central-bank-forward-guidance, dollar-smile-theory, us-bank-failure-probability |
| Sports | sports-market-liquidity-signal, sports-pairing-chemistry |
| Europe | swiss-direct-democracy (dual-majority initiative system, EU treaty constraint) |
| France | french-electoral-dynamics (two-round system, cohabitation risk, cordon sanitaire, RN ceiling) |
| Southeast Asia | head-of-state-icc-arrest-dynamics (post-exit cooperation, enforcement chain, ASEAN precedent) |
| Technology | generational-replacement, platform-owner-amplification, national-security-tech-ban |

## Recent Forecasts (Cataloged in Runs Index)

|All recent forecasts are now cataloged in the [[runs/_index]] table (55 runs total). Notable recent additions:

||| Forecast | p_yes | Date | Market | Key Insight |
||----------|-------|------|--------|-------------|
|| Raúl Castro US custody by June 30 | 0.005 | 2026-05-22 | Polymarket $99K | [[domains/global/concepts/structural-improbability-check/_concept|Structural impossibility]]: 5 mechanisms checked, zero active paths. New canonical case for [[domains/global/concepts/market-vault-structural-divergence|market-vault-structural-divergence]] (#3: zero-mechanism type) |
|| US strike on Colombia by Dec 31 | 0.12 | 2026-05-22 | Polymarket $993K | [[domains/global/concepts/theater-specific-strike-base-rates|Theater-specific strike base rates]]: Latin America is inactive theater (zero strikes in 35+ years). Divergence case #4 (theater-level base rate override). |
|| Colombia 1st round winner outright | 0.08 | 2026-05-21 | Polymarket $6.1M | Fragmented right wing; Cepeda ceiling at ~46-48% |
|| MV Hondius: 5+ off-ship (tournament) | 0.24 | 2026-05-21 | Metaculus #43465 | Same outbreak as cup run (0.20), different source platform; Andes H2H ceiling keeps P below 0.25 |
|| Tereza Cristina wins Brazil 2026 | 0.005 | 2026-05-21 | Polymarket $994K | Crowded right-wing field; Centrao placeholder |
|| US enacts AI safety bill before 2027 | 0.38 | 2026-05-23 | Polymarket $98.8K | Dynamic convergence case: market corrected 49.5%→41.5% toward vault's 38% over 2 days. [[domains/global/concepts/market-vault-structural-divergence|Market-vault convergence]] Case 6. |
|| Cloobeck wins CA Governor | 0.002 | 2026-05-22 | Polymarket $994K | Vault-market convergence case: vault 0.002 is within ±0.001 of Polymarket YES (0.0015). Contrast to Raúl Castro divergence. 5 structural blockers identified. |
|| Switzerland &quot;No to 10M&quot; initiative | 0.22 | **RESOLVED NO** (Jun 14) | Swiss vote | [[domains/europe/concepts/swiss-direct-democracy|Swiss direct democracy]]: dual-majority + EU treaty + Fed Council opposition → structural NO. Outcome: ~55% NO. Brier: 0.0484. |
|| Colombia pres. runoff (Jun 21) | 0.35-0.40 Cepeda | 2026-05-23 | Polymarket $29.9M | [[domains/global/concepts/two-round-runoff-dynamics/_concept|Two-round runoff dynamics]]: right-wing consolidation advantage, Cepeda ceiling, first-round margin as leading indicator. Largest active political market. |
|| AI safety bill swing (↓8.5pp) | 0.38 | 2026-05-23 | Polymarket $98.8K | Swing report confirming vault analysis: market corrected from 49.5% to 41.5% toward vault's 38% as structural gridlock factors crystallized. |
|| Kamala Harris CA Governor | 0.005 | 2026-05-23 | Polymarket $994K | Two resolved announcement markets (NO) create near-deterministic NO on CA governorship. Near-zero structural impossibility despite high-profile name. |
||| Michael Younger CA Governor | 0.002 | 2026-05-23 | Polymarket $993K | Zero-name-recognition + zero-fundraising + zero-endorsements = structural near-zero in top-two primary. Vault created entity stub during this cycle. |
||| Yair Golan next PM of Israel | 0.004 | 2026-06-16 | Polymarket $988K | [[domains/global/concepts/leader-succession-compounding-probability/_concept|Leader-succession-compounding-probability]]: 4-gate chain (center-left win × Golan leads × coalition accepts × Netanyahu not blocking) ≈ 0.27%. New canonical case for Israeli succession forecasting. |
||| Amir Ohana next PM of Israel | 0.005 | 2026-06-16 | Polymarket $990K | Same compounding chain (Likud win × Netanyahu departure × Ohana selected over 3+ rivals × coalition approves ≈ 0.17%). Market at 0.5% within 3× of vault estimate. |
||| Trump speaks to al-Sharaa in June | 0.15 | 2026-06-16 | Polymarket $99K | [[domains/global/concepts/short-window-expiration-cluster/_concept|Short-window expiration cluster]]: 14 days + HTS terrorist designation + Iran deal bandwidth = structural NO. Market at 32.5% appears overpriced. |
|||| Díaz-Canel out as leader by June 30 | 0.07 | 2026-06-16 | Polymarket $992K | [[domains/global/concepts/short-window-expiration-cluster/_concept|Short-window expiration cluster]]: 14-day base rate (~0.2%) vs market (7.5%) = 37× overpricing. No visible health/coup/resignation indicators. |
|||| AI safety bill swing (Jun 15) | 0.15 | 2026-06-15 | Polymarket $98.8K | Major swing: market crashed 41.5%→17.0%. Vault at 15% aligns with new market. Gridlock analysis vindicated. |
|||| US-Iran next meeting in Switzerland | 0.68 | 2026-06-15 | Polymarket $991K | Switzerland's 45-year protecting power monopoly creates structural advantage. Market at 63.7% may underweight neutral venue infrastructure. |
|||| Kharg Island by June 30 | 0.01 | 2026-06-15 | Polymarket $995K | Near-structural NO: 3 scenarios examined (military seizure, IRGC revolt, Khamenei succession) all near-zero. Part of June 30 expiration cluster. ||
|||| Israel withdraws from Lebanon by June 30 | **0.07** | 2026-06-15 | Polymarket $994K | [[runs/20260615-israel-lebanon-withdrawal|Timeline-arithmetic structural NO]]: 14-day window &lt;&lt; 4-8 week median withdrawal cycle. Hezbollah proxy autonomy post-Khamenei gives Israel continuing security rationale. Market at 94.5% NO — strong liquid consensus. |
||||| US-Iran diplomatic meeting by July 31 | **0.89** | 2026-06-18 | Polymarket $994K | Post-peace-deal structural necessity: peace deal requires implementation meeting. 3-chain probability (deal holds × meeting scheduled × occurs by deadline ≈ 0.89). Pakistan as natural mediator host. Market well-calibrated. [[runs/20260618-us-iran-diplomatic-meeting-july31]] |
||||| Kristersson next PM of Sweden | **0.22** | 2026-06-18 | Polymarket $99.6K | Left bloc polling lead (~50-52% vs right bloc ~46-48%) + SD kingmaker paradox creates coalition fragility. Market at 22% properly reflects structural dynamics. [[runs/20260618-sweden-kristersson-pm]] |
||| **Russia-Ukraine ceasefire by Dec 31** | **0.48** | 2026-06-18 | Polymarket $99.3K | Peace-broker bandwidth cascade: Iran deal completion (Jun 14) frees NSC capacity for Ukraine push. Vault at 48% vs market 32.5% — divergence driven by cascade thesis. Alaska Summit template (Aug 2025) provides negotiation framework. Korea-style armistice is most probable format. [[runs/20260618-russia-ukraine-ceasefire]] |
||| **Andy Burnham next UK PM** | **0.35** | 2026-06-21 | Polymarket $991K | Massive market-vault divergence: vault at 35% vs market at 95.3%. Burnham faces 3 sequential hurdles (by-election → Labour contest → OMOV win) within 6.5 months. No non-MP has ever won Labour leadership. Market may be pricing insider knowledge or resolution quirk. [[runs/20260621-andy-burnham-uk-pm]] |
||| **Cuban regime falls in 2026** | **0.18** | 2026-06-21 | Polymarket $987K | Market-aligned structural NO: 67-year one-party regime durability, security apparatus loyalty, emigration pressure valve (500K+ since 2022). Tail risks: Díaz-Canel health, economic collapse, natural disaster. [[runs/20260621-cuba-regime-falls]] |
||| **UAE leaves GCC in 2026** | **0.10** | 2026-06-21 | Polymarket $98.5K | Calibration consensus: vault aligns with market 10%. GCC 43-year zero-exit base rate (since 1981). US-Iran peace deal reduces Gulf tensions, making exit LESS likely. Tail risk: major Saudi-UAE rupture. [[runs/20260621-uae-leave-gcc]] |

## See Also
- [[_macro_gaps]] — Identified coverage gaps
- [[_procedure]] — Cross-domain structural reasoning procedure
- [[runs/_index]] — Full run catalog
- [[events/_index]] — Event files index
