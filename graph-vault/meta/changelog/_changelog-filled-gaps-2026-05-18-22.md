---
type: changelog-archive
tags: [meta, archive]
created: 2026-05-22
source: _macro_gaps.md
purpose: "Archived filled-gaps log from _macro_gaps.md (May 18-22, 2026). Kept for historical reference."
---

# Archived Filled Gaps Log (May 18-22, 2026)

Archived from `_macro_gaps.md` during the May 22, 2026 curation pass. The live `_macro_gaps.md` now only tracks current open gaps. This archive preserves the full chronological record of every gap identification and remediation since the vault's creation.

## Filled Gaps (This Curation Pass — May 21, 2026 — Graph Vault Reflection 3: France & Macro-Economic)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — No Marine Le Pen entity stub** — RN figurehead, three-time presidential candidate, EU funds legal case (2026 verdict), eligibility uncertainty for 2027 | Created: `domains/france/entities/marine-le-pen.md` — full entity with legal case detail (EU parliamentary assistant funds trial, 5-year electoral ban risk), candidacy decision framework (step aside for Bardella vs fight), forecasting significance section, and cross-wikilinks to all France entities, thread, and concept. | 2026-05-21 |
| **Priority 2 — No Édouard Philippe entity stub** — Horizons leader, former PM (2017-2020), Mayor of Le Havre, potential 2027 center-right candidate (12-18% polling) | Created: `domains/france/entities/edouard-philippe.md` — centrist positioning analysis, relationship to Renaissance succession, centrist fragmentation dynamics, cohabitation implications, head-to-head polling vs RN performance. | 2026-05-21 |
| **Priority 2 — No Gabriel Attal entity stub** — Former PM (Jan-Sep 2024), youngest in French history, Renaissance figure | Created: `domains/france/entities/gabriel-attal.md` — brief premiership analysis, 2027 candidacy feasibility (age, centrist field congestion, name recognition), political rhythm calculation, and relationship to wider centrist field. | 2026-05-21 |
| **Priority 2 — No French electoral dynamics concept** — Vault had 2027 election thread but no structural framework for two-round system, cohabitation risk, or cordon sanitaire erosion | Created: `domains/france/concepts/french-electoral-dynamics.md` — comprehensive 6-section concept: two-round mechanics, cordon sanitaire historical erosion (2002:82%→2017:66%→2022:58%), 5th Republic constitutional structure (semi-presidential system, Article 6 term limit, cohabitation risk), RN ceiling projection (projected 26-28% first round / 48-52% runoff at current trend rate), left fragmentation dynamics, centrist fragmentation patterns, and 6-step forecasting checklist. Cross-linked to all 5 entity files, thread, and europe domain. | 2026-05-21 |
| **Priority 2 — Existing France entities lacked cross-wikilinks** — Macron entity had no links to Le Pen/Philippe/Attal; Bardella entity had no wikilinks section at all | Macron entity enriched: added marine-le-pen, edouard-philippe, gabriel-attal, french-electoral-dynamics to Wikilinks section. Bardella entity enriched: added full Wikilinks section with domain, le-pen, macron, thread, concept. Thread enriched: added Key Entities section referencing all 5 entities + concept. France _domain.md rewritten: added Attal entity, electoral-dynamics concept, expanded Coverage History. France domain grew from 4 to 8 files. | 2026-05-21 |
| **Priority 3 — No yield-curve-dynamics concept** — Vault discussed yield curve inversion in macro indicators but had no dedicated structural framework | Created: `domains/economics/concepts/yield-curve-dynamics.md` — 4 yield curve regimes (normal, flat, inverted, steepening) with driving mechanisms, inversion signal interpretation table mapping curve state to Fed direction and confidence, recession forecasting methodology (inversion duration analysis, structural change caution for post-QE era), canonical cases (2022-24 longest inversion ever without recession, 2007-08 inversion-to-recession 16-month lag), dollar/EM channel linkage, and cross-references to monetary-policy-cycle-phases, forward-guidance, and midterm-legislative-bandwidth concepts. | 2026-05-21 |
| **Brazil thread wikilinks used flat paths** (broken after domain migration) — 8 wikilinks in `domains/latin-america/threads/brazil-2026-presidential-election/_thread.md` referenced `[[luiz-inacio-lula-da-silva]]` instead of `[[domains/latin-america/entities/luiz-inacio-lula-da-silva]]` | All 8 entity wikilinks corrected to domain-prefixed paths (key actors + wikilinks section). Same fix applied to forecast file `forecasts/2026-05-21-tereza-cristina-brazil-2026.md`. | 2026-05-21 |
| **No Michelle Bolsonaro entity stub** — referenced in Brazil thread as `[[michelle-bolsonaro]]` but file didn't exist | Created: `domains/latin-america/entities/michelle-bolsonaro.md` — former first lady, PL candidate, evangelical appeal, 3.3% PM price | 2026-05-21 |
| **No Eduardo Bolsonaro entity stub** — referenced in Brazil thread as `[[eduardo-bolsonaro]]` but file didn't exist | Created: `domains/latin-america/entities/eduardo-bolsonaro.md` — Federal Deputy, ZERO risk candidate, Bolsonaro campaign surrogate | 2026-05-21 |
| **Latin America domain index missing Brazil thread** — had Mexican and Venezuelan threads but no Brazil entry despite active presidential race | Added thread ref to frontmatter; added Active Threads and Active Entities sections to body | 2026-05-21 |
| **Root _index.md bloated to 974 lines** (~920 lines of accumulated changelogs from May 18-21 curation passes) | Trimmed to 105-line lean navigational index. Full changelogs archived to `meta/changelog/_changelog-2026-05-18-21.md` | 2026-05-21 |
| **Runs index missing 2 new forecasts** — Tereza Cristina (p_yes=0.005) and AI Safety Bill (p_yes=0.38) had no entries in the runs catalog | Added rows 21-22 to runs table, updated header to "22 runs", added Other Forecasts section | 2026-05-21 |

## Filled Gaps (May 20, 2026 — Graph Vault Reflection Pass)

| Gap | Resolution | Date |
|-----|-----------|------|
| No Europe domain index | Created: `domains/europe/_domain.md` with 25 file catalog, UK/Slovenia/ECB entity map, domain status notes | 2026-05-20 |
| No MENA domain index | Created: `domains/mena/_domain.md` with 19 file catalog, event surface, cross-domain dependency notes | 2026-05-20 |
| Economics _domain.md outdated (missing 10+ entities and concepts) | Rewrote: full entity catalog (Fed, SEC, crypto currencies/institutions, ETF issuers), added missing concepts (monetary-policy-cycle-phases, bitcoin-halving-cycle, post-covid-inflation-surge, regulatory-precedent-cascade), added eurozone thread | 2026-05-20 |
| Duplicate entity: fed-chair-jerome-powell.md (thin stub) and jerome-powell.md (real entity) | Merged: enriched jerome-powell.md with 2022-2026 cycle timeline, Trump 2nd-term independence pressure, easing cycle context. Converted fed-chair-jerome-powell.md to redirect stub | 2026-05-20 |
| No events index | Created: `events/_index.md` surfacing all 6 event files nested under thread directories; identified 5 high-value missing events | 2026-05-20 |
| No runs index | Created: `runs/_index.md` cataloging 2 run files with calibration summary | 2026-05-20 |
| No Bitcoin Feb 2026 drawdown event | Created: `events/bitcoin-feb-2026-drawdown.md` documenting $97K→$60K drawdown as canonical crypto-macro linkage failure case | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Per-Question Reflection, Q5 Turkish Central Bank)

| Gap | Resolution | Date |
|-----|-----------|------|
| No TCMB entity stub | Created: `domains/mena/entities/turkish-central-bank-tcmb.md` with governance structure, historical regimes, forecasting considerations | 2026-05-20 |
| No Fatih Karahan entity stub | Created: `domains/mena/entities/fatih-karahan.md` — current TCMB governor | 2026-05-20 |
| No Mehmet Simsek entity stub | Created: `domains/mena/entities/mehmet-simsek.md` — Finance Minister and "political shield" | 2026-05-20 |
| No Turkish monetary policy thread | Created: `domains/mena/threads/turkish-monetary-policy-normalization/_thread.md` tracking the 2023-present normalization cycle with causal framework for forecasting rate decisions | 2026-05-20 |
| No EM central bank normalization concept | Created: `domains/mena/concepts/em-central-bank-credibility-normalization/_concept.md` documenting the 5-phase pattern and forecast application | 2026-05-20 |
| Central-bank-rate-decision procedure was G4-only | Expanded to cover EM central banks with separate approach/calibration sections | 2026-05-20 |
| _procedure.md step 15 only listed G4 central banks | Extended to include TCMB, BCB, CBR, SARB, plus EM Central Bank Extension checklist | 2026-05-20 |
| _spec.md rule 14 only covered advanced-economy central banks | Added EM Central Bank Extension with specific coverage requirements | 2026-05-20 |

## Filled Gaps (May 19, 2026 — Graph Vault Reflection Pass — Round 2)

| Gap | Resolution | Date |
|-----|-----------|------|
| benjamin-netanyahu.md overwritten by vault_probe output | Restored original from git, enriched with full Appears In backlinks to threads, concepts, and quarters | 2026-05-19 |
| Top-15 entity files missing bidirectional backlinks | Added/enriched "Appears In" sections in: russia (9 threads + 5 concepts), nato (5 threads + 1 concept), bitcoin (corrected wrong format links, added missing threads/concepts), federal-reserve-system, keir-starmer, openai, volodymyr-zelenskyy, elon-musk, xi-jinping, jerome-powell | 2026-05-19 |
| Existing Appears In sections used wrong wikilink paths | bitcoin.md had `[[us-crypto-regulation]]` instead of `[[threads/us-crypto-regulation]]`, `[[2026-Q1]]` instead of `[[timeline/2026-Q1]]` — all corrected | 2026-05-19 |
| General backlink nutrition gap | Backlink coverage audit: 12/143 entity files had Appears In. After this pass ~27/143 have correct backlinks. Remaining 116 need future passes. | 2026-05-19 |

## Filled Gaps (May 18, 2026 — Graph Vault Reflection Pass)

| Gap | Resolution | Date |
|-----|-----------|------|
| No CME entity stub | Created: `entities/cme.md` | 2026-05-18 |
| No crypto exchange entity stubs | Created: `entities/microstrategy.md` (largest corporate BTC holder), `entities/tether.md` (USDT issuer), `entities/circle.md` (USDC issuer) | 2026-05-18 |
| No concept for question interpretation ambiguity | Created: `concepts/question-interpretation-ambiguity.md` | 2026-05-18 |
| No sports pairing chemistry concept | Created: `concepts/sports-pairing-chemistry.md` | 2026-05-18 |
| THIN Brazil entities (Lula, Bolsonaro, Tarcisio) | Enriched with active market context, relationships, forecasting variables | 2026-05-18 |
| THIN AI entities (OpenAI, Anthropic, Mistral, Google DeepMind) | Enriched with market position, competitive dynamics, regulatory context | 2026-05-18 |

## Filled Gaps (May 20, 2026 — Graph Vault Reflection Pass 2)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 1**: No crypto-market thread tracking price action, ETF flows, macro-crypto linkage | Created: `domains/economics/threads/crypto-market/_thread.md` — dedicated thread distinguishing market dynamics from regulatory coverage. Documents the Feb 2026 drawdown as canonical macro-crypto linkage failure case. | 2026-05-20 |
| Duplicate entity: gary-gensler in both economics/ and usa/ | Merged: `domains/usa/entities/gary-gensler.md` as canonical. `domains/economics/entities/gary-gensler.md` converted to redirect stub. | 2026-05-20 |
| No sports domain — Cervia tennis forecast (Brier 0.0001) was a freebie (zero vault signal) | Created: `domains/sports/_domain.md` — nascent domain documenting the gap. No actual sports content yet. | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Graph Vault Reflection Pass 3)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — Sports domain still empty after 2 passes** | Created: `domains/sports/entities/jannik-sinner.md` (ATP No.1, Sinner effect), `domains/sports/entities/cervia-challenger.md` (Challenger tournament), `domains/sports/concepts/sports-market-liquidity-signal/_concept.md` (liquidity threshold for filtering noise markets). Updated `_domain.md` to reflect 3-file coverage. | 2026-05-20 |
| **Priority 2 — fed-chair-jerome-powell redirect used inconsistent format** | Standardized to `entity-redirect` type with full `redirect_target` wikilink, matching the gary-gensler redirect convention established in Pass 2. | 2026-05-20 |
| **Structural: _macro_gaps.md frontmatter typo** | Fixed `tags: [meta]e` → `tags: [meta]` (trailing 'e' was breaking YAML parsing) | 2026-05-20 |
| **Structural: bitcoin-feb-2026-drawdown.md broken thread wikilinks** | Fixed 2 wikilinks: `[[...threads/us-crypto-regulation]]` → `[[...threads/us-crypto-regulation/_thread]]` (same for monetary-policy thread). These were dangling due to missing `/_thread` suffix. | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Per-Question Reflection, Q48 H-1B)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 1 — No US immigration policy content at all** | Created: `domains/usa/threads/trump-immigration-policy/_thread.md` — master thread for Trump 2nd-term immigration policy with H-1B faction analysis (nativist Miller wing vs tech Musk wing), first 100-days action probability table, and H-1B elimination feasibility analysis | 2026-05-20 |
| **Priority 1 — No framework distinguishing program restriction vs elimination** | Created: `domains/usa/concepts/program-restriction-vs-elimination/_concept.md` — structured analysis of why elimination is structurally harder than restriction, with Youngstown legal framework, APA timeline constraints, political economy ("ally fallacy"), historical examples table, and diagnostic questions for forecasters | 2026-05-20 |
| **Priority 1 — No "first 100 days" time-horizon concept** | Created: `domains/usa/concepts/first-100-days-action-horizon/_concept.md` — what fits and doesn't fit in 100 days, with budget constraint analysis, empirical track record table, and H-1B elimination application showing near-zero probability | 2026-05-20 |
| **Priority 1 — No Stephen Miller entity** | Created: `domains/usa/entities/stephen-miller.md` — Dep. Chief of Staff for Policy, architect of Trump's immigration restriction agenda, H-1B positions, influence assessment, and constraint mapping against tech coalition | 2026-05-20 |
| **Priority 1 — No USCIS entity** | Created: `domains/usa/entities/uscis.md` — agency administering H-1B, fee-funded structure, policy discretion levers, backlog-as-restriction dynamic | 2026-05-20 |
| **Priority 1 — No executive action feasibility assessment procedure** | Created: `domains/usa/procedures/executive-action-feasibility-assessment.md` — 8-step procedure: action type classification, statutory basis check, time horizon assessment, coalition constraint evaluation, precedent check, legal vulnerability assessment, Bayesian updating from market prices, combine and output | 2026-05-20 |
| **Elon Musk entity missing H-1B advocacy role** | Updated: `domains/usa/entities/elon-musk.md` — added High-Skilled Immigration (H-1B) Advocacy section documenting Tesla/SpaceX reliance on H-1B, public pro-immigration statements, Musk-Ramaswamy alignment, and faction counterweight analysis vs Miller | 2026-05-20 |
| **Donald Trump entity missing immigration policy connections** | Updated: `domains/usa/entities/donald-trump.md` — linked to new immigration thread, concepts, entities, and procedure via Wikilinks section | 2026-05-20 |

## Filled Gaps (Q51 AI Safety Bill Reflection - May 20, 2026)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 1 — No federal AI safety regulation thread** | Created: `domains/usa/threads/us-ai-safety-regulation-federal/_thread.md` — tracks legislative journey from 2023 through 2026, structural barriers, key actors, forecasting significance | 2026-05-20 |
| **Priority 1 — No concept for why comprehensive tech regulation fails** | Created: `domains/usa/concepts/comprehensive-tech-regulation-gridlock/_concept.md` — six-barrier framework (polarization, lobbying, competing approaches, China competition fear, preemption deadlock, EO substitution) with 7 diagnostic questions and default priors | 2026-05-20 |
| **Priority 2 — No Chuck Schumer entity stub** | Created: `domains/usa/entities/chuck-schumer.md` — Senate Majority Leader, SAFE Innovation Framework architect, AI Insight Forum convener | 2026-05-20 |
| **Priority 2 — No NIST AI Safety Institute entity stub** | Created: `domains/usa/entities/nist-ai-safety-institute.md` — Biden-created, Trump-defanged government body | 2026-05-20 |
| **Critical — 2025-Q1 timeline had zero AI legislation coverage** | Updated: `timeline/2025-Q1.md` — added AI Safety Legislation subsection documenting Trump's EO revocation of Biden's framework, bipartisan bill reintroduction without markup, and six-barrier gridlock dynamics | 2026-05-20 |
| **Cross-links — State-level AI thread missing federal reference** | Updated: `domains/usa/threads/state-level-ai-regulation/_thread.md` — added federal thread to Related Threads section | 2026-05-20 |
| **Cross-links — USA domain index missing thread & concept** | Updated: `domains/usa/_domain.md` — added federal AI regulation thread and comprehensive-tech-regulation-gridlock concept | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Per-Question Reflection, Q50 Israel Ceasefire Announcement)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 1 — Ceasefire-announcement-ratification-gap concept conflated mediator announcements with party announcements** | Rewrote `domains/global/concepts/ceasefire-announcement-ratification-gap.md` — added "WHO Announces" dimension with four-date tracking (framework acceptance, mediator announcement, party announcement, ratification, effective). Corrected Jan 2025 entry to distinguish mediator framework announcement (Jan 15, by US/Qatar) from Israel's formal announcement (Jan 17, cabinet approval). Added gold_50 error to Validated By table. | 2026-05-20 |
| **Priority 1 — No concept for the commitment trap when a superpower publicly announces a ceasefire framework before the local party's ratification** | Created: `domains/mena/concepts/public-framework-announcement-commitment/_concept.md` — documents the mechanism by which a public mediator announcement makes party rejection politically impossible (commitment trap), with Bayesian priors (~90-95% P of ratification within 1-3 days), canonical case analysis of gold_50 error, and forecasting application. | 2026-05-20 |
| **Priority 1 — No entity stubs for key envoys who coordinated the Jan 2025 ceasefire transition** | Created: `domains/usa/entities/steve-witkoff.md` (Trump's Middle East envoy), `domains/usa/entities/brett-mcgurk.md` (Biden's Middle East coordinator) — both document their role in the Jan 2025 transition-period ceasefire coordination, with significance for forecasting future transition-window ceasefires. | 2026-05-20 |
| **Priority 1 — Ceasefire-announcement-forecast procedure lacked a "check for prior mediator announcement" step** | Rewrote `domains/mena/procedures/ceasefire-announcement-forecast.md` — added Phase 0 (Resolution Criteria Analysis) with explicit "WHO has already acted?" check, decision rule for when a mediator has announced but party hasn't ratified, and load-instruction for new public-framework-announcement-commitment concept. | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Graph Vault Reflection Pass)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — No tournament draw verification procedure in sports domain** | Created: `domains/sports/procedures/tournament-draw-verification.md` — 7-step procedure formalizing the method that resolved the Cervia tennis run correctly (draw check + liquidity filter). Covers ATP/Challenger/WTA draw sources, round compatibility, liquidity thresholds, and probability calibration. | 2026-05-20 |
| **Priority 2 — SBR concept had no explicit canonical failure case** | Enriched: `domains/economics/concepts/strategic-bitcoin-reserve.md` — added Canonical Failure Case section documenting the Feb 2026 drawdown ($97K→$60K) as the clearest instance of expectation-without-delivery. Also fixed 6 broken wikilinks using unqualified paths and added cross-references to the BTC run and drawdown event. | 2026-05-20 |
| No cross-domain concept for campaign-promise-to-market-disappointment pattern | Created: `domains/global/concepts/policy-expectation-without-delivery.md` — 4-phase pattern (Promise→Hype, Expectation Lock-In, Delay Without Communication, Disappointment Cascade) with probability calibration table and canonical cases across SBR, tariff policy, and H-1B domains. | 2026-05-20 |

## Filled Gaps (May 20, 2026 — Q21/30 PIT Blind Test Reflection, Israel Ceasefire Jan 17)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — No event file for the May 2024 ceasefire framework** (the pre-existing framework cited across 7+ files that Hamas accepted in May 2024 but couldn't activate until Jan 2025) | Created: `domains/mena/threads/gaza-ceasefire-negotiations-2025/events/may-2024-ceasefire-framework.md` — documents three-phase proposal, timeline of its 8-month stall, why it failed in May and succeeded in Jan, and five forecasting lessons. | 2026-05-20 |
| **Priority 2 — No concept for "pre-negotiated framework awaiting activation"** (pattern where a detailed framework exists but needs political conditions to align — distinct from negotiation from scratch) | Created: `domains/global/concepts/pre-negotiated-framework-activation/_concept.md` — formalizes the pattern with activation delay formula (0-30d to 18mo+), Bayesian priors (60-80% P within 3 months of blocking condition resolution), canonical Israel-Hamas case, and cross-conflict applicability (Colombia, Sudan, Yemen, Ukraine). | 2026-05-20 |
| **Priority 3 — Missing MENA entity stubs**: Khalil al-Hayya (Hamas lead negotiator post-Sinwar), Yoav Gallant (Israeli defense minister, ICC warrant), Mohammed Deif (Hamas military commander, killed July 2024) | Created: Three entity stubs at `domains/mena/entities/khalil-al-hayya.md`, `yoav-gallant.md`, `mohammed-deif.md` — each documents role in ceasefire dynamics and forecasting significance. | 2026-05-20 |
| **Priority 2 — 2025-Q1 timeline ceasefire entry too brief** — one line saying "Jan 15 — Israel and Hamas approve ceasefire" risked gold_50-type confusion between mediator and party announcements | Updated: `timeline/2025-Q1.md` — expanded entry with four-date tracking, WHO-announces distinction, explicit forecasting note, and links to pre-negotiated framework activation concept. | 2026-05-20 |
| **Priority 2 — Transition-window concept lacked cross-conflict contrast** — could be applied without distinguishing transition-window vs superpower-combatant-entry mechanisms | Updated: `domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept.md` — added Iran-Israel June 2025 as negative case, with comparison table across 5 dimensions (trigger, key actor, timeline, P range, preconditions). | 2026-05-20 |

## Filled Gaps (May 21, 2026 — Graph Vault Reflection Pass)

### Health Domain: Hantavirus Coverage Gap

The two Metaculus Cup runs on hantavirus (MV Hondius off-ship cases p_yes=0.2, hantavirus 100+ cases p_yes=0.07) exposed the health domain's complete lack of non-respiratory disease content. The vault had zero hantavirus-specific files despite this being a real-world outbreak with active forecasting questions.

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 1 — No hantavirus entity stub** | Created: `domains/health/entities/hantavirus.md` — full pathogen entity with strain table, historical outbreak base rates, forecasting rules, cross-refs to runs | 2026-05-21 |
| **Priority 1 — No MV Hondius event file** | Created: `events/mv-hondius-hantavirus-outbreak-2026.md` — outbreak chronology, risk factors (Andes H2H on cruise ship), mitigation factors, canonical case analysis | 2026-05-21 |
| **Priority 2 — No transport-vector outbreak concept** | Created: `domains/health/concepts/transport-vector-outbreak.md` — two-phase pattern (index cluster + secondary dispersion), contrast table with agricultural spillover model, forecasting checklist, Diamond Princess/Grand Princess canonical cases | 2026-05-21 |
| — | **runs/_index.md critically outdated** (2 of 21 runs cataloged) | Rewrote: full catalog of 21 runs with table, calibration summary, gold set domain breakdown, closed-loop remediation tracking | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Graph Vault Reflection Pass 2)

This pass enriched WHO/CDC entities with hantavirus context, created Oceanwide Expeditions stub, and formalized the market-vault-structural-divergence pattern observed across 4 gold-set runs.

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — Health domain index appeared stale** | Verified: health _domain.md already lists hantavirus entity + transport-vector concept. Gap was already filled in pass 1; _macro_gaps.md now corrected. | 2026-05-21 |
| **Priority 2 — No Oceanwide Expeditions entity** | Created: `domains/health/entities/oceanwide-expeditions.md` — MV Hondius operator, Dutch expedition cruise company, vessel fleet, liability/disclosure incentive analysis | 2026-05-21 |
| **Priority 2 — WHO entity respiratory-only, no hantavirus role** | Enriched: `domains/health/entities/who.md` — added Hantavirus Surveillance Role section (IHR notification, risk assessment, multi-country coordination, Andes tracking gap). Added wikilinks to hantavirus entity, transport-vector concept, MV Hondius event. | 2026-05-21 |
| **Priority 2 — CDC entity respiratory-only, no hantavirus surveillance context** | Enriched: `domains/health/entities/cdc.md` — added Hantavirus Surveillance Context section (notifiable disease status, contact tracing role, Special Pathogens Branch, Sin Nombre baseline, Vessel Sanitation Program). Added wikilinks to hantavirus, transport-vector, MV Hondius event. | 2026-05-21 |
| **Priority 2 — MV Hondius event file lacked operator wikilink** | Added: `[[domains/health/entities/oceanwide-expeditions|Oceanwide Expeditions]]` wikilink to MV Hondius event file overview. | 2026-05-21 |
| **Priority 2 — No concept for vault diverging downward from market price** | Created: `domains/global/concepts/market-vault-structural-divergence.md` — formalizes pattern from SCOTUS TikTok (PM=0.24→vault=0.07) and Venezuela Maduro (PM=0.655→vault=0.12). 5 diagnostic questions for when divergence is justified, relationship to Rule 9 exception. | 2026-05-21 |
| **Priority 3 — Hantavirus entity missing WHO/CDC backlinks** | Added: `domains/health/entities/who` and `domains/health/entities/cdc` to hantavirus.md Appears In section with role annotations. | 2026-05-21 |

## Filled Gaps (Q28 Reflection — May 20, 2026)

| Gap | Resolution | Date |
|-----|-----------|------|
| No monetary-policy-cycle-phases concept | Created: `domains/economics/concepts/monetary-policy-cycle-phases/_concept.md` — 5-phase framework with default next moves, the "6-month plateau rule" making hikes near-impossible, and first-move magnitude pattern. | 2026-05-20 |
| Forward-guidance concept had no "prerequisite phase analysis" step | Updated `central-bank-forward-guidance.md` — added "Relationship to Cycle Phases" section and Step 0 to Forecasting Application. | 2026-05-20 |
| Procedure step 19 started with meeting mapping — no structural phase check before forward guidance | Updated `_procedure.md` — added "FIRST — identify the monetary policy cycle phase" sub-step. | 2026-05-20 |

## Weather Domain Gaps (May 18, 2026)

- No weather-operational-forecaster agent role stub created
- No geographic entities for Madrid, Spain, AEMET
- No weather thread tracking Madrid daily temperature series
- No concept file for exact-value integer temperature forecasting
- No seasonal-scale climate outlook for Iberian Peninsula

## Vault Creation Notes

- Bitcoin entity stub created by macro-economic-analyst on 2026-05-18
- bitcoin-halving-cycle concept created by technology-trajectory-analyst on 2026-05-18
- crypto-macro-linkages concept created by crypto-financial-markets-specialist on 2026-05-18
- US monetary policy thread updated through Jan 2026 by macro-economic-analyst
- US macro indicators table updated to Q1 2026 values by sibling agents
- MicroStrategy, Tether, Circle, and CME entity stubs created 2026-05-18
- 2024-Q1 timeline: Added Bitcoin ETF approval event (Jan 10, 2024)

## Filled Gaps (May 21, 2026 — Graph Vault Reflection Pass, Cross-Run Patterns)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — No concept for high-confidence YES under procedural lock-in** | Created: `domains/global/concepts/short-horizon-procedural-certainty/_concept.md` | 2026-05-21 |
| **Priority 2 — No concept for endemic pathogen criteria ambiguity** | Created: `domains/global/concepts/endemic-pathogen-criteria-ambiguity/_concept.md` | 2026-05-21 |
| **Priority 2 — No event file for Trump 2024 election win** | Created: `events/trump-2024-election-win.md` | 2026-05-21 |
| **Priority 2 — No event file for Assad regime collapse** | Created: `events/assad-regime-collapse-december-2024.md` | 2026-05-21 |
| **Priority 3 — Paired gold set runs lacked cross-references** | Added Cross-References sections linking sibling runs and concepts | 2026-05-21 |
| **Priority 3 — Runs index lacked cross-run pattern synthesis** | Enriched `runs/_index.md` with paired-question analysis and procedural-certainty cluster | 2026-05-21 |
| **Priority 3 — Events index missing Trump election and Assad collapse** | Updated `events/_index.md` with both events | 2026-05-21 |
| **Ethereum ETF run missing cross-references** | Added Cross-References section | 2026-05-21 |
| **Ethereum ETF and hantavirus runs missing endemic-pathogen-ambiguity concept ref** | Added concept ref to hantavirus 100-cases run | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Graph Vault Curation Pass)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Duplicate Rule 12 numbering** in _forecast_instructions.md | Renamed second Rule 12 → Rule 15 | 2026-05-21 |
| **_macro_gaps.md** current-priority gaps buried after 220+ lines of filled-gap logs | Restructured with `## Current Open Gaps` section at top | 2026-05-21 |
| **Hezbollah Ceasefire event file** | Created: `events/hezbollah-ceasefire-november-2024.md` | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Graph Vault Reflection 2)

| Gap | Resolution | Date |
|-----|-----------|------|
| **France domain had no _domain.md** | Created: `domains/france/_domain.md` | 2026-05-21 |
| **Emmanuel Macron entity was THIN** | Enriched with 2027 Election Context, term-limit, succession dynamics | 2026-05-21 |
| **_index.md claimed "14 domains"** | Fixed: domain count 14→12 | 2026-05-21 |
| **Priority 2: No BTC ETF flow tracking** | Created: `domains/economics/concepts/bitcoin-etf-flow-price-driver.md` | 2026-05-21 |
| **Priority 2: No Bitwise entity stub** | Created: `domains/economics/entities/bitwise.md` | 2026-05-21 |
| **Priority 2 gap list was stale** (claimed Fidelity/ARK 21Shares missing) | Audit-confirmed both exist; gap closed as stale | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Entity and Event Remediation)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Duplicate entities: pete-buttigieg/peter-buttigieg** | Merged into `pete-buttigieg.md` (canonical); `peter-buttigieg.md` → redirect stub | 2026-05-21 |
| **Duplicate entities: chuck-schumer/charles-schumer** | Merged into `chuck-schumer.md` (canonical); `charles-schumer.md` → redirect stub | 2026-05-21 |
| **No event file for Argentina $LIBRA Cryptogate (Feb 2025)** | Created event file | 2026-05-21 |
| **No event file for Argentina ANDIS Corruption Scandal (Aug 2025)** | Created event file | 2026-05-21 |
| **No concept for midterm legislative bandwidth** | Created: `domains/global/concepts/midterm-legislative-bandwidth.md` | 2026-05-21 |
| **events/_index.md had stale "remaining gaps"** | Cleaned up stale rows; count 15→17 | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Graph Vault Reflection 4)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2: No Alaska Summit event file** | Created: `events/alaska-summit-2025.md` | 2026-05-21 |
| **Priority 3: No dollar-smile-theory concept** | Created: `domains/economics/concepts/dollar-smile-theory.md` | 2026-05-21 |
| **Structural: Runs index table corrupted** | Rewrote `runs/_index.md` with clean formatting | 2026-05-21 |
| **Structural: _index.md Recent Forecasts section stale** | Replaced with "Cataloged in Runs Index" | 2026-05-21 |
| **Structural: _macro_gaps.md row 6 dollar-smile-theory status** | Updated to FILLED | 2026-05-21 |

## Filled Gaps (May 21, 2026 — Graph Vault Reflection 5: Taylor Rule & Navigation Fixes)

| Gap | Resolution | Date |
|-----|-----------|------|
| **Priority 2 — No taylor-rule-calibration concept file** | Created: `domains/economics/concepts/taylor-rule-calibration.md` | 2026-05-21 |
| **Priority 2 — Economics domain index missing 4 concepts** | Added to frontmatter and Concepts section | 2026-05-21 |
| **Priority 2 — Latin America domain missing Colombia thread** | Added Colombia thread to _domain.md | 2026-05-21 |
| **Priority 3 — Events index missing TikTok SCOTUS event** | Added to events index; count 18→19 | 2026-05-21 |
