---
type: forecast-instructions
tags: [meta, forecast-instructions]
version: 1.2
date: 2026-05-21
purpose: "Behavioral rules for the forecasting agent, evolved through reflection. Each rule is a lesson learned from a previous mistake."
---
---
---
# Forecast Instructions

This file contains behavioral rules that the forecasting agent MUST follow when making predictions. These rules are the direct product of the reflection loop — each one was added after a mistake revealed a systematic failure mode.

## Vault layout (live vs history vs meta)

- **Live graph** (`timeline/2022+`, `threads/`, `entities/`, `concepts/`): traverse for every forecast.
- **history/**: pre-2022 or analog-only research. **Never** add new historical actor files to `entities/` at root — use `history/entities/` or `history/threads/`.
- **meta/reflections/**: session logs; not part of the forecast graph. Orphan nodes under `forecasts/`, `runs/`, `agent-roles/`, and `meta/` are expected in Obsidian.

---

## How to Use

Before making any forecast:
1. Read this file in full
2. Check each rule against the current question
3. If any rule applies, document in your reasoning how you satisfied it
4. If no rule applies, state "No forecast instruction rules triggered for this question"

---

## Meta-Rule A: Multi-Agent Orchestration

**Context:** Added 2026-05-18 with the creation of the sub-agent ecosystem (14 agent role files in `graph-vault/agent-roles/`).

**Rule:** Before forecasting any complex geopolitical, economic, or multi-domain question:

1. **Read the orchestrator's prerogatives**: Read `graph-vault/agent-roles/_orchestrator_prerogatives.md` in full. This document defines the meta-level workflow for selecting, spawning, and synthesizing sub-agents.

2. **Check the agent roster**: Run `search_files("agent-roles/*.md", path="graph-vault/")` to see all available agent roles. Each role file has:
   - `domain` tags — does it match the question's domain?
   - `region` tags — does it match the question's geography?
   - `Trigger Conditions` — does the question match any trigger?

3. **Decide whether to multi-agent**: Not every question needs sub-agents. Apply this triage:
   - **SIMPLE** (single domain, one region, yes/no factual): Forecast directly. No sub-agents needed.
   - **MODERATE** (single domain but multi-actor or high uncertainty): Use 2-3 agents for perspective diversity.
   - **COMPLEX** (multi-domain, multi-region, geopolitical, high-stakes): Use the full pipeline — 3-5 agents + contrarian.

4. **Execute the orchestration workflow** (from _orchestrator_prerogatives.md):
   - SELECT → SPAWN (parallel) → COLLECT → CONTRARIAN → SYNTHESIZE → FORECAST → REVIEW → UPDATE

5. **Record which agents were used** in your forecast reasoning: "Consulted agents: [china-actor-simulation, conflict-escalation-analyst, contrarian-debater]"

6. **If no matching agent exists** for the question's domain/region, note this as a roster gap and consider creating a minimal agent role stub.

**Rationale:** Multi-perspective forecasting reduces blind spots. Each agent role sees the question through a different lens — actor simulation reveals decision calculus, game theory reveals strategic structure, escalation analysis reveals conflict dynamics. Together they produce forecasts that are more robust than any single perspective. The orchestrator chooses WHICH lenses to apply based on the question's characteristics.

---

## Meta-Rule B: Stub Agent Completion During Reflection

**Context:** Added 2026-05-18 after the first orchestrated scan created crypto-financial-markets-specialist (complete but marked `stub`) and digital-asset-markets-analyst (true stub, overlapped with the first, deleted as duplicate).

**Rule:** During the reflection step after each orchestration cycle:

1. **Scan for stubs**: `search_files("agent-roles/*.md", path="graph-vault/")` and check each file's frontmatter for `status: stub`.

2. **Assess each stub**:
   - Does it have a full persona, expertise, methodology, trigger conditions, and output format? → Set `status: active`.
   - Does it overlap with an existing active agent? → Delete the duplicate. The active agent wins.
   - Is it a true stub (minimal frontmatter, no methodology)? → Fill in the methodology based on its domain tags and trigger conditions. Use the agent's purpose to generate persona, expertise domains, methodology steps, trigger conditions, output format, and rules.
   - Does it address a genuinely new domain not covered by existing agents? → Flesh it out fully, set `status: active`.
   - Is the domain too narrow (e.g., "layer-2 scaling" when there's already a "crypto" agent)? → Fold the unique trigger conditions into the broader agent and delete the stub.

3. **Default action**: Every reflection cycle must leave zero stubs in `agent-roles/`. Stubs represent uncompleted work. Either complete them, merge them, or delete them.

4. **Log the action**: Record which stubs were completed/merged/deleted in the reflection entry.

**Rationale:** The orchestrator creates stubs as a bookmark — "we need coverage here" — but stubs are useless until filled. The reflection cycle is the natural place to complete them because it has the context of what forecasts were just made and what domain knowledge was needed. Leaving stubs uncompleted means the system is aware of gaps but taking no action to close them.

---

## Rule 1: Central Bank Questions Require Forward Guidance Analysis

**Context:** Added after pm_eco_01 (correct YES prediction, but vault had no structural support).

**Rule:** Before forecasting any central bank rate decision question:
1. Check the latest central bank statement and dot plot (for the Fed) for explicit signals about the next meeting
2. Identify the date of the next scheduled meeting — is it before or after key data releases?
3. Check market pricing (Fed funds futures / CME FedWatch) as a consensus anchor
4. Apply the baseline: central banks strongly prefer to pre-announce changes; surprise moves are rare
5. **EM Central Bank extension**: For EM central banks with political constraints (TCMB, CBR, BCB, etc.), additionally check:
   - Is the political shield (finance minister) still in place?
   - Is the central bank in a normalization phase or politically captured?
   - Has the currency depreciated 5%+ since the last meeting?
   - See the [[domains/mena/concepts/em-central-bank-credibility-normalization]] concept

**Rationale:** Central banks use a structured communication pipeline (statements, press conferences, dot plots, minutes, speeches) that makes rate decisions unusually forecastable. The vault should capture this pipeline in the [[concepts/central-bank-forward-guidance]] concept. For EM central banks, the forward guidance is less structured and political context matters more — requiring a separate analytical framework documented in [[domains/mena/concepts/em-central-bank-credibility-normalization]].

---

## Rule 2: Domestic Politics Gap Check

**Context:** Added after pm_pol_04 (correct YES prediction, but vault had zero US political coverage).

**Rule:** Before forecasting any question about US politics, elections, or leadership changes:
1. Verify that the vault's quarter summaries cover US domestic politics for ALL relevant quarters
2. The blind-build process systematically under-covers US domestic stories in favor of international events
3. If US domestic coverage is missing from any quarter summary, flag it as a vault gap
4. Supplement with general knowledge, but record the gap for reflection remediation

**Rationale:** The blind-build subagent prioritized international elections (India, Mexico, EU Parliament) over the US 2024 presidential election. This is a systemic coverage bias.

---



## Rule 4: Geographic Coverage Gap Check

**Context:** Added after live Brazil 2026 election forecasts (vault had ZERO Brazil content despite $11M+ markets).

**Rule:** Before forecasting any foreign election or geopolitical question:
1. Verify the vault has coverage of that country — thread file, entity stubs for key candidates, political dynamics
2. If coverage is missing, flag it as a geographic vault gap
3. The vault is currently US/Middle East-heavy. Election questions about Brazil, France, Japan, India, etc. may have zero vault signal
4. Create minimal coverage (thread + 2-3 entity stubs) during reflection even for correct forecasts

**Rationale:** The vault was built on gold_30 questions that skewed US/Middle East. Geographic expansion is mandatory for global forecasting coverage.

---

## Rule 5: NATO/US Alliance Questions Require Thread + Entity Coverage

**Context:** Added May 18, 2026 after identifying that the vault had no US-NATO relations thread and no NATO entity file despite an active $989K prediction market ("Will US withdraw from NATO before 2027?" at 8.4% YES).

**Rule:** Before forecasting any question about NATO, US alliance commitments, European security architecture, or US withdrawal from multilateral security agreements:
1. Verify the vault has a [[threads/us-nato-relations]] thread or equivalent
2. Verify the [[nato]] entity stub exists
3. Check the most recent quarter summary for NATO-relevant events (Article 4 invocations, defense spending announcements, US policy statements)
4. Distinguish between formal withdrawal (treaty denunciation requiring 2/3 Senate approval) and functional withdrawal (funding cuts, deployment delays, ignored Article 5 requests) — these have different probabilities
5. Apply the transactionalism vs alliance framework: Trump's second-term foreign policy operates through bilateral deal-making (Alaska Summit, Armenia-Azerbaijan, Gaza ceasefire) rather than multilateral alliance frameworks. This structural preference shapes NATO outcomes regardless of specific policy statements.

**Rationale:** NATO questions are structurally distinct from general US foreign policy questions. The alliance has 32 members, specific treaty obligations, and bipartisan Congressional support that constrains presidential action even for a transactional administration. A vault that lacks basic NATO infrastructure cannot assess these constraints.

---

## Rule 6: Section 6.1 — Switzerland as a Mandatory Coverage Gap Check

**Context:** Added May 18, 2026 after identifying that the vault had zero Swiss political coverage despite an active $99K prediction market ("Will the 'No to ten million' initiative be approved?" on June 14, 2026). The previous hypothesis (hyp-20260518-002) explicitly noted this gap.

**Rule:** Before forecasting any question about Swiss politics, referendums, or initiatives:
1. Verify the vault has at minimum an [[entities/switzerland]] entity stub
2. Check whether a Swiss politics thread exists
3. Apply the baseline initiative passage rate: 10-15% historically (since 1891, ~22 of ~220 initiatives approved)
4. For population/immigration-limiting initiatives specifically, note the mixed track record: the 2014 "Against Mass Immigration" initiative passed (50.3%) but subsequent tightening attempts failed
5. Check the initiative's cantonal majority requirement — restrictive measures often fail the cantonal vote even if they win the popular vote

**Rationale:** Swiss direct democracy follows predictable structural patterns that differ from electoral politics in representative democracies. Treating Swiss initiative votes like generic opinion polls produces systematically wrong forecasts. The vault must at minimum document the initiative passage baseline and specific historical precedent for the initiative type.

---

## Rule 7: AI/Tech Company Coverage Gap Check

**Context:** Added May 18, 2026 (polymarket scan) after detecting active coding AI benchmark markets ($35K+ volume) with zero vault coverage of any AI company — no entity stubs for Anthropic, OpenAI, Mistral, Google DeepMind, or xAI.

**Rule:** Before forecasting any question about AI model rankings, LLM capabilities, or tech company competition:
1. Verify the vault has entity stubs for at least the top 3 contenders relevant to the question
2. Check whether the relevant benchmark leaderboard (lmarena, SWE-bench, HumanEval, etc.) has current data accessible from the cutoff date
3. AI model rankings change rapidly — a 3-month-old entity file may have stale capability assessments
4. Distinguish between "best" and "second-best" questions — the top position is often concentrated (Anthropic dominates coding), while second-best is more competitive
5. For chatbot arena / lmarena questions specifically, check the most recent leaderboard snapshot before the cutoff — rankings can shift with new model releases

**Rationale:** The vault was built on geopolitical and economic prediction markets and has zero AI/tech company coverage despite $35K+ daily volume on coding AI markets. AI capability forecasting follows different dynamics (rapid release cycles, benchmark concentration, open-source disruption) than political forecasting.

---

## Rule 8: Local/Municipal Election Coverage Gap Check

**Context:** Added May 18, 2026 (polymarket scan) after finding the vault had a California gubernatorial thread but zero coverage of the LA mayoral election (a top-10 US city election with active prediction markets).

**Rule:** Before forecasting any US local/municipal election question:
1. Verify the vault covers the specific jurisdiction — a state-level thread (e.g., California governor) does not substitute for city-level coverage
2. For non-partisan races (like LA mayor), check whether candidate name recognition and media coverage are the primary drivers of odds — celebrity candidates (e.g., Spencer Pratt at 70.5% for LA 2nd place) can dominate market prices
3. For primary systems (top-two, jungle primary, runoff), understand the specific rules — these dramatically change the strategic landscape
4. Create city-level entity stubs for major municipalities with active markets (Los Angeles, New York, Chicago, etc.)

**Rationale:** Local elections follow different dynamics (lower turnout, celebrity candidate effects, non-partisan structures, incumbency advantages) than state/federal races. The vault's US political coverage is state-level and misses this entire category.

---

## Rule 9: Polymarket Calibration Mode (PIT price alignment)

**Context:** Added 2026-05-19 after PIT market-calibration probes showed large misses when the vault contained post-cutoff narrative (e.g. ceasefire announced on cutoff day, Biden withdrawal documented in election thread) while Polymarket YES remained far from 1.0.

**Rule:** When the prompt includes Polymarket YES at cutoff or says "calibration mode":

1. **Primary target is the market price**, not terminal resolution and not post-hoc vault certainty.
2. Output `p_yes` within **±0.05** of the stated Polymarket YES unless you document strong **pre-cutoff** conjunctural evidence the market was mispriced (rare).
3. **Vault post-hoc threads do not override the market** — if `threads/` describe an outcome on or before cutoff but PM is still 0.15–0.70, traders had not fully priced it; align with PM unless pre-event signals were public before cutoff.
4. PIT snapshots may omit thread bullets after cutoff; do not infer omitted events as certain.
5. For structural forecasts without a PM anchor, use vault conjuncture stances (e.g. shadow-war escalation ≥0.65) and state which concept/thread drove the number.

**Rationale:** The system must learn to track **market-implied probability at time T**, which supervises calibration on interesting geopolitical graphs. Resolution Brier and market-alignment are different objectives; conflating them produces 0.98 vs 0.16 errors.

---

## Rule 10: Reflection Must Not Leak Outcomes Into Graph Docs

**Context:** Added 2026-05-19. After market-calibration misses, reflection sometimes "fixed" the vault by adding sentences like "Biden withdrew on July 21" or "ceasefire announced October 8" to threads — which teaches the forecaster to output `p_yes≈1` instead of aligning with Polymarket at cutoff.

**Rule:** When editing `threads/`, `concepts/`, `timeline/`, or `entities/` after a calibration or Brier miss:

1. **Do not** add terminal outcomes or post-cutoff resolution facts to fix a miss.
2. **Do** add `pit_body_cutoff`, trim bullets after cutoff, split PIT conjuncture files, or sharpen mechanisms that explain **market price at T**.
3. **Do not** use `runs/` or `forecasts/` as sources during reflection — they contain hindsight reasoning.
4. A miss with `forecaster_p >> market` usually means **vault hindsight or leakage**, not "the market was wrong."

**Rationale:** Reflection supervises **structure and PIT hygiene**, not transcribing what eventually happened.

---

## Meta-Rule B2: PIT Research Librarian (historical cutoffs)

**Context:** Added 2026-05-19. Historical forecasts used the full vault and leaked post-cutoff outcomes into otherwise-correct reasoning.

**Rule:** When `cutoff` is before today or `enforce_pit=True`:

1. The harness spawns **pit-research-librarian** first (read-only). It returns a PIT brief — not a forecast.
2. Forecaster and orchestrated sub-agents must **prefer the brief** over re-reading threads that may contain hindsight.
3. Domain agents (MENA, macro, etc.) may still use **current** vault for live questions; for backtests they consume the PIT brief only unless explicitly in live mode.
4. Optimize leakage in the librarian prompt and PIT snapshot — not in every specialist role.

**Rationale:** Separating "what was knowable at T" from "who decides p_yes" lets us tune retrieval once and keep specialist agents stable.

---

## Rule 11: Public Event Before Cutoff — Market Price Supersedes Event Certainty

**Context:** Added 2026-05-19 after gold-gold_18 (Biden DNC dropout probe). Forecaster output 0.990 while Polymarket was 0.715, reasoning "Biden already withdrew on July 21 — this is resolved YES." The market was pricing residual procedural uncertainty (delegate release mechanics, virtual roll call timing, reversal tail risk) that post-hoc narrative hides.

**Rule:** When a public event relevant to the question's outcome has already occurred before cutoff, but the Polymarket YES price at cutoff is below 0.90:

1. **Do NOT treat the question as resolved** just because the event "happened." The resolution of a prediction market question depends on resolution criteria (often procedural/contractual), not journalistic fact.
2. **Output p_yes within ±0.05 of the Polymarket YES price.** The market is pricing residual uncertainty that the vault's post-hoc narrative does not capture.
3. **Document the residual uncertainties** the market is pricing (procedural mechanics, reversal risk, scope ambiguity) — these are the forecaster's calibration signal, not noise.
4. **When Polymarket YES ≥ 0.90**, residual uncertainty is negligible; alignment may track event-certainty within vault bounds.
5. **Never write "the market was wrong"** to justify overriding the price — the market is the calibration target, not a noisy estimate to be corrected.

**Rationale:** The vault's resolved-thread format presents terminal outcomes as certain narrative. This creates a systematic bias toward p≈1 for events that "already happened" at cutoff, even when the market prices real residual procedural uncertainty. Rule 11 forces the forecaster to align with market-implied probability, not post-hoc certainty. The residual uncertainties the market prices (delegate mechanics, resolution criteria, reversal risk) are the calibration signal — documenting them teaches the forecaster why 0.715 ≠ 1.0.

---

## Rule 12: Horizon-Matched Base Rates — Long-Horizon Concepts Must Not Zero Out Short-Deadline Market Prices

**Context:** Added 2026-05-19 after gold-gold_03 (Hamas Jul 15 ceasefire probe). The forecaster applied Key Dynamic #8 from [[threads/gaza-ceasefire-negotiations-2025]] — a 7-10 week diplomatic bandwidth refocusing lag — to a 7-day deadline question. This crushed p_yes to 0.15 while Polymarket was 0.49, because the lag concept (developed for Oct breakthrough timing) was structurally about long-horizon announcement windows, not short-fuse market dynamics.

**Rule:** Before applying any time-lag, refocusing, or accumulation concept to a question:

1. **Check the concept's native time horizon.** A concept derived from a 7-10 week pattern (diplomatic refocusing) is structurally about **medium-term breakthrough timing**, not **short-fuse (≤14 day) deadline questions.**
2. **Short-horizon (≤14 days to deadline) questions have different dynamics:** active shuttle diplomacy, Iran-ceasefire spillover, pre-deadline diplomatic packages — these can sustain PM at 0.45-0.55 even when diplomatic infrastructure has not yet reached full maturity.
3. **When a concept says "X weeks minimum" and the deadline is shorter than X:** the concept vetoes a near-term breakthrough at the specific deadline, but it does NOT justify a p_yes below the Polymarket price. The market is pricing cumulative pressure vectors, not a specific diplomatic timetable.
4. **Split long vs short horizon reasoning explicitly in your forecast:**
   - Long horizon (6+ weeks out): lag/accumulation concepts apply fully; estimate is concept-driven
   - Short horizon (≤14 days): lag concepts bound probability at ~0.10-0.20 floor but do not force the price to that floor — cumulative pressure vectors can sustain ~0.45-0.55 even without mature diplomatic infrastructure
5. **Always check: does my reasoning zero out a market price that is clearly non-zero?** If so, re-examine whether the concept is being applied at the wrong time scale.

**Rationale:** Long-horizon diplomatic concepts (bandwidth refocusing, pressure accumulation, leadership decapitation windows) are about when a breakthrough becomes structurally possible — not whether a market is pricing a specific short-term deadline correctly. Applying a 7-10 week refocusing lag as a veto on a 7-day question treats timing uncertainty as impossibility. The market at ~0.50 on a short deadline reflects: (a) pre-deadline diplomatic packages can still be delivered in compressed timeframes, (b) the question resolution criteria may be broad enough to cover low-grade agreements, (c) cumulative pressure has multiple vectors that don't all require mature diplomatic infrastructure. The lag concept should cap optimism (not push to 0.15) while the market provides the calibration anchor.

---

## Rule 13: Rare-Event Base Rate Assessment (Nuclear Detonation, Coup, Disaster)

**Context:** Added 2026-05-20 after Question 4/84 (nuclear weapon detonation by June 30, 2023). The correct NO prediction was a freebie — vault contributed zero signal. The vault had no base-rate data, no thread tracking nuclear posture, and no entities for nuclear governance bodies.

**Rule:** Before forecasting any question about a truly rare catastrophic event (nuclear weapon detonation, accidental nuclear launch, successful terrorist acquisition of WMD, catastrophic infrastructure failure, or any event with fewer than 3 historical occurrences in the modern era):

1. **Determine the event type and check historical frequency:** Classify the event by type (wartime use, testing, accidental, terrorist) and look up the historical frequency:
   - Wartime nuclear use: last occurred 1945; 0 occurrences in 78+ years
   - P5 nuclear testing: last occurred 1996 (France/China); none by any CTBT signatory in 27+ years
   - Accidental nuclear detonation: 0 occurrences in history (multiple broken arrows but zero nuclear yield)
   - Terrorist nuclear detonation: 0 occurrences in history
   - Nuclear test by non-signatory state (North Korea): 6 tests (2006-2017), clusters in periods of diplomatic tension

2. **Start from the base rate, not from "maybe":** The default answer for any "will X rare event happen by [short horizon]?" question is NO with >95% baseline confidence unless specific leading indicators are present. The reasoning must start from the base rate and adjust upward only with evidence.

3. **Distinguish posture signaling from use preparation:** Nuclear posture events (treaty suspensions, weapons sharing announcements, nuclear modernization reveals) are typically political signaling, not preparation for use. These adjust the base rate by at most a 2x multiplier. Only operational preparation (test site activity, weapon dispersal from storage, alert level changes, C2 modification) justifies larger adjustments.

4. **Check the [[concepts/nuclear-use-base-rates]] framework:** Apply its step-by-step framework:
   - Identify the detonation type (wartime, test, accidental, terrorist)
   - Check the base rate for that type in the specific time window
   - Survey the five leading indicators (explicit escalation, test prep, CTBT withdrawal, C2 breakdown, terrorist acquisition)
   - If no leading indicator is active, the probability stays at base rate (<2% per year, often <0.1%)

5. **Document the negative case explicitly:** State clearly which leading indicators were checked and found absent. An unexplained "no" is not a vault-supported forecast — the reasoning must demonstrate that the relevant mechanisms were examined. Template:

   ```
   Leading indicator check for [window]:
   - Explicit nuclear threat with operational preparation: [not present / present — detail]
   - Nuclear test preparation indicators: [not present / present — detail]
   - CTBT withdrawal or violation preparations: [not present / present — detail]
   - Nuclear C2 breakdown or unauthorized movement: [not present / present — detail]
   - Credible terrorist fissile material acquisition: [not present / present — detail]
   Result: No leading indicators active. Base rate of [X%] applies. Probability: [p_yes].
   ```

6. **For questions that already resolved and are confirmed NO:** Even with a correct prediction, create or update the following vault assets during reflection:
   - Verify the [[concepts/nuclear-use-base-rates]] concept exists with accurate base-rate data
   - Verify the [[threads/nuclear-weapons-posture]] thread exists and covers the question's window
   - Verify entity stubs exist for CTBTO, NPT, and IAEA
   - Add this forecast to the concept's "Validated By" table as a post-hoc entry

**Rationale:** Rare-event questions are the most deceptive "freebie" traps in forecasting. The correct answer (NO) is obvious from general knowledge, so the vault appears to have contributed. But without structured base-rate data and a leading-indicator framework, the vault provides zero analytical signal — the forecaster is relying on general knowledge that may be inaccurate or outdated. A vault that cannot systematically assess "will a nuclear weapon detonate?" by checking each leading indicator against historical base rates is not a forecasting vault — it's a collection of news summaries. This rule ensures that rare-event questions leave the vault strictly better than they found it, with the same depth of analytical structure as for elections, rate decisions, or ceasefire questions.

---

## Rule 3: Short-Window Military Strike Questions Require Window-Adjusted Calibration

**Context:** Added after gold_16 (correct NO — Israel attacked Iran by Feb 15, 2024, prediction was NO). The prediction was correct but the vault lacked structured analytical support for narrow-window military strike forecasting.

**Rule:** Before forecasting any "Will X attack Y by date Z?" question where Z is within 6 weeks of the forecast date:

1. **Measure the window length**: W = days from forecast to deadline. If W < 30 days, this is a short-window question requiring the short-window framework.

2. **Check for a tripwire event**: Has a trigger event occurred within the 14 days before the window opens? If no, the base rate drops by ~80% from the 6-month rate. Trigger events include:
   - Direct attack on the potential attacker's territory causing casualties
   - Killing of senior military commanders of the potential attacker
   - Attack on diplomatic facilities
   - Adversary nuclear breakout announcement
   - Collapse of diplomatic off-ramps

3. **Apply the precedent penalty**: First-ever strikes (attacker has never struck target's soil) have a much lower short-window base rate than repeat strikes. A state that has already crossed the threshold can do so again with less friction.

4. **Assess decision cycle friction**: Can the attacker plan, authorize, and execute a strike within W days? Israel requires 2-5 weeks minimum for a sovereign-state strike. Shorter windows mean a decision must already have been made before the window opens.

5. **Check military bandwidth**: Is the attacker's force already committed elsewhere? Multi-front constraints reduce the probability of initiating a new front.

6. **Assess patron posture**: Is the superpower patron signaling green/yellow/red light? Patron containment reduces probability; patron clearance increases it.

7. **Apply the framework**: Read and apply [[concepts/short-window-military-strike-probability]] for the base rate calibration by window length. Read [[concepts/shadow-war-to-direct-escalation]] to map the current escalation stage. For Israel-specific questions, apply the [[procedures/israel-strike-forecast]].

8. **Document the negative case explicitly**: State which factors were checked and found absent. Template:
   ```
   Short-window strike check for [target] by [deadline]:
   - Window length: [W] days
   - Tripwire event in prior 14 days: [present / absent]
   - First-ever strike: [yes / no] → precedent penalty [applied / not applied]
   - Decision cycle feasibility: [feasible / not feasible within W]
   - Attacker military bandwidth: [constrained / unconstrained]
   - Patron posture: [supporting / containing / ambiguous]
   - Shadow war stage: [stage 0-8]
   Result: [P(strike within window)] based on short-window framework.
   ```

9. **During reflection after resolution**: Even with a correct NO prediction, verify that [[concepts/short-window-military-strike-probability]] and [[concepts/shadow-war-to-direct-escalation]] exist with accurate base rates and stage mappings. Verify that relevant entity stubs (attacker, target, key decision-makers) exist. Update the concept's calibration tables with this forecast's outcome.

**Rationale:** Short-window military strike questions are deceptive because the medium-term probability (e.g., P(Israel attacks Iran in 2024)) can be substantial while the short-window probability (e.g., P(Israel attacks Iran by Feb 15)) is near-zero. Without a framework that systematically accounts for window length, tripwire presence, decision cycle friction, and bandwidth constraints, the vault provides no structured support. The vault must distinguish between "this is unlikely within 3 weeks" and "this is unlikely at all" — a distinction that collapses without explicit calibration.

---

## Rule 14: Asymmetric Ceasefire Questions Require War Aims Compatibility Assessment

**Context:** Added after Question 17/84 (correct NO — Israel-Hamas ceasefire by Feb 29, 2024). The prediction was correct but the vault lacked structured analytical support for asymmetric ceasefire forecasting. The vault had `short-window-military-strike-probability` but no equivalent for ceasefires — which require mutual consent from both parties and are fundamentally different from unilateral military action.

**Rule:** Before forecasting any ceasefire question involving a state and a non-state armed group (Hamas, Hezbollah, Houthis, etc.):

1. **Classify the conflict type**: Is this state-vs-state or state-vs-non-state?
   - State-vs-non-state → apply the asymmetric framework below
   - State-vs-state with superpower → use [[concepts/escalation-bargaining-termination]]
   - State-vs-state without superpower → use [[concepts/diplomatic-pressure-tipping-point]]

2. **Assess war aims compatibility**: Read [[concepts/war-aims-incompatibility]] and classify the stronger party's stated war aim:
   - "Destroy" / "Eliminate" → ceasefire structurally impossible until the aim is either achieved or redefined. This is the single strongest forecasting input.
   - "Degrade" / "Deter" → ceasefire possible once sufficient degradation is achieved
   - Document the exact official language, not journalist summaries.

3. **Apply the mutual-consent penalty**: Ceasefires require both parties to agree. For any short window (< 3 months), the base rate is structurally lower than for military strikes because two parties must say "yes."

4. **Measure the window**: W = days from forecast to deadline. Apply [[concepts/short-window-ceasefire-probability]]:
   - W < 30 days and no pre-existing framework: P(ceasefire) < 0.01 (absent superpower imposition)
   - W < 30 days with framework active: P(ceasefire) ~0.02-0.05
   - Check the temporary pause exception: would a hostage/prisoner deal satisfy resolution criteria?

5. **Map the mediation structure**: Who mediates and what leverage does each mediator have on each party?
   - Single mediator with leverage on both: fastest path (rare in asymmetric conflicts)
   - Multiple mediators each with leverage on one side (Israel-Hamas pattern: US↔Israel, Qatar↔Hamas): the leverage gap is a structural constraint that blocks rapid agreements.

6. **Assess non-state actor's strategic calculus**: Is the non-state actor pursuing an attrition strategy (prolonging war increases pressure on the state)? If yes, the non-state actor has NO incentive for a quick ceasefire.
   - Check if the non-state actor's hardline leader is alive (if yes, no decapitation window)
   - Check if the non-state actor holds hostages or POWs (if yes, hostage-ceasefire deadlock may be active)

7. **Check domestic political constraints on the state leader**: Does the leader face coalition pressure against ceasefire? If the leader's government depends on hardline parties that oppose ceasefire, this creates a near-deterministic veto.

8. **Assess military trajectory**: Is the stronger party preparing a major new offensive or winding down?
   - Preparing escalation → P(ceasefire) caps at 0.02
   - Winding down → P(ceasefire) elevated

9. **Apply the framework**: Read and apply [[concepts/short-window-ceasefire-probability]] for base rate calibration. Apply [[procedures/asymmetric-ceasefire-forecast]] for the full step-by-step assessment.

10. **Document the negative case explicitly**: State which factors were checked and found absent. Template:
    ```
    Asymmetric ceasefire check for [state] vs [non-state] by [deadline]:
    - Conflict type: [state-vs-non-state]
    - War aims: [classification] — [compatible / incompatible]
    - Window: [W] days — base rate: [X%]
    - Mediation structure: [single-leverage / leverage-gap] — [mediator-1: leverage on whom, mediator-2: leverage on whom]
    - Non-state actor calculus: [attrition / exhaustion / capitulation]
    - State domestic constraints: [constrained / unconstrained]
    - Military trajectory: [escalation / plateau / de-escalation]
    - Temporary pause exception: [active / not active]
    Result: [P(ceasefire within window)] based on asymmetric ceasefire framework.
    ```

11. **During reflection after resolution**: Even with a correct NO prediction, verify that [[concepts/short-window-ceasefire-probability]] and [[concepts/war-aims-incompatibility]] exist with accurate base rates. Verify that the [[procedures/asymmetric-ceasefire-forecast]] procedure is updated. Verify that entity stubs for the non-state actor (e.g., [[entities/hamas]]), the state leader (e.g., [[entities/benjamin-netanyahu]]), and key mediators ([[entities/qatar]], [[entities/egypt]]) exist with coverage of the relevant negotiation period.

---

## Rule 15: US-Russia Diplomatic Relations Thread Required for Forecasting

**Context:** Added 2026-05-20 after creating the [[domains/global/threads/us-russia-diplomatic-relations]] thread. The vault had the Alaska Summit as an isolated event entity but no continuous thread tracking the bilateral relationship across quarters.

**Rule:** Before forecasting any question about US-Russia summits, Trump-Putin meetings, Ukraine peace negotiations, or US-Russia sanctions policy:

1. **Read the US-Russia diplomatic relations thread**: Read `domains/global/threads/us-russia-diplomatic-relations/_thread.md` in full. This thread tracks the arc from post-2022 freeze to Trump-era re-engagement.

2. **Check the Alaska Summit entity**: Read `domains/global/entities/alaska-summit.md` for the precedent-setting August 2025 summit.

3. **Assess the bilateral vs multilateral dynamic**: The Alaska Summit demonstrated Trump's preference for bilateral leader-level negotiation without NATO/EU intermediation. Questions about "EU country" venues must account for European opposition to being excluded from the format.

4. **Cross-reference the NATO relations thread**: Read `domains/global/threads/us-nato-relations/_thread.md` for how US-Russia bilateral engagement affects NATO cohesion.

5. **Apply the transactionalism vs alliance framework**: Trump's second-term foreign policy operates through bilateral deal-making rather than multilateral alliance frameworks. This structural preference shapes US-Russia outcomes regardless of specific policy statements.

**Rationale:** US-Russia relations are among the most consequential and active forecasting domains. The vault now has structural support for this question class through the dedicated thread, but the forecasting agent MUST read it to benefit from the compiled knowledge about venue politics, European reactions, and bilateral negotiation dynamics. Without reading the thread, the agent would miss critical context about why an EU-hosted summit is structurally unlikely.

---

**Rationale:** Asymmetric ceasefire questions are deceptive because they appear to be about "will there be a ceasefire?" when the real question is "is a ceasefire structurally possible given war aims, mediation structure, and incentives?" Without a framework that systematically accounts for the mutual-consent penalty, war-aims incompatibility, mediation leverage gaps, and non-state actor attrition strategies, the vault provides zero analytical signal. The vault must distinguish between "no ceasefire because it hasn't happened yet" and "no ceasefire because it cannot happen until one side's position changes" — a distinction that only structured analysis can provide.

---

## Structural Reasoning (Procedure)

Before producing p_yes for any binary question, follow [[procedures/structural-reasoning]].
This procedure models event structure across three dimensions — time, chain, anchor —
and subsumes the domain-specific checks that previously required individual rules.
The reflection agent owns and evolves this procedure based on forecasting outcomes.

---

## Rule 16: Mechanism Calibration Tables Override Base-Rate Blending

**Context:** Added 2026-05-21 after the 18-run gold set resolved with 100% directional accuracy but Brier scores ranging from 0.0001 to 0.3969. The weakest forecasts (Venezuela Gonzalez p=0.37, Taiwan Lai p=0.82) were directionally correct but under-confident because the agent's weighted blending formula compressed probabilities toward 0.5. Concepts that correctly identify structural situations need empirical calibration data to anchor probability estimates.

**Rule:** After identifying the active mechanism(s) for a forecast question, query the mechanism's concept file for an `## Empirical Calibration` table BEFORE applying the base-rate blending formula.

1. **Load the mechanism concept**: If the situation-identification step returns "FPTP fragmentation active," read `domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md`. If "procedural certainty active," read `domains/global/concepts/short-horizon-procedural-certainty/_concept.md`. If "authoritarian electoral facade active," read `domains/usa/concepts/authoritarian-electoral-facade.md`.

2. **Query the calibration table**: Find the row matching the current indicator combination. Note the Hit Rate and Sample count (N).

3. **Override the blend if calibration is sufficient**:
   - If `Hit Rate = 100%` and `N ≥ 2`: set `p_yes = 0.90-0.95` (anchor adjusted for epistemic uncertainty with small sample). Do NOT apply the 0.35/0.25/0.15/0.25 blend.
   - If `Hit Rate = 0%` and `N ≥ 3`: set `p_yes ≤ 0.02`. Do NOT blend.
   - If `Hit Rate = 100%` and `N = 1`: set `p_yes = 0.80-0.90` (wider uncertainty band).
   - If `N = 0` or no calibration table exists: fall back to standard forecast methodology including base-rate blending and market-price anchoring.

4. **Pool when the native concept has too few samples**: If the matched concept has N < 3, check the `## Empirical Calibration` section for pooling guidance. The concept's maintainer will have documented which adjacent concept provides pooled calibration (e.g., authoritarian-electoral-facade pools into the broader electoral model with 8+ samples). Use the pooled concept's calibration table instead.

5. **Document the calibration query in reasoning**: State which concept was queried, which indicator row matched, the hit rate and sample count, and whether the calibration table overrode or supplemented the standard blend. Template:
   ```
   Calibration query: [[concepts/divided-opposition-plurality-win]] → "Front-runner 35-45% + FPTP" → Hit Rate 100% (2/2). 
   Calibration override active: p_yes anchored at 0.90-0.95 instead of blended formula.
   ```

6. **Never override calibration with base-rate blending**: The whole point of mechanism calibration is that the empirical hit rate is a better probability anchor than any weighted formula. If the calibration table says 100% hit rate with 2+ samples, the agent's job is to apply the anchor, not to second-guess it with base rates.

**Rationale:** The 18-run gold set revealed a systematic pattern: the agent correctly identifies the active mechanism but then compresses the probability toward 0.5 through weighted blending. The FPTP fragmentation mechanism was correctly identified for Lai (p=0.82), LLA (p=0.78), and the third-party ceiling cases (p≤0.02). But the blend formula dragged p_yes down from the structural certainty it should have had. Mechanism calibration tables solve this by providing empirical anchors that override the blend: when the vault has seen this exact indicator combination before and it resolved YES 100% of the time with multiple samples, use that as the forecast anchor. The calibration table is the statistical pipeline — the agent's job is situation identification + indicator extraction, not probability blending.

