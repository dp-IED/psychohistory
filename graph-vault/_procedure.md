---
type: procedure-root
tags: [meta, procedure]
version: 2.0
date: 2026-05-18
author: hermes-agent
purpose: "Define the workflow for writing, reviewing, and maintaining graph vault summaries"
---
---
---
# Vault Procedure v2.0

## When to Write Summaries

Summaries are written per-request. **Contemporary** (2022+) quarter files go in `timeline/`. **Historical** or analog research goes in `history/timeline/` — never at vault root. See `history/README.md`.

## Research Protocol (per quarter)

### Phase 1: Scoping (before writing)
1. **Identify the major domains**: War/conflict, politics, science/tech, culture/society, disasters, **terrorism/security**, births/deaths.
2. **Check continuity**: Read the previous quarter's summary and ALL active thread files to understand ongoing arcs.
3. **Identify key entities**: For **contemporary** quarters only — which actors need `entities/` stubs for live forecasting? For **historical** quarters, use `history/entities/` or thread cast tables only; do not create root-level entity files for historical figures unless promoting to a live market question.
4. **Set PIT boundary**: The cutoff is the last day of the quarter. Nothing after that date is admissible as knowledge.
5. **Check thread statuses**: Review `threads/` for active threads that should be updated. A thread with `status: active` MUST be followed up in the current quarter if relevant events occurred.
6. **Check for upcoming forecast-relevant elections**: Before writing a contemporary quarter file (post-2020), check the calendar for any major elections (presidential, parliamentary, or otherwise consequential) scheduled within the next 2 quarters. If an election is upcoming:
   - Create placeholder wiki links for major candidates who lack entity stubs
   - Note the electoral system type (single-round plurality, two-round runoff, PR)
   - Plan to include a dedicated election campaign subsection in the quarter file
   - Flag the candidate field and opposition coordination status for coverage
   - This ensures the vault has PIT campaign coverage BEFORE the election occurs, not just post-hoc results. The Taiwan 2024 election gap (see _spec.md Rule 22) demonstrated that pre-election coverage is routinely missed.

6a. **Audit existing quarter files for pre-election campaign coverage**: AFTER writing any quarter file that covers a period before a scheduled major election, verify that the PREVIOUS quarter(s) also contain the mandated campaign subsection. The Argentina 2025 legislative election case is the canonical example of a systematic failure: the vault's quarter files (2025-Q1, 2025-Q2, 2025-Q3) contained zero pre-election campaign coverage despite the October 26 election being Argentina's most consequential political event of 2025. The thread then documented excellent post-hoc results, but a mid-2025 forecaster would have found no campaign context in the timeline. The audit must check:
   - Does Q-2 (six months before election) mention the upcoming election and the candidate field?
   - Does Q-1 (quarter immediately before election) have a dedicated campaign subsection with polling data, scandal tracking, and candidate status?
   - If either is missing, treat it as a vault gap requiring remediation regardless of whether the election outcome is already known.
   - The verification is simple: search the quarter file for the election's country name and the word "election" or "campaign." If zero results, the gap exists.

7. **Check for major terrorism/security events in belligerent states**: Before writing any contemporary quarter file covering a period when a state was actively at war, search for major terrorist attacks (>20 casualties) or security crises on that state's territory. If found:
   - Document the attack details, the actual perpetrator (if known), and the regime's public attribution
   - Note whether the regime blamed its war adversary (the [[domains/global/concepts/wartime-blame-shifting]] dynamic)
    - Create or update the relevant [[domains/global/threads/isis-resurgence]] or analogous thread
    - Create entity stubs for any named perpetrator organization not yet in the vault
    - The Crocus City Hall attack (March 2024) is the canonical case: a major terror attack on a belligerent state where blame-shifting occurred. Future quarter files covering states at war must not repeat this gap.

8. **Check for ongoing infectious disease outbreaks with case count milestones**: Before writing any contemporary quarter file, check whether major infectious disease outbreaks (H5N1 bird flu, mpox, seasonal influenza, COVID-19 variants) are ongoing in the US or globally. If they are:
   - Record the CDC's official human case count for US outbreaks (or WHO count for global outbreaks) as of the quarter's end date
   - Note the CDC public health risk assessment (low/moderate/high) as a leading indicator for trajectory
   - Check whether any step-change event occurred (human-to-human transmission cluster, mammalian adaptation mutation, vaccine breakthrough variant)
   - Verify that the ongoing outbreak is connected to a thread file (e.g., [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]]) — create or update the thread if needed
   - This ensures that any future forecasting question about outbreak case counts finds PIT quantitative trajectory data in the thread, not just a single mention in a quarter file

9. **Check for major policy domains missing active threads**: Before writing any contemporary quarter file (especially post-2020, where the executive branch has policy-setting power), identify whether there are active threads for each major policy domain of the administration in power. For a US presidency, these would include:
   - Immigration policy (border enforcement AND legal immigration reform)
   - Trade policy (tariffs, trade agreements)
   - Energy/climate policy
   - Health policy
   - Technology regulation
   - Antitrust/competition policy (FTC + DOJ enforcement against Big Tech)
   - Defense/foreign policy
   - Budget/debt/fiscal policy
   
   If a thread does NOT exist for a major policy domain, create it as a stub thread with the following minimum content:
   - The administration's campaign platform on the issue
   - Key internal coalition factions and their policy preferences
   - Key personnel with influence (agency heads, advisors, congressional allies)
   - First-year legislative/executive agenda
   - Cross-reference to relevant [[concepts/program-restriction-vs-elimination]] and [[concepts/first-100-days-action-horizon]] for feasibility analysis
   
   This prevents the post-hoc gap where a forecasting question arrives and the vault has no threaded analysis of the policy domain. The Trump immigration policy gap (filled per-question reflection Q48) is the canonical case: before H-1B questions appeared, the vault had zero immigration content despite it being the central domestic policy focus of Trump's campaign.

10. **Check for central bank monetary policy developments**: Before writing any contemporary quarter file (post-2020), check whether any major central bank (Fed, ECB, BoJ, BoE) held a scheduled meeting during the quarter. For each meeting identified:
    - Record the rate decision (hike, hold, or cut) with the exact basis point change
    - Note the forward guidance language shift from the previous meeting
    - For the Fed with quarterly SEP meetings (March, June, September, December), record the dot plot median projection shift
    - Note any dissenting votes and the dissenter's identity (creates entity requirement)
    - Record incoming CPI/PCE data releases and market repricing during the inter-meeting period
    - Connect to existing thread (e.g., [[domains/economics/threads/us-monetary-policy-cycle-2022-2026]]) or update thread with new meeting data
    
    This ensures that quarter files provide PIT monetary policy context for any future rate-decision forecasting question. The Q25 gold_25 reflection (Fed increases after July 2024) is the canonical case: the June 2024 SEP shift from 3 cuts to 1 cut was a critical signal documented in the pit_blind_test quarter files but absent from the graph-vault's timeline files, forcing a post-hoc remediation. A vault that covers the 2024 election campaign comprehensively but misses the June 2024 dot plot hawkish shift is materially incomplete for monetary policy questions with mid-2024 PIT cutoffs.

### Phase 2: Writing the Quarter File
1. **Frontmatter**: Fill in type, year, label, date_range, prev, next, pit_cutoff, source.
2. **Overview**: Write 1-2 paragraphs that frame the quarter's significance.
3. **Major sections**: Order by geopolitical importance first, then science/culture.
   - Each section starts with `##` header.
   - Within sections, use month-based subsections with `###` for long arcs.
   - Event entries use `- **Date**: Description` format with [[wikilinks]].
4. **Cross-Domain Threads**: At the bottom, write 3-7 thematic analyses using `###` headers. These are the forecasting value — pattern recognition across domains.
5. **Wikilinks Created**: Comprehensive list at bottom. This is both a reference and a checklist for entity creation.
6. **Births/Deaths**: Integrated into month sections. Prioritize people with forecasting relevance.

### Phase 3: Update Thread Files

AFTER writing the quarter file, update ALL active thread files:
1. Read the thread's existing content
2. Append new developments from the current quarter
3. Update the thread's `status` and `span` fields
4. Add wikilinks to the current quarter file
5. Do NOT delete old content — threads are cumulative
6. **Verification step**: After updating, run this check: for every thread with `status: active`, confirm a new entry was appended for this quarter OR add a documented skip reason. If any active thread has been skipped for 2+ consecutive quarters, change its status to `fading` with a rationale. This verification must be completed before marking the quarter file as done.

## Per-Forecast Cycle (for each prediction question)

Every forecast question is a test of the vault. When a question arrives, perform this rapid vault-audit before predicting:

### Pre-Forecast Audit (before predicting)
1. **Map the question domain**: Is this historical or contemporary? Identify relevant quarter files, threads, concepts, and entities.
2. **Check contemporary coverage**: If the question is about current events (post-2020), verify that contemporary quarter files have active threads. If not — flag this as a vault gap for post-forecast remediation.
3. **Check entity coverage for key actors**: Identify ALL named persons, parties, organizations, and coalitions in the question. Check if entity files exist. If any key actor lacks a file, create an entity STUB (frontmatter + 1-2 paragraph summary) BEFORE forecasting — this ensures the reasoning references existing vault structure rather than ephemeral knowledge.

3a. **Comprehensive exclusion list audit** — If the question follows the "Will another/an unlisted [entity] achieve [outcome]?" pattern with a specific exclusion list of N named entities:

   - **Count the exclusion list**: Record the total number of named entities on the exclusion list.
   - **Classify each entry** using the [[domains/usa/concepts/comprehensive-exclusion-list-forecast]] framework:
     - Category A (genuine contenders): 50-100% plausible — people actually under consideration
     - Category B (plausible mentions): 10-50% plausible — media speculation without formal vetting
     - Category C (padding): 0-5% plausible — constitutionally ineligible or no realistic path
   - **Calculate the effective exclusion list**: Category A + (0.3 × Category B). Ignore Category C entirely — padding inflates the list without narrowing the candidate pool.
   - **Assess the plausible unlisted pool**: Are there any plausible candidates NOT on the list? List them explicitly. If none, the universe is saturated.
   - **Check for process changes since list creation**: Did a major event (nominee change, scandal, withdrawal) fundamentally alter the selection process after the list was created? If yes, the list may be outdated.
   - **Apply the calibration table** from the comprehensive-exclusion-list-forecast concept: high-transparency process + effective exclusion > 10 + no plausible unlisted candidates → P(YES) < 3%.
   - **Document the exclusion list analysis explicitly** in the reasoning. Count the entries, classify them, calculate the effective list, and state the calibrated probability. The most common forecasting error is treating a 15-name exclusion list as covering 15 plausible candidates, when in reality 5-6 are genuine contenders and the rest are padding or speculation.
   - **Create entity stubs for EVERY named entity on the exclusion list** before forecasting — Spec Rule 9 requires this regardless of whether they seem implausible. An entity stub for Mark Cuban is still mandatory even though Cuban had zero chance of being VP. The stub documents why the entity was on the list and how to categorize it for exclusion-list analysis.
   - **Cross-reference with other frameworks**: If the question involves VP selection, also load [[domains/usa/concepts/veepstakes-electoral-signal]] and [[domains/usa/concepts/gender-balancing-ticket-composition]] — the exclusion-list framework gives the structural baseline; these frameworks provide the selection-process mechanics that determine the actual outcome.

   This step exists because the "Will another man be the 2024 Democratic VP nominee?" question (correct NO) was answered correctly but the vault had no formalized concept for analyzing exclusion-list questions. After this step, every future "another X" question will trigger systematic exclusion-list analysis with calibrated probability before forecasting.

4. **Terrorist-attribution audit**: If the question asks whether a state adversary was responsible for a terrorist attack (e.g., "Was Ukraine responsible for the Moscow Crocus attack?"):
   - Check whether the actual perpetrator (usually ISIS-K or another non-state actor) claimed responsibility and is a known entity — create an entity stub if missing
   - Check whether the accusing state is at war with the alleged responsible state — if yes, apply the [[domains/global/concepts/wartime-blame-shifting]] framework
   - Check whether the accusing state received advance warnings about the attack that it failed to act on — this creates a domestic incentive for blame-shifting
   - Check whether any independent sources (allies, intelligence partners, journalists) support the state's attribution — the diplomatic consensus is a strong signal
   - Forecast NO (adversary NOT responsible) unless specific independent evidence exists, because wartime blame-shifting is the default regime response pattern
5. **Check domain thread**: Verify that a thread file exists for the question's primary domain (e.g., Argentina politics, Israel-Iran conflict, US elections). If no thread exists, note this as a pre-forecast vault gap that requires remediation regardless of prediction outcome.
6. **Named entity sweep**: Extract every proper noun from the question text (e.g., "HNP", "Chamber of Deputies", "Argentina") and verify each has a vault stub or is linked to an existing entity. The question's named entities are the MINIMUM coverage bar. Create any missing stubs before forecasting.

6a. **Content-loaded entity sweep**: After loading any concept, thread, or procedure file (via skill_view, read_file, or direct load), scan the loaded content for named wikilinks to entities ([[entity-name]] or [[path/to/entity|Entity Name]]). For EACH named entity referenced in the loaded content, verify that a vault file exists at the referenced path. If a referenced entity lacks a stub, create it before forecasting. This sweep catches gaps that the question-text sweep (step 6) misses — concept files routinely reference entities (party elders, donors, historical figures, institutional actors) that do not appear in the question text but are critical to the analytical framework. The canonical violation found in the Biden dropout reflection: the incumbent-withdrawal-cascade concept referenced [[Nancy Pelosi]] (file existed but wikilink path was wrong) and [[George Clooney]] (no file existed) as key cascade actors, but neither was flagged by a question-text sweep because neither name appeared in the question "Biden drops out of presidential race?" Every named entity in a loaded concept, thread, or procedure MUST have a resolvable vault stub before forecasting.
7. **Structural feasibility check for third-way/regional parties**: If the question asks whether a party, coalition, or individual can win a national plurality or majority:
   - Check the party's national vote share history — has it ever exceeded 15%? If no, a national plurality is structurally improbable without a prior collapse of the dominant blocs.
   - Check whether the party's power base is regional: is its vote share concentrated in one province/state? If the home-region share is 3x+ the national share, the party cannot scale to national dominance.
   - Check the polarization trend: is the national electorate bipolarizing between two dominant blocs? If the combined share of the top two blocs exceeds 70% and is growing, third-way/centrist/regional parties are structurally squeezed.
   - Apply the [[regional-third-way-squeeze]] concept framework: estimate the squeeze magnitude (30-60% vote share decline over 1-2 cycles) and assess whether the party can plausibly reach "most seats" territory.
   - Document the feasibility assessment explicitly in the reasoning. For the Argentina 2025 questions, this check would have shown that HNP (7.73% national, Córdoba-only base, squeezed by Milei-Kicillof polarization) had a ~0% probability of winning most seats — a structural impossibility that general reasoning confirmed. For far-left parties (FIT-U in the same election, 3.90%), the same structural ceiling applies: doctrinaire left parties cannot win national pluralities in polarized two-bloc systems. Apply the [[far-left-marginalization-polarization]] concept for the ceiling heuristic (<5% for Trotskyist/revolutionary-socialist parties, <12% for populist-left parties).
8. **Check candidate count and opposition fragmentation for election questions**: If the question asks whether a specific candidate will win an election:
   - **Count the credible candidates**: How many candidates are polling >10%? If 3+ candidates are viable, the opposition fragmentation effect is active.
   - **Assess the electoral system**: Is the final election single-round plurality (first-past-the-post) or majority-runoff? Only single-round plurality is subject to fragmentation-driven plurality wins. Document the system type explicitly.
   - **Check opposition alliance negotiations**: Are trailing candidates discussing a joint ticket? Have registration deadlines passed? If negotiations failed or deadlines lapsed, fragmentation is locked in.
   - **Assess the front-runner's polling structure**: If the leader is at 30-45% with the opposition split below them, this is structural strength from fragmentation, NOT weakness. The front-runner will likely win with <50%.
   - **Apply the [[concepts/divided-opposition-plurality-win]] framework**: In a single-round plurality election with 3+ viable candidates and the front-runner polling at 30-45%, forecast a win at 85-95% confidence absent a last-minute opposition consolidation.
   - **Create entity stubs for ALL named candidates and their parties**: This is mandatory. The Taiwan 2024 question named Lai Ching-te and the DPP — neither had a vault stub. The implicit actors (Ko Wen-je, Hou Yu-ih, KMT, TPP) also lacked files.
   - **Document the structural rationale explicitly**: Unlike policy-driven forecasting (which requires analyzing voters, platforms, and issues), fragmentation-driven forecasting is structural — it follows from the electoral system and the candidate count. The reasoning should state these mechanics.
   - **Classify structural vs. performance variables explicitly**: Before calibrating probability, write a classification sentence: "Structural variables that dominate this forecast: electoral system ([X]), candidate count ([Y]), opposition coordination status ([Z]). Performance variables that are secondary: approval trends, scandal impact, campaign quality. The winner will be determined by the structural variables because the opposition cannot coordinate under single-round plurality."
   - **Check historical precedent within the SAME electoral system**: For the country in question, look up all prior elections under the same electoral rules. Taiwan 2000 (3-way, 39.3%) and Taiwan 2024 (3-way, 40.05%) are the canonical pair — a 24-year replication of the same divided-opposition pattern. If historical precedent shows plurality winners, the current race's front-runner is structurally dominant regardless of current polling trends against them.
   - **Check for the reasoning trap**: Apply the [[domains/global/concepts/plurality-race-reasoning-trap/_concept]] diagnostic. If you are predicting NO for a front-runner at 30-45% in a 3-way FPTP race, load the reasoning trap concept, run the self-diagnostic (see "Diagnostic: Are You Falling Into the Trap?" table), and document your assessment of whether the trap applies. A NO prediction under these conditions requires an explicit rebuttal of the trap — showing HOW the opposition will coordinate despite negotiation failure, deadline passage, and voter psychology.
 
  This step exists because the Taiwan 2024 presidential election (gold_26) was predicted NO (wrong) when all structural variables pointed to a YES. The error was treating the race as a two-candidate popularity contest (DPP fatigue, Lai's low approval) while ignoring the decisive structural variable: three candidates in single-round plurality with opposition coordination already failed. The reasoning trap concept was created in response. Every future election forecast must systematically avoid this error by classifying structural vs. performance variables before calibrating probability.

8a. **Classify state-level electoral reliability for US presidential questions**: If the question asks whether a specific party will win a US state in a presidential election:
   - **Classify the state** using the [[domains/usa/concepts/state-electoral-reliability/_concept]] framework: identify whether the state is Safe/Likely/Lean for either party or Tossup based on recent election history (last 2-3 cycles).
   - **Check if the state was ever seriously contested**: Was the state in the tossup column of any major election forecaster (Cook, Sabato, 538, Nate Silver) during the cycle? If no major forecaster ever rated it as competitive, the prediction is near-deterministic.
   - **Determine the national margin needed**: What national popular vote margin would be required for the opposing party to flip this state? Use the state's partisan lean (Cook PVI or similar) to calculate: flip threshold ≈ (state's PVI) × 2. A state with D+5 PVI requires R+10 national margin to flip, which is historically rare.
   - **Assess whether that national margin is plausible**: Given the national polling, candidate dynamics, and historical precedent, is a national landslide even plausible? Most US elections are decided by <5 points nationally, which cannot flip states with >5-point partisan lean.
   - **Check for unique local conditions**: Are there state-specific factors (ballot measures, scandal, natural disaster, demographic change) that could override the structural baseline? If none, the structural baseline dominates.
   - **Apply the default rule**: For states classified as "Safe" or "Likely" for a party, and no major forecaster rated them competitive, and no unique local condition overrides the baseline: predict the party that holds the state. The probability is >95% for Safe states and >90% for Likely states.
   - **Document the classification explicitly**: State the state's category, the national margin threshold, and the assessment of whether that threshold is plausibly reachable. This forces systematic reasoning rather than gut-feel national-level analysis.
   
   This step exists because the New Mexico question (gold_115, a correct NO prediction) was answered correctly on general knowledge, not vault content. The state was never considered competitive in 2024, but the vault had no framework for classifying state-level reliability. Adding this step ensures future state-level questions have systematic vault-driven analysis.
   
   This step is distinct from step 7 (regional third-way squeeze). Step 7 applies to parties that are regionally strong but nationally weak. Step 8 applies when 3+ candidates are competing nationally and the opposition is fragmented. The Taiwan 2024 election is the canonical case of step 8: a national three-way race where opposition fragmentation guaranteed a plurality win for the front-runner.

8b. **Forecast House seat ranges using the vote-seat conversion framework**: If the question asks whether a specific party will win a specific seat count or range in a US House election:

   - **Establish the current seat baseline**: Before the election, how many seats does each party hold (including vacancies)? This determines the "starting point" for the expected seat change.

   - **Assess the generic ballot projection**: What is the national House popular vote polling average? Use the generic ballot as the primary input — not presidential polling, which has a distinct conversion relationship to House outcomes unless coattails are strong.

   - **Apply the seat-vote conversion function**: Translate the generic ballot projection into a seat range using the [[domains/usa/concepts/generic-ballot-seat-conversion/_concept]] framework. A tied generic ballot produces a Republican seat floor of approximately 213-216 due to the gerrymandering advantage. Adjust for any expected change from the baseline.

   - **Assess presidential coattail intensity**: Use the [[domains/usa/concepts/presidential-coattail-variability/_concept]] three-factor framework (candidate novelty, margin above party baseline, ticket-splitting environment) to determine whether the presidential race will add or subtract House seats for the winning party. In 2024, all three factors were near-zero, yielding negligible coattails.

   - **Combine to produce an expected seat range**: The expected range = baseline + coattail adjustment + seat-vote conversion from generic ballot. In 2024: pre-election baseline 222 GOP seats - minimal coattails (~0 seats) + tied generic ballot (GOP seat floor at ~213-216) = expected Republican range of 215-218, with 220 as the plausible upper edge.

   - **Evaluate the question's range against the expected range**: Is the question's range centered on the plausible outcome or at the edge? The 215-219 range for Republicans was centered on 217 — a plausible midpoint — but the upper bound of 219 was structurally too low. The gerrymandering advantage and the narrow pre-election majority meant 220 was structurally plausible as the upper edge. A forecast of NO was correct because 220 was the most likely outcome if the GOP held the edge of the expected range.

   - **Check district-level factors**: Are there court-ordered map changes, unusual retirement waves, or scandal effects in specific districts that could shift the conversion function?

   - **CRITICAL — Classify the question type BEFORE applying the distribution model**: House numerical outcome questions come in four types, each requiring different methodology:
     - **Exact-count** ("exactly N seats") → Load [[domains/usa/procedures/exact-seat-count-forecast]] and use within-bin distribution. Do NOT use the bin-level table.
     - **Range** ("between A and B seats") → Continue with this procedure (bin-level distribution from generic-ballot-seat-conversion).
     - **Threshold** ("at least N seats") → Use cumulative probability from the bin-level distribution (P(X >= N)).
     - **Binary control** ("will party X control the House?") → Use P(seats >= 218) from the bin-level distribution.
     
     The most common methodology error is treating an exact-count question like a range question. An exact-count question asking about 223 seats has P(223) ≈ 5%, while a range question asking about 220-224 seats has P(220-224) ≈ 35%. Applying the range framework to an exact-count question overestimates YES probability by 5-10x. Load [[domains/usa/concepts/exact-count-vs-range-forecast/_concept]] for the full distinction and [[domains/usa/procedures/exact-seat-count-forecast]] for the exact-count methodology.

8c. **Check for external threat/interference effects on elections**: If the question asks about the outcome of an election in a country with a known external adversary that has recently escalated pressure:

   - **Identify the adversary and escalatory action**: Has a foreign power taken visible hostile action (military exercises, sanctions, trade war, cyberattacks, hostile diplomacy) within the past 6 months that targets or pressures the country?

   - **Assess partisan alignment**: Is the incumbent party/coalition clearly more hawkish on this adversary than the opposition? Does the opposition have policies that could be framed as conciliatory? The effect requires a clear hawk/dove distinction to produce an electoral advantage.

   - **Check timing**: Is the escalatory action within 3 months of the election? The closer to election day, the stronger the effect. If the action was 6+ months prior, the salience will have decayed and the effect may be marginal.

   - **Check for blowback**: Did the incumbent's own actions plausibly provoke the escalation? If yes, the "external threat" framing is contested and the effect may be neutral or reversed.

   - **Check for economic pain**: Does the external threat cause economic harm (sanctions, trade disruption, energy price spikes) that voters will blame on the incumbent? If the economic channel dominates, the threat may hurt rather than help the incumbent.

   - **Apply [[domains/global/concepts/external-threat-incumbency-boost/_concept]]**: Load the full concept and calibrate the magnitude:
     - Strong (3-8 pp boost): visible escalation <3 months, clear hawk/dove distinction, no blowback, no major economic pain
     - Moderate (1-3 pp): less visible or more distant escalation
     - Neutral/reversed: chronic threat, blowback, economic pain, or opposition equally hawkish

   - **Document the external-threat assessment explicitly**: Write a sentence like "PRC military exercises 2 months before the Taiwan election activate the external-threat-incumbency-boost pattern at a moderate level (1-3 pp boost to DPP), reinforcing the structural advantage from opposition fragmentation."

   - **Separate from the opposition fragmentation analysis**: The external-threat effect is a **secondary amplifier**, not the primary driver. In Taiwan's 2024 election, the primary variable was opposition fragmentation (3-way race in FPTP guaranteed DPP win). The external-threat effect (PRC pressure) was a secondary factor that may have increased Lai's margin from ~36% to 40%. Do NOT confuse the two — they operate through distinct mechanisms and should be assessed independently before combining into a final probability.

   This step exists because the Ko Wen-je question (Q27, correct NO) was correctly predicted using the third-party-ceiling and divided-opposition concepts, but the vault had no formalized concept for the external-threat-incumbency-boost dynamic that helps explain the DPP's structural advantage and the precise margin. Every future election forecast in a country facing an adversarial external actor must assess this dynamic as a secondary amplifier.

8d. **Apply the CR-governance shutdown framework for US government shutdown questions**: If the question asks whether a US government shutdown will occur (or whether government funding will pass by a deadline):

   - **Identify the exact deadline**: Sep 30 (FY change), Dec (lame-duck CR), or Mar (full-year or CR extension). Mark the exact date and time.
   
   - **Read the thread**: Load [[domains/usa/threads/us-government-shutdown-crises/_thread]] to understand the current status of the CR cycle and previous episodes.
   
   - **Read the concept**: Apply [[domains/usa/concepts/cr-governance-shutdown-dynamics/_concept]] to estimate the structural variables: Freedom Caucus defection count (F), Democratic leadership posture (D), external actor intervention (E), Johnson's procedure track choice (J), and time-bound policy clocks (T).
   
   - **Check entity coverage for all key actors**: Create stubs for Mike Johnson ([[domains/usa/entities/mike-johnson]]), Hakeem Jeffries ([[domains/usa/entities/hakeem-jeffries]]), House Freedom Caucus ([[domains/usa/entities/house-freedom-caucus]]), and any external actors (Elon Musk, Donald Trump) if they are actively intervening — these stubs MUST exist before forecasting, as they document the institutional roles and leverage mechanisms that determine the outcome.
   
   - **Read the procedure**: Load [[domains/usa/procedures/us-government-shutdown-forecast]] and apply the 7-step forecasting algorithm to produce a calibrated probability estimate.
   
   - **Analyze the resolution text carefully**: This is the single most important step for Polymarket shutdown questions. The resolution text may define "shutdown" differently from government practice — some markets consider any funding lapse (even hours) a shutdown, others require OMB shutdown activation, and others reference an executive order trigger. The distinction between a funding lapse (deadline passed by minutes/hours before bill signed) and a formal shutdown (OMB furloughs employees) can be outcome-determinative. Document the resolution definition explicitly and adjust the probability accordingly (broad definition => 15-25% higher YES probability; narrow definition => 15-25% lower).

   - **Distinguish shutdown from debt ceiling**: The resolution text may conflate "government shutdown" with "debt ceiling crisis." If the question asks about a shutdown but the resolution text mentions default risk, apply BOTH this procedure and the [[domains/usa/procedures/debt-ceiling-forecast]] procedure. They are distinct mechanisms that require separate analysis.
   
   - **Apply the cascade sequence model**: Determine which stage of the shutdown cascade the process is currently in (Stage 0: status quo; Stage 1: partisan proposal; Stage 2: partisan fails; Stage 3: clean bipartisan CR; Stage 3V: external intervention; Stage 4: last-minute resolution; Stage 5: full shutdown). Each stage has a different probability of advancing to the next stage. Knowing the current stage dramatically improves the forecast's temporal accuracy.
   
   - **Document the structural variables explicitly**: Write the forecast with each variable (F, D, E, J, T) assessed and the final probability calculated from the algorithm, rather than a narrative judgment. This makes the forecast auditable and the vault accountable.

   This step exists because Question 37 (gold_37, US government shutdown, correct YES) relied on general knowledge of the Dec 2024 CR crisis rather than vault content. The vault had no thread, concept, entities, or procedure for US government shutdowns despite _spec.md Rule 10 mandating US domestic budget coverage. The correct prediction was a coincidence of general knowledge, not vault-driven analysis — and the vault's gap was as real as if the prediction had been wrong.

   - **Document the reasoning**: State each step explicitly — the generic ballot projection, the seat-vote conversion estimate, the coattail assessment, and how the question's range relates to the expected outcome.

   This step exists because the House seat range question (question 34, correct NO prediction) was answered correctly on general knowledge (a tied generic ballot + narrow pre-election majority = GOP around 217-220; 215-219 range was too tight), but the vault contributed zero structured input. After this step, every House seat range question will trigger systematic seat-vote conversion analysis, parallel to the state-level reliability framework in step 8a.
9. **Identify escalation ladders**: For conflict-related questions, map the escalation thresholds already crossed. Each threshold crossed increases the probability of further escalation. Use the [[escalation-bargaining-termination]] concept as a framework.
10. **Identify patron dynamics**: For conflicts involving a client state and superpower, assess whether direct patron entry would accelerate or slow ceasefire.
11. **Track diplomatic signals**: For ceasefire-related forecasts, identify ALL of the following before predicting:
   - Does a concrete peace proposal or framework exist? Who proposed it and when?
   - Have indirect talks or backchannel communications begun?
   - Is international pressure accumulating (multilateral statements, legal findings, sanctions, recognitions)?
   - Is the belligerent's political leadership facing domestic pressure to end the conflict?
   - Has the patron signaled a change in posture (distancing from client, proposing terms)?
   - **Track the exact sequence of official announcement vs. ratification**: For date-specific ceasefire questions, distinguish between:
     * **First official announcement of agreement** (executive/PMO statement confirming the deal) — this is the date that resolves Polymarket-style "first announce" questions
     * **Cabinet/parliamentary ratification** (formal approval vote) — this is an internal process that typically follows 1-7 days after the announcement
     * **Ceasefire effective date** (when the halt in military engagement actually begins)
     * Document ALL three dates separately in the reasoning. Conflating ratification dates with announcement dates is a common forecasting error.
   - **CITICAL — Check ceasefire definition scope (temporary vs. enduring)**: The term "ceasefire" in Polymarket resolution criteria means ANY publicly announced and mutually agreed halt in military engagement, including temporary humanitarian pauses, hostage-exchange truces, and multi-day lulls. A ceasefire is not invalidated by its subsequent expiration or collapse. Before predicting:
     * Read the FULL resolution text, not just the question title
     * Check for qualifier words: "permanent," "comprehensive," "lasting," "end to the war," "ending hostilities"
     * If NO qualifier is present, assume ANY temporary halt qualifies (even 4-day pauses)
     * Check whether a humanitarian pause, hostage deal, or temporary truce occurred in the relevant period
     * Check media/international organization descriptions — if they call it a "ceasefire," treat it as one for resolution purposes
     * Apply the [[concepts/temporary-vs-enduring-ceasefire]] framework: when a temporary pause exists and the resolution text is unqualified, predicate YES even if the pause later expires
     * See gold_8 error (Israel-Hamas ceasefire 2023? predicted NO, actual YES) — the canonical case of this gotcha
   Record these as a "diplomatic signals inventory" in the reasoning. Missing diplomatic signals is the most common cause of wrong ceasefire predictions.
11a. **Assess inter-state ceasefire structural feasibility**: If the ceasefire question involves two states (not a state vs. non-state actor), apply the protracted-war-stalemate feasibility framework BEFORE assessing diplomatic signals or timing. Load [[domains/global/procedures/inter-state-ceasefire-feasibility]] and run all 8 steps:
    - **Step 1: Territorial Incompatibility** — Are the parties' territorial demands mutually exclusive? If yes, ceasefire is structurally unlikely without a decisive battlefield outcome.
    - **Step 2: Military Trajectory** — Is front-line movement <5km/month? If positional stalemate, ceasefire is unlikely.
    - **Step 3: Mutually Hurting Stalemate** — Is either side's regime at risk of collapse from war costs? If no, no incentive to settle.
    - **Step 4: Credible Mediator** — Does an external actor have leverage over BOTH parties AND willingness to use it? If no, no diplomatic mechanism exists.
    - **Step 5: External Sustainment** — Can both sides continue fighting through patron supplies? If yes, continuation is feasible.
    - **Step 6: Political Deadline** — Is a transition or election approaching that could force a breakthrough?
    - **Step 7: Synthesis** — If 4+ factors point toward continuation, P(ceasefire within 12 months) < 0.15.
    - **Step 8: Context-Specific Adjustments** — Nuclear power conflict? Use diplomatic/internal framework, not escalation-bargaining-termination.
    Document each step's assessment explicitly. If 4+ factors point to continuation, the default forecast is NO and diplomatic signals (Step 11) primarily serve as confirmation, not as countervailing evidence. This framework correctly predicted NO for "Russia x Ukraine Ceasefire in 2024?" — all five primary factors favored continuation.
    This step fills the gap between the general diplomatic signals inventory (Step 11) and the asymmetric-conflict leadership decapitation analysis (Step 12). Inter-state ceasefires follow fundamentally different dynamics from state-vs-non-state ceasefires, and this framework prevents applying the wrong pattern.
12. **Assess leadership decapitation impact**: For ceasefire questions involving non-state armed groups (Hamas, Hezbollah, etc.), check whether a leadership decapitation event occurred within the last 6 months:
    - Was the most hardline leader killed? (Decapitation of a moderate EMPOWERS hardliners and reduces ceasefire probability)
    - If yes, how many days has it been since the decapitation?
      * 0-30 days: Ceasefire probability DECREASED (retaliation phase)
      * 30-60 days: Return to baseline (reassessment phase)
      * 60-120 days: Ceasefire probability ELEVATED (negotiation window opens)
      * 120+ days: Window closing — successor likely consolidating authority
    - Does the successor have equal authority to the predecessor? If yes, the window may not open.
    - Does a pre-existing diplomatic framework exist? Decapitation alone does not create a ceasefire — there must be terms for the successor to accept.
    - Is the organization facing military exhaustion (conflict 12+ months)? Exhaustion amplifies the decapitation effect.
    - Apply the [[concepts/leadership-decapitation-negotiation-window]] framework: calibrate probability based on days-since-decapitation + successor authority + diplomatic framework existence.
    
    Document this assessment explicitly in the reasoning. This step creates a systematic bridge between military events (leadership decapitation) and diplomatic outcomes (ceasefire timing).
12. **Check political deadlines as ceasefire catalysts**: For ceasefire-by-[specific date] questions, BEFORE calibrating timing with the diplomatic-pressure-tipping-point framework, check whether a known political deadline (inauguration, election, transition, legal judgment, summit) is approaching within the next 3 months. If such a deadline exists:
   - **Identify the deadline type**: US presidential transition (strongest catalyst), election (moderate), summit/symbolic date (weak-moderate), legal deadline (variable).
   - **Assess the direction of expected change**: Will the post-deadline policy environment be more or less favorable to each belligerent? If one side expects improvement, they have incentive to delay — reducing the deadline's forcing effect.
   - **Check preconditions**: The deadline alone cannot create a ceasefire. Verify that the preconditions (military exhaustion, diplomatic framework, leadership changes) are present. The deadline COMPRESSES the timeline for an already-ripening deal.
   - **Measure the deadline gap**: If the target date is 1-21 days before the deadline → strongest signal. If 22-60 days → moderate signal. If 61+ days → weak signal.
   - **Apply the [[concepts/political-deadline-ceasefire]] framework**: The pre-inauguration effect (January 2025) is the canonical example — a US presidential transition created a structural forcing function that compressed ceasefire negotiations into a ~10-week window.
   
   Document the political deadline assessment explicitly in the reasoning. This step fills the gap between leadership decapitation analysis (step 11) and timing calibration (step 13). A ceasefire forecast that ignores an approaching political transition is systematically underestimating the temporal compression effect.

13. **Calibrate timing**: For ceasefire-by-deadline questions, apply timing calibration rules from the relevant concept (typically [[diplomatic-pressure-tipping-point]] or [[political-deadline-ceasefire]]):
   - Calculate months elapsed since last ceasefire collapse or conflict start
   - If < 3 months: forecast NO (insufficient accumulation time) unless an escalation-bargaining dynamic is in play
   - If 3-6 months: count active pressure vectors — this determines whether the timeline is feasible
   - If > 6 months and no off-ramp exists: forecast NO until a concrete proposal emerges
   - Check whether the patron superpower is distracted by another crisis — if yes, add 1-2 months to expected timeline
   - Distinguish which ceasefire pattern applies: [[escalation-bargaining-termination]] (days) vs [[diplomatic-pressure-tipping-point]] (months) — using the wrong pattern's timing will produce the wrong forecast

   Document the timing assessment explicitly in the reasoning. This step was the critical factor in correctly predicting NO for the "Israel x Hamas ceasefire by July 15?" question — ~3.5 months from March collapse was at the minimum edge for the diplomatic-pressure-tipping pattern.
14. **Assess point-in-time accuracy**: Read the most recent relevant quarter file and verify its claims are accurate by cross-referencing with external sources available at that PIT cutoff.
15. **Check geographic economic coverage parity**: If the question involves economic data (inflation, interest rates, GDP growth, unemployment) for a non-US economy or region, verify that the vault has coverage structurally parallel to what exists for the US:
    - **Central bank entity check**: Does an entity stub exist for the relevant central bank (ECB, BOJ, BOE, PBOC, RBI, TCMB, BCB, CBR, SARB, etc.) and its chair/governor?
    - **Central bank regime check**: Does the central bank operate under political constraints? If yes, does an EM-central-bank-credibility-normalization concept exist to frame the policy dynamics? See [[domains/mena/concepts/em-central-bank-credibility-normalization]] for the template.
    - **Macro thread check**: Does the vault have a thread tracking that economy's key indicators (parallel to [[domains/economics/threads/us-macro-economic-indicators]])? If the question is about the eurozone, check for [[domains/economics/threads/eurozone-macro-economic-indicators]]. If about Japan, check for a Japan macro thread. If about the UK, check for a UK macro thread. If about Turkey, check for [[domains/mena/threads/turkish-monetary-policy-normalization]].
    - **Inflation measurement check**: If the question uses a specific inflation metric (HICP, CPI, RPI, PCE), does a concept file exist documenting what the metric measures and how it differs from other metrics?
    - **Causal chain check**: Does a concept file exist for the dominant driver of the question's economic dynamic (e.g., [[domains/economics/concepts/post-covid-inflation-surge]] for 2021–2023 inflation questions, [[domains/mena/concepts/em-central-bank-credibility-normalization]] for politically constrained EM central banks)?
    - **Quarter file check**: Do quarter files exist for the relevant time period? If the question references October 2021, verify that 2021-Q3 and 2021-Q4 exist and contain relevant economic data.
    - **Action**: For any identified gap, create the missing stub/thread/concept before forecasting. A vault with US macro depth but zero eurozone macro coverage cannot produce reliable forecasts for EU inflation questions.
    
    This step exists because Question 1 of the PIT blind test (EU HICP inflation >= 4.3% in October 2021) exposed a vault with robust US macro coverage (Fed, FOMC, Powell, CPI/PCE) but zero eurozone coverage (no ECB, no Lagarde, no HICP concept, no eurozone macro thread, no 2021 quarter files). The correct prediction was based on general knowledge, not vault signal — a violation of Spec Rule 8 ("no freebie predictions"). Every future economic question about a non-US economy must find vault coverage of comparable structural depth.
    
    **EM Central Bank Extension (Added after pm_eco_05 — Turkish Central Bank rate question):** For questions about EM central banks, the vault must additionally check for:
    - An entity stub documenting the central bank's governance, political constraints, and historical policy regimes (see [[domains/mena/entities/turkish-central-bank-tcmb]] as the template)
    - A thread tracking the economy's monetary policy normalization or cycle (see [[domains/mena/threads/turkish-monetary-policy-normalization]])
    - A concept file documenting the credibility-normalization dynamic (see [[domains/mena/concepts/em-central-bank-credibility-normalization]])
    - Entity stubs for the key political shield (finance minister) and the central bank governor
    - The correct procedure for rate decisions that accounts for EM-specific dynamics (see [[domains/economics/procedures/central-bank-rate-decision]])
16. **Assess leadership persistence AND withdrawal dynamics — with nomination-status gate**: If the question asks whether a leader will resign, withdraw from a race, or step down:

    - **FIRST — Classify the candidate type and apply the governing framework**: Before any persistence/withdrawal analysis, determine which structural framework governs this candidate:
    
      - **Framework A: Post-nomination / post-selection structural lock-in** — Applies when the candidate has secured the party's nomination (or formal selection) and is NOT the incumbent. Load [[concepts/post-nomination-persistence-baseline]] and [[procedures/candidate-withdrawal-probability]]. Default baseline: <1% withdrawal probability. This applies to non-incumbent US presumptive nominees and equivalent positions in comparable party-based systems.
      
      - **Framework B: Leadership persistence under threat** — Applies when the leader faces compounding legal jeopardy (pending charges, convictions, credible prosecution threat). This can compound with Framework A or override Framework C. Load [[concepts/leadership-persistence-under-threat]].
      
      - **Framework C: Internal pressure withdrawal cascade** — Applies when the leader faces NO legal jeopardy and pressure comes from within their own party/institution. This applies primarily to incumbents or pre-nomination candidates. Load [[concepts/incumbent-withdrawal-cascade]].
      
      - **Document the framework selection and justification explicitly.** The most common forecasting error is applying Framework C to a Framework A candidate — which overestimates withdrawal probability by an order of magnitude.
      
    - **Then, assess the PERSISTENCE frame** (leader stays in power):
      - **Nomination-status gate**: Has the candidate secured the nomination? If yes AND non-incumbent → the post-nomination baseline (<1% withdrawal) is the dominant variable. Skip directly to documentation — no further assessment needed barring total incapacitation.
      - Identify whether the leader faces compounding legal jeopardy (pending charges, convictions) — this creates existential motivation to stay (office = legal protection). Legal jeopardy is the strongest single predictor of persistence.
      - Verify whether the leader has already secured their party's nomination or formal endorsement — structural lock-in (delegates, ballot access, campaign infrastructure) makes withdrawal extremely difficult.
      - Check for recent assassination attempts or physical threats — these generally harden rather than deter a candidate's resolve.
      - Determine whether the pressure to withdraw is internal (party leaders, donors) or external (opponents, media, legal system). Internal pressure without legal jeopardy is the most plausible path to withdrawal.
      - Assess whether a viable successor exists who can absorb campaign infrastructure — the Biden→Harris transition (72-hour absorption in July 2024) establishes a benchmark.
      - Apply the [[leadership-persistence-under-threat]] concept framework: the combination of legal jeopardy + nomination lock-in produces near-deterministic persistence (<5% withdrawal probability).
    - **SIMULTANEOUSLY assess the WITHDRAWAL frame** (leader might step down) using the [[concepts/incumbent-withdrawal-cascade]] framework:
      - Check ALL 5 conditions for withdrawal probability:
        1. **Legal jeopardy absent** — no pending charges that would be enforced upon loss of office
        2. **Internal party pressure present** — calls from donors, elected officials, party elders
        3. **Trigger event occurred** — debate disaster, primary near-loss, health scare, corruption revelation
        4. **Viable successor exists** — someone who can absorb campaign infrastructure quickly
        5. **Electoral position weak** — trailing in polls, dragging down-ballot races
      - Calibrate: 0 conditions → <5% withdrawal; 3 conditions → 30-50%; 5 conditions → >70%
      - Historical precedent check: compare to Truman 1952, LBJ 1968, Biden 2024 — all incumbents with no legal jeopardy who withdrew after trigger events.
      - **PRE-TRIGGER SUB-STEP — Cumulative probability for aging incumbents**: If NO trigger event has occurred yet AND the leader is 70+ AND the question asks about withdrawal over a horizon of 3+ months, do NOT default to "no trigger = status quo." Instead:
      - Run the [[domains/usa/procedures/proc-aging-incumbent-early-warning]] procedure's 6-signal inventory (age concern, no legal jeopardy, party doubt, low approval, successor ready, party not restructured around leader).
      - Calculate cumulative trigger probability: P(any trigger in N months) = 1 - (1 - monthly_rate)^N. For 80+ leaders with 4+ YES signals, monthly_rate = 7-10%.
      - Multiply by cascade completion rate (~85%): P(withdrawal pre-trigger) = P(any trigger) × 0.85.
      - **The critical insight**: A "NO" forecast for an aging incumbent's withdrawal over a multi-month horizon is a bet that NO trigger event will materialize. For a leader with 4+ vulnerability signals over 10+ months, this is a roughly 40-60% event — not a low-probability tail. You must explicitly state the cumulative trigger probability in the reasoning.
      - **Counter-frame documentation**: If the leader is 70+ with 4+ signals and you still forecast NO, document EXACTLY why no trigger will occur (e.g., no debates scheduled, party has suppressed dissent, leader has survived previous scares). Otherwise the default is P(withdrawal) = P(trigger cumulative) × cascade rate.
    - **Key insight**: Legal jeopardy is the binary gate. If present, persistence is ~deterministic. If absent, withdrawal is possible if internal pressure and a trigger event converge.
    - **Historical baseline check — non-incumbent nominees**: If the candidate has secured the nomination and is NOT an incumbent, check the post-nomination persistence baseline: zero withdrawals by non-incumbent presumptive nominees since 1972 (12 cases). This baseline OVERRIDES the cascade framework's estimates. Even if 2-3 cascade conditions appear met, the structural lock-in makes withdrawal effectively impossible for this candidate type. The cascade framework applies only to incumbents.
    - **Track cascade velocity once trigger event occurs**: If a trigger event (poor debate, primary setback, health scare) has already occurred, track the cascade velocity to estimate time-to-withdrawal:
      - **Day 0-5 after trigger**: First individual defections (safe-seat members, retiring legislators). This wave is necessary but NOT sufficient for withdrawal. Apply the Stage 0 (denial) pattern — the leader will publicly deny any intention to withdraw.
      - **Day 5-14 after trigger**: If only individual members are defecting without institutional leadership engagement, the cascade is still containable. Track whether donor/surrogate defections occur (op-eds, public statements from major fundraisers) — these precede institutional leader defection by 3-7 days (see Biden 2024: [[domains/usa/entities/george-clooney|George Clooney]] op-ed at day 13, Pelosi/Jeffries at days 14-15).
      - **Day 7-21 after trigger**: If party leadership (caucus chairs, committee chairs, former leaders) begins privately or publicly signaling withdrawal, the cascade has passed the point of no return. Expect withdrawal within 7-14 days of institutional leader engagement.
      - **External pause events** (assassination attempt, foreign crisis, national tragedy) can briefly shift media attention but do NOT reset the cascade once institutional leadership has engaged. The Biden-Trump assassination attempt (day 16 of Biden's cascade) paused public attention but did not change trajectory.
      - **Velocity benchmarks by trigger type** (from [[concepts/incumbent-withdrawal-cascade]] Cascade Velocity Benchmarks section):
        - Primary loss: ~18 days trigger-to-withdrawal (Truman 1952)
        - Primary near-loss: ~19 days trigger-to-withdrawal (LBJ 1968)
        - Debate/performance failure: ~24 days trigger-to-withdrawal (Biden 2024)
      - **Calibrate time-to-withdrawal estimate** using the above benchmarks, adjusting for: latency-of-vulnerability (higher baseline approval = slower cascade), party cohesion (fractured party = faster cascade), and successor readiness (ready VP = structurally easier withdrawal).
    - **DEADLINE-CONSTRAINED WITHDRAWAL — explicit sub-step for "before [deadline]" questions**: If the question asks "will X withdraw BEFORE [deadline]?" (a convention, filing deadline, vote, nomination, or any fixed date), the forecast is a compound probability that depends on both the trigger timing AND the cascade completion time. This is structurally different from "will X withdraw?" without a deadline:
      - **Step 1: Identify the deadline and the cascade completion clock**: Record the deadline date. From the cascade velocity benchmarks (above), set the expected cascade completion time given the trigger type. For convention/binding votes, the relevant deadline is the last procedural moment the withdrawal changes the nomination outcome — typically the first day the convention gavels in (for conventions) or the ballot access filing deadline (for other processes).
      - **Step 2: Compute the effective trigger deadline**: The trigger must occur by `deadline - cascade_completion_upper_bound` (using the upper bound of the velocity range). For a convention 53 days away and 24-day cascade: effective trigger deadline = convention - 24 days. For a convention 30 days away: effective trigger deadline = convention - 24 days = only 6 days remain for a trigger — drastically lower probability.
      - **Step 3: Assess whether a trigger has already occurred**: If YES → the deadline constraint is not binding (cascade has been running and has 24 days; if remaining time > 24 days → constraint is irrelevant). If NO trigger has occurred → the deadline constraint reduces the effective window for trigger occurrence from the full forecast horizon to `deadline - cascade_time`. This can dramatically reduce P(withdrawal before deadline) vs P(withdrawal without deadline).
      - **Step 4: Compound probability calculation**: P(withdrawal before deadline) = P(any trigger by effective_deadline) × cascade_completion_rate (~85%). If the question's cutoff is before any trigger, the effective window is `cutoff_date to effective_deadline`. If already post-trigger, the window is `trigger_date to effective_deadline` — and since cascade takes ~24 days, if remaining time > 24 days the constraint is not binding: P(withdrawal before deadline) ≈ P(withdrawal).
      - **Document the deadline explicitly in the reasoning**: State the deadline date, the cascade completion benchmark used, the effective trigger deadline, and the resulting compound probability. Failure to document this timing dimension is the primary error mode for deadline-constrained withdrawal questions.
      - **Canonical example — Biden before DNC (August 19, 2024)**: The DNC deadline was Aug 19. The debate trigger occurred June 27 (day 0). The cascade completed July 21 (day 24) — 29 days before the DNC. At a post-debate cutoff, the deadline constraint was NOT binding (53 days remaining > 24-day cascade). At a pre-debate cutoff, the effective trigger deadline was ~July 26 (Aug 19 - 24 days). The question was: P(any trigger by July 26) × 85% cascade rate. With the 6-vulnerability signals at full YES and a 10-month horizon, the compound probability was ~45-55% — not a flat NO.
    - **CRITICAL: Overweight structural conditions, underweight stated intentions**: The leader's public statements denying any intention to withdraw are NOT a reliable signal. The Stage 0 denial pattern (see [[concepts/incumbent-withdrawal-cascade]]) shows that leaders who ultimately withdraw deny intention up to the moment of withdrawal — Truman (1952), LBJ (1968), and Biden (2024) all did this. Base your probability on the 5-condition framework and historical precedent, not on the leader's press statements or campaign assurances.
    - Document BOTH assessments explicitly in the reasoning. The most common forecasting error is assessing only one frame — seeing why a leader would stay (persistence factors) without also checking why they might leave (withdrawal factors).
16. **Audit US budget shutdown dynamics**: If the question involves a US government shutdown, funding deadline, or budget crisis:
   - Check whether there is an active [[us-budget-shutdown-dynamics]] thread. If not, the vault has a structural gap — create the thread before forecasting.
   - Identify the current funding status: is the government operating on a CR? When does it expire?
   - Map the congressional dynamics: is the House majority narrow? Is an HFC-style hardline faction active? Is the Speaker relying on Democratic votes to pass funding?
   - Check for external disruptors: Is there an unelected actor (mega-donor, president-elect, social media influencer) with leverage over one party who could kill a bipartisan deal?
   - Assess transition/lame duck status: Is this a period between an election and the new government taking office? If yes, shutdown risk is structurally higher.
   - Apply the [[budget-brinkmanship-hostage-dynamics]] concept framework: estimate the pain tolerance ratio, identify the disruption vulnerability, and calibrate the probability accordingly.
   - For questions about a future shutdown: the baseline probability of a funding lapse in any given Congress with a narrow House majority is 15-25%. Add 25-35% if an external disruptor is present. Multiply by 1.5-2x during lame duck periods.
   - **Check for Speaker crisis / succession dynamics**: Has the Speaker been removed, resigned, or faced a credible removal threat within the last 90 days?
     - If a Speaker was recently removed or resigned (within 30 days): apply the [[speaker-crisis-paradox]] framework. Short-term (first 30-60 days) shutdown risk DECREASES by 15-25% from baseline due to the prove-competence incentive. The new Speaker has maximum incentive to avoid a shutdown on their first funding test.
     - If there is an ongoing Speaker vacancy (no Speaker elected for 14+ days): this is a PARALYSIS scenario — no legislative business can pass. If a funding deadline falls during the vacancy, shutdown probability approaches 100% regardless of other factors.
     - If the Speaker faces a credible removal threat but has not been removed: baseline risk applies with a small upward adjustment (5-10%) for uncertainty. The threat alone does not trigger the paradox — the prove-competence effect activates only after succession has occurred.
     - If the Speaker is an incumbent with no active threat: use baseline dynamics only. The paradox does not apply.
   - **Always document**: (1) Whether a Speaker succession has occurred, (2) how many days since succession if applicable, (3) whether the next funding deadline falls within the prove-competence window (30-60 days after succession), and (4) whether this is a regular session or transition/lame duck period. The most common error is assuming surface-level dysfunction (Speaker removal) increases near-term shutdown risk when the paradox shows the opposite.

16b. **Audit US debt ceiling dynamics**: If the question involves the US debt ceiling — whether it will be raised, suspended, reinstated, or default reached by a specific date:

   - **FIRST — Distinguish debt ceiling from shutdown questions**: Debt ceiling questions involve the statutory borrowing limit and the risk of sovereign default. Shutdown questions involve the appropriations process and funding lapses. They have DIFFERENT mechanics (extraordinary measures vs. immediate funding lapse), DIFFERENT resolution pathways (reconciliation vs. appropriations bills), and DIFFERENT timelines (months of buffer vs. immediate). Do NOT conflate them. If the question mentions both or is ambiguous, resolve each independently.

   - **Establish the current debt ceiling state**: Is the ceiling suspended (date-based sunset), binding (dollar limit hit, extraordinary measures running), or at reinstatement (just resumed after suspension)? This determines the baseline timeline.

     - If suspended: Check the reinstatement date. No action needed until that date — the ceiling is not binding.
     - If binding: Identify the binding date. Extraordinary measures began then.
     - If at reinstatement: Extraordinary measures are running from reinstatement date.

   - **Estimate the extraordinary measures runway**: When did extraordinary measures begin? What is the current X-date estimate? (Check Bipartisan Policy Center, CBO, or Treasury letter — these are the canonical references.) The runway is typically 4-6 months from binding/reinstatement. The first 30-60 days have near-zero probability of resolution because there is no crisis urgency.

   - **Identify available legislative vehicles**: Is there a must-pass bill (CR, reconciliation, budget) that debt ceiling can be attached to? If reconciliation is active and will include debt ceiling language, the reconciliation timeline is the resolution timeline. If no vehicle exists, probability of a standalone debt ceiling bill is low — these are always politically costly to pass as standalone.

   - **Apply the six-factor (VWUAPE) model** from [[domains/usa/concepts/debt-ceiling-mechanics/_concept]]: V (Vehicle), W (Window length), U (Urgency to X-date), A (Alignment/political control), P (Political cost of vote), E (Economic pressure). If 4+ factors point to NO, P(YES) < 10%. If 4+ factors point to YES, P(YES) > 90%.

   - **Check for the "no urgency" trap**: Extraordinary measures mean Congress does NOT need to act in the short term. The most common error is overestimating YES probability for short windows because the ceiling is \"binding\" without checking how much runway remains.

   - **Load the procedure**: Apply [[domains/usa/procedures/debt-ceiling-forecast]] and [[domains/economics/threads/us-debt-ceiling-crises/_thread]] before forecasting. Document the six-factor assessment explicitly in the reasoning.

   - **Create entity stubs for key actors**: Treasury Department ([[domains/usa/entities/us-department-of-treasury]]), Treasury Secretary ([[domains/usa/entities/janet-yellen]] for the 2021-2025 period, [[domains/usa/entities/scott-bessent]] for 2025+). These stubs MUST exist before forecasting — they document the operational actor and communicator of X-date estimates.

   This step exists because the debt ceiling question (question 45, correct NO prediction) was correctly predicted but the vault contributed zero structured signal — no dedicated debt ceiling thread, no concept for extraordinary measures/X-date mechanics, no entity stubs for Treasury or the Treasury Secretary, and no forecast procedure. The correct NO prediction relied on reasoning about the impossibly short 18-day window and the failed CR vehicle, but the vault had no framework for systematically evaluating these variables. Every future debt ceiling question must find a vault with structural analytical depth parallel to what exists for shutdowns.

17. **Assess legal timeline dynamics**: If the question asks whether a legal proceeding (trial, verdict, sentencing, ruling) will reach a milestone before a political deadline (election, inauguration, end of term):
   - **Map the defendant's delay incentive**: Does the defendant benefit from pushing the proceeding past the deadline? If the defendant is a political candidate facing charges, the default assumption is a strong incentive to delay (victory can moot prosecution). Document the specific benefits of delay.
   - **Identify available delay mechanisms**: What procedural avenues can the defendant use?
     - Interlocutory appeals (especially on immunity, qualified immunity, or jurisdictional questions)
     - Motions to dismiss (raise novel constitutional or procedural questions)
     - Discovery disputes (document requests, privilege claims)
     - Recusal motions (challenge the judge's impartiality)
     - Venue change motions
     - Continuance requests
   - **Check for automatic stays**: Does an appeal automatically stay district court proceedings? If yes, the appellate timeline becomes the controlling variable.
   - **Calculate the appellate timeline**: For cases reaching the Supreme Court:
     - Cert grant to oral argument: ~2-3 months
     - Oral argument to decision: ~2-3 months
     - Total SCOTUS timeline: ~4-8 months from cert grant to mandate
     - For Circuit Courts: 6-12 months average depending on complexity
   - **Assess remand time**: Even after an appellate ruling, the district court needs time (2-4 months) to apply the new legal framework. 
   - **Add the delay budget**: Sum the minimum realistic duration of each procedural step. If the sum exceeds the time available before the political deadline → the milestone WILL NOT be reached.
   - **Assess electoral mooting probability via OLC doctrine**: If the defendant could win a federal election and thus become president (or is already president), DOJ Office of Legal Counsel policy prohibits federal prosecution of a sitting president. This moots ALL federal cases upon the defendant assuming office, regardless of their pre-election legal posture. Check the [[entities/doj-office-of-legal-counsel]] entity for the two controlling OLC opinions (1973, 2000). This creates asymmetric incentives — the defendant's legal strategy is optimized for delay, not trial preparation. For state cases, OLC doctrine does NOT apply — state prosecutions continue independently.
   - **Distinguish state vs. federal**: State-level prosecutions are structurally more likely to proceed because:
     - No federal immunity doctrines apply
     - State appellate timelines are typically faster
     - The defendant cannot use SCOTUS's docket as a delay mechanism
     - State judges may have different scheduling incentives
   - **Map the prosecutor's timing constraints (SYMMETRICAL to defense-delay analysis)**: Independent of defense delay, the prosecutor faces institutional constraints that limit pre-election trial feasibility:
     - **DOJ 60-day rule**: The DOJ's informal policy against overt steps within ~60 days of a federal election creates a hard institutional deadline. Even if the court schedules a trial, the DOJ will resist taking visible action after roughly early September of an election year.
     - **AG posture**: An independent AG (like Garland) enforces the 60-day rule strictly. An AG appointed by the defendant's party may apply it more loosely. Assess the AG's institutional independence.
     - **Comey effect**: Post-2016, DOJ leaders are hyper-cautious about any election-year action that could be perceived as influencing an election. This makes the 60-day rule binding in practice for high-profile cases, even if technically waivable.
     - **State exception**: State prosecutors are NOT bound by DOJ policy. State cases can proceed into the pre-election window. Always check whether the charges are state or federal for this dimension.
     - **Load [[concepts/prosecutorial-election-year-timing]]** for the full framework. The combined effect: the defense delay consumes the pre-September window, and the 60-day rule blocks the September-November window. Even if the defense's delay fails, the prosecution's constraint independently prevents trial.
   - **Apply the [[concepts/judicial-timing-political-deadline]] framework**: calibrate probability based on:
     - Novelty of the constitutional question (novel = slow, more delay)
     - Availability of automatic stays (present = trial paused during appeal)
     - Judge's scheduling practice (some judges expedite, others routinely grant continuances)
     - Defendant's electoral viability (plausible path to winning = stronger delay incentive)
   - **Document the timing assessment explicitly** in the reasoning, including each procedural step and its estimated duration. The most common error is underestimating the cumulative delay of successive procedural steps — each step consumes 2-4 months, and 3-4 steps can consume a full year.
   - **Run the structured estimation function**: Load and apply the [[functions/estimate-legal-timeline]] function with the gathered parameters to produce a calibrated probability estimate. Cross-reference the output with the [[procedures/proc-legal-timeline-estimation]] procedure for the full analytical workflow. The function provides a reproducible calculation; the procedure provides the step-by-step methodology.

17b. **Assess SCOTUS intervention feasibility in state court proceedings**: If the question asks whether the US Supreme Court will block, stay, or review a STATE court proceeding (sentencing, trial, subpoena, contempt order, gag order):

   - **FIRST — Classify the proceeding as state vs. federal**: This is a binary gate. If the underlying proceeding is in state court (even if federal questions are raised), SCOTUS's ability to intervene is fundamentally constrained by the adequate and independent state grounds doctrine. Federal proceedings have no such barrier. Misclassifying a state proceeding as "SCOTUS-reviewable" is the single most common error for this question type.

   - **Apply the adequate and independent state grounds barrier**: Load [[concepts/adequate-independent-state-grounds]] before forecasting. The key question: does the state court judgment rest on state law that is independent of any federal question? If the underlying charge is a purely state-law crime (e.g., NY falsifying business records), the state-grounds barrier is strong. Even if a federal question is raised about evidence admissibility or constitutional immunity, the state court may have an independent state-law basis for its ruling that SCOTUS cannot review.

   - **Identify the federal question**: What specific federal constitutional or statutory claim is the applicant raising? Assess:
     - Is this a pure federal question (First Amendment, due process, equal protection)? → Weaker barrier, 15-25% chance SCOTUS intervenes.
     - Is this a claim that federal law preempts the state proceeding? → Moderate barrier, 5-15% chance.
     - Is this a claim that a federal immunity doctrine applies to a state proceeding? → Strong barrier (<5% chance) — as with Trump's attempt to apply the *Trump v. US* immunity ruling to block NY state sentencing.
     - Is this a fact-bound evidentiary claim with a federal gloss? → Near-zero barrier penetration.

   - **Assess the emergency stay standard**: If the question is about SCOTUS blocking an imminent action (sentencing, execution, trial start), apply the four-factor Nken test:
     1. **Likelihood of success on merits**: Is the federal question strong enough that SCOTUS would likely reverse? If no → stay denied.
     2. **Irreparable harm**: Would the applicant suffer harm that cannot be compensated? For a routine state sentencing, the harm is speculative. For a state execution, the harm is irreparable (death).
     3. **Balance of equities**: Does the state's interest in proceeding outweigh the applicant's interest in delay? State interest in finality nearly always prevails for routine proceedings.
     4. **Public interest**: Would a stay serve or harm the public interest? Federalism and state sovereignty weigh against SCOTUS intervention in state proceedings.

   - **Check the timing of the application**: 
     - If the application is filed <48 hours before the action the applicant seeks to block → P(SCOTUS grants stay) < 1%. The Court will not be rushed on a weak question.
     - If filed 3-14 days before → P(SCOTUS grants stay) = 5-15% depending on the strength of the federal question.
     - If filed >14 days before → the applicant has time for a more deliberate procedural path (cert petition, not emergency stay).

   - **Check the vote composition**: If SCOTUS has already ruled on a similar application (e.g., a stay denial vote of 5-4), analyze the split:
     - Did the conservative justices split? (Roberts and Barrett crossing the conservative bloc is a significant signal — it shows the institutionalist center rejecting an aggressive claim of presidential immunity from state process.)
     - Unanimous procedural votes are the strongest signal; 5-4 splits indicate the legal question is genuinely contested.
     - The specific vote composition from a PRIOR similar application is the highest-confidence signal for a SUBSEQUENT similar application.

   - **Calibrate using the [[concepts/scotus-procedural-signals]] framework**: Apply the "Emergency Stay Applications in Ongoing State Proceedings" section's calibration table:

     | Condition | P(SCOTUS grants emergency stay) |
     |-----------|--------------------------------|
     | Federal proceeding, clear federal question | 20-30% |
     | State proceeding, weak federal question | <5% |
     | State proceeding, strong federal constitutional claim | 15-25% |
     | President-elect seeking stay of state sentencing | <3% |
     | Application filed <48 hours before the event | <1% |

   - **Document the SCOTUS intervention assessment explicitly** in the reasoning, including: (1) whether the proceeding is state or federal, (2) the federal question and its strength, (3) the adequate state grounds assessment, (4) the four-factor Nken test application, (5) the filing timing, and (6) the calibrated probability. The most common error is treating "SCOTUS asked to intervene" as a signal that the question is close, when in reality SCOTUS intervention in state proceedings is structurally improbable.

18. **Assess sentencing feasibility for post-conviction scenarios**: If the question asks whether a convicted political figure will receive a specific sentence (especially incarceration):

   - **Identify the defendant's status at expected sentencing time**: Is the defendant a sitting president, president-elect, or likely to become one before sentencing? If yes → incarceration probability drops below 10%. Is the defendant a former officeholder with no prospect of returning? → Standard sentencing factors apply.
   
   - **Map the timing relative to political deadlines**: Has sentencing been delayed past an election? If the defendant won → practical obstacles dominate. Pre-election sentencing → standard factors plus political calculation.
   
   - **Assess the prosecution's posture**: Has the prosecution changed its sentencing recommendation since a status shift (election victory, inauguration)? If the prosecution concedes that incarceration is not "practicable," the judge nearly always follows. The prosecution's sentencing posture is the single most actionable indicator.
   
   - **Check the offense severity and felony class**: For low-level felonies (NY Class E: max 4 years; standard range 0-16 months for first-time non-violent offenders), incarceration is already unlikely for a first offender. For higher classes, incarceration is more plausible but still constrained for officeholders. Apply the [[domains/global/concepts/forecast-range-plausibility-filter]] double-filter framework: the specified range must be structurally plausible for the conviction class (Filter B) independent of whether any prison sentence occurs (Filter A).
   
   - **Identify practical obstacles to incarceration**: Secret Service protection requirements (no detention facility can accommodate a protective detail), constitutional questions about presidential capacity, the logistical challenges of incarcerating someone who must perform official duties, and the institutional conflict between the judiciary and the executive.
   
   - **Assess the judge's demonstrated tendencies**: Has the judge imposed harsh sentences in comparable non-political cases? Has the judge shown procedural flexibility toward this defendant? Count the number of sentencing delays — apply the [[domains/global/concepts/sentencing-delay-cascade]] framework: each delay after the first is a Bayesian update toward leniency. Judges facing unprecedented sentencing questions default to the least novel option (unconditional discharge or symbolic fine).
   
   - **Apply the [[concepts/presidential-sentencing-dynamics]] framework**: Calibrate based on:
     - President-elect or sitting president, post-election: P(incarceration) < 5%. P(unconditional discharge or symbolic sentence) > 80%.
     - Former officeholder with no return prospect: Standard factors dominate.
     - The key insight is that the sentencing phase becomes a distinct dynamic from the trial phase once the defendant's status shifts — the prosecution loses leverage, the judge's options narrow, and practical obstacles dominate.
   
   - **Document the sentencing assessment explicitly** in the reasoning, including the defendant's status, the timing relative to political deadlines, the prosecution's posture, and the practical obstacles. The most common error is treating "sentencing" as a pure legal question without accounting for the structural constraints of officeholder status.

18. **Distinguish "winning the vote" from "assuming office" in authoritarian-election questions**: If the question asks whether a candidate "wins" an election in an authoritarian or semi-authoritarian context:

   - **Identify the resolution criteria**: Polymarket and similar prediction markets resolve election questions based on who is declared the winner by the relevant official body OR, if official results are disputed, based on widely accepted evidence of the actual vote outcome. The key question is: does the market follow the official result or the actual vote? This varies by market and must be determined from the specific resolution text.
   
   - **Check the resolution text for ambiguity**: If the resolution text says "wins the election" without specifying "takes office" or "is inaugurated," the default interpretation is that winning the vote (receiving the most votes) constitutes winning the election. The market does NOT require the winner to assume office for resolution — the Venezuela 2024 market is the canonical example where González "won" the election (got the most votes) despite Maduro remaining in power.
   
   - **Assess opposition vote-monitoring infrastructure** (applying [[domains/latin-america/concepts/authoritarian-electoral-facade/_concept]]):
     - Does the opposition have a parallel vote tabulation (PVT) capability?
     - Can it collect tally sheets from a majority of polling centers?
     - Has the opposition credibly demonstrated this capability in prior elections?
     - If yes: the opposition can document the true outcome regardless of official results.
     - If no: the regime can fabricate results without credible contradiction.
   
   - **Check for candidate disqualification** (applying [[domains/latin-america/concepts/late-candidate-substitution/_concept]]):
     - Has the regime barred the most popular opposition candidate?
     - If yes: apply the late-candidate-substitution framework to assess whether the replacement can inherit the barred figure's support.
     - Key success factors: Can the barred figure still campaign for the replacement? Is the opposition coalition unified? Is there sufficient time before election day?
     - If the regime has NOT barred anyone but still controls the electoral apparatus, the authoritarian-electoral-facade framework applies without the substitution complication — the opposition's challenge is to get the regime to accept the outcome, not to find a candidate.
   
   - **Calibrate the two dimensions separately**:
     - P(candidate wins the vote) — based on polling, turnout enthusiasm, opposition unity, regime manipulation tools
     - P(candidate assumes office after winning) — based on military loyalty (see below), international pressure, regime fallback options
     - DO NOT conflate these. The Venezuela 2024 error was assuming that because González wouldn't take office, he couldn't "win."

   - **Assess military loyalty systematically** using the six-factor model from [[domains/latin-america/concepts/authoritarian-electoral-facade/_concept]] (Military Loyalty Mechanisms section):
     1. **Economic co-optation**: Do security forces control state enterprises or corruption networks that would collapse if the regime falls? (Yes = stronger loyalty)
     2. **Loyalty-based promotion**: Has the regime purged non-loyalist officers and promoted based on regime loyalty? (Yes = stronger loyalty)
     3. **Shared criminal liability**: Is the officer corps complicit in human rights abuses, repression, or fraud that creates prosecution risk after transition? (Yes = stronger loyalty)
     4. **Ideological indoctrination**: Is military education aligned with the regime's political ideology? (Yes = stronger loyalty)
     5. **Factional management**: Does the regime actively rotate commanders and manage internal military rivalries to prevent a unified defection bloc? (Yes = stronger loyalty)
     6. **Exit-blocking isolation**: Have international sanctions or arrest warrants targeted the officer corps, removing the "exit with impunity" option? (Yes = stronger loyalty)
     - Key entity to check: [[domains/latin-america/entities/vladimir-padrino-lopez]] — the institutional face of military loyalty in Venezuela; his public posture (visible/affirmative vs. silent/absent) is a leading indicator.
     - If 4+/6 mechanisms are active: P(assumes office after winning) < 10% — the regime can absorb an electoral defeat without losing power. If only 1-2 mechanisms are active and the military is divided: P(assumes office) can reach 30-50%.
   
   - **Document both assessments explicitly** in the reasoning. The most common error is treating "wins the election" as a single binary outcome when it has two separable dimensions: vote outcome and power transition.

19. **Assess central bank rate decision dynamics**: If the question asks whether a central bank (especially the Federal Reserve) will change interest rates at a specific meeting or within a specific timeframe:

   - **FIRST — identify the monetary policy cycle phase** (load [[domains/economics/concepts/monetary-policy-cycle-phases]]):
     - Determine whether the Fed is in a tightening cycle, early plateau, late plateau, easing cycle, or extended hold
     - The phase provides the **default next move** — the action most likely absent a major data surprise. A late-plateau Fed (6+ months at peak with declining inflation) has a near-zero probability of hiking — the structural phase constrains what is possible before forward guidance or data enter the analysis.
     - If the question asks about a hike and the phase is late plateau or active easing, the answer is structurally NO regardless of forward guidance signals — the Fed would need to establish an entirely new tightening cycle, which takes 2-3 meetings of new guidance.
     - Document the phase identification explicitly in the reasoning. This is the structural baseline that forward guidance then refines.
   
   - **Then map the decision to a specific meeting**: Central banks operate on published schedules. Identify the exact FOMC meeting date (or dates) that fall within the question's timeframe. Questions asking "Will the Fed cut rates before [date]?" span multiple meetings — each must be assessed independently.

   - **Check the most recent statement language**: Did the previous statement signal the direction of the next move? Key phrases:
     - "Further" (implies more moves in same direction)
     - "Patient" / "Data-dependent" (implies a hold)
     - "Gaining confidence" / "Time has come" (implies a pivot is near)
     - "Not yet confident" (rules out the immediate next meeting)

   - **Check the most recent dot plot (if applicable)**: For quarterly SEP meetings (March, June, September, December), the dot plot shows the median FOMC member's rate path projection. If the median shows no change at the upcoming meeting, a change will NOT happen.

   - **Check the Chair's most recent press conference**: Did the Chair explicitly or implicitly rule out the next meeting? The Chair's press conference is the most detailed forward-guidance signal. A statement like "we need to see more progress on inflation before reducing rates" rules out the immediate next meeting.

   - **Check market-implied probabilities**: Use CME FedWatch or equivalent. If the market-implied probability of a rate change is below ~40% one week before the meeting, the change will NOT happen. If above 80%, it almost certainly will (direction, not magnitude).

   - **Distinguish direction from magnitude**: The Fed telegraphs whether a move will happen (direction) but may not telegraph how large it will be (magnitude). The September 2024 50bp cut (vs 25bp expected) is the canonical example. For magnitude questions, the base rate is the previous move size; surprise moves tend to be larger (first cut of a new cycle) or smaller (when approaching neutral).
   
     **Special case: magnitude-specific questions.** When a question asks about a specific cut/hike size (e.g., "Fed decreases by 25bps") rather than just direction ("Fed cuts rates"):
     - Check whether the meeting is the first move of a new cycle. If yes, the actual magnitude may be larger than the standard 25bp increment — meaning a 25bp-specific question may resolve NO even though a cut happens. The September 2024 cut is the canonical example: the Fed cut 50bp, so a question about "25bp cut after July 2024" resolved NO.
     - If the meeting IS the first move of a new cycle and the question specifies 25bp (the standard increment), assign lower probability than market-implied probability for "a cut" because the Fed may use a larger increment to recalibrate.
     - For subsequent moves in an established cycle, 25bp is the norm and magnitude-specific questions about 25bp are more likely to match actual outcomes.
     - Calibrate using CME FedWatch's magnitude distribution (probability of "no change" vs "25bp" vs "50bp" vs "75bp") — if the specified magnitude is not the dominant probability node (>50%), the question is less likely to resolve than a simple direction question.
     
     This distinction is critical because the same meeting can generate two questions — one about increases and one about decreases — both resolving NO for different reasons (no hike expected vs. cut happened but at wrong size). The [[domains/economics/concepts/central-bank-forward-guidance]] concept provides the full framework at step 8.

   - **Identify potential dissents**: Check whether any FOMC member has publicly dissented from the consensus direction. A governor dissenting (like Michelle Bowman in September 2024) is more significant than a regional bank president dissenting, because governors are Board members appointed by the President and less prone to signaling disagreement.

   - **Apply the [[concepts/central-bank-forward-guidance]] framework**: The Fed's structured communication pipeline makes rate decisions unusually forecastable. If the forward guidance signals are pointing toward a move, it will likely happen. If they are silent or pointing away, it will not.

   - **Document the rate decision assessment explicitly** in the reasoning, including the most recent FOMC statement language, dot plot, market pricing, and Chair guidance. The most common error is treating a rate decision as a guessing game when the Fed has already telegraphed the outcome.

20. **Assess SCOTUS procedural signals**: If the question asks whether the US Supreme Court will block, delay, or uphold a government action (especially one with a statutory deadline):

   - **Identify the procedural posture**: Is the case before the Court on:
     - Certiorari (standard review) — indicates normal pace, no unusual urgency
     - Cert before judgment — extremely rare (~5 cases/term). The Court believes the question is so urgent and clear-cut that it cannot wait for the circuit court. **Strong signal** that the Court intends to resolve on the merits before the deadline.
     - Emergency application for a stay — the challenger is asking the Court to halt enforcement while review proceeds. The Court's response (grant, deny, or refer to full Court) is the single most informative signal.
     - Standard appeal from a circuit ruling — most common posture; signals routine review.

   - **Map the timeline relative to the statutory or constitutional deadline**:
     - How many days/weeks remain until the deadline?
     - How much time does SCOTUS need to issue a ruling? (Compressed: 3-6 weeks from cert. Standard: 4-8 months.)
     - If the deadline is <3 months away, the Court's procedural choices — not the legal merits — will determine the outcome.

   - **Track the Court's procedural choices as they become known**:
     - **Cert before judgment granted**: P(stay granted) < 10%. The Court is expediting merits review, not delaying enforcement.
     - **Stay denied while merits expedited**: P(uphold) > 95%. The strongest possible signal — the Court is saying the law stands while it quickly decides.
     - **Stay granted**: P(reversal or remand) 40-60%. The Court sees something worth preserving in the status quo.
     - **Standard schedule with no stay**: Ambiguous — proceed to legal merits analysis.
     - **Compressed briefing/argument schedule**: Signals the Court believes the legal question is clear. P(decided before deadline) > 90%.
     - **Post-argument ruling within 14 days**: Votes were locked before arguments. Unanimous or near-unanimous decision likely.

   - **Distinguish "will Court delay?" from "will the ban persist?"**: These are fundamentally different questions governed by different frameworks:
     - **Judicial delay** (Q22: "Will Supreme Court delay the TikTok ban?"): Does the Court issue a stay, injunction, or take too long to rule, pushing enforcement past the deadline? Apply the SCOTUS procedural signals framework above. The answer depends on the Court's procedural trajectory (cert before judgment? compressed schedule? stand-alone stay?).
     - **Executive delay** (Q10: "Will the ban persist after it takes effect?"): Will the executive branch (president, agency) decline to enforce the law after it legally takes effect? Apply the [[domains/global/concepts/executive-enforcement-delay/_concept]] framework. The answer depends on political conditions (administration change, consumer backlash, enforcement capacity).
     - **Two-delay sequence**: The TikTok case proves these can operate independently in sequence on the same legal timeline. SCOTUS did NOT delay (judicial: correct NO prediction on Q22), but Trump DID delay enforcement after the fact (executive: ban still resolved YES because legal effect was satisfied). A "no" on judicial delay does NOT imply a "no" on executive delay — they are independent events.
     - **Decision rule for ambiguous questions**: If the question asks "will X be delayed?" without specifying the actor, check: (a) resolution text for actor identification, (b) position on timeline relative to effective date, (c) whether both types could apply in sequence. Document which type(s) are relevant.

   - **Apply the trajectory principle**: A case's procedural trajectory is path-dependent. Once the Court commits to a fast track (cert before judgment + compressed schedule + no stay), the probability of a stay or delay drops to near zero. The trajectory is itself the outcome signal.

   - **Contrast with the defendant-delay pattern** ([[concepts/judicial-timing-political-deadline]]):
     - Court-accelerated review (this step) = the Court compresses timelines. Signals the Court wants to resolve before a deadline.
     - Defendant-driven delay = a litigant uses procedural mechanisms. Signals the defendant fears an unfavorable ruling.
     - If a defendant is seeking delay AND the Court is accelerating review, the defendant's strategy is likely to fail.

   - **Check the law's bipartisan support**: Laws passed with supermajority support (especially on national security matters) are less likely to be overturned or delayed by the Court. Unanimous or near-unanimous procedural votes from the justices similarly predict lopsided substantive rulings.

   - **Apply the [[concepts/scotus-procedural-signals]] framework**: Calibrate probability based on the specific combination of procedural signals observed. The calibrated table in that concept provides quantitative estimates for each signal combination.

   - **Document the procedural signal assessment explicitly** in the reasoning, including the specific procedural posture, timeline, and signal interpretation. The most common error is forecasting based on legal merits analysis while ignoring the more informative procedural signals.

21. **Assess VP selection (veepstakes) dynamics**: If the question asks about who will be selected as a Vice Presidential nominee, or involves a VP selection as a subcomponent of a broader electoral question:

   - **Map the finalist pool**: Identify the 3-6 candidates who undergo formal vetting. Track public reports of vetting — these are typically intentional signals from the campaign. Sources: major campaign reporters, anonymous campaign official leaks, and the nominee's public statements about what they're looking for.

   - **Identify the elimination cascade**: VP selections follow a predictable pattern:
     1. Broader list (10-15) → filtered by obvious disqualifiers (background issues, state law conflicts, willingness to serve)
     2. Short list (3-6) → filtered by vetting depth, chemistry with nominee
     3. Finalist (2-3) → filtered by the nominee's core priority
     4. The Pick (1)
     
     Track which candidates are eliminated and why — "vetting problems" that surface publicly are often decisive and eliminate candidates with 70-80% probability.

   - **Categorize the selection model**: VP picks follow one of four models:
     - **Balancing model**: compensates for nominee's weakness (geographic, demographic, ideological) — traditional, moderately predictable
     - **Reinforcement model**: amplifies nominee's strength — modern, harder to predict (more candidates fit the frame)
     - **Governing model**: signals readiness to assume presidency — experienced elected official
     - **Campaign model**: optimized for message/attack — less experienced, often media-savvy
     
     The 2024 paired case (Harris→Walz, Trump→Vance) shows both parties adopting the reinforcement model, a potential structural shift from the historical balancing model.

   - **Assess media narrative reliability**: Media consensus 7+ days before the pick is <30% predictive of the actual selection. VP searches are intentionally opaque; media frontrunners are often decoys or the result of name recognition bias. The actual pick can emerge from outside the media-hyped frontrunner group 2-3 days before announcement.

   - **Identify the strategic rationale**: What electoral problem is the pick solving?
     - Swing-state vulnerability → expect pick from that state
     - Demographic weakness → expect representative of that group
     - Enthusiasm gap → expect ideological firebrand
     - Credibility deficit → expect experienced statesperson

   - **Apply the [[concepts/veepstakes-electoral-signal]] framework**: The concept provides calibrated forecasting rules for each stage of the VP selection process, including probability estimates for vetting-stage signals and selection-model probabilities.

   - **Exclusion-list filter**: If the question uses a "will another [category] be VP?" format with an exclusion list of named candidates, refer to step 25 (comprehensive exclusion list assessment) before beginning the veepstakes analysis. If the exclusion list is exhaustive, the question resolves independently of the VP selection dynamics.

   - Document the veepstakes assessment explicitly in the reasoning, including the finalist pool, elimination cascade, selection model, and strategic rationale. The most common error is treating media speculation as a reliable signal when VP searches are designed for opacity.

22. **Thread continuity enforcement**: After writing each quarter file, the agent MUST explicitly verify that every thread with `status: active` has been updated with new developments from the current quarter OR has a documented rationale for no update. This verification is a blocking step — do not proceed to the next quarter or to forecasting until all active threads are current. The most common vault failure mode is creating thread infrastructure and then failing to maintain it as quarters progress. A thread that has not been updated in 2+ consecutive quarters should have its status changed to `fading` with a note explaining why.

23. **Quarterly Fed decision audit**: Each contemporary quarter file (post-2020) MUST include a subsection recording every FOMC meeting that occurred during that quarter, with the rate decision, vote tally, any dissents, and the Chair's forward guidance signal. This is not optional — it is mandated by spec principle #14. The template for this section is:

    ```
    ### Federal Reserve Rate Decisions
    - **[Date]**: [Decision] (X-Y vote). [Key details: dissenters, statement language, dot plot changes, market reaction]. [[federal-reserve-system]] [[jerome-powell]] [[central-bank-forward-guidance]] [[us-monetary-policy-cycle-2022-2026]]
    ```
    
    This standardization ensures consistency across quarters and enables automated parsing of the Fed decision timeline.

   - **Determine the nominee's gender**: This is the primary variable. If the presidential nominee is a woman, the probability of a female VP pick drops structurally (see [[concepts/gender-balancing-ticket-composition]]).
   
   - **Check for explicit gender pledges**: If the nominee (regardless of gender) has pledged to select a running mate of a specific gender, assess the credibility of the pledge:
     - Male nominee who pledged to pick a woman → P(compliance) > 90% (Biden 2020 established this precedent)
     - Female nominee who pledged to pick a woman → P(compliance) 30-50%; strategist pushback is likely to weaken the pledge
     - No pledge → default to structural baseline probabilities
   
   - **Map the available candidate pool by gender**: 
     - How many women with national profiles, elected office experience, swing-state value, and vetting clearance exist in the party?
     - If the question includes an exclusion list (e.g., "a woman other than [list of 9 women]"), note that the exclusion list likely captures the entire viable female pipeline — the probability of a woman OUTSIDE the list being selected is structurally lower than the probability of a woman ON the list.
     - **Use step 25 (comprehensive exclusion list assessment)**: If the question uses an exclusion-list format, the four-part diagnostic should be applied before the gender dynamics assessment. If the list is exhaustive of all viable candidates of the relevant gender, the question may resolve NO independently of the gender balancing dynamic.
   
   - **Apply the gender balancing framework** (from [[concepts/gender-balancing-ticket-composition]]):
     - Male nominee → baseline P(female VP) = 15-35% (rises to 90%+ with pledge)
     - Female nominee → baseline P(female VP) = 5-12% (rises to 30-50% with pledge)
     - Female nominee + shallow female candidate pool → P(female VP) < 5%
   
   - **Check the ticket composition pattern**: Does a woman+man ticket look "balanced" in a way a woman+woman ticket would not? This is the central strategic calculus for a female nominee — the historical pattern (Clinton 2016, Harris 2024) confirms female nominees choose male VPs.
   
   - **Consider the cascade effect**: If the nominee is not yet determined (e.g., primary still ongoing), the VP question's answer depends on which gender the nominee is. Map both scenarios separately: P(woman VP | man nominee) * P(man nominee) + P(woman VP | woman nominee) * P(woman nominee).
   
   - **Document the gender dynamics assessment explicitly** in the reasoning, including the nominee's gender, any pledges, the available candidate pool, and the structural baseline probability. The most common error is treating a "woman VP" question as a generic speculation question without accounting for the nominee's gender as the dominant variable.

22. **Audit financial regulation and SEC product approval dynamics**: If the question asks whether a novel financial product (crypto ETF, new asset class, new security type) will be approved or begin trading within a specific timeframe:

   - **Map the legal/regulatory precedent chain**: Has a court recently ruled against the SEC on a similar product? The Grayscale v. SEC DC Circuit ruling (August 29, 2023) forced Bitcoin ETF approval (January 10, 2024), which forced Ethereum ETF approval (May-July 2024). Each prior approval increases the probability of subsequent similar approvals — this is the [[regulatory-precedent-cascade]] dynamic.

   - **Identify the statutory deadline forcing the decision**: Under SEC rules (Section 19(b) of the Securities Exchange Act), the SEC has 240 days to approve or deny an ETF application. When multiple applications are pending, the applicant with the EARLIEST statutory deadline determines the approval date. For Bitcoin ETFs, ARK 21Shares' application had the earliest deadline (January 10, 2024). Because the DC Circuit ruling gave the SEC no good-faith basis to deny, this deadline became the mandatory approval date. ALWAYS identify which applicant has the earliest deadline — that date is the effective decision date when court pressure is active.

   - **Identify which regulatory step the process is at**: SEC ETF approvals have two stages:
     - 19b-4 approval (exchange rule change): This is the substantive hurdle. Once approved here, the product is highly likely to proceed.
     - S-1 registration statement approval (issuer registration): This is typically a paperwork phase. Once 19b-4 is approved, S-1 approval follows within 1-3 months.
     - Questions asking "will trading begin by [date]" require BOTH stages to be completed. If only the 19b-4 stage is passed, the S-1 stage still needs time.

   - **Check the SEC Chair's public posture**: Does the Chair oppose the product class? If yes, the SEC will use procedural delay (extended comment periods, requests for additional information) before ultimately approving — the question is when, not whether. If the Chair is neutral or supportive, the timeline is shorter.

   - **Check the institutional applicant identity**: Not all applicants are equal. BlackRock (575+ ETF approvals, nearly zero denials) entering the race is a leading indicator — the SEC cannot easily deny the world's largest asset manager without facing reputational harm and legal challenge. Distinguish between:
     - **Incumbent institutional applicants** (BlackRock, Fidelity, Invesco): Their participation structurally raises the probability of approval.
     - **Crypto-native applicants** (Grayscale, ARK Invest): Their legal challenges can compel approval but they lack the institutional credibility to raise baseline probability on their own.
     - The canonical pattern: a regulatory logjam breaks when (a) a court rules against the agency AND (b) an incumbent applicant enters the race. Either alone is insufficient.

   - **Check for pending litigation**: Are ETF applicants currently suing the SEC over denial of a similar product? Active litigation creates judicial deadlines that can force SEC action.

   - **Apply the [[concepts/regulatory-precedent-cascade]] framework**: Calibrate probability based on:
     - P(product approved) = baseline probability (depends on product novelty and legal precedent)
     - P(approved by deadline) = P(approved) * P(timing sufficient) — if the legal case is clear but the deadline is tight, the question shifts from "will it happen" to "when will it happen"

   - **Distinguish "will it be approved" from "will it begin trading"**: These are two different events with separate timelines. "Approved" can mean just the 19b-4 stage; "begins trading" requires both stages plus exchange listing preparation (typically 1-5 days after S-1 approval). The Ethereum ETF case: 19b-4 approved May 23 → S-1 approved July 22 → trading began July 23.

   - **Document the regulatory precedent assessment explicitly** in the reasoning, including the precedent chain, the applicant with the earliest deadline, the current procedural stage, the institutional applicant identity, and the timeline calibration. The most common errors are: (1) treating each ETF approval as independent when it is a cascade; (2) ignoring statutory deadlines as the mechanism that converts legal compulsion into concrete dates; and (3) treating all applicants as equal when institutional incumbent applicants are leading indicators.

   ➡ **LOAD the dedicated procedure**: [[domains/economics/procedures/sec-product-approval-forecast]] for the step-by-step checklist. This procedure MUST be loaded (not just referenced) when the question involves any SEC crypto product approval timeline — it contains the statutory deadline identification, institutional tier analysis, and regulatory stage distinction that vault content alone does not enforce as a workflow. The gold_54 error (Ethereum ETF by June 30, predicted NO, actual YES) occurred because this procedure was not loaded despite the vault having all the underlying content.

23. **Assess tech ban resolution dynamics**: If the question asks whether a technology product, service, or company will be "banned," "prohibited," "blocked," "restricted," or "outlawed" in a jurisdiction:

   - **FIRST — Load the dedicated procedure**: [[domains/global/procedures/ban-resolution-checklist]] MUST be loaded before forecasting. This procedure provides the 5-step structured assessment for determining what "banned" means in a prediction market resolution context. The TikTok case proved that the everyday meaning ("ongoing prohibition") differs from the market resolution meaning ("legal prohibition that took effect, even if enforcement is later delayed").

   - **Distinguish legal status from practical enforcement**: Answer these three sub-questions separately:
     - Did the law/order legally take effect? (Passage + legal review + deadline reached)
     - Did enforcement actions occur? (App store removal, service shutdown, fines, orders to cease operations)
     - Is enforcement persisting? (Ongoing vs. suspended/delayed)
     **Resolution rule**: Prediction markets resolve on (1) AND (2). Condition (3) is almost never required unless the resolution text explicitly specifies permanence.

   - **Classify the ban type**: Different ban mechanisms have different legal and forecasting characteristics:
     - **Legislative ban** (most durable): Passed by Congress with bipartisan support, signed into law. Survives unanimous SCOTUS review. Most likely to satisfy resolution criteria and resist executive reversal.
     - **Executive order ban** (vulnerable): Issued by president alone, subject to legal challenge on procedural/due process grounds. Higher reversal probability. Trump's 2020 TikTok EOs were blocked by courts; the 2024 legislation succeeded where EOs failed.
     - **Regulatory ban** (agency action): Issued by federal agency under existing statutory authority. Subject to APA challenge. CFIUS orders, export controls from Commerce/BIS.
     - **State-level ban** (patchwork): Individual state prohibitions. Face stronger First Amendment challenges and can be preempted by federal law. Montana's TikTok ban was blocked; federal law preempted state action.

   - **Map the lifecycle stage** using [[domains/global/concepts/national-security-tech-ban]]:
     1. Threat framing: Is the security establishment actively identifying this product as a risk?
     2. Political mobilization: Does bipartisan support exist for action?
     3. Legislative/executive action: Has a bill been introduced or an order been issued?
     4. Legal challenge: Is litigation pending or expected? Legislative bans survive review; executive orders often do not.
     5. Implementation: Has the deadline passed? Has enforcement started?
     6. Alliance pressure: Are allies adopting similar restrictions?
     7. Adaptation/resilience: Is the targeted company adapting (divestiture, alternative supply)?

   - **Check for executive enforcement delay**: If the ban legally took effect but enforcement was delayed or suspended:
     - Is the delay formal (executive order) or informal (non-enforcement)?
     - Is it time-limited or indefinite?
     - Does the resolution text distinguish legal effect from enforcement?
     - Apply [[domains/global/concepts/executive-enforcement-delay/_concept]]: a formal delay does NOT negate the ban for resolution purposes — the TikTok case is the canonical example where the ban legally took effect despite Trump's Jan 20 EO.

   - **Assess legal vulnerability**:
     - Does the ban target ownership structure (most durable — survived SCOTUS review) or content (First Amendment challenge)?
     - Was it passed with bipartisan supermajorities (legislative) or by executive order alone?
     - Is there pending litigation that could block enforcement but not retroactively undo the legal effect?
     - The bipartisan + legislative combination is the strongest structural predictor of survival.

   - **Check the resolution text for specificity**: Document the exact phrasing:
     - "Banned for download and/or use" → app store removal satisfies the criterion
     - "Banned from operating" → may require ongoing enforcement
     - "Banned with no exemptions" → higher bar than simple prohibition
     - "Banned or restricted" → lower bar; restriction alone may suffice
     - If the text says "permanently banned" → enforcement persistence matters. If simply "banned" → legal effect + enforcement action is sufficient.

   - **Document the ban resolution assessment explicitly** in the reasoning, including: the ban type, current lifecycle stage, legal/bipartisan durability, any executive enforcement delay, and the exact resolution text criteria. The most common error is treating executive enforcement delay as retroactively negating the ban, when resolution markets resolve on the ban's legal effect and enforcement action, not on enforcement persistence.

   This step exists because future "will X be banned?" questions (about WeChat, Shein, Temu, or any Chinese-owned platform) will need to replicate the TikTok analysis framework. The vault already has the concepts and entities — this procedure step ensures they are systematically applied.



24. **Assess presidential term continuity**: If the question asks whether a specific individual will be President of the United States (or hold any other US office) on a given date:

   - **Map all five removal mechanisms**: Death in office, resignation, impeachment + conviction, 25th Amendment Section 4 (incapacity), assassination. For each mechanism, assess whether it is plausibly activatable within the timeframe between now and the question's target date.
   - **Start from baseline continuity probability**: >95% for any sitting US president whose term covers the target date. Only adjust downward when specific evidence for a removal mechanism exists.
   - **Check age and health**: Document the president's age and any public health indicators. For presidents 70+: note that annual mortality risk rises to ~2-4%. For presidents 80+: document specific health incidents (falls, hospitalizations, cognitive concerns) that further elevate risk.
   - **Check impeachment posture**: Has any House member introduced articles? Has the Speaker opened an inquiry? If yes, assess whether the structural factors predict inquiry success or failure using the [[concepts/impeachment-inquiry-failure-mode]] framework. Even if the inquiry advances to articles, conviction requires a 2/3 Senate supermajority — impossible if the president's party holds 34+ seats. An inquiry alone is a weak signal; the structural factors (majority margin, direct evidence, committee unity, election proximity, Senate composition) determine whether articles reach the floor.
   - **Check resignation pressure**: Are party leaders, donors, or elected officials publicly or privately calling for resignation or withdrawal? Is there a trigger event (debate failure, scandal, health crisis)? Apply the [[concepts/incumbent-withdrawal-cascade]] framework if resignation pressure is active.
   - **Document the absence of mechanisms explicitly**: The most common correct prediction will be "yes, still in office." The reasoning must still demonstrate that all five mechanisms were checked and found inactive. An unexplained "yes" is not a vault-supported forecast.
   - **Apply the [[concepts/us-presidential-term-continuity]] framework**: use its baseline probability table and adjustment rules to calibrate the specific forecast.
   - **Document the continuity assessment explicitly** in the reasoning, including each mechanism's status and the final calibrated probability. The most common error is failing to distinguish between "nothing suggests he'll leave" (which is a valid forecast) and "I don't know" (which is not).

24. **Aging-incumbent pre-trigger vulnerability assessment**: If the question involves an incumbent leader aged **70+** in a re-election campaign, or asks whether an aging leader will remain in office through a specific future date, the agent MUST run the [[domains/usa/procedures/proc-aging-incumbent-early-warning]] framework BEFORE calibrating the final probability. This applies EVEN IF no trigger event, internal pressure cascade, or visible defection has yet occurred. The gold_12 error (Biden dropout 2024, predicted NO at Q2 2023 cutoff, actual YES) demonstrates that the cumulative probability of a trigger event over a 10-month horizon for an 81-year-old incumbent can be 40-55% even when the current moment looks stable. Applying only the persistence frame (status-quo bias) will produce systematically wrong forecasts for aging incumbents with no legal jeopardy.

   - **Check the leader's age as the gate condition**: If age >= 75, the mandatory assessment triggers. If age 70-74, moderate elevation — assess the vulnerability signals and apply at the agent's discretion. If age < 70, baseline withdrawal dynamics apply (use the standard [[domains/usa/procedures/proc-incumbent-withdrawal]] procedure).

   - **Run the 6-signal vulnerability inventory** from [[domains/usa/procedures/proc-aging-incumbent-early-warning]] Step 2:
     1. Age concern — are polls showing 50%+ voters concerned?
     2. No legal jeopardy — leader faces no existential charges?
     3. Party doubt — does the leader's own party want someone else?
     4. Low approval — stuck below 45% without acute crisis?
     5. Successor ready — is a natural successor (VP, cabinet member) available?
     6. Party not restructured around leader — would party function without them?
   
   - **Calculate the cumulative trigger probability** over the forecast horizon using the compound formula from Step 4 of the procedure. For an 80+ year-old leader with 4+ vulnerability signals and a 10-month horizon, the baseline P(any trigger) is approximately 55-65%.

   - **Apply the stated-intention discount** from Step 5: The leader's public statements denying any intention to withdraw are NOT valid evidence for a NO forecast. All three canonical withdrawers (Truman, LBJ, Biden) denied intention up to the moment of withdrawal. Overweight structural vulnerability signals, underweight stated intentions.

   - **Document BOTH frames explicitly**: (1) Why the leader might persist (current stability, institutional inertia, campaign investment) AND (2) why a trigger could materialize (cumulative probability, the six vulnerability signals, historical precedent). The most common error in aging-incumbent questions is adopting a single "stable" frame and ignoring the cumulative trigger risk.

   - **Reference the three canonical cases**: Truman (1952), LBJ (1968), Biden (2024) — all incumbents aged 60+ with no legal jeopardy who withdrew after trigger events. The cross-case pattern validates the pre-trigger vulnerability framework.

   - **Cross-link** to [[domains/usa/procedures/proc-aging-incumbent-early-warning]] and [[domains/usa/concepts/incumbent-withdrawal-cascade]] in the reasoning.

25. **Assess comprehensive exclusion list in prediction-market questions**: If the question uses an exclusion-list format — asking whether "another" or "a candidate OTHER THAN" a list of named entities will be selected or occur:

   - **Identify the format**: Does the resolution text list multiple named entities as excluded from "another"? Examples: "Will another man be the 2024 Democratic VP nominee?" excluding 13+ men, or "Will another woman be the 2024 Democratic VP nominee?" excluding 9+ women.

   - **Run the four-part diagnostic** from [[concepts/comprehensive-exclusion-list-forecast]]:
     1. **Coverage test**: Does the list include every plausible candidate in the category? List all known viable candidates and compare to the exclusion list. Any gap where a plausible candidate is missing?
     2. **Surprise test**: Would the "another" selection be a genuine surprise to informed observers? If yes → P(YES) is structurally low. If no → the list missed a plausible outcome.
     3. **Pipeline test**: How deep is the next-tier candidate pool outside the list? Is there a sharp quality/qualification drop-off? If the next tier is materially less viable → P(YES) is low.
     4. **Context test**: Do selection constraints (campaign pledges, institutional norms, party rules, gender balancing) further restrict the pool? If the context independently narrows the feasible set → P(YES) is even lower.

   - **Check for context changes after question creation**: The most common vulnerability of exclusion-list questions is that events after the list was written change the selection context. In the 2024 case: Biden's withdrawal made Harris the presidential nominee, which changed the VP selection dynamic entirely. Always check: has the selection process or context changed since the question was created? If yes, re-apply the diagnostic in the new context.

   - **Calibrate using the comprehensive-exclusion-list framework**:
     | Condition | P("another" = YES) |
     |-----------|---------------------|
     | List covers ALL plausible candidates + context constrains further | <5% |
     | List covers most but 1-2 plausible candidates missing | 10-20% |
     | List covers only a few candidates | 30-50%+ |
     | Context changed after question creation | Variable — re-run diagnostic |

   - **Document the exclusion list assessment explicitly** in the reasoning, including each diagnostic test's result and the calibration rationale. The most common error is treating "another [category]" as an open-ended question when the exclusion list has already captured the entire plausible candidate pool.

   - **Apply this step BEFORE the veepstakes assessment (step 21)** when the question involves a VP selection with an exclusion list. The exclusion-list analysis is a meta-level filter that can make the domain-specific analysis moot — if the list is exhaustive, the question resolves NO regardless of the specific VP selection dynamics.

   - **Cross-link** to [[domains/usa/concepts/comprehensive-exclusion-list-forecast]] and [[domains/usa/concepts/veepstakes-electoral-signal]] in the reasoning.

26. **Assess state-level tech bill passage dynamics**: If the question asks whether a state-level technology regulation bill (AI safety, privacy, content moderation) will be passed, become law, or be signed by a specific date:

   - **FIRST, classify the jurisdiction**: Is this California (the canonical bellwether) or another state? California has the most developed tech regulation ecosystem and the most defined legislative calendar. Other states (New York, Colorado, Washington) have different calendars and dynamics — but California is the default reference case.
   
   - **Determine the bill's current stage** (see procedure step): Introduced → Committee → Passed one chamber → Passed both chambers → On governor's desk. Each stage has a different baseline passage probability. For a bill still in committee, the baseline passage probability is typically 10-30%.
   
   - **Check the legislative calendar**: For California, the Aug 31 deadline (even-numbered years) is the hard binary cliff. A bill introduced within 3 months of this deadline faces a 30-50% penalty on baseline passage probability. If the deadline has already passed, the bill is dead — the question resolves NO regardless of support level.
   
   - **Assess the governor's position using the governor-veto-tech-bill-dynamics framework** ([[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]]): Check the governor's national ambition signals, alternative executive action pathways, industry opposition intensity, bill novelty, and override capacity. The governor's posture is the dominant variable once the bill reaches the desk.
   
   - **Map the political landscape**: Which party factions support/oppose? Is there a notable intra-party split (like Pelosi opposing SB 1047 within the Democratic caucus)? Which industry groups are lobbying and at what intensity?
   
   - **Load the dedicated procedure**: [[domains/usa/procedures/state-level-tech-bill-forecast]] — this formalizes the 7-step assessment (bill stage → calendar → veto point → governor ambition → political landscape → probability calibration → reasoning documentation). The procedure MUST be loaded when the question involves state-level tech legislation.
   
   - **Document the full assessment explicitly**: Include the bill stage, calendar deadline, governor posture, override assessment, and the key uncertainty variable in the reasoning.
   
   - For the canonical case (SB 1047), the forecast was NO and the actual outcome was NO — the bill failed to pass both chambers before the Aug 31 deadline, with Governor Newsom's veto posture and alternative executive order serving as additional barriers. The key variables were: bill still in process through Q3, heavy industry opposition, intra-party split, and Newsom's national ambition signaling a preference for executive action over legislation.
   
   This step exists because the SB 1047 question (Question 36) was correctly predicted but the vault had zero technology policy content. Every future state-level tech regulation question must find structured analytical support.

27. **Audit mass-sociogenic-event and government-confirmation-requirement questions**: If the question asks about the cause or nature of a mystery event (drone waves, UFO sightings, unexplained phenomena) or whether an official government confirmation will occur:

   - **FIRST — determine if the question requires government confirmation**: Read the resolution text carefully. Does it require an official government statement (\"confirms,\" \"admits,\" \"acknowledges,\" \"officially states\")? If yes, the government confirmation bar applies — see step 27a below. If the question asks about an event existing regardless of government confirmation, standard probability assessment applies.

   - **Classify the event type**: Is the question about:
     - A mass sociogenic event (mystery drone waves, UFO flaps, social panics) — apply the [[domains/usa/concepts/mass-sociogenic-event/_concept]] framework
     - A verified security incident with official tracking (military base incursions, confirmed surveillance) — use standard incident assessment
     - Mixed — where civilian reports and genuine incidents coexist. Load the distinction guidance from the mass-sociogenic-event concept and assess each layer separately.

   - **Check age of the event**: How many days since initial reports emerged? The probability of official confirmation DECLINES with time:
     - 1-7 days: Early stage — official position not yet formed. P(confirmation) is at maximum but still low (10-25% for Type A).
     - 8-21 days: Official statements begin — typically \"no evidence of threat.\" Each passing day without confirmation halves the remaining probability.
     - 22+ days: Investigation conclusions released. If the joint statement says \"nothing anomalous,\" P(confirmation) drops to <2%.

   - **Check for pre-existing official denials**: Has any federal agency already denied the theory? A prior denial from NNSA, DHS, FBI, or DoD is near-deterministic — the agency would have to admit it was wrong to confirm later. This is the single most important variable.

   - **Map the theory's origin**: Did the theory originate from:
     - Federal investigators (most credible — raises confirmation probability)
     - Local officials (plausible but low credibility — local officials rarely have national security clearance to make such claims)
     - Social media/online speculation (lowest credibility — the existence of a theory does not correlate with confirmation probability)

   - **Apply the government-confirmation-requirement concept** ([[domains/global/concepts/government-confirmation-requirement/_concept]]) framework:
     - Classify the confirmation type (Type A: vulnerability admission; Type B: adversary attribution; Type C: technical fact; Type D: error admission)
     - Type A questions (this canonical case): default P(yes) < 10%
     - Check whether external compulsion (lawsuit, subpoena, leak, unanimous bipartisan pressure) could force confirmation — if none, the voluntary confirmation probability is near zero

   - **Check entity stub coverage**: Entity stubs MUST exist for:
     - The lead investigating agency (FBI for most domestic security incidents)
     - Any agency specifically mentioned in the question or theory (NNSA for nuclear search, DHS for homeland security, DoD for military incidents)
     - The agency head (if relevant for confirmation authority)
     - Create missing stubs before forecasting — these are named actors in the question

   - **Document the assessment explicitly** in the reasoning, including: the confirmation type, the age of the event, any pre-existing denials, the theory's origin, and the calibrated probability. The most common error is treating \"will the government confirm X?\" as equivalent to \"is X true?\" — the government confirmation requirement adds a structural barrier that typically halves or quarters the naive probability.

   This step exists because the Mystery Drones question (Question 44 of the PIT blind test, correct NO prediction) was correctly predicted on the government confirmation bar, but the vault had no framework for recognizing a Type A government confirmation question, no mass-sociogenic-event concept, and no entity stubs for any of the agencies involved. After this step, every future question about a domestic security incident or government confirmation of a speculative security claim will trigger systematic framework-based analysis.

### Post-Forecast Reflection (after outcome known)
1. **Diagnose error**: Was it a PIT error (factually wrong vault content), a missing-thread error (causal chain untracked), a missing-concept error (pattern not recognized), or a missing-entity error (key actor dynamics unanalyzed)? For correct predictions, also diagnose: which vault component enabled the correct reasoning? This validates the feedback loop.
   - **For numerical range questions** (e.g., "Will X party have between Y and Z seats?"), diagnose separately: was the error in the point estimate (wrong modal prediction), the distribution (wrong probability spread), or the boundary assessment (range boundaries mispositioned relative to the distribution)? A wrong prediction on a range question that had the correct distribution shape is a calibration issue, not a framework failure — but the distribution should still be updated with the new data point.
2. **Assess vault contribution for correct predictions**: If the prediction was correct, identify whether the vault contributed non-trivial signal or whether the correct answer came from general knowledge alone. A correct prediction that relied solely on general knowledge reveals a vault gap as surely as a wrong one — the vault did not contribute to the forecast. Create or update threads, entities, and concepts for the question's domain regardless of prediction correctness.
3. **Score vault contribution**: Assign a vault contribution score to every forecast (correct or wrong) on this scale:
   - **0% (freebie)**: Correct prediction came from general knowledge alone. The vault had no relevant threads, no entity stubs for named actors, and no concept files for the domain. Full remediation required — create at minimum threads and entity stubs for the domain.
   - **Partial (10-80%)**: Some vault assets existed and contributed, but coverage was incomplete. Example: a thread existed but entity stubs for the question's named actors were missing. The partial gap percentage should reflect what fraction of the needed context the vault supplied. Remediate missing components.
   - **Full (100%)**: Every named entity in the question had a vault stub. A relevant thread existed and was up to date. A relevant concept file covered the pattern. The vault provided the majority of reasoning signal. Only minor updates needed (e.g., adding this forecast to the concept's Validated By table).

4. **Scan for recursive entity completeness**: After assessing primary entity coverage, scan the body text of ALL entity files related to the question for named individuals who lack entity stubs. Entity files often list "Key Figures," "Leadership," or "Notable Members" without creating stubs for them. If any named individual in a related entity file could plausibly appear in a forecast question (deputy, candidate, faction leader, spokesperson), create a stub. This step prevents the "second-tier entity gap" where the primary subject (e.g., FIT-U) has a stub but the named individuals within it (e.g., Myriam Bregman, Romina Del Plá, Christian Castillo) do not. Stub creation cost is negligible (~2 min each) and prevents information cascades.
   
   The score forces honest assessment of vault health per domain. Trend over cycles should show increasing scores as domains are covered.
4. **Fix the timeline**: If any PIT quarter file had factual errors, fix them immediately.
5. **Create missing threads**: If the causal chain wasn't tracked, create a thread file. Create or update a thread for the question's domain even if the prediction was correct — a correct prediction with no thread is a structural gap.
6. **Create missing concepts**: If a recurring dynamic wasn't captured, create a concept file.
7. **Create missing entities**: If key actors were absent, create entity stubs.
8. **Write forecast entry**: Create a `forecasts/YYYY-MM-DD-slug.md` file documenting the question, prediction, actual outcome, and vault gaps found. For correct predictions, explicitly note what vault component(s) enabled the correct reasoning, or note that the vault contributed no signal.
9. **Update _index.md**: Add new threads, concepts, entities, and forecasts to the index.
10. **Validate concepts**: If a correct prediction was enabled by a concept file, add a "Validated By" entry to that concept tracking the forecast. This builds the concept's track record over time and identifies which patterns are most reliable for forecasting. If a correct prediction was NOT enabled by a concept but a new concept was created post-hoc, add the forecast to the new concept's "Validated By" table with a note that it was created retroactively.
11. **Write reflection**: Update `_reflection-YYYY-MM-DD.md` with lessons learned.

12. **Check for question battery saturation**: After every forecast (correct or wrong), check whether this question belongs to a known battery — a set of questions about the same event or domain:
    - **Battery test**: Do the last 3-5 questions share a domain tag (e.g., `argentina`, `gaza`, `fed-rate`)? Do they reference the same election, conflict, or institution? If yes, this is a battery.
    - **Saturation test**: Does the vault already have (a) a complete thread, (b) entity stubs for all named actors, and (c) at least one concept for the dynamic? If all three → saturated.
    - **1st battery question**: Full reflection, create threads, entities, concepts.
    - **2nd-3rd battery questions**: Fill residual gaps, extract domain-specific concepts.
    - **4th+ battery question**: Shift effort to abstraction. The domain is saturated. Instead of creating more domain-specific content, ask "What cross-national or cross-domain pattern does this question expose?" Create concepts and procedures in `domains/global/` rather than `domains/[region]/`.
    - **Document saturation**: State "Nth question in [domain] battery. Domain is saturated. Effort shifted to [abstraction target]."
    
    This step exists because the Argentina 2025 legislative election generated 5 separate questions (FIT-U, 3x HNP, LLA). By the 5th question the domain was fully saturated → further reflection on domain-specific content had zero marginal value. The vault must recognize saturation and shift effort to abstraction automatically.

12a. **Blind-test battery detection**: In a blind test or unknown-distribution forecasting competition, you cannot rely on "last 3-5 questions share a domain tag" because the question sequence is randomized and you don't know the prior questions' domains. Use these alternative signals:

    - **Vault saturation test (independent of question sequence)**: Before committing to reflection effort, check whether the vault already has a complete thread covering the question's subject, entity stubs for all named actors, and at least one concept explaining the dynamic. If all three are present, the question is in a saturated domain regardless of question number.
    - **Structural improbability pre-check**: Apply the [[structural-improbability-check]] concept before any reflection. If the YES outcome requires 2+ independent failures of structurally larger actors (p < 0.01), the question describes a structurally-impossible outcome. The vault cannot improve calibration by adding domain-specific content for questions with p(yes) < 0.01 — the marginal information gain is zero. Shift effort to methodology instead.
    - **Trace the chain**: If the vault thread ended with "Phase N of M" or "Next test: [year]", the question is likely asking about the next phase — check for saturation proactively even if no other questions from the current batch share the domain tag.
    - **Four-level effort allocation in blind tests**:
      1. Domain is new to vault + structurally possible → full effort (thread, entities, concepts, procedure)
      2. Domain is new + structurally improbable → create concept for the improbability pattern, skip domain-specific content
      3. Domain is saturated + structurally possible → minimal domain-specific fixes (residual entity stubs, pending thread updates)
      4. Domain is saturated + structurally improbable → no domain-specific effort; shift to cross-domain abstraction only

12b. **Pre-forecast structural improbability check**: Before any analysis of a question about a minor party / long-shot candidate winning a plurality or majority, run the [[structural-improbability-check]] decision tree (Steps 1-5). If p(YES) < 0.01, do not build domain-specific reflection content — the prediction is structurally determined. Document the check result in the reflection and move to cross-domain abstraction. This prevents wasted effort on questions whose resolution reveals no vault content gap because the outcome was not contingent on any information the vault could have contained.

### Thread Status Guidelines
- `nascent`: Just identified, few data points
- `active`: Clearly unfolding, multiple data points per quarter
- `climaxing`: Approaching a peak or turning point
- `fading`: Losing momentum, winding down
- `resolved`: Ended or transformed into a different thread

### Phase 4: Create/Update Entity Files
After writing each quarter (or in batch), create entity files for:
- Wars, treaties, major political events referenced by wikilinks
- Scientists, inventors, and thinkers whose work will matter later
- Political leaders and statesmen
- Cultural figures with long-term significance
- Places of strategic importance

Entity file naming: lowercase slug with hyphens, no articles, no periods.
  - `boxer-rebellion.md`, `max-planck.md`, `second-boer-war.md`

Entity priority order:
1. Entities referenced in 2+ quarters (create first — highest connectivity)
2. Entities central to active threads
3. Entities with direct forecasting relevance
4. Single-quarter entities (lowest priority — create only if heavily referenced)

### Phase 5: Review Concept Files
Check if any existing concept files now have new canonical examples. If so, update them. If a new recurring pattern emerged, create a new concept file.

### Phase 6: Verification
1. **Wikilink integrity**: Every `[[wikilink]]` in a quarter file should point to an existing entity file, or be a known forward-reference (entity not yet created).
2. **Date accuracy**: Verify key dates, especially for events with differing sources.
3. **PIT compliance**: Check that no information uses post-period knowledge.
4. **Consistency**: Same event across consecutive quarters should use the same phrasing and framing.
5. **Thread continuity**: Every open thread from the previous quarter that has developments in this quarter must be updated.

### Phase 7: Commit
1. Stage all new/changed files.
2. Commit message format:
   - `summary: YYYY-QN` for new quarter
   - `entities: entity1, entity2` for entity batch
   - `threads: thread1, thread2` for thread updates
   - `vault: description` for structural changes
   - `fix: description` for corrections

22. **Assess dominant-party election dynamics**: If the question asks whether a candidate from a dominant party (a party that has won 3+ consecutive presidential elections or holds >45% party identification) will win an upcoming presidential election, apply the dominant-party election framework:

   - **Identify the system type**: Is this a dominant-party system? Check: (1) Has the incumbent party won 3+ consecutive presidential elections? (2) Does it hold >45% party identification? (3) Does it control majorities in the national legislature and most state/provincial governments?
   - **Check term limits**: Can the incumbent president run for re-election? If term-limited (e.g., Mexico's single-term presidency), the successor dynamic is active. If eligible and running, the re-election dynamic applies.
   - **Measure incumbent approval**: What is the outgoing (or running) incumbent's approval rating from PIT-compatible polling sources? Thresholds:
     - >55%: Strong successor dominance — forecast the successor win at >95% confidence
     - 40-55%: Competitive — forecast depends on opposition quality and economic conditions
     - <40%: Successor penalty — opposition likely wins unless fragmented
   - **Calculate approval-to-vote efficiency**: In Latin American dominant-party systems, the successor's final vote share typically approximates the incumbent's approval rating, with 5-15% wastage (some approval voters don't transfer). In Mexico 2024, wastage was near zero. In Venezuela 2013 (Chávez→Maduro), wastage was ~4.5%.
   - **Count credible opposition candidates**: In single-round plurality systems, each additional opposition candidate splits the anti-dominant-party vote. Apply the [[domains/east-asia/concepts/divided-opposition-plurality-win/_concept]] framework if 3+ candidates are viable.
   - **Assess opposition coalition coherence**: Is there a unified opposition candidate or multiple competing candidates? An ideologically incoherent coalition (e.g., PAN-PRI-PRD in Mexico 2024) is less effective because voters face a choice between the known dominant-party candidate and an unknown coalition compromise.
   - **Apply the [[domains/latin-america/concepts/incumbent-successor-dominant-party/_concept]] framework**: The successor-dominance dynamic operates through three mechanisms (approval transfer, incumbent campaigning, welfare continuity signal). Document which mechanisms are active.
   - **Load the [[domains/latin-america/procedures/dominant-party-election-forecast]] procedure**: Follow its 5-phase assessment for comprehensive probability calibration.
   - **Document the structural mechanics explicitly**: Unlike competitive-election forecasting (which requires analyzing voters, platforms, and issues), dominant-party election forecasting is structural — it follows from the approval-transfer mechanism. The reasoning should state these mechanics and distinguish them from fragmentation dynamics.
   - **Create entity stubs for ALL named actors**: The candidate, the incumbent, the dominant party, each opposition candidate and their party. Entity stubs are the minimum coverage bar.
   - **Document both the structural case and the risk case**: The structural case (why the dominant party is expected to win) AND the risk case (what could plausibly cause an upset, with estimated probability).

   This step exists because the 2024 Mexican presidential election question was correctly predicted but the vault contributed zero signal: no Latin American domain, no entity stubs, no thread, no concept. The prediction relied on general knowledge, not vault content. Every future dominant-party election question must trigger this assessment before forecasting.

## Periodic Maintenance

### Every 4 quarters (year-end)
- Write annual summary (`timeline/YYYY.md`) linking all 4 quarters
- Create or update topic files for major cross-cutting themes
- Review entity files for completeness — fill gaps from wikilinks
- Review thread files — merge any that overlap, close resolved ones

### Per reflection cycle
- Read the vault's reflection files to identify recurring problems
- Update _procedure.md with new patterns discovered
- Prune or merge entity files that overlap
- Check that wikilinks resolve — orphaned links should either get entity files or be removed
- Audit thread statuses — update as threads progress

## Writing Quality Checklist

[ ] Overview paragraph captures the quarter's character
[ ] Major event arcs are chronological and linked across sections
[ ] Births/deaths include forecasting-relevant notables
[ ] At least 3 cross-domain threads
[ ] Every proper noun of significance is a [[wikilink]]
[ ] Wikilinks are case-consistent
[ ] No future knowledge leaks into PIT framing
[ ] Entity files exist for all major entities
[ ] Previous quarter's open threads are continued or resolved
[ ] Thread files updated with new quarter's developments
[ ] Concept files reviewed for new examples or new patterns
[ ] Wikilinks use minimal pipe syntax — prefer canonical names

## Pitfalls to Avoid

- **Status-quo bias (underestimating disruption probability)**: Empirically, the vault's wrong predictions have ALL been NO (predicted status quo/continuation, actual was disruptive change: Israel-Iran ceasefire, Israel-first ceasefire announcement, US gov shutdown, Biden dropout, Venezuela election). This suggests a systematic tendency to overweight the stability of the current state and underestimate the probability of structural breaks. When the question asks whether a change will occur, consciously ask: "Am I defaulting to 'no change' just because change requires multiple conditions to align? Are those conditions actually nearer to alignment than I think?" Apply the affirmative frame FIRST (case for change) before applying the skeptical frame.
  - **Authoritarian-election sub-pattern**: A specific manifestation of status-quo bias in authoritarian contexts — assuming the regime will prevent the opposition from winning. This conflates "winning the vote" with "assuming office." In contexts where the opposition has parallel vote tabulation infrastructure, the true vote outcome may be knowable and the opposition may have won even if the regime prevents a power transition. See [[concepts/authoritarian-electoral-facade]] and procedure step 17.
- **Retrocausation**: Writing "X would become important later" instead of describing X as it was then.
- **Scale imbalance**: Spending 500 words on a minor cultural event while a war gets 200 words.
- **Orphaned wikilinks**: Linking to nothing — either create the entity or remove the link.
- **Date drift**: Listing events in approximate order rather than exact dates.
- **Lost threads**: A civil war mentioned in Q1 should be followed up in Q2 unless it ended.
- **Missing the frame**: Every quarter should explain WHY the events matter, not just WHAT happened.
- **Pipe-syntax overload**: `[[actual|display]]` hides the canonical name from the graph. Use sparingly.
- **Thread neglect**: The most common failure — writing a quarter file without updating the threads that quarter feeds.
- **Entity bloat**: Not every minor figure needs an entity file. Focus on entities with multi-quarter relevance.
- **Narrow range trap**: When a question asks whether an outcome will fall in a specific narrow numerical range (e.g., "220-224 seats"), avoid two errors: (1) treating the range as unlikely just because it's narrow, and (2) treating the range's probability as the sum of equally likely outcomes. Instead, build a probability distribution over the full outcome space and assess how much mass falls in the range. The key parameters are: where the range sits relative to the distribution's mode, whether the distribution is skewed by structural biases, and how many competitive "coin flips" determine the outcome. A range that includes the mode can have 25-35% probability even if it's only 1% of the seat space. Always load the [[domains/usa/procedures/house-seat-range-forecast]] procedure when the question involves a numerical House seat range. See the generic-ballot-seat-conversion concept for the canonical example: the 220-224 range captured ~35% of the probability mass despite being a 1.1%-of-possible-outcomes range, because it covered the mode of a right-skewed distribution.

## Recurring Tasks

| Frequency | Task |
|-----------|------|
| Per quarter | Write quarter summary, update threads, create/update entities, review concepts |
| Per year | Write annual summary, create topic files, review coverage, merge threads |
| Per reflection | Read reflections, update procedures, restructure, run wikilink integrity audit (verify all concept/entity/procedure refs resolve) |
| On structural change | Update _spec.md if schema changes |

## Lessons from Cycle 4 (2026-05-18)

### Frontmatter Drift Prevention
After every batch of new files, run `grep` audits to catch non-standard frontmatter before committing. In this cycle, `quarters/1901-Q3.md` used `title:`/`slug:`/`period_start:` instead of `label:`/`date_range:` — despite the spec being clear. The most common drift patterns: `span:` instead of `inception:`/`conclusion:` for threads; `tags:` instead of or in addition to standard fields for quarters.

### Backlinks Should Be Batched
Adding `## Appears In` sections to entity files is high-value but tedious. Batch them at the end of each cycle for entities that appear across 2+ quarters. Prioritize entities in active threads first.

### Threads vs Sub-threads
When a thread contains a distinct storyline that spans multiple quarters with its own dynamics (e.g., the Philippine-American War inside american-imperial-expansion), elevate it to its own thread file. The criterion: does the sub-storyline have its own causal chain, key events, and forecasting significance that would be lost if only the parent thread existed?

### Cross-Domain Threads in Quarter Files
The Cross-Domain Threads section at the bottom of each quarter file should link to concept files using [[wikilinks]]. This connects quarter-level analysis to the vault's pattern-matching layer.

### Dual Directory Audit
To prevent the `quarters/` vs `timeline/` confusion from recurring: run `ls` on both directories after every quarter commit. If both exist and have content in both, something has gone wrong. The canonical location is `timeline/`.

| ### "Winning" in Authoritarian Contexts Is Ambiguous
|The Venezuela 2024 error (predicted NO, actual YES — González won the vote) exposed a critical framing failure: equating "wins the election" with "assumes office." In prediction markets, "wins" defaults to the electoral outcome (received the most votes), not the political outcome (took power). This distinction matters most in authoritarian contexts where the regime can falsify official results but the opposition can document the true outcome. The [[concepts/authoritarian-electoral-facade]] concept and procedure step 16 now formalize the dual-dimension assessment.
|
|### Dangling Concept/Entity References Undermine Graph Integrity
|The Question 32 per-question reflection (woman VP nominee, correct NO prediction) surfaced a structural integrity issue that was invisible at forecast time: the concept `veepstakes-electoral-signal` was referenced from 5+ vault files (the 2024 election thread, the gender-balancing concept, the campaign-pledge concept, the comprehensive-exclusion-list concept, and _spec.md Rule 19) but had NO actual `_concept.md` file. The vault claimed to have a veepstakes framework with "calibrated forecasting rules" but no such framework existed — the references were dead edges in the graph that created an illusion of coverage.
|
|**Root cause**: Concepts and entities were being referenced in wikilinks during the writing process without checking whether the target file existed. This is a pipeline problem, not an analytical one — the vault's structural integrity checks were insufficient.
|
|**Fix**: _spec.md now includes Rule 36 (no dangling concept/entity references). The per-question reflection process must include a wikilink integrity audit: check all concept, entity, and procedure references in files used during the forecast. Missing files must be created or references removed. This audit is now a mandatory step in every reflection cycle.
|
|**Additional finding**: 6 VP finalist entity stubs were missing despite being named actors in the VP selection process: tim-walz (the actual VP pick), mark-kelly, andy-beshear, jb-pritzker, marco-rubio, tim-scott, elise-stefanik. Rule 19 already mandated these stubs, but the rule was not enforced because no audit mechanism existed. The wikilink integrity audit would have caught these.


### Parallel Vote Tabulation Is a Forecasting-Significant Variable
The Venezuela opposition's ConVzla operation — collecting tally sheets from 81% of polling centers — was the decisive factor that made González's victory knowable despite CNE falsification. The presence or absence of opposition PVT infrastructure is a material forecasting input for authoritarian election questions. The new [[concepts/authoritarian-electoral-facade]] concept captures this variable for future forecasts.

### Pre-Election Coverage Is the Most Common Omission in Contemporary Quarters
The Taiwan 2024 election question exposed a critical vault gap: the 2023-Q3 and 2023-Q4 timeline files had extensive campaign coverage for dozens of events (Gaza war, US politics, European elections) but ZERO coverage of the Taiwan presidential election campaign — despite it being held just 2 weeks after the Q4 cutoff. The vault had post-hoc entities (Ko Wen-je, Lai Ching-te, TPP, DPP) and a concept file (divided-opposition-plurality-win) but no PIT pre-election quarter entries that a forecaster could have used at the time. This omission rendered the vault useless for pre-election forecasting despite having correct post-hoc analysis.

The fix: every contemporary quarter file must include a dedicated subsection for any major election scheduled within the next 2 quarters. The candidate field, electoral system, opposition coordination status, and polling data are mandatory fields. See _spec.md Rule 22 for the detailed requirements. This rule applies to ALL quarter-writing agents regardless of whether the election outcome seems predictable — the vault's value is in providing point-in-time evidence, not post-hoc explanation.

### Technology Policy Was a Zero-Coverage Domain

The SB 1047 question (Question 36, correct NO prediction) exposed a major structural gap: the vault had no technology policy domain, no AI regulation thread, no California entity stubs (Newsom, Wiener, CA Legislature), and no concept for state-level tech regulation dynamics. This is a blind spot for an entire domain of recurring prediction market questions — state-level AI regulation, privacy bills, content moderation laws, and the California bellwether dynamic.

**Root cause**: The vault's coverage was organized around traditional geopolitical domains (war, elections, macroeconomics) but not technology policy. Unlike financial regulation (which was added after the Ethereum ETF gap), state-level tech regulation was simply never considered a mandatory domain because no prior question had exposed the gap — until SB 1047.

**Fix**: Added Spec Rule 39 (state-level technology regulation as mandatory coverage), a dedicated thread (state-level-ai-regulation), two concepts (state-level-tech-regulation-bellwether, governor-veto-tech-bill-dynamics), three entity stubs (Newsom, Wiener, CA State Legislature), and a forecasting procedure (state-level-tech-bill-forecast). The California tech regulation ecosystem is now a vault domain with structural analytical depth parallel to electoral dynamics and macroeconomic coverage.

**Future prevention**: The per-forecast audit's "named entity sweep" should now catch questions about technology actors (Newsom, Wiener, tech CEOs) and trigger the tech regulation framework. The domain coverage checklist should include "technology policy" alongside war, politics, economics, and finance. Any quarter file covering a period with significant state-level tech regulation developments (CCPA passage, SB 1047 debate, Newsom executive orders) must now include a technology regulation subsection.

---

### Cabinet Formation Coverage Is Missing Despite Being a Recurring Forecast Domain

The Rubio Secretary of State question (Question 41, correct YES prediction) revealed a structural gap: the vault had no systematic coverage of cabinet formation dynamics, no concept for second-term cabinet selection patterns, and no procedure for predicting presidential personnel picks. This is a recurring domain — prediction markets routinely ask about cabinet nominations, especially during presidential transitions.

**Root cause**: The vault's US political coverage was organized around electoral dynamics (veepstakes, state reliability, generic ballot conversion) and governance dynamics (shutdowns, impeachment, budget brinkmanship) but not personnel selection. The VP finalist data existed in the 2024 election thread but was never structurally connected to cabinet formation — the Rubio entity mentioned his VP candidacy but not his actual Secretary of State appointment. The vault had the building blocks (VP finalist pool, Trump personnel patterns) without the analytical framework to connect them.

**Fix**:
1. **New concept**: `domains/usa/concepts/trump-rival-to-ally-pipeline/_concept.md` — Documents the 4-stage sequence (rivalry → endorsement → loyal service → appointment) that governs Trump personnel selection, with canonical cases and forecasting rules.
2. **New concept**: `domains/usa/concepts/second-term-cabinet-formation/_concept.md` — Documents structural differences between first-term and second-term cabinet formation, the VP-finalist-to-cabinet pipeline, the establishment-credibility threshold, and confirmation dynamics.
3. **New procedure**: `domains/usa/procedures/trump-cabinet-selection-patterns.md` — Step-by-step framework for predicting Trump cabinet nominations, including the portfolio-type categorization, personnel pool identification, probability calibration tables, and confirmed 2024-2025 transition patterns.
4. **Entity update**: `domains/usa/entities/marco-rubio.md` — Added Secretary of State section documenting the VP-finalist-to-cabinet pipeline pattern, foreign policy credentials, Senate confirmation advantage, and rival-to-ally arc. Connected to new concepts via wikilinks.
5. **Spec update**: Added Rule 40 (cabinet formation as mandatory coverage) to _spec.md.

Forecast rule: For ANY question about a presidential nominee for a cabinet position (any president, not just Trump), the first step is to check the VP finalist pool. If the person was a VP finalist who was eliminated on procedural grounds, their cabinet nomination probability is 60-80% — the dominant variable in the forecast. A vault with complete veepstakes coverage but no cabinet-formation framework is incomplete: the veepstakes data is the primary input to cabinet forecasting.

**Entity stub creation**: Named entity stubs for cabinet nominees must exist before forecasting on cabinet questions. For Trump's 2024-2025 transition, the minimum entity stub coverage for any named cabinet nominee includes [[domains/usa/entities/marco-rubio]], [[domains/usa/entities/doug-burgum]], [[domains/usa/entities/elise-stefanik]], [[domains/usa/entities/jd-vance]], and [[domains/usa/entities/tim-scott]]. All of these now exist.

**Future prevention**: The per-forecast audit's named entity sweep must now include cabinet nominees along with elected officials, legal actors, and candidates. Any question naming a potential or actual cabinet nominee triggers the second-term-cabinet-formation concept and the appropriate personnel procedure. Quarter files covering post-election transitions (2024-Q4, 2028-Q4, etc.) MUST include a subsection on cabinet formation, documenting the nominee list, Senate confirmation timeline, and the policy-direction signal of the picks.

---

### Religious Institutions Coverage Was Missing Despite Recurring Forecast Domain

The Pope Francis successor question (gold_28, Q59, wrong YES prediction) exposed a structural gap larger than any single domain omission: the vault had ZERO coverage of religious institutions, specifically the Catholic Church — a global institution with 1.4 billion members, a sovereign state, a defined succession process, and recurring prediction-market questions about papal health and transitions.

**Root cause**: The vault's coverage was organized entirely around secular geopolitical domains (war, elections, macroeconomics, technology policy, financial regulation). Religious institutions were never considered a mandatory domain because the vault was implicitly built on a secular-state ontology — the Pope was not treated as a head of state despite being one. The vault tracked the Argentina Chamber of Deputies and the Turkish Central Bank but not the Pope, the Vatican, or the College of Cardinals.

**Fix**:
1. **New domain**: `domains/religion/_domain.md` — Documents the Catholic Church and other major religious institutions as a forecasting domain.
2. **Entity stubs**: `domains/religion/entities/pope-francis.md` and `domains/religion/entities/pope-leo-xiv.md` — Document birth dates, health trajectories, and forecasting-relevant data for both the late pope and his successor.
3. **New thread**: `domains/religion/threads/papal-succession/_thread.md` — Tracks the health decline, death, conclave, and election sequence.
4. **New concept**: `domains/religion/concepts/elderly-leader-mortality-risk/_concept.md` — Provides age-based base rates, comorbidity multipliers, and functional decline signals for calibrating mortality probability over defined time horizons.
5. **New procedure**: `domains/religion/procedures/elderly-leader-mortality-assessment.md` — Step-by-step protocol for translating health data into calibrated probability.
6. **Spec update**: Rules 27 and 28 added to _spec.md mandating religious institution coverage and elderly leader health assessment.

**Forecast rule**: For ANY question about whether an elderly leader will survive through a defined period (or whether a "new" leader will be installed), the first step is to load the elderly-leader-mortality-assessment procedure and apply the age-comorbidity-functional-decline framework. The question of whether Pope Francis would survive 2025 was poorly calibrated because the leader's age (88), documented respiratory vulnerability, reduced mobility, and recurrent hospitalizations were not systematically converted into a mortality probability. After this fix, every elderly leader in a forecast question will trigger a structured health assessment before forecasting.

**Entity stub creation**: Named religious institution entities must exist before forecasting on questions involving religious leaders. For the Catholic Church, minimum stub coverage includes the current pope, the preceding pope, and any named cardinals/curial officials in the question.

**Future prevention**: The per-forecast audit's named entity sweep must now include religious leaders alongside political, legal, and financial actors. Any question naming a pope, cardinal, archbishop, religious leader, or religious institution (Vatican, Holy See, Catholic Church) triggers the religion domain's threads, concepts, and procedures. Quarter files covering papal transitions (2025-Q2) must now connect to the papal-succession thread.

---

### Tariff Coverage Was Scattered Without Connecting Thread or Structural Framework

Q61 (US tariffs on European cars before May 2025) was correctly predicted (YES), but the vault's tariff coverage was entirely in timeline quarter files as discrete events with no connecting thread, no causal mechanism concept, and no entity stubs for the actors implementing tariff policy.

**Root cause**: The vault had no dedicated trade policy thread. Tariff events were captured in quarter files (2025-Q1, 2025-Q2, 2025-Q3, 2026-Q1, 2026-Q2) as standalone items under "Economics & Trade" sections, but there was no narrative thread connecting the Feb 11 Section 232 action to the March Canada/Mexico tariffs to the April 2 Liberation Day to the May 28 CIT vacatur to the ongoing appeal. Each quarter read independently; the arc was invisible. Additionally:
- Trump's specific tariff escalation-bargaining pattern was not formalized as a concept (the five-phase cycle is distinct from normal protectionism)
- The three-layer legal framework (Section 232 / Section 301 / IEEPA) was not documented, despite being critical for forecasting which tariffs survive legal challenge
- The US Court of International Trade had no entity stub despite being the primary judicial forum for trade challenges and vacating the Liberation Day tariffs
- Key trade actors (USTR Jamieson Greer, Commerce Secretary Howard Lutnick) had no entity stubs
- The EU dimension (retaliatory tariffs, US-EU negotiations, European auto sector exposure) was mentioned in passing in 2026-Q2 but not structurally connected

**Fix**:
1. **New thread**: `domains/global/threads/us-trade-policy-tariffs/_thread.md` — Traces the full tariff arc from Jan 2025 through 2026, covering Section 232, IEEPA, Liberation Day, CIT vacatur, EU dimension, auto sector, and Fed interaction.
2. **New concept**: `domains/global/concepts/trump-tariff-escalation-bargaining/_concept.md` — Five-phase escalation-bargaining cycle with sector vulnerability ranking, market signal interpretation, and probability calibration factors.
3. **Entity stubs**: `domains/usa/entities/jamieson-greer.md`, `domains/usa/entities/howard-lutnick.md`, `domains/usa/entities/united-states-court-of-international-trade.md` — Key trade actors now documented.
4. **Spec update**: Rule 50 added to _spec.md mandating trade policy thread for tariff/auto questions, with six-factor mandatory pre-forecast checks and default heuristics.
5. **Thread connectivity**: Linked new trade policy thread from existing us-china-tech-decoupling thread.

**Forecast rule**: For ANY question asking whether Trump will impose tariffs on a specific sector or trading partner:
1. Load the US trade policy thread to check existing tariff actions in the relevant sector
2. Identify the legal authority (Section 232/301/IEEPA) and assess the vulnerability of each layer
3. Assess which phase of the escalation-bargaining cycle the administration is in (announcement, blowback, retreat, negotiation, repeat)
4. Check EU retaliation capacity and auto sector specificity
5. Apply the default heuristic: Section 232/301 authority with campaign rhetoric → >50%; IEEPA-only (post-CIT) → <30%; allied country → -20pp adjustment

**Entity stub creation**: Named trade policy actors must exist before forecasting on tariff questions. For any question naming a tariff action or trade negotiation, minimum stub coverage includes the USTR, Commerce Secretary, and the CIT.

**Future prevention**: The per-forecast audit's named entity sweep must now include trade policy actors alongside political, legal, financial, and religious actors. Quarter files covering trade policy events (tariff announcements, CIT rulings, EU retaliatory measures) must now connect to the us-trade-policy-tariffs thread.

---

### Ceasefire Questions Must Be Pathway-Classified or the Error Is Structural

The Iran-Israel ceasefire before July question (gold_01) exposed a forecasting error pattern distinct from missing entities or domains: **applying the wrong causal model to a ceasefire question.** The question was about a state-on-state ceasefire, but the available evidence showed "no ceasefire negotiations in progress" — evidence that applies to Pathway A (diplomatic) but is irrelevant to Pathway B (war-termination via superpower entry). The forecaster treated the question as diplomatic, when it was actually about whether a war would start and be terminated within the window.

**Root cause**: The vault had all the building blocks for a correct prediction — the escalation ladder concept, the escalation-bargaining termination concept with the 48-hour rule, the IAEA entity, the Iran entity with nuclear latency analysis — but NO mechanism to tell the forecaster which building blocks to connect. The escalation-bargaining concept and the "ceasefire by date X" question type were never structurally linked. The forecaster could read the 48-hour rule about US entry producing rapid ceasefire without realizing that this rule was the correct model for answering the question.

**This is a graph-connectivity problem, not a content problem.** The vault had the right content but the wrong edges. The escalation-bargaining concept was connected to the Iran-Israel escalation thread but NOT to the ceasefire-timing procedure, the inter-state-ceasefire-feasibility procedure, or the short-window-ceasefire-probability concept. A forecaster loading the ceasefire procedure would not see the escalation-bargaining concept as a primary reference.

**Fix:**
1. **New concept**: `domains/global/concepts/ceasefire-pathway-decomposition/_concept.md` — Three-pathway framework (A: diplomatic, B: war-termination, C: none) with the decomposition formula P(ceasefire) = P(war in window) × P(termination | war). The single most important forecasting insight for state-on-state ceasefire questions.
2. **New procedure**: `domains/global/procedures/state-on-state-ceasefire-decomposition.md` — Step-by-step decomposition procedure that replaces the assumption of diplomacy with the escalation-ladder-to-war-termination analysis.
3. **New entity**: `domains/mena/entities/israeli-security-cabinet.md` — Missing ratification body entity; its crisis-accelerated approval process (0 hours vs 1-2 days standard) is a critical timing variable for war-termination ceasefires.
4. **Spec update**: Rule 11 and 11a added to _spec.md mandating ceasefire pathway classification before probability estimation, and ceasefire entity completeness for security councils and ratification bodies.

**Forecast rule**: For ANY ceasefire question:
1. Check if the conflict is state-on-state AND has an identifiable escalation ladder AND a superpower patron with escalation dominance.
2. If YES to all three, classify as Pathway B (war-termination). P(ceasefire) = P(war in window) × P(termination | war). Do NOT look at ceasefire negotiations or diplomatic pressure — those are Pathway A signals that do not apply.
3. If NO to any, classify as Pathway A (diplomatic). Use the short-window-ceasefire-probability base rates, political deadline adjustments, and pressure accumulation timeline.
4. For any state actor with a ceasefire ratification body (security cabinet, war cabinet, SNSC), verify entity stub coverage and the body's crisis-acceleration timeline.

**Entity stub creation**: Named ceasefire ratification bodies must have entity stubs before forecasting on ceasefire questions involving that state. The Israeli Security Cabinet is the canonical case. Iran's Supreme National Security Council (SNSC), Russia's Security Council, India's Cabinet Committee on Security, and Pakistan's National Security Council are anticipated future cases.

**Future prevention**: The per-question reflection for any ceasefire question must now include a pathway classification check as the first diagnostic step. The question "Was the ceasefire question Pathway B?" should be asked before assessing whether the vault had adequate coverage. If the question was Pathway B and the forecaster applied Pathway A reasoning, the gap is not in the content but in the connectivity — and the fix is to add edges between the escalation-bargaining concept and the ceasefire procedures, not to add more content.
