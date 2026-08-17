---
type: spec
tags: [meta, spec]
version: 2.0
date: 2026-05-18
author: hermes-agent
purpose: "Define the graph vault schema — file types, directory structure, frontmatter conventions, and graph connectivity rules"
---
---
---
# Graph Vault Specification v2.0

## Principles

1. **Point-in-time (PIT)**: Every entry is scoped to information available at the cutoff date. No retrocausation.
2. **Threads as primary nodes**: Long-running narratives are the vault's backbone. Quarter files populate threads; threads connect across quarters.
3. **Contemporary coverage alongside historical**: The vault MUST maintain parallel coverage of contemporary (post-2020) events, not just historical periods. Contemporary quarter files MUST be connected to thread files, entity files, and concept files — same structural rigor as historical coverage. A vault that only tracks 1900 cannot forecast 2025.
4. **Concepts as frameworks**: Concept files capture recurring patterns for forecasting — they are the vault's highest-leverage output for future prediction.
5. **Entities as graph vertices**: Entity files are the nodes that threads and concepts reference. Every entity referenced across 2+ quarters should have a file.
6. **Signal over noise**: Prioritize events, entities, and patterns with forecasting value.
7. **Forecast as feedback loop**: Each forecast is a test of the vault's quality. Wrong predictions reveal structural gaps that must be fixed immediately (threads, concepts, entities, timeline accuracy).
8. **No freebie predictions**: A correct forecast that relied solely on general knowledge (not vault content) reveals a vault gap as surely as a wrong one. The vault must provide non-trivial, domain-specific signal for every forecast it supports. After every forecast cycle — whether correct or wrong — assess whether the vault contributed signal and remediate any gaps found.
9. **Named entity stub completeness**: Every named person, party, coalition, or organization that appears in a forecast question MUST have a vault entity file — even if the prediction is trivially obvious. If a question asks about "FIT-U" and no `entities/fit-u.md` exists before the forecast is made, the vault is incomplete. The HNP gap (gold_28) was the original canonical example of this rule. The FIT-U gap (gold_56) is the reinforcing example: the correct NO prediction was a freebie relying on general knowledge, not vault content. After reflection, the entity file was created and the concept [[domains/global/concepts/far-left-marginalization-polarization]] was added to abstract the far-left ceiling pattern.

    **Recursive completeness**: Named individuals listed within entity files as "Key Figures" or "Leadership" MUST also have their own entity stubs if they could plausibly appear in a forecast question. If an entity file names 3 key figures and none have stubs, the vault has a recursive completeness gap. This rule is established by the FIT-U entity file (created in Cycle 8) naming Myriam Bregman, Romina Del Plá, and Christian Castillo without stubs until Cycle 10 — these figures could appear in questions about FIT-U leadership succession, faction dynamics, or electoral viability. Entity authors MUST scan the entity's body text for named individuals and create stubs for any whose absence would degrade future forecast quality. Stub creation cost is negligible (~2 minutes each) and prevents information cascades where a missing stub becomes a missing thread becomes a missing concept.

10. **US domestic budget and political dynamics are mandatory coverage**: The vault MUST systematically cover US federal budget processes (appropriations, CRs, debt ceiling, shutdowns) and domestic political dynamics (congressional factionalism, leadership battles, transition periods). These are among the most common subjects of forecasting questions and directly affect global markets, geopolitics, and risk assessment. Every contemporary quarter file (post-2020) MUST include a section on US domestic budget/political developments, even when foreign affairs seem more consequential. A vault that covers Middle East escalations but misses a US government shutdown is dangerously incomplete.

10a. **US government shutdown forecasting requires institutional-actor entity stubs**: Because US government shutdowns in the 118th-119th Congress are driven by the interaction of specific institutional actors with defined leverage mechanisms, the vault MUST maintain entity stubs for the following as the absolute minimum coverage:

    - **Speaker of the House** (Mike Johnson): Documents the Speaker's procedural options (suspension-of-rules vs regular order), Freedom Caucus constraint, and bipartisan-coalition-operating pattern.
    - **House Minority Leader** (Hakeem Jeffries): Documents the de facto governing coalition leader's vote-delivery mechanism, concession price, and signaling apparatus (the "green card" floor signal).
    - **House Freedom Caucus**: Documents the procedural veto mechanism (Rules Committee seats, one-member motion-to-vacate), defection baselines, and internal factional splits.
    - **Elon Musk**: Documents the emergent external-actor intervention dynamic where a non-officeholder can kill a bipartisan funding deal through social media pressure on the President-elect.
    - **President-elect / President** (as relevant): Documents the influence mechanism over congressional Republicans on budget votes, the timing relative to transitions, and the relationship with the Speaker.

    Additionally, the vault MUST maintain:
    - A thread file ([[domains/usa/threads/us-government-shutdown-crises/_thread]]) tracking the full CR-deadline sequence with dates, vote counts, and intervention events.
    - A concept file ([[domains/usa/concepts/cr-governance-shutdown-dynamics/_concept]]) defining the structural variables (F, D, E, J, T) and the cascade-stage model.
    - A procedure file ([[domains/usa/procedures/us-government-shutdown-forecast]]) with a calibrated probability algorithm.

    The threshold for "systematic coverage" in Rule 10 is not met by a paragraph in a quarter file. It requires graph-structure coverage with entities, threads, concepts, and procedures that can be loaded and applied on question arrival. A vault with no entity stub for the House Speaker, no thread for shutdown cycles, and no concept for CR-governance dynamics is not in compliance with Rule 10 regardless of what the quarter files contain. This rule is established by Question 37 (gold_37, US government shutdown, correct YES) — the correct prediction was supported by general knowledge, not vault content, because the vault had none of these files.

10b. **US debt ceiling forecasting requires dedicated thread, concept, entities, and procedure**: The debt ceiling is a distinct forecasting domain from government shutdowns. While both involve US budget brinkmanship, the debt ceiling operates through fundamentally different mechanics — extraordinary measures, X-date estimation, suspension vs. increase, reconciliation pathway — that require separate analytical frameworks. The vault MUST maintain:

    - **A thread file** ([[domains/economics/threads/us-debt-ceiling-crises/_thread]]) tracking the full sequence of debt ceiling episodes (2011-present), including the FRA 2023 suspension/reinstatement cycle, the Dec 2024 CR with failed debt ceiling suspension, and the 2025 reconciliation resolution.
    
    - **A concept file** ([[domains/usa/concepts/debt-ceiling-mechanics/_concept]]) documenting the six-factor VWUAPE forecasting model (Vehicle, Window, Urgency/X-date, Alignment, Political cost, Economic pressure) and the key institutional mechanics (extraordinary measures, X-date estimation, suspension vs. increase, reconciliation pathway).
    
    - **Entity stubs** for the US Treasury Department ([[domains/usa/entities/us-department-of-treasury]]) as the actor administering extraordinary measures and communicating X-date estimates, and the Treasury Secretary ([[domains/usa/entities/janet-yellen]] for 2021-2025, [[domains/usa/entities/scott-bessent]] for 2025+) as the author of X-date public letters.
    
    - **A procedure file** ([[domains/usa/procedures/debt-ceiling-forecast]]) with the step-by-step pre-audit (debt ceiling state, vehicle existence, X-date proximity, six-factor model) and the 8-step calibration methodology.
    
    The threshold for "systematic debt ceiling coverage" in Rule 10 is not met by mentioning the debt ceiling in the budget-brinkmanship concept or noting the Dec 2024 CR fight in a quarter file. It requires a dedicated thread tracking debt ceiling episodes with causal chains, a concept defining the distinct mechanics, entity stubs for the institutional actors, and a procedure that can be loaded and applied on question arrival. A vault with no debt ceiling thread, no extraordinary-measures concept, and no Treasury entity stub is not in compliance with this rule regardless of how much shutdown coverage exists. This rule is established by Question 45 (debt ceiling raised/suspended by inauguration, correct NO) — the correct prediction was supported by general knowledge about the failed CR vehicle and the short transition window, not vault content, because the vault had none of these files.

10c. **Lame-duck legislative feasibility is a mandatory pre-forecast check for any question with a post-election window**: When a forecasting question asks about a legislative action (debt ceiling change, major reform, treaty ratification, budget deal) occurring between a general election and the seating of the next Congress or inauguration, the vault MUST apply the [[domains/usa/concepts/lame-duck-legislative-feasibility/_concept]] framework before calibrating probability. The key principle: **structural reform has <5% probability in a lame-duck session unless pre-negotiated before the election**. The default expectation for any major legislative action in a lame-duck window is NO.

    The lame-duck framework must be applied to two distinct question types:
    
    - **Routine legislation (CRs, NDAA, disaster relief)**: These are the normal business of lame-duck sessions and can pass if a vehicle exists. The framework's Category 1 actions can have >90% probability.
    - **Structural reform (debt ceiling abolition, entitlement reform, constitutional changes)**: These require months of pre-negotiation, committee work, and institutional consensus. Category 4-5 actions (new proposals) have <5% baseline unless pre-election negotiation is confirmed.
    - **Debt ceiling suspension/increase**: These are Category 1-2 actions in normal circumstances but were Category 3-4 in Dec 2024 because the vehicle failed. The procedure's six-factor model handles the vehicle-dependent probability.
    
    This rule exists because the Q46 question (debt ceiling abolished before Trump inauguration, correct NO) was correctly predicted but the vault had no structured framework for assessing why abolition specifically was impossible in the 18-day lame-duck window. The existing debt-ceiling-mechanics concept categorized abolition as a mechanism type but provided no probability framework. The lame-duck legislative feasibility concept fills that gap. Every future question about a legislative action in a post-election window must trigger this framework.

11. **Dual-frame analysis**: Every forecast must consider BOTH the affirmative case (why event X will happen) AND the countervailing case (why event X will NOT happen). The most common forecasting errors come from adopting a single frame — seeing only the case for persistence without checking for withdrawal, or only the case for escalation without checking for de-escalation. Concept pairs that are mirror images ([[concepts/leadership-persistence-under-threat]] ↔ [[concepts/incumbent-withdrawal-cascade]]) MUST be assessed together. The procedure step for each domain should explicitly require both frames to be documented in the reasoning.

12. **Stated intentions are unreliable signals for withdrawal forecasts**: Leaders who will ultimately withdraw from a race or office nearly always deny intention to do so publicly and forcefully up to the moment of withdrawal. The denial is both genuine (the leader intends to continue) and strategically performed (admitting vulnerability would accelerate the cascade). This pattern held for Truman (1952), LBJ (1968), and Biden (2024) — all three denied withdrawal intention and all three withdrew. **Forecasters must overweight structural vulnerability conditions** (age, approval, party doubt, legal jeopardy absence, successor availability) and **underweight stated intentions** — the denial pattern is itself a signal of vulnerability rather than resilience. The [[concepts/incumbent-withdrawal-cascade]] concept formalizes this as Stage 0 (Stated-Intention Denial Phase).

13. **Legal proceedings as mandatory coverage**: The vault MUST systematically track legal proceedings against political figures, especially when they overlap with electoral timelines. High-profile criminal cases against political candidates involve procedural dynamics (appeals, stays, immunity doctrines, judicial scheduling) that are independent of the campaign narrative and must be tracked as their own thread. Every named legal actor (prosecutor, judge, defendant) in a forecast question about legal proceedings MUST have a vault entity stub. The [[concepts/judicial-timing-political-deadline]] concept and the [[trump-criminal-cases]] thread establish this minimum coverage. A vault that covers the 2024 election campaign but not the four criminal cases that ran parallel to it is missing material context for questions about trial timing, legal jeopardy, and electoral consequences of prosecution.

13a. **SCOTUS state-court intervention limits are mandatory pre-forecast coverage**: The vault MUST maintain a dedicated concept ([[concepts/adequate-independent-state-grounds]]) documenting the Supreme Court's structurally limited jurisdiction over state court proceedings. When a forecasting question asks whether SCOTUS will block, stay, or review a STATE court action (sentencing, trial, subpoena, execution), the adequate and independent state grounds doctrine is the single most important structural constraint and MUST be assessed before any merits analysis. The default probability for SCOTUS intervention in state proceedings should be <5% absent a strong pure-federal constitutional question.

    Specifically, the vault MUST maintain:
    - A concept file documenting the adequate and independent state grounds doctrine and its forecasting application ([[concepts/adequate-independent-state-grounds]])
    - An updated SCOTUS procedural signals concept that includes the emergency stay application framework and the state-proceeding barrier analysis ([[concepts/scotus-procedural-signals]])
    - An entity stub for the relevant state court system (e.g., [[entities/new-york-state-court-system]]) when the proceeding is in a state whose court structure is unfamiliar or uses confusing naming conventions (e.g., "New York Supreme Court" = trial court)
    - A procedure step in the per-forecast audit (Procedure step 17b) that triggers the state-vs-federal classification and applies the adequate-state-grounds barrier before any SCOTUS intervention forecast

    This rule exists because the SCOTUS hush-money sentencing question (Q49, correct NO prediction) could have been answered with equal accuracy on general knowledge alone — the vault's existing legal coverage (SCOTUS procedural signals, judicial-timing-political-deadline, presidential-sentencing-dynamics) did not directly address whether SCOTUS would block a state sentencing. The correct prediction was based on reasoning about the state-court barrier that was not formalized in any vault file. After this rule, every future question about SCOTUS intervention in a state proceeding will trigger systematic state-vs-federal classification and adequate-state-grounds analysis before forecasting.

14. **Central bank and monetary policy as mandatory coverage**: The vault MUST systematically track central bank policy for both advanced-economy central banks (especially the Federal Reserve) AND systemically important EM central banks (TCMB, RBI, BCB, CBR, etc.), because interest rate decisions are among the most common and consequential forecasting questions in prediction markets. Every contemporary quarter file (post-2020) MUST include a section on central bank monetary policy decisions — rate changes, forward guidance signals, dot plot shifts, and dissenting votes. The Fed's forward guidance apparatus (FOMC statements, press conferences, dot plots) makes rate decisions unusually forecastable compared to most geopolitical questions, so the vault must capture this structured communication pipeline. A vault that covers Middle East escalations and US budget fights but lacks entity stubs for the Federal Reserve, the FOMC, and Jerome Powell, and lacks a concept for [[concepts/central-bank-forward-guidance]], is missing a high-leverage forecasting domain. Entity stubs MUST exist for: the central bank itself, its policy-setting committee, and its chair/governor.

    **EM Central Bank Extension**: For EM central banks that operate under political constraints or have a history of unorthodox policy (e.g., TCMB, CBR, BCB), the vault MUST additionally provide:
    - An entity stub documenting governance structure, political context, and historical policy regimes (see [[domains/mena/entities/turkish-central-bank-tcmb]] as template)
    - A thread tracking monetary policy normalization or cycle (see [[domains/mena/threads/turkish-monetary-policy-normalization]])
    - A concept for the credibility-normalization dynamic (see [[domains/mena/concepts/em-central-bank-credibility-normalization]])
    - Entity stubs for the political shield (finance minister/equivalent) and the central bank governor
    - Updated procedures that account for EM-specific dynamics (see [[domains/economics/procedures/central-bank-rate-decision]])
    
    The vault's central bank coverage is not complete if it only covers advanced-economy central banks. Turkey, Brazil, India, Nigeria, and Egypt all have active prediction market rate questions and all operate under political constraints that make their rate decisions less forecastable by the Fed's forward-guidance framework alone. The vault must have structurally parallel coverage for any central bank that appears in a forecast question — whether the Fed, the ECB, or the TCMB.

15. **Authoritarian election forecasting requires resolution-criteria clarity**: Election questions in authoritarian or semi-authoritarian contexts are among the most deceptive forecasting traps in prediction markets. The term "win" is ambiguous — it can mean receiving the most votes (electoral outcome) or assuming office (political outcome). The vault MUST distinguish these systematically. When a question about an authoritarian election arrives:

    - **Immediately check the resolution text**: Does it say "wins the election" or "takes office" / "is inaugurated"? If the former, the default resolution is based on who actually received the most votes, regardless of who assumes power.
    - **Create entity stubs for ALL named actors in the question**: candidate, incumbent, opposition coalition, electoral commission — before forecasting.
    - **Assess the opposition's vote-monitoring infrastructure**: Without independent tally-collection, the opposition cannot prove its victory. With it (e.g., ConVzla in Venezuela 2024), the true outcome is documentable.
    - **Apply the [[domains/latin-america/concepts/authoritarian-electoral-facade/_concept]] concept**: The regime's institutional control (electoral commission, judiciary, military) determines whether it can falsify results and remain in power, but NOT necessarily whether the opposition can win the vote.
    - **Check whether the regime has barred the most popular opposition candidate**: If yes, apply the [[domains/latin-america/concepts/late-candidate-substitution/_concept]] concept. The replacement candidate may inherit the barred figure's voter support — this dynamic is distinct from the standard electoral-facade pattern and has its own success conditions (credible endorsement, unified coalition, sufficient campaign time, barred figure's continued political activity).
    - **Create entity stubs for ALL named actors**: candidate, incumbent, opposition coalition, electoral commission, AND the key regime figures behind the electoral facade (chief negotiator, hardliner faction leader, electoral commission head) — before forecasting.
    
    A vault that has no entity stub for the president of Venezuela, no thread tracking Venezuelan politics, and no concept for authoritarian electoral dynamics is dangerously incomplete for a question about the Venezuelan presidential election. This rule is established by the Grade 19 error (Venezuela 2024 prediction of NO when González actually won the vote).

16. **Ceasefire date-resolution criteria MUST be systematically tracked**: Questions about ceasefires are among the most common and error-prone forecasting domains. The vault MUST distinguish between three distinct dates for every ceasefire:
    - **Announcement date**: When a party first officially states it has agreed to a ceasefire (executive/PMO statement). This is the date that resolves Polymarket-style "first announcement" questions.
    - **Ratification date**: When the agreement is formally approved by a cabinet, parliament, or governing body. This is an internal process that typically follows 1-7 days after announcement.
    - **Effective date**: When the halt in military engagement actually begins. This may differ from the announcement date by hours or days.
    
    Every ceasefire thread and every forecast-entry about a ceasefire MUST record all three dates separately. Conflating ratification dates with announcement dates is a recurring forecasting error (see gold_16: predicted YES for Oct 9 announcement when actual first announcement was Oct 8). The [[gaza-ceasefire-negotiations-2025]] thread is the canonical example of proper date tracking.

    **CRITICAL ADDENDUM — Temporary vs. Enduring Ceasefire Definition**: The term "ceasefire" in Polymarket resolution criteria means ANY publicly announced and mutually agreed halt in military engagement, regardless of duration, and regardless of whether fighting later resumes. Temporary humanitarian pauses (4-day, 7-day, etc.) qualify as ceasefires under standard resolution criteria UNLESS the resolution text explicitly qualifies the term with "permanent," "comprehensive," "lasting," "end to the war," "ending hostilities," or similar qualifiers. A ceasefire that expires or collapses after its announced duration is still a ceasefire for resolution purposes — the subsequent resumption does not retroactively invalidate it. The November 2023 humanitarian pause is the canonical example: a 4-7 day pause with hostage/prisoner exchanges that expired on Nov 30, but still resolved "YES" for a question about a 2023 ceasefire.

    **Pre-forecast audit for any ceasefire question**:
    - Read the FULL resolution text, not just the title
    - Check for qualifier words: "permanent," "comprehensive," "lasting," "end to the war"
    - If no qualifier present, assume ANY temporary halt qualifies
    - Document whether a temporary pause or humanitarian truce occurred in the relevant window
    - Check media and international organization descriptions — if they call it a "ceasefire," treat it as one for resolution purposes
    - See [[concepts/temporary-vs-enduring-ceasefire]] for the full framework

17. **Political deadlines as ceasefire forcing functions**: Ceasefire timing is systematically influenced by known political deadlines (inaugurations, elections, transitions, legal judgments). The vault MUST track political deadlines as a distinct forcing function alongside leadership decapitation and diplomatic pressure. Every ceasefire thread MUST include a section documenting relevant political deadlines within 3 months of any ceasefire event, and MUST assess whether a [[concepts/political-deadline-ceasefire]] dynamic was active. The key distinction: leadership decapitation creates organizational willingness to negotiate, diplomatic pressure creates external cost for continuing war, but political deadlines create a TEMPORAL COMPRESSION effect — parties accelerate negotiations because the post-deadline environment is uncertain. A ceasefire that overlaps with a known political transition (e.g., January 2025 ceasefire announced 5 days before Trump's inauguration) is structurally different from one that occurs in a stable policy environment. When forecasting a ceasefire-by-[specific date] question, ALWAYS check for political deadlines within the next 3 months and assess whether the deadline effect is accelerating, decelerating, or neutral.

18. **Leadership decapitation as ceasefire leading indicator**: The vault MUST systematically track leadership decapitation events in asymmetric conflicts as potential ceasefire leading indicators. When a non-state armed group's most hardline leader is killed, a 2-4 month window typically opens during which the successor leadership is more willing to accept ceasefire terms. This pattern — formalized in [[concepts/leadership-decapitation-negotiation-window]] — is a high-leverage forecasting input that was embedded in the vault's data but not extracted as a predictive framework until late in the forecast cycle. Every ceasefire thread MUST include a section tracking leadership decapitation events and their timing relative to ceasefire milestones.

18a. **War aims incompatibility as mandatory ceasefire assessment**: The vault MUST systematically assess whether war aims are compatible with a negotiated ceasefire when forecasting asymmetric conflicts (state vs non-state). The strongest single predictor of whether a ceasefire is achievable is whether the stronger party's stated war aim is compatible with a negotiated settlement. Every ceasefire thread MUST include a section documenting:
    - The stronger party's exact official war aim language (not journalist summaries)
    - A classification on the compatibility spectrum (degrade/deter → compatible; remove → partial; destroy/eliminate → incompatible)
    - Whether any resolution pathway (military achievement of aim, redefinition, leadership decapitation, patron imposition) is active
    - How the incompatibility affects the expected ceasefire timeline
    - References to [[concepts/war-aims-incompatibility]] and [[procedures/asymmetric-ceasefire-forecast]]

    A ceasefire forecast that does not reference war aims incompatibility when the stronger party has stated a destruction-oriented war aim is structurally incomplete. The canonical case: Israel's stated aim to "destroy Hamas" made any ceasefire that left Hamas standing politically impossible from Oct 2023 through late 2024 — this was the dominant factor in the Jan-Feb 2024 ceasefire question window and remained the dominant blocking factor until the leadership decapitation cascade (Haniyeh + Sinwar) created a resolution pathway.

19. **VP selection (veepstakes) as mandatory US election coverage**: The vault MUST systematically cover Vice Presidential selection processes in US elections because VP choices are among the most common prediction-market questions within electoral cycles. The VP selection process reveals electoral strategy, factional power within the party, candidate decision-making style, and coalition-building priorities. Every US election thread (post-2020) MUST include coverage of:

    - **The finalist pool**: The 3-6 candidates who undergo formal vetting, with their key strengths and vulnerabilities documented
    - **The elimination cascade**: Which finalists were eliminated and why (vetting problems, state-law conflicts, ideological incompatibility)
    - **The selection model**: Whether the pick follows the balancing model (compensates for nominee's weakness) or the reinforcement model (amplifies nominee's strength) — see [[concepts/veepstakes-electoral-signal]]
    - **Strategic rationale**: What electoral problem the pick was intended to solve
    - **Contrast with the other party's selection**: Paired analysis of both VP picks (when both nominees have selected) provides the richest analytical signal
    - **Exclusion-list question analysis**: "Will another [gender/person] be VP?" questions often include a comprehensive exclusion list of named candidates. Before analyzing the VP selection dynamics, the agent MUST assess whether the exclusion list is exhaustive using the [[concepts/comprehensive-exclusion-list-forecast]] framework. If the list covers ALL plausible candidates of a given category, the question may resolve NO independently of the VP selection process.

    Every named VP finalist who appears in a forecast question MUST have a vault entity stub — even if they were not ultimately selected. The 2024 election demonstrated this need: the Polymarket "Will another man be the 2024 Democratic VP nominee?" question listed Mark Kelly, Josh Shapiro, Roy Cooper, Andy Beshear, Tim Walz, Pete Buttigieg, JB Pritzker, Mark Cuban, Wes Moore, and others — most of whom lacked entity stubs despite being named actors in a forecast question. A vault that tracks presidential campaigns but not the VP selection process that runs parallel to them is missing a high-leverage forecasting domain.

20. **Gender dynamics in ticket composition as mandatory electoral coverage**: The vault MUST systematically cover gender composition dynamics in executive-office selection (presidential tickets, VP selections, and equivalent positions in other democracies) because these are among the most common and deceptive forecasting domains within electoral cycles. Gender imbalances in ticket composition follow asymmetric structural rules that differ for male vs female nominees, and the vault must capture these rules rather than treating "will X gender be VP?" as a generic speculation question. Every US election thread (post-2020) MUST include coverage of:

    - **The nominee's gender as a primary variable**: How did the nominee's gender change the VP selection calculus? For male nominees, picking a woman is coalition-expanding. For female nominees, picking a woman is structurally constrained by the "balanced ticket" convention and risk-aversion on identity concentration.
    
    - **Gender pledges and their reliability**: Did the nominee pledge to pick a woman? For male nominees, such pledges have near-100% compliance (Biden 2020). For female nominees, pledges are less reliable due to strategist pushback and the balancing dynamic (Clinton 2016 did not pledge; Harris 2024 did not pledge).
    
    - **The excluded-list effect**: When a prediction-market question includes an exclusion list of prominent women (e.g., "a woman other than [9 named women]"), this list typically captures the entire viable female pipeline. The probability of a woman outside the list being selected is structurally lower than the probability of a woman ON the list — but accounting for gender balancing, even the women on the list are unlikely to be selected if the nominee is a woman.
    
    - **Paired cross-cycle analysis**: The 2016 (Clinton→Kaine) and 2024 (Harris→Walz) cases form a 2-case pattern that must be documented for cycle-over-cycle comparison. Both female major-party nominees chose male running mates — this is not coincidental but structural.
    
    Every question that asks about a VP pick by gender MUST trigger an assessment using the [[concepts/gender-balancing-ticket-composition]] framework and procedure step 21. A vault that covers veepstakes dynamics without accounting for gender is missing the dominant variable in "will a woman be VP?" style questions.

21. **Thread continuity is mandatory**: All threads with `status: active` MUST be updated in each subsequent quarter file where relevant events occur. Creating a thread and then failing to maintain it is a structural vault failure — it creates the illusion of coverage without the reality. The procedure's Phase 3 (Update Thread Files) is not optional. A thread that has no updates in 2+ consecutive quarters must have its status changed to `fading` or `resolved` with a documented rationale. The quarterly writing checklist MUST include explicit verification of active thread statuses before marking a quarter file as complete.

22. **Financial regulation and securities policy as mandatory contemporary coverage**: The vault MUST systematically track major financial regulatory developments, especially SEC decisions on novel financial products (ETFs, crypto products, market structure changes), because these are recurring domains for prediction market questions. Every contemporary quarter file (post-2020) MUST include a section on significant financial regulatory actions — SEC ETF approvals/denials, SEC enforcement actions against major platforms, Congressional crypto or financial legislation, and significant court rulings on financial regulatory authority. Entity stubs MUST exist for the SEC, its Chair, and any other named regulatory body or official in a forecast question about financial regulation. The SEC's crypto ETF decisions (Bitcoin in January 2024, Ethereum in May-July 2024) are canonical examples of events that must be tracked in quarter files. A vault that tracks Fed rate decisions but not SEC product approval decisions is missing a parallel and comparably forecastable domain.

    The minimum coverage bar for any contemporary quarter with financial regulatory events:
    - SEC ETF approvals or denials with dates and list of products approved
    - Major SEC enforcement actions against publicly traded companies or major platforms
    - Congressional financial legislation with bipartisan or party-line breakdown
    - Federal court rulings on SEC authority (Grayscale v. SEC, etc.)
    - Links to [[sec]] [[us-crypto-regulation]] and [[regulatory-precedent-cascade]] where applicable

    This rule exists because the Ethereum ETF question exposed a vault where the January 2024 Bitcoin ETF approval was mentioned in a single line in the Economy section of 2024-Q1, but the entire May-July 2024 Ethereum ETF regulatory process was completely absent from 2024-Q2 and 2024-Q3 — despite being a major financial market development with direct prediction market relevance. A vault that covers geopolitical crises but not financial regulatory decisions that prediction markets routinely ask about is structurally incomplete.

23. **Pre-election coverage is mandatory in contemporary quarter files**: When writing a contemporary quarter file (post-2020), the agent MUST check whether any major election (presidential, parliamentary, or otherwise consequential) is scheduled within the next two quarters. If an election is upcoming and is forecast-relevant (i.e., could appear in a prediction market question), the quarter file MUST include a dedicated subsection documenting:

    - **Candidate field**: Who has declared/nominated, their party affiliations, and their polling trajectory
    - **Electoral system**: Single-round plurality, two-round runoff, or proportional representation — this determines the structural dynamics
    - **Opposition coordination status**: Whether opposition parties are negotiating alliances, joint tickets, or vote-splitting arrangements
    - **Polling data**: Point-in-time polling snapshots showing each candidate's support level
    - **Key campaign events**: Debates, controversies, withdrawals, foreign interference incidents
    - **Forecasting significance**: A paragraph analyzing what the pre-election evidence implies about the likely outcome

    This rule exists because the most common vault failure for prediction questions is having post-hoc coverage (result documented) with no pre-election coverage (campaign dynamics absent). The Taiwan 2024 election question was the canonical case: the vault had extensive post-election entities and concepts but 2023-Q3 and 2023-Q4 lacked any campaign coverage — meaning the vault could not have supported a pre-election forecast. See [[ko-wen-je]] [[lai-ching-te]] [[hou-yu-ih]] [[taiwan-people-party]] [[kuomintang]] [[democratic-progressive-party]] and [[concepts/divided-opposition-plurality-win]].

24. **Candidate count and opposition fragmentation are mandatory pre-forecast assessments for ANY election question**: Single-round plurality elections with 3+ credible candidates produce structurally different outcomes than two-candidate races. The vault MUST assess opposition fragmentation dynamics before every forecast about a presidential or legislative election in a single-round plurality system. Specifically:
    - **Count the number of credible candidates** (those polling >10%). If 3+ candidates are viable, the opposition fragmentation effect is active.
    - **Check for opposition alliance negotiations**: Are the trailing candidates discussing a joint ticket? If negotiations are reported but unresolved, monitor. If they have failed or registration deadlines have passed, fragmentation is locked in.
    - **Assess the electoral system**: Does the final election use single-round plurality (first-past-the-post) or a two-round runoff? Only single-round plurality is subject to fragmentation-driven plurality wins.
    - **Apply the [[concepts/divided-opposition-plurality-win]] framework**: For front-runners polling at 30-45% with a split opposition, forecast a win at 85-95% confidence absent a last-minute opposition consolidation.
    - **Create entity stubs for ALL candidates** named in the question and all major party organizations. The 2024 Taiwan election question named Lai Ching-te (no stub), the DPP (no stub), and implicitly involved Kou Wen-je, Hou Yu-ih, the KMT, and the TPP — none of which had vault files. This is the minimum coverage bar.
    - **Document the structural rationale in the reasoning**: Unlike policy-driven forecasting (which requires analyzing voters, platforms, and issues), fragmentation-driven forecasting is structural — it follows from the electoral system and the candidate count. The reasoning should explicitly state the structural mechanics.

    A vault that covers candidate platforms but not the number of viable candidates or the electoral system's fragmentation mechanics is missing the dominant variable in single-round plurality elections. Rule 24 establishes that candidate count and opposition coordination status are mandatory pre-forecast data points, equivalent to checking the electoral system type before the question type.

25. **Statutory and regulatory deadlines as forcing functions**: When a regulatory agency is compelled by a court ruling to approve a product class, the specific approval date is determined by the applicant with the earliest statutory deadline — not by agency discretion. SEC rules give the agency 240 days to decide on ETF applications; the applicant with the earliest filing date has the earliest final deadline. This deadline converts legal compulsion into a concrete date. The vault MUST track statutory deadlines alongside court rulings in regulatory precedent cascades. Every [[concepts/regulatory-precedent-cascade]] entry MUST document:
    - The relevant statutory deadline for the first applicant in the cascade

26. **State-level electoral reliability as mandatory US election coverage**: For any question about a party winning a specific US state in a presidential election, the vault MUST provide state-level classification using the [[domains/usa/concepts/state-electoral-reliability/_concept]] framework. The 2024 election confirms a structural pattern: ~43 of 50 states have near-deterministic presidential outcomes based on their partisan baseline, with only ~7 swing states being genuinely competitive. Questions about non-swing states (e.g., "Will a Republican win New Mexico?") are effectively asking about national landslide scenarios and must be analyzed as such, not treated as generic competitive elections. Every state-level forecast MUST document:

    - **The state's current category** (Safe D/Likely D/Lean D/Tossup/Lean R/Likely R/Safe R) based on recent election results
    - **The national popular vote margin required** to flip the state (flip threshold ≈ state's Cook PVI × 2)
    - **Whether that margin is plausible** given the national polling environment
    - **Whether any major forecaster rated the state competitive** during the cycle

    This rule exists because the New Mexico question in the PIT blind test (gold_115) was answered correctly on general knowledge alone — the vault had no state-level classification framework, no entity stub for New Mexico, and no concept for state electoral reliability. The vault made no structural contribution to the prediction. After this rule, any future question about a party winning a US state will trigger systematic state-level classification that would be impossible to ignore.
    - How the deadline interacts with the court ruling's timing
    - Whether the agency could delay past the deadline (e.g., by requesting more information or reopening comment periods)
    - The applicant with the earliest deadline and their institutional track record

    This rule exists because forecasting "will [product] be approved by [date]?" is fundamentally different from forecasting "will [product] be approved at some point?" — the deadline determines the date, the court ruling determines the inevitability. The Bitcoin ETF approval (January 10, 2024 = ARK 21Shares statutory deadline) and Ethereum ETF approval (May 23, 2024 = VanEck/BlackRock 19b-4 deadline) are canonical examples. A vault that tracks court rulings but not the statutory deadlines that convert them into dates is missing the temporal forecasting variable.

    **MANDATORY PROCEDURE LOADING**: Any question involving SEC approval of a novel financial product MUST trigger loading of the [[domains/economics/procedures/sec-product-approval-forecast]] procedure. This is not optional. The procedure contains the step-by-step workflow (precedent chain check, statutory deadline identification, institutional tier analysis, regulatory stage distinction, probability calibration) that the spec's structural rules alone do not enforce as a runtime process. If the procedure references entity stubs ([[van-eck]], [[fidelity]], [[blackrock]], [[ark-invest]], [[grayscale]], [[sec]], [[gary-gensler]]), those MUST also be loaded. Failure to load the procedure before forecasting is treated as a procedure violation — the forecast cannot claim vault support if the vault's own analytical method was not activated.

26. **Institutional applicant identity as regulatory leading indicator**: Not all applicants for regulatory approval are equal. The vault MUST distinguish between:

27. **Presidential term-continuity assessment for US leader-status questions**: Questions asking "Will [President] be [title/position] on [date]?" are among the most common political forecasting domains and follow a systematic structural pattern that the vault MUST formalize. At any point during a US president's term, the default state is continuity (>95% baseline probability of remaining in office at any given future date within the term). The forecasting task is to identify and assess the small set of removal mechanisms that could disrupt this default. The vault MUST maintain a [[concepts/us-presidential-term-continuity]] concept file documenting the five removal mechanisms (death in office, resignation, impeachment + conviction, 25th Amendment Section 4, assassination) with historical frequencies, baseline probabilities, and key leading indicators. Before forecasting any "will X be president on [date]?" question:

    - **Map all available removal mechanisms**: For each of the five mechanisms, assess whether it is plausibly activatable within the timeframe. Most of the time, none will be active — document this explicitly.
    - **Start from baseline continuity probability**: Do NOT default to "I don't know" — start from >95% and adjust downward only upon evidence of a specific mechanism.
    - **Check age and health**: The most common deviation from baseline is presidential age (60-70 → slight elevation; 70-80 → moderate; 80+ → significant). Document any public health indicators.
    - **Check impeachment status**: If the House has launched an inquiry or articles have been introduced, document the House and Senate compositions and assess conviction probability. Apply the [[concepts/impeachment-inquiry-failure-mode]] framework to determine whether the inquiry is structurally likely to produce articles — a narrow majority, lack of direct evidence, split-committee investigation, election proximity, and Senate conviction impossibility all predict inquiry failure regardless of the inquiry's public intensity.
    - **Check resignation pressure**: Internal party calls for resignation, combined with a trigger event and no legal jeopardy, create the [[concepts/incumbent-withdrawal-cascade]] pattern. Document presence or absence.
    - **Document the absence of mechanisms explicitly**: The most common correct forecast will be "yes, he's still president." The vault's reasoning must still demonstrate that all five mechanisms were checked and found inactive. An unexplained "yes" is not a vault-supported forecast.
    - **Create entity stubs for removal mechanism actors**: If impeachment is active, entity stubs MUST exist for the House Speaker, relevant committee chairs, and lead House managers. If a resignation crisis is active, entity stubs MUST exist for party leaders signaling pressure.

    Every contemporary quarter file MUST reference the [[concepts/us-presidential-term-continuity]] concept at least once per year to document whether any removal mechanisms were active. This rule exists because the most fundamental forecasting question in US politics — "will the president be president on [date]?" — was completely unsupported by the vault, which had no concept file, no entity stubs for removal mechanisms, and no thread tracking the president who was the subject of the question. A vault that can forecast Taiwan elections and eurozone inflation but cannot systematically answer "will the US president still be president?" has a foundational gap in its US political coverage.

28. **Eurozone macroeconomic coverage as mandatory parallel to US coverage**: The vault MUST maintain Eurozone macroeconomic coverage structurally parallel to US macroeconomic coverage, because (a) the eurozone is an economic bloc of comparable global weight (~$15T GDP), (b) prediction markets routinely ask about EU inflation, ECB rate decisions, and eurozone growth indicators, and (c) the vault's US-centric coverage creates a blind spot for the second-largest economy. Specifically:

29. **Commodity price spike dynamics as mandatory coverage**: The vault MUST systematically cover the dynamics of commodity price spikes driven by geopolitical shocks, because (a) these are among the most common and financially consequential prediction market questions, (b) the spike-reversion pattern follows a consistent structural sequence with high forecasting value, and (c) without this coverage, forecasters default to simple extrapolation of the spike (predicting the spike persists) or contrarian reversion (predicting immediate normalization) without structured reasoning. The vault MUST maintain:

    - **A commodity spike-reversion concept** ([[concepts/geopolitical-commodity-spike-reversion]]) documenting the four-phase pattern: Trigger → Fear Spike → Reality Calibration → New Equilibrium, with calibration rates and key variables
    - **Entity stubs for the global benchmark** (Brent Crude) and US benchmark (WTI) — parallel to the Fed/SEC coverage requirements
    - **Entity stubs for the key institutional actors**: OPEC+ (supply management), IEA (coordinated SPR releases), and relevant national SPR programs
    - **A concept for the strategic petroleum reserve mechanism** ([[concepts/strategic-petroleum-reserve]]) documenting its price suppression effect, anticipation dynamics, and diminishing returns
    - **A thread for the canonical modern oil market crisis** (2022) with quarterly price ranges, key supply/demand events, and policy responses
    - **A procedure for oil price forecasting** ([[procedures/oil-price-forecast]]) formalizing the 7-step assessment framework

    Every contemporary quarter file covering a period that includes a significant commodity price shock (oil crossing $100, natural gas price surge, critical mineral supply disruption) MUST document:
    - The price trajectory (baseline → spike → calibration → new equilibrium)
    - The trigger event and its duration
    - Policy responses (SPR releases, sanctions, price caps)
    - The calibration timeline and key reversion drivers

    This rule exists because Question 3 of the PIT blind test (crude oil >= $115 on March 15, 2022) was correctly predicted but the vault contributed zero signal: no Brent Crude entity (despite WTI existing), no commodity spike concept, no IEA entity, no OPEC+ entity, no global oil market thread, and no forecast procedure. The correct prediction was based on general knowledge of the Feb-Mar 2022 oil spike and reversion pattern, not vault content. The vault must provide structured, non-trivial signal for every future commodity price question.

    - **ECB entity stub**: An entity stub MUST exist for the European Central Bank, its President (Christine Lagarde), and its policy-setting Governing Council — parallel to the Fed/Jerome Powell/FOMC stubs required by Rule 14.
    - **Eurozone macro thread**: A thread MUST exist ([[threads/eurozone-macro-economic-indicators]]) tracking eurozone HICP inflation, ECB policy rates, GDP growth, unemployment, and energy prices — parallel to the US macro thread required by implicit Rules 4/14 precedent.
    - **HICP tracking**: Every contemporary quarter file MUST include eurozone HICP inflation data (flash estimate and final) as a standard data point, parallel to US CPI/PCE tracking.
    - **HICP/Eurostat concept**: A concept file MUST exist ([[concepts/hicp-eurostat-inflation-measurement]]) documenting how HICP differs from US CPI — particularly the higher energy weight, the exclusion of owner-occupied housing, and the monthly flash publication cycle.
    - **Post-COVID inflation concept**: A concept file MUST exist ([[concepts/post-covid-inflation-surge]]) documenting the causal chain of base effects, supply-chain disruption, energy price shock, pent-up demand, and labor market tightening that drove the 2021-2023 global inflation cycle.
    - **ECB forward guidance tracking**: Every quarter file covering 2021+ MUST document the ECB's forward guidance stance (rate path signals, PEPP/APP purchase trajectories, TLTRO conditions) parallel to Fed dot-plot and press-conference coverage.

    The minimum coverage bar for any contemporary quarter:
    - Eurozone HICP headline and core rates (flash estimate date, final value)
    - ECB policy rate decisions (any change to DFR, MRO, or MLF rates)
    - Key ECB communication (Governing Council statement, Lagarde press conference signals)
    - Energy price context (TTF gas, Brent oil) as primary driver of eurozone inflation
    - Links to [[entities/european-central-bank]] [[entities/christine-lagarde]] [[threads/eurozone-macro-economic-indicators]] [[concepts/hicp-eurostat-inflation-measurement]]

    This rule exists because Question 1 of the PIT blind test (EU HICP inflation >= 4.3% in October 2021) was correctly predicted but the vault contributed zero signal: no ECB entity, no Lagarde entity, no eurozone macro thread, no 2021 quarter files, no HICP concept, no post-COVID inflation concept. The vault had robust US macro coverage but literally nothing for the eurozone — a structural imbalance that this rule remediates. Every future eurozone inflation or ECB rate question must find a vault with parallel analytical depth to what exists for the US.
    - **Incumbent institutional applicants**: Firms with deep regulatory credibility and a long track record of approved applications (e.g., BlackRock with 575+ ETF approvals and nearly zero denials). Their applications are leading indicators that raise the probability of approval because the agency fears inconsistency and reputational harm from denying a well-regarded incumbent.
    - **Crypto-native or novel applicants**: Firms without an established regulatory track record (e.g., Grayscale, ARK Invest in crypto). Their applications face higher skepticism and their legal challenges are necessary to compel approval, but they lack the institutional credibility to raise the baseline approval probability on their own.
    
    The canonical pattern: a regulatory logjam on a novel product class breaks when (a) a court rules against the agency AND (b) an incumbent institutional applicant enters the race. Either factor alone is insufficient — the court ruling provides the legal compulsion, the incumbent applicant provides the institutional credibility. The vault MUST track which institutional applicants have entered a regulatory race and document their historical approval track record. Entity stubs for major institutional financial firms (BlackRock, Fidelity, Vanguard, State Street) SHOULD include their ETF approval track record as a forecasting-relevant data point.

30. **Aging-incumbent pre-trigger vulnerability assessment as mandatory coverage**: The vault MUST systematically assess pre-trigger vulnerability for any incumbent leader aged 70+ who is the subject of a forecasting question about term continuity, re-election, or withdrawal. The gold_12 error (Biden dropout 2024, predicted NO at Q2 2023 cutoff, actual YES) demonstrated the danger of applying only a persistence frame to an aging incumbent: Biden had all six vulnerability signals present even at Q2 2023 (age 81, no legal jeopardy, party doubt, low approval, successor ready, party not restructured around him), but the forecaster saw no trigger event and predicted NO — missing the 40-55% cumulative trigger probability over a 10-month horizon. The vault MUST maintain:

    - **An aging-incumbent early-warning procedure** ([[domains/usa/procedures/proc-aging-incumbent-early-warning]]) formalizing the 6-signal vulnerability inventory, cumulative trigger probability calculation, stated-intention discount, and trigger scenario simulation.
    - **A concept file for the withdrawal cascade** ([[domains/usa/concepts/incumbent-withdrawal-cascade]]) that documents both pre-trigger vulnerability assessment AND post-trigger cascade velocity — not only retrospective pattern recognition but proactive vulnerability detection.
    - **Entity stubs for all canonical withdrawal cases**: Every canonical case of an incumbent withdrawing under party pressure (Truman 1952, LBJ 1968, Biden 2024) MUST have an entity stub that explicitly connects to the withdrawal-cascade concept and documents the pre-trigger vulnerability signals that were visible before the trigger event. These entity stubs serve as analogical references for any future aging-incumbent question.
    - **Dual-frame documentation in every aging-incumbent forecast**: The reasoning MUST document BOTH the persistence case (why the leader might stay) AND the withdrawal case (cumulative trigger probability over the horizon). The most common error is adopting a single "stable" frame and ignoring the cumulative trigger risk.
    - **Stated-intention discount rule embedded in the procedure**: A leader's public statements denying any intention to withdraw MUST be discounted as evidence (they are Stage 0 behavior per the cascade concept, observed in all three canonical withdrawers). Structural vulnerability signals (age, approval, party doubt, legal jeopardy absence) MUST be overweighted relative to public statements.
    - **Cumulative trigger probability model**: For any aging incumbent (70+) with 4+ vulnerability signals, the pre-trigger assessment MUST include a compound probability calculation over the forecast horizon. The base rate is 5-8% per month trigger risk for an 80+ leader; over a 10-month horizon, this yields 40-55% cumulative probability of at least one trigger event.

    Every contemporary quarter file covering a period with an aging incumbent leader (70+) in office MUST document the pre-trigger vulnerability signals visible at that time, even if no trigger event has yet occurred. The vault's coverage of an aging-incumbent term is incomplete if it only documents the withdrawal event itself without tracking the vulnerability signals that preceded it. A vault that captures the exact date of Biden's withdrawal but has no pre-trigger vulnerability analysis in the preceding quarters is missing the forecasting-relevant content — the vulnerability trajectory, not just the outcome event.

31. **Impeachment-specific forecasting as mandatory coverage**: Questions asking "Will [President] be impeached before [date]?" or "Will articles of impeachment be approved before [date]?" follow a distinct structural pattern that differs from term-continuity questions. The vault MUST maintain a [[concepts/impeachment-inquiry-failure-mode]] concept file documenting the distinction between impeachment inquiries and articles of impeachment, and the structural factors that determine whether an inquiry produces articles. This rule exists because the most common forecasting error in this domain is treating an impeachment inquiry as a leading indicator for articles when it is a weak signal at best. Before forecasting any "will X be impeached on [date]?" question:

    - **Determine the stage**: Has the House launched an inquiry? Introduced articles? Held a floor vote? The stage determines the baseline probability — an inquiry means articles are possible but unlikely; articles introduced means the question is whether the House will pass them; articles passed means the question shifts to Senate conviction.

    - **Apply the five structural factors**: Using the [[concepts/impeachment-inquiry-failure-mode]] framework, assess (1) majority margin, (2) direct evidence linking the president, (3) committee unity, (4) election proximity, and (5) Senate composition. If 3+ factors predict failure, the inquiry's existence does not materially elevate articles probability.

    - **Differentiate inquiry-based from evidence-based impeachment**: Impeachment driven by direct presidential action (Trump-Ukraine call, Clinton-Lewinsky perjury, Nixon tapes) has fundamentally different probability than impeachment driven by indirect association (Biden family business dealings). The distance between the president and the alleged misconduct is the single strongest predictor of articles probability.

    - **Track Speaker ownership**: A Speaker who initiates an inquiry personally has higher incentive to see it through. A Speaker who inherits an inquiry from a predecessor has lower incentive. Speaker succession mid-inquiry typically reduces articles probability.

    - **Create entity stubs for inquiry leadership**: If an impeachment inquiry is active, entity stubs MUST exist for the relevant committee chairs (Oversight, Judiciary), the House Speaker, and any central figures in the investigation. These stubs MUST document the chair's approach to the inquiry (methodical vs. combative), which correlates with articles probability.

    Every contemporary quarter file covering a period with an active impeachment inquiry MUST reference the [[concepts/impeachment-inquiry-failure-mode]] concept and document which of the five structural factors are present. The quarter file MUST also document whether articles were introduced or voted on, and if not, explain which factors prevented it. This rule exists because the Biden impeachment inquiry (September 2023-June 2024) was the most significant active impeachment process in US politics during the 2023-2024 period, yet the vault had no dedicated concept, no entity stubs for the key actors, and minimal thread coverage — despite the vault correctly forecasting NO to "Biden impeached before 2024 election." The vault's correct prediction was based on general knowledge, not vault content, which violates the "no freebie predictions" principle.

32. **Latin American politics as mandatory coverage**: The vault MUST systematically cover Latin American political developments, especially major elections and dominant-party dynamics in the region's largest economies (Mexico, Brazil, Argentina, Chile, Colombia), because (a) Latin American presidential elections are among the most common forecasting questions in prediction markets outside US politics, (b) the region's dominant-party systems and populist cycles create recurring structural patterns with high forecasting value, and (c) the vault had zero Latin American coverage at the time of the Mexican presidential election question — a structural blind spot for an entire continent with ~650M people.

    The vault MUST maintain:
    - **A Latin America domain entry** ([[domains/latin-america/_domain]]) documenting the region's key forecasting-relevant characteristics (dominant-party dynamics, commodity dependence, left-right pendulum swings, criminal violence / state capacity gaps) and listing major countries with year-by-year forecasting relevance.
    - **Entity stubs for ALL named actors in any Latin American forecast question**: president, candidate, party, electoral commission, central bank governor. The minimum coverage bar is the named-entity set from the question itself plus the region's major institutional actors (rotating-party alliances, dominant party, opposition coalition). For the Mexican 2024 question, this meant entity stubs for [[domains/latin-america/entities/claudia-sheinbaum]], [[domains/latin-america/entities/andres-manuel-lopez-obrador]], [[domains/latin-america/entities/morena]], [[domains/latin-america/entities/xochitl-galvez]], [[domains/latin-america/entities/pan]], [[domains/latin-america/entities/pri]], [[domains/latin-america/entities/prd]], and [[domains/latin-america/entities/jorge-alvarez-maynez]] — none of which existed.
    - **A concept for incumbent-party successor dominance** ([[domains/latin-america/concepts/incumbent-successor-dominant-party/_concept]]) documenting the structural advantage of a dominant-party successor when the outgoing incumbent is term-limited and popular (approval >55%). The canonical case: Mexico 2024 (AMLO 60% approval → Sheinbaum 60% vote share). This concept is distinct from [[domains/east-asia/concepts/divided-opposition-plurality-win/_concept]] because the dominant mechanism is positive approval transfer, not opposition fragmentation.
    - **A procedure for dominant-party election forecasting** ([[domains/latin-america/procedures/dominant-party-election-forecast]]) formalizing the 5-phase assessment: system identification → approval transfer → opposition assessment → structural factors → probability calibration.
    - **A thread tracking the dominant-party consolidation** ([[domains/latin-america/threads/mexican-politics/_thread]]) for the country in question, documenting electoral outcomes, institutional control metrics, and opposition dynamics.

    **MANDATORY PRE-FORECAST CHECKS for any Latin American election question**:
    1. Determine the electoral system (single-round plurality, two-round runoff, or PR). In single-round plurality systems, dominant-party candidates are especially hard to dislodge.
    2. Measure the outgoing incumbent's approval rating. If >55%, the successor dominance framework applies; if <40%, the "successor penalty" applies (the opposition is favored).
    3. Count the credible opposition candidates. Each additional candidate in a single-round plurality system further elevates the dominant party's win probability, but the primary driver remains the approval transfer effect.
    4. Assess opposition coalition coherence. An ideologically incoherent coalition (e.g., PAN-PRI-PRD in Mexico 2024) is structurally weaker than a coherent one.
    5. Distinguish "winning the vote" from "assuming office" if the question involves an authoritarian or semi-authoritarian context (Venezuela, Nicaragua) — apply Spec Rule 15 (authoritarian election forecasting) in parallel.

    This rule exists because Question 13 of the PIT blind test (Mexican presidential election: will Sheinbaum win?) was correctly predicted but the vault contributed zero signal: no Latin America domain, no entity stubs for any Mexican political actor, no thread tracking Mexican politics, no concept for the incumbent-successor dynamic that structurally determined the outcome. The correct prediction was based on general knowledge, not vault content — a violation of the "no freebie predictions" principle (Spec Rule 8). Every future Latin American election question must find a vault with substantive, structured coverage of the region's political dynamics and key actors, parallel to the coverage depth that exists for the US, East Asia, Europe, and MENA.

    **Entity stub creation is mandatory before forecasting**: If a Latin American candidate, party, or institution appears in a question's named-entity set and lacks a vault stub, the stub MUST be created before the forecast is rendered. This is the minimum coverage bar — the named entities in the question define the floor, not the ceiling, of required vault coverage.

    **Cross-application with existing rules**: Latin American elections in authoritarian contexts (Venezuela, Nicaragua) require parallel application of Spec Rule 15 (authoritarian election forecasting, resolving the "wins" vs "assumes office" ambiguity). Latin American elections in dominant-party systems (Mexico, potentially Bolivia) require parallel application of this rule's successor-dominance framework. The two frameworks are not mutually exclusive — Venezuela combines authoritarian electoral dynamics with a dominant-party successor (Maduro's post-Chávez succession), but the dominant mechanism is regime coercion rather than popular approval transfer.

33. **DOJ internal policies as mandatory coverage for federal prosecution questions**: The vault MUST systematically track the US Department of Justice's internal policies that constrain federal prosecutions, because these policies are often more consequential for case outcomes than substantive criminal law. Every forecast question about a federal prosecution of a political figure MUST reference:

    - **The OLC doctrine on presidential immunity from prosecution**: The Office of Legal Counsel's opinions (1973, 2000) that a sitting president cannot be indicted or prosecuted are the actual mechanism that moots federal cases against presidential winners. This is distinct from constitutional immunity (SCOTUS Trump v. United States, July 2024) and operates independently of any court ruling. The [[entities/doj-office-of-legal-counsel]] entity documents this doctrine.

    - **The 60-day rule**: DOJ policy against election-influencing investigative steps within 60 days of a federal election. This creates a hard institutional deadline that defense teams can exploit.

    - **The Special Counsel regulations (28 CFR Part 600)**: The framework governing independently appointed prosecutors. Special Counsels have operational autonomy but remain bound by OLC doctrine and DOJ guidelines.

    - **State vs. federal distinction**: DOJ policies (OLC doctrine, 60-day rule) apply ONLY to federal prosecutions. State prosecutors operate under independent state law and are not bound by any of these constraints. The NY hush-money case (pre-election conviction) demonstrates that state cases can proceed where federal cases cannot.

    Every named DOJ entity appearing in a forecast question (Special Counsel, Attorney General, OLC) MUST have a vault entity stub. The [[entities/us-department-of-justice]] entity provides the institutional documentation. The [[entities/doj-office-of-legal-counsel]] entity provides the OLC doctrine documentation. The [[functions/estimate-legal-timeline]] function provides the structured timeline estimation tool. A vault that covers the electoral campaign but not the DOJ policies that determined the outcome of parallel criminal cases is missing material information for questions about legal case timing, sentencing, and post-election legal jeopardy.

    **MANDATORY PRE-FORECAST CHECKS for any federal prosecution timing question**:
    1. Determine whether the defendant is a current or former officeholder. If current or plausibly future president, the OLC doctrine applies post-election.
    2. Assess electoral viability: Is the defendant a competitive candidate who could win the presidency? If yes, federal prosecution will end upon victory regardless of pre-election legal posture.
    3. Check for active appeals: Is there an interlocutory appeal on immunity or jurisdictional grounds? If yes and it carries an automatic stay, the trial is paused until the appeal resolves.
    4. Calculate the appellate timeline: Estimate minimum delay from appeal filing to final ruling (including remand). Compare to the election deadline.
    5. Distinguish state vs. federal: If the case is in state court, none of the above DOJ policies apply. State cases proceed independently of federal immunity doctrines.
    6. Apply the [[functions/estimate-legal-timeline]] function for structured timeline calculation.
    7. Load the [[procedures/proc-legal-timeline-estimation]] procedure before forecasting — failure to activate the procedure is a procedure violation.

    This rule exists because the Trump election interference trial timing question was correctly predicted, but the vault at the time of the initial forecast had no structured DOJ policy coverage, no OLC entity, no DOJ entity, and no legal timeline estimation function. The correct prediction was supported by the trump-criminal-cases thread and judicial-timing-political-deadline concept (created in a prior reflection), but the institutional policies that actually mooted the federal cases — the OLC opinions preventing prosecution of a sitting president — were only implicitly referenced. Future federal prosecution timing questions must find explicit DOJ policy documentation in the vault.

34. **Post-nomination persistence baseline as mandatory pre-forecast assessment for US candidate withdrawal questions**: For ANY question asking whether a US presidential candidate will withdraw from the race, the vault MUST first classify the candidate type and apply the appropriate governing framework. The single most important structural variable is nomination status combined with incumbency. The vault MUST maintain:

    - **A post-nomination persistence baseline concept** ([[concepts/post-nomination-persistence-baseline]]) documenting the structural baseline: since the modern primary system (1972), zero non-incumbent presumptive major-party nominees have withdrawn. This baseline is the dominant variable for any non-incumbent who has secured the nomination — it overrides all other factors (polling, donor confidence, internal pressure) except total incapacitation.
    
    - **An integrated candidate withdrawal probability procedure** ([[procedures/candidate-withdrawal-probability]]) synthesizing the three governing frameworks: post-nomination baseline (non-incumbents), leadership-persistence-under-threat (legal jeopardy present), and incumbent-withdrawal-cascade (internal pressure absent + trigger). This procedure MUST be loaded before any candidate-withdrawal forecast.
    
    - **A nomination-status gate** in every candidate-withdrawal forecast: has the candidate secured the nomination? If yes and non-incumbent → the baseline is <1% withdrawal probability. This gate must be the first step in the reasoning, not an afterthought.
    
    - **Candidate-type differentiation**: The vault MUST distinguish between non-incumbent nominees (governed by the post-nomination baseline), incumbent nominees (governed by the withdrawal-cascade concept), and primary-phase candidates (governed by general leadership-persistence dynamics). Applying the wrong framework to the wrong candidate type is the most common forecasting error in this domain — using the incumbent-withdrawal-cascade for a non-incumbent overestimates withdrawal probability by an order of magnitude.
    
    - **Historical reference cases**: Every withdrawal question MUST document the relevant historical baseline — the complete list of presumptive nominees since 1972, which withdrawals occurred (3 incumbents: Truman, LBJ, Biden) and which did not (all 12 non-incumbents). This historical reference anchors the probability estimate.
    
    Every candidate withdrawal forecast MUST document:
    - The candidate type classification (non-incumbent nominee / incumbent nominee / primary-phase candidate)
    - The governing framework and why it applies
    - Nomination status (clinched / presumptive / primary-phase)
    - Legal jeopardy status (present / absent) — the binary gate
    - Internal pressure assessment and its intensity
    - A stated-intention discount statement explicitly noting that the candidate's denials carry zero evidentiary weight
    - The combined probability estimate with supporting reasoning
    
    This rule exists because the Trump dropout question (question 22, correct NO prediction) was correctly predicted but the vault provided support only through the leadership-persistence-under-threat and incumbent-withdrawal-cascade concepts — neither of which documents the structural baseline that zero non-incumbent nominees have ever withdrawn since 1972. The vault's forecast would have been equally correct if it had relied on this baseline alone, without any of the additional legal-jeopardy or threat-hardening analysis. The post-nomination baseline is the most foundational variable for this question type, and its absence from the vault was a structural gap.

35. **Candidate-type framework differentiation as mandatory pre-forecast check for ANY withdrawal question**: When a question asks whether a political leader (any country, not just the US) will withdraw from a race or office, the vault MUST explicitly classify the leader's situation against the three governing frameworks before estimating probability:

    - **Framework A: Post-nomination/Post-selection structural lock-in** — Applies when the candidate has secured the party's nomination or formal selection and is NOT the incumbent. The dominant variable is the institutional lock-in of delegate commitments, ballot access, campaign infrastructure, and fundraising mechanics. Withdrawal is structurally nearly impossible unless the candidate faces total incapacitation.
    
    - **Framework B: Leadership persistence under threat** — Applies when the leader faces compounding legal jeopardy (pending charges, convictions, credible prosecution threat after loss of office). Legal jeopardy creates existential motivation to retain office, making the persistence baseline >95% regardless of other factors. This can compound with Framework A (as in Trump 2024) or override Framework C (as would apply to an incumbent facing charges).
    
    - **Framework C: Internal pressure withdrawal cascade** — Applies when the leader faces NO legal jeopardy and pressure comes from within their own party/institution. The 5-condition framework (no legal jeopardy, internal pressure present, trigger event occurred, successor ready, weak electoral position) determines withdrawal probability. This framework applies primarily to incumbents or pre-nomination candidates.
    
    The most common forecasting error is applying Framework C to a candidate who is actually governed by Framework A or B — which overestimates withdrawal probability by an order of magnitude. Every withdrawal forecast MUST explicitly state which framework governs the candidate AND why the other two frameworks do not apply.
    
    This rule exists because the difference between non-incumbent nominees and incumbent candidates is not a minor calibration — it is a factor-of-100 difference in baseline probability (from <1% to >70%). A vault that treats "will Trump drop out?" as legally analogous to "will Biden drop out?" without classifying the candidate type is making a structural error even before the first calculation.

36. **No dangling concept/entity references — every wikilink MUST resolve to an existing file**: The graph vault's connectivity depends on every wikilink pointing to an actual file. A wikilink to a concept or entity that does not exist as a file creates a "dead edge" in the graph — the reference implies analytical support that the vault cannot provide, creating an illusion of coverage. This is especially dangerous when a concept is referenced from multiple files (creating an implicit claim of structured knowledge) but has never been written. The following MUST hold at all times:

    - **Every concept name referenced in `related_concepts:` frontmatter or `[[concepts/...]]` inline wikilinks MUST have an existing `_concept.md` file.** If the concept does not yet exist, the reference MUST be removed or the concept file MUST be created before the change is committed.
    - **Every entity name referenced in inline wikilinks from threads, concepts, timeline files, or other entities MUST have an existing entity stub.** If the entity does not yet exist, either create the stub or remove the reference.
    - **Every procedure referenced in `[[procedures/...]]` or `[[.../procedures/...]]` MUST have an existing procedure file.**
    - **Per-forecast audit**: Before or during every per-question reflection, check all concept and entity wikilinks in files that were used during the forecast. If any point to non-existent files, either create the missing files or remove the references.
    - **Batch resolution**: When a new concept or entity file is created, check which other vault files reference that slug and ensure the references resolve correctly (matching directory structure and file naming conventions).

    **Canonical violation found during Question 32 reflection**: The concept `veepstakes-electoral-signal` was referenced from 5+ vault files (the 2024 election thread, the gender-balancing concept, the campaign-pledge concept, the comprehensive-exclusion-list concept, and the spec itself) but had no `_concept.md` file. The vault claimed to have a veepstakes framework, provided links to it, and directed readers to its "calibrated forecasting rules" — but no such framework existed. This is a structural integrity failure that undermines the vault's credibility as a knowledge graph, regardless of whether individual forecasts using the references were correct or wrong.

    **Detection mechanism**: Before any batch of changes is committed, use `search_files` to identify all concept and entity references, then verify each maps to an existing file. This is the minimum automated integrity check for the graph vault.

37. **Withdrawal irreversibility as mandatory pre-forecast assessment for reinstatement questions**: For ANY question asking whether a withdrawn candidate could be reinstated, returned to the race, or re-nominated, the vault MUST apply the [[domains/usa/concepts/nominee-withdrawal-irreversibility]] framework before calibrating probability. The single most important structural variable is that **withdrawal is a one-way door in party-based nomination systems** — once a presumptive nominee has withdrawn, endorsed a successor, and released delegates, reinstatement is procedurally, legally, politically, and institutionally blocked by the four irreversibility locks (delegate release, ballot access transfer, party consolidation, convention ratification). The vault MUST maintain:

    - **A nominee-withdrawal-irreversibility concept** ([[domains/usa/concepts/nominee-withdrawal-irreversibility]]) documenting the four blocking mechanisms, the zero historical precedent, and the distinction from the withdrawal cascade concept (which covers *how* a leader withdraws — this covers *why it can't be reversed*).

    - **An entity stub for the convention body** ([[domains/usa/entities/democratic-national-committee]] for the DNC, or the analogous body for the other party and other countries' political parties) documenting its institutional role as a ratifying body, not a decision-making body, for nominee selection.

    - **A procedural-stage gate**: Has the successor been formally nominated (virtual roll call, convention roll call, or equivalent)? If yes → reinstatement probability is <0.1%. If no → reinstatement is theoretically possible but still structurally unlikely (<5%) because delegate release and party consolidation have already occurred.

    - **Historical zero-base-rate reference**: There is zero precedent for a withdrawn candidate being reinstated in US presidential history, and zero precedent in comparable party-based nomination systems. The Eagleton 1972 case (VP nominee replaced, never reinstated) is the closest analog and confirms the one-way pattern.

    - **Prohibition on treating reinstatement as a generic "political possibility"**: A question about reinstating a withdrawn candidate is NOT a question about political viability, party sentiment, or elite support. It is a question about procedural mechanics — whether any mechanism exists to reverse a one-way institutional process. The procedural analysis (are there any rules or precedents for reinstatement?) dominates all other variables. If no mechanism exists, the probability is <1% regardless of how much party elites might (theoretically) want reinstatement.

    Every reinstatement forecast MUST document:
    - Whether the candidate voluntarily withdrew or was removed
    - Whether a successor has been endorsed, formally nominated, or both
    - Which of the four irreversibility locks are engaged
    - The historical baseline (zero precedents for reinstatement)
    - The procedural gate status — is there any mechanism that could theoretically reverse the replacement?

    This rule exists because the Biden reinstatement question (question 33, correct NO prediction) was a 0%-vault-contribution freebie — general knowledge of "once you drop out you can't come back" was sufficient, but the vault had no structured framework for analyzing reinstatement questions. The concept and entity created in the Question 33 reflection are the minimum coverage bar. Every future reinstatement question must find a vault with explicit reasoning about the irreversibility locks.

38. **US House election dynamics as mandatory coverage**: The vault MUST systematically cover US House of Representatives elections, seat distributions, narrow-majority governing dynamics, and the relationship between presidential and congressional races, because (a) House seat counts and control questions are among the most common US political prediction market questions, (b) the relationship between the national popular vote and seat distribution follows a structural conversion function that can be calibrated in advance, (c) the narrow-majority era (2018-present) produces distinct governing dynamics (Speaker fragility, HFC veto power, bipartisan CR necessity) that differ from previous periods, and (d) the vault's US political coverage was entirely presidential-election-centric with zero House election content at the time of the 2024 House seat range question. The vault MUST maintain:

    - **A US House elections thread** ([[domains/usa/threads/us-house-elections/_thread]]) tracking House seat dynamics, the narrow-majority era, and the governing consequences of a 215-222 seat majority. This thread MUST document each Congress's seat margin, key contested districts, and the structural forces (gerrymandering, geographic sorting, nationalization) that determine seat outcomes.

    - **A generic ballot to seat conversion concept** ([[domains/usa/concepts/generic-ballot-seat-conversion/_concept]]) documenting the asymmetric vote-to-seat relationship that has favored Republicans since 2010. This concept MUST provide a seat-vote regression formula or heuristic calibrated on recent elections, enabling forecasters to translate a generic ballot projection into a seat range estimate.

    - **A presidential coattail variability concept** ([[domains/usa/concepts/presidential-coattail-variability/_concept]]) documenting the structural factors that determine whether a presidential candidate generates downballot lift for their party's House candidates. This concept MUST provide the three-factor framework (candidate novelty, margin above party baseline, ticket-splitting environment) and calibration tables for the expected coattail strength by candidate type.

    - **A house-seat-range-forecast procedure** ([[domains/usa/procedures/house-seat-range-forecast]]) formalizing the 4-step assessment: generic ballot projection, seat-vote conversion, coattail adjustment, and range plausibility assessment.

    - **Entity stubs for House leadership**: The House Speaker ([[domains/usa/entities/mike-johnson]] already exists), House Minority Leader ([[domains/usa/entities/hakeem-jeffries]]), and relevant committee chairs (NRCC/DCCC chairs if named in a forecast question) MUST have entity stubs documenting their leadership role, margin constraints, and relationship to the narrow majority.

    Every contemporary quarter file covering a US election year (2022, 2024, 2026, etc.) MUST include a subsection on House election outcomes documenting:
    - The final seat count and majority margin
    - The national House popular vote and the seat-vote gap
    - Key districts that were won or lost by the majority
    - The governing implications of the resulting seat distribution (Speaker vulnerability, shutdown probability, legislative capacity)
    - Links to the House elections thread and relevant concepts

    Every US election year quarter file that covers the presidential race but omits the concurrent House election violates this rule. The 2024-Q4 quarter file is the canonical violation: it details the presidential outcome (Trump 312-226, popular vote margins, swing state analysis) but provides zero information on the House election outcome — despite its direct relevance to governance questions in the same quarter (the December 2024 shutdown) and the 119th Congress.

    **MANDATORY PRE-FORECAST CHECKS for any House seat range or control question**:
    1. Establish the current seat baseline (pre-election seats held by each party, including vacancies and party-switches)
    2. Assess the generic ballot projection (national House popular vote polling averages)
    3. Apply the seat-vote conversion function: calculate the expected seat range from the generic ballot projection, accounting for the gerrymandering advantage
    4. Assess presidential coattail intensity using the three-factor framework
    5. Combine steps 2-4 to produce an expected seat range
    6. Evaluate the question's specific range against the expected range — is it centered on the plausible outcome or at the edge?
    7. Assess district-level factors that could shift the conversion function (incumbent retirements, court-ordered map changes, scandal effects in specific districts)
    8. Document the reasoning explicitly: state the generic ballot projection, the expected seat range, the coattail assessment, and the final probability estimate

    This rule exists because the House seat range question (question 34, correct NO) was correctly predicted on general knowledge (a tied generic ballot produces a very narrow GOP majority, and the 215-219 range was too tight given the gerrymandering floor). The vault contributed zero structured input: no House thread, no seat-vote conversion concept, no coattail concept, no Hakeem Jeffries entity — despite these being directly relevant to the question's resolution. Every future House seat question must find a vault with structural analytical depth parallel to the vault's presidential election coverage. A vault with detailed coverage of VP selection dynamics, state electoral reliability, and gender balancing — but nothing on whether a party controls the House — is dangerously unbalanced in its US political coverage.

39. **Exact-count vs. range question classification as mandatory pre-forecast methodology check for House seat questions**: When a forecasting question asks about a specific numerical outcome in a US House election (or any multi-seat election), the vault MUST first determine whether the question is asking about an exact integer, a range, a threshold, or binary control — because each type requires a different probability calibration methodology. The most common methodology error is applying range-based reasoning (bin-level seat distribution) to exact-count questions, which overestimates YES probability by a factor of 5-10x. The vault MUST maintain:

    - **An exact-count concept** ([[domains/usa/concepts/exact-count-vs-range-forecast/_concept]]) documenting the distinction between exact-count, range, threshold, and binary-control question types, with the structural insight that no single seat count in a 435-seat chamber exceeds ~12% probability at the mode, and exact-count questions systematically require within-bin distribution estimation rather than bin-level aggregation.

    - **An exact-count procedure** ([[domains/usa/procedures/exact-seat-count-forecast]]) providing the step-by-step within-bin disaggregation methodology: classify question type → build within-bin distribution → calibrate exact probability → apply baseline discount. This procedure MUST be loaded alongside the existing [[domains/usa/procedures/house-seat-range-forecast]] whenever a House numerical outcome question arrives, because the question type determines which procedure to use.

    - **A question-type gate**: Every House numerical outcome forecast MUST begin with a question-type classification step. This is not optional — the classification determines which distribution model (bin-level or within-bin) governs the probability estimate.

    **MANDATORY PRE-FORECAST CHECKS for any House seat numerical outcome question:**

    1. **Classify the question type**: Is it exact-count ("exactly N"), range ("between A and B"), threshold ("at least N"), or binary control ("win the House")? Document the classification explicitly in the reasoning.
    2. **Load the correct procedure**: If exact-count → load [[domains/usa/procedures/exact-seat-count-forecast]]. If range → load [[domains/usa/procedures/house-seat-range-forecast]]. If threshold → use cumulative distribution from the bin-level table. If binary control → use the P(≥218) aggregate probability.
    3. **Apply the exact-count baseline discount**: If the question is exact-count, note that even the most likely single seat count has P < 12-15%. The default prediction is NO unless extraordinary evidence of a specific seat configuration exists. Do NOT treat a 5-bin range's probability (e.g., 35% for 220-224) as informative about any individual exact count within it.
    4. **Document within-bin distribution**: If using exact-count methodology, explicitly provide the within-bin probability estimates (using the normal approximation with σ ≈ 3.5) and show the calculation of p_yes for the specific integer.
    5. **Check for non-uniformity**: The within-bin distribution is NOT uniform. Probability is concentrated near the bin edge closest to the distribution mode. For GOP-skewed distributions, exact counts at the low end of the 220-224 bin (220-221) are ~2-3x more probable than counts at the high end (223-224).

    This rule exists because the "exactly 223 seats" question (question 42, correct NO) was correctly predicted on general knowledge (the probability of any exact seat count is inherently low), but the vault contributed zero structured exact-count methodology: no exact-count concept, no within-bin distribution model, no question-type classification procedure. The vault's bin-level distribution (5-seat ranges from generic-ballot-seat-conversion) would have actively misled a forecaster who tried to use it for exact-count reasoning — the 220-224 bin's 35% probability creates an illusion of plausibility for 223 when its individual probability is ~5%. Every future House seat exact-count question must find a vault with methodology designed for exact-count reasoning, not just range reasoning.

40. **Donor/surrogate defection as mandatory cascade-signal tracking**: The vault MUST systematically track donor and surrogate defection events as a distinct leading indicator within any incumbent-withdrawal or leadership-resignation cascade. The Biden 2024 case established the canonical pattern: the George Clooney NYT op-ed (July 10, 2024) preceded institutional leadership engagement (Pelosi/Jeffries signals on July 11-12) by 1-2 days and the withdrawal itself (July 21) by 11 days. This pattern is consistent across all three canonical withdrawal cases: donor/surrogate defection is the signal that a cascade has entered Phase 2 (institutional engagement imminent) and that withdrawal within 7-21 days is probable.

    The vault MUST maintain:

    - **Donor/surrogate defection as a formal trigger scenario** in the [[domains/usa/procedures/proc-aging-incumbent-early-warning]] trigger scenario simulation table (Step 6). The defection of a major fundraiser, celebrity surrogate, or loyalist donor with proximate-witness credibility must be tracked as a distinct signal with a 3-7 day leading indicator window before institutional leadership engagement.

    - **Entity stubs for key donor/surrogates** who appear in cascade narratives. The [[domains/usa/entities/george-clooney]] entity stub documents the canonical case. Any named donor, surrogate, or fundraiser who publicly defects in a future cascade MUST receive an entity stub before the forecast is rendered, documenting:
      - The nature of their relationship to the incumbent (loyalty level, history)
      - The credibility of their proximate-witness claims (did they recently observe the leader directly?)
      - The timing of their defection relative to the trigger event and other defections
      - Whether their defection precedes, coincides with, or follows institutional leadership engagement

    - **A donor-defection timing rule**: In any withdrawal cascade, when a donor/surrogate with proximate-witness credibility defects publicly AND the leader faces no legal jeopardy:
      - P(withdrawal within 21 days) > 80%
      - Expected timeline: institutional leadership engagement within 3-7 days, withdrawal within 7-21 days
      - This timing rule is calibrated on the Biden case (Clooney op-ed day 13, Pelosi/Jeffries days 14-15, withdrawal day 24) and validated against LBJ 1968 (Clifford/wise men advisory 3 days before withdrawal) and Truman 1952 (party elite signal 7-14 days before withdrawal).

    - **Documentation in every contemporary quarter file**: If a donor/surrogate defection event occurs in a quarter covering an incumbency-withdrawal crisis, the quarter file MUST document:
      - The surrogate's identity, relationship to the incumbent, and public statement
      - The timing relative to the trigger event (days since trigger)
      - Whether it preceded or followed institutional leadership engagement
      - Comparison to the three canonical cases' timing patterns

    This rule exists because the gold_12 miss (Biden dropout 2024, predicted NO, actual YES) was partly driven by failing to track the donor/surrogate defection signal. The vault now has the George Clooney entity stub (created in reflection), the cascade concept documenting the donor-defection timing, and the procedure's trigger simulation table — but without a spec rule requiring systematic donor/surrogate tracking, future forecasts may still miss this signal.

## Directory Structure

```
graph-vault/
  _index.md                   # Root index — vault purpose, navigation, status
  _spec.md                    # This file — schema definitions
  _procedure.md               # Workflow instructions for writing/updating summaries
  _forecast_instructions.md   # Agent behavioral rules
  
  agent-roles/                # Agent behavior definitions (orchestrator roster)
    _orchestrator_prerogatives.md
    <role-name>.md
  
  timeline/                   # Time-based quarter files (YYYY-QN.md)
  
  domains/                    # KNOWLEDGE: all content organized by domain
    <domain>/                 # Domain directory (usa, global, economics, etc.)
      _domain.md              # Domain overview — entry point
      entities/               # Actors, orgs, places relevant to this domain
      concepts/               # Recurring patterns / frameworks
        <concept>/
          _concept.md          # Concept definition + canonical examples
          procedures/          # Analytical methods for this concept
      procedures/             # Shared procedures for the domain
      functions/              # Shared function references (code tools)
      threads/                # Narrative arcs
        <thread>/             # Each thread is a directory
          _thread.md          # Thread overview + timeline
          entities/           # Entities tied to this thread
          events/             # Occurrences within this thread
          procedures/         # Analytical methods for this thread

  history/                    # NON-LIVE: pre-2022 / analog research
  
  forecasts/  runs/  meta/    # Artifacts (not part of the knowledge graph)
```

### Live vs history vs meta (locked)

| Zone | Cutoff | Agent write rule |
|------|--------|------------------|
| **Live** (`timeline/`, `threads/`, `entities/`, `concepts/`) | 2022+ contemporary | Default for all forecast work. Entity only when market/thread needs it. |
| **history/** | Before 2022 or analog-only | **Mandatory** for historical quarters, retired arcs, historical actor research. Never create pre-2022 `entities/*.md` at vault root. |
| **meta/** | N/A | Reflections and session logs only. |

Promoting history → live: distill into a `concepts/` file with a 2022+ example, or a subsection on a contemporary thread — not a bulk entity import.

## Obsidian Graph Tags (mandatory)

Every vault file MUST have `type:` and `tags:` in its YAML frontmatter for Obsidian graph coloring and filtering. The `tags:` field is a YAML list (e.g. `tags: [concept, entity]`).

| Tag | Applies to | Graph color |
|-----|-----------|-------------|
| `#domain` | `domains/<d>/_domain.md` | Domain overviews |
| `#concept` | `domains/<d>/concepts/*.md` or `*_concept.md` | Cross-domain patterns |
| `#thread` | `domains/<d>/threads/<t>/_thread.md` | Ongoing dynamics |
| `#entity` | `domains/<d>/entities/*.md`, `threads/<t>/entities/*.md` | Actors, orgs, places |
| `#event` | `domains/<d>/threads/<t>/events/*.md` | Specific events |
| `#procedure` | `domains/<d>/procedures/*.md`, `threads/*/procedures/`, `concepts/*/procedures/` | Analytical toolchains |
| `#function` | `domains/<d>/functions/*.md` | Callable code tools |
| `#agent-role` | `agent-roles/*.md` | Agent behavior definitions |
| `#timeline` | `timeline/*.md` | Quarter/year files |
| `#reflection` | `meta/reflections/*.md` | Reflection/audit outputs |
| `#meta` | Root config files (`_spec.md`, `_procedure.md`, etc.) | System configuration |
| `#probe` | `probes/*.md` | Analysis probes |

Tag enforcement is done by `scripts/tag_vault_nodes.py`. Run it after creating any new vault files to ensure consistent tagging.

## File Types

### 1. Domain Entry Files (`domains/<domain>/_domain.md`)

Frontmatter:
```yaml
---
type: domain
title: "Display Title"
slug: domain-slug
subjects:  # wikilinks to relevant concepts
  - "[[domains/global/concepts/escalation-bargaining-termination]]"
procedures:
  - "[[domains/global/procedures/proc-escalation-forecast]]"
threads:
  - "[[domains/global/threads/russia-ukraine-war/_thread]]"
---
```

Structure:
- **Overview**: 1-2 paragraphs describing the domain's scope
- **Navigation**: How to traverse entities, concepts, threads, and procedures within this domain

### 2. Timeline Quarter Files (`timeline/YYYY-QN.md`)

Frontmatter:
```yaml
---
type: quarter
year: YYYY
label: "YYYY-QN"              # MUST use this exact format — no descriptions
date_range: "YYYY-MM-DD to YYYY-MM-DD"
prev: YYYY-Q(N-1) | null
next: YYYY-Q(N+1) | null
pit_cutoff: YYYY-MM-DD        # last day of quarter — PIT boundary
source: web-research | vault-synthesis | mixed
---
```

Structure:
- **Overview**: 1-2 paragraphs setting the quarter in global context
- **Major Event Sections**: Ordered by geopolitical importance first, then science/culture
  - Month-based subsections (`###`) for long arcs
  - Event entries: `- **Date**: Description` format with [[wikilinks]]
- **Cross-Domain Threads**: 3-7 thematic analyses connecting events across domains
- **Wikilinks Created**: Comprehensive list at bottom for reference
- **Births/Deaths**: Tables integrated into month sections

### 3. Thread Files (`domains/<domain>/threads/<thread>/_thread.md`)

Thread files are the PRIMARY NODES of the vault. They track causal chains, ongoing dynamics, and multi-quarter developments.

Frontmatter:
```yaml
---
type: thread
title: "Display Title"
slug: thread-slug
span: "YYYY-MM-DD to YYYY-MM-DD" | # optional: date range
inception: YYYY-MM-DD | YYYY        # when the thread began
conclusion: YYYY-MM-DD | YYYY | ongoing | null
status: nascent | active | climaxing | fading | resolved
---
```

Structure:
- **Overview**: What this thread is and why it matters
- **Timeline / Key Events**: Chronological entries linked to quarter files
- **Key Dynamics**: Structural forces, repeated patterns, inflection points
- **Forecasting Significance**: What pattern-matching this thread enables
- **Related Threads**: Wikilinks to other threads
- **Wikilinks**: Entities and quarters referenced

Rules:
- Update threads each quarter — append new developments, update status
- Do NOT remove old content when appending; threads are cumulative
- Threads should reference quarter files via `[[YYYY-QN]]` wikilinks
- A thread's `status` field should be updated each quarter
- **Sub-thread elevation**: When a thread within a thread develops its own causal chain spanning 2+ quarters with distinct dynamics and forecasting significance, elevate it to its own thread file. Example: The Philippine-American War was initially inside American Imperial Expansion; it became its own thread when Balangiga, the Insular Cases, and the Sedition Act formed a distinct counterinsurgency narrative.

### 4. Concept Files (`domains/<domain>/concepts/<concept>/_concept.md`)

Concept files capture recurring frameworks, patterns, and dynamics that repeat across time periods. They are the vault's highest-value output for forecasting.

Frontmatter:
```yaml
---
type: concept
title: "Display Title"
slug: concept-slug
first_observed: YYYY | ~YYYY # approximate earliest observation
domain: geopolitics | military-strategy | risk-assessment | science | economics | culture
related_concepts: [slug1, slug2]  # optional
---
```

Structure:
- **Definition**: 1-2 paragraphs defining the concept
- **Canonical Examples**: Concrete historical cases from the vault
- **Pattern Archetype**: The structural dynamics — what to look for
- **Forecasting Application**: Specific indicators to watch for
- **Validated By**: Table of forecasts that tested this concept — prediction, actual outcome, and whether the concept correctly predicted it. This builds the concept's track record and identifies which concepts are most reliable.
- **Wikilinks**: Entities and quarters that exemplify the concept

Rules:
- Concepts should be generalizable — applicable across multiple eras, not just 1900
- Every concept should have at least one canonical example from the vault
- Update concepts when new examples emerge from later periods

### 5. Entity Files (`domains/<domain>/entities/<entity>.md` or `domains/<domain>/threads/<thread>/entities/<entity>.md`)

Frontmatter:
```yaml
---
type: entity
kind: person | event | concept | place | organization | treaty | technology
title: "Entity Display Name"
slug: entity-slug
born: YYYY-MM-DD | YYYY | null     # for persons
died: YYYY-MM-DD | YYYY | null     # for persons
date_start: YYYY-MM-DD | YYYY | null  # for events/organizations
date_end: YYYY-MM-DD | YYYY | null    # for events/organizations
pit_cutoff: YYYY-MM-DD             # PIT boundary for the entity summary
---
```

Structure:
- **Summary**: 1-3 paragraphs — PIT-compliant for the cutoff
- **Significance**: Why this entity matters for forecasting
- **Timeline**: Key dates (bullet list)
- **Wikilinks**: [[Links]] to related entities, threads, and quarter files

Entity priority for creation:
1. Entities referenced in 2+ quarters (highest connectivity value)
2. Entities central to a thread (war leaders, key inventors)
3. Entities with direct forecasting relevance (future decision-makers, pattern-setters)

### 5. Topic Files (`topics/topic-slug.md`)

Optional cross-cutting syntheses. Examples: `science-1900.md`, `geopolitics-1900.md`. Summarize developments across a domain across an entire year or period.

## Wikilink Conventions

- Every named entity of significance in a thread or quarter file should be a [[wikilink]]
- Wikilink targets are resolved by searching: domain entities/, thread entities/, domain concepts/, thread _thread.md files, timeline/ files, root _*.md files
- Wikilink case: Use sentence case matching the entity's `title` field
- Pipe syntax `[[actual|display]]`: AVOID where possible. It hides the canonical name from the graph. Use only for disambiguation or where the display text must differ from the canonical name.
- Entity files link back to the thread and quarter files where they appear
- Thread files link to quarter files, entity files, and related threads
- Concept files link to canonical examples in threads, entities, and quarters
- Cross-domain wikilinks: use full path `[[domains/east-asia/entities/xi-jinping]]` to avoid ambiguity
- Within the same domain, short names resolve automatically (e.g., [[emmanuel-macron]] in any france/ file)

### 6. Contemporary Events and Forecast Context (`type: forecast-entry`)

A special file type for recording the causal reasoning behind a forecast at the time it was made. This bridges historical pattern-matching to real-time prediction.

Optional frontmatter:
```yaml
---
type: forecast-entry
date: YYYY-MM-DD
question: "Israel x Iran ceasefire before July?"
prediction: "NO"
actual: "YES"
pit_cutoff: YYYY-MM-DD
---
```

Structure:
- **Question**: The exact forecast question
- **Reasoning at time**: What the evidence suggested at forecast time
- **Actual outcome**: What actually happened
- **Diagnosis**: Why was the reasoning right or wrong
- **Vault gaps**: What was missing from the vault that would have helped

Forecast entries live in `forecasts/YYYY-MM-DD-slug.md`. They are the feedback loop that improves the vault.

## Quality Standards

- **PIT fidelity**: All information must be knowable as of the cutoff date. No retroactive attribution of future significance.
- **Cross-domain connections**: Every quarter should draw at least 3 cross-domain threads.
- **Proportional coverage**: Major wars get more space than minor cultural events, but small events with large future consequences get highlighted.
- **Neutral tone**: Descriptive, analytical, not celebratory or condemnatory.
- **Thread continuity**: Every thread referenced in a quarter should be followed up in subsequent quarters until resolved.
- **Entity hygiene**: No orphaned wikilinks — either create the entity file or remove the link.

## Backlink Conventions

- Entity files should include a `## Appears In` section listing the quarters and threads where they appear
- This is the most important graph-connectivity practice: it makes the vault browsable in both directions
- **Priority**: Add backlinks first to entities in active threads, then entities appearing in 2+ quarters, then single-quarter entities
- **Batch method**: Collect all entities that need backlinks at the end of each cycle; add them in a single batch to minimize context switches

39. **State-level technology regulation as mandatory coverage**: The vault MUST systematically cover major state-level technology regulation, especially in California (which acts as a de facto national regulator in the absence of federal action), because (a) state-level AI, privacy, and content moderation bills are recurring prediction market questions, (b) the California legislative process (part-time legislature with tight end-of-session deadlines, gubernatorial signature deadline 30 days after passage) creates distinct timing dynamics that differ from federal legislation, and (c) the relationship between state tech regulation and federal preemption/innovation concerns follows a structural pattern that can be calibrated. The vault MUST maintain:

    - **A state-level AI regulation thread** ([[domains/usa/threads/state-level-ai-regulation/_thread]]) tracking major state-level AI bills, their sponsorship, committee trajectories, gubernatorial signals, and outcomes. This thread MUST include SB 1047 as the canonical case and track subsequent bills.
    - **A California bellwether concept** ([[domains/usa/concepts/state-level-tech-regulation-bellwether/_concept]]) documenting the structural pattern of California as a tech regulation pioneer: when Congress is gridlocked on tech policy, California acts first (privacy — CCPA 2018, AI — SB 1047 2024, content moderation — AB 587 2022). The concept MUST track three mechanisms (regulatory vacuum fill, executive national ambition, industry lobbying pressure) and their interaction.
    - **A veto-bill dynamics concept** ([[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]]) documenting the pattern of California governors vetoing significant tech regulation bills: the governor faces competing pressures between progressive safety/consumer advocates and the tech industry (a major state economic driver). Key variables include the governor's national ambition (tougher regulation = national credibility), state economic exposure (weaker regulation = industry friendliness), bill novelty (first-of-its-kind = higher veto probability), and legislator override capacity (California's 2/3 override threshold is nearly impossible).
    - **Entity stubs for key actors**: California Governor ([[domains/usa/entities/gavin-newsom]]), bill authors ([[domains/usa/entities/scott-wiener]]), the California State Legislature ([[domains/usa/entities/california-state-legislature]]), and any named tech company or advocacy group in a forecast question.
    - **Entity stubs for major tech trade groups**: The California Chamber of Commerce, TechNet, and similar organizations that lobby on tech bills should have entity stubs documenting their lobbying spending and bill-position track records.
    - **A procedure for state-level tech bill forecasting** ([[domains/usa/procedures/state-level-tech-bill-forecast]]) formalizing the 6-step assessment: bill stage → legislative calendar → veto point analysis → override assessment → governor national ambition signal → probability calibration.

    Every state-level tech bill forecast MUST document:
    - The bill's current stage (introduced, committee, passed one chamber, passed both, on governor's desk)
    - The legislative calendar deadline (CA bills must pass by Aug 31 in 2nd year of session; governor signs or vetoes by Sep 30)
    - The governor's public posture on the bill and on tech regulation generally
    - The governor's national ambition level (running for higher office, hosting national convenings on the topic)
    - The intensity and source of opposition (industry, labor, civil liberties coalitions — each has different leverage)
    - The override probability (CA requires 2/3 of both chambers — nearly impossible for controversial tech bills)
    - The key uncertainty variable (e.g., for SB 1047: Newsom's position was the unresolved variable; bill passage through the legislature was always uncertain but the veto was the final blocking point)

    **MANDATORY PRE-FORECAST CHECKS for any state-level tech regulation question**:
    1. Determine the bill stage — is it still in committee, has it passed one chamber, or is it on the governor's desk? Each stage has a different baseline passage probability.
    2. Check the legislative calendar — is the end-of-session deadline approaching? California's Aug 31 deadline creates a binary cliff that compresses negotiations.
    3. Identify the governor's position — has the governor publicly stated a position? Is the governor positioning for national office (which creates incentive for moderation on industry-sensitive issues)?
    4. Assess the override capacity — California's 2/3 legislative override threshold is extremely high; a governor's veto is effectively the final word for any controversial bill.
    5. Map the lobbying landscape — which industry groups are opposing and what is their spending/leverage? Tech industry opposition in California is the strongest single predictor of a bill's failure or modification.
    6. Check for legislative maneuver alternatives — is the bill's substance being added as an amendment to another bill (gut-and-amend) to bypass committee or procedural obstacles?

    This rule exists because the SB 1047 question (Question 36 of the PIT blind test, correct NO prediction) was a 0%-vault-contribution freebie — the vault had no technology policy domain, no AI regulation thread, no California entity stubs, no bellwether concept, and no forecasting procedure. The correct prediction relied on general knowledge of Newsom's veto, tech industry opposition, and the Democratic split on AI regulation. Every future state-level tech regulation question must find vault coverage of comparable structural depth to what exists for financial regulation, central bank policy, and electoral dynamics.

40. **Cabinet formation and presidential personnel selection as mandatory coverage**: The vault MUST systematically cover cabinet formation dynamics during presidential transitions, because (a) cabinet nomination questions are among the most common prediction market questions during transition periods, (b) the VP-finalist-to-cabinet pipeline is a robust structural pattern that makes cabinet nominations highly forecastable, and (c) cabinet picks signal the president's policy priorities and governance approach. The vault MUST maintain:

    - **A second-term cabinet formation concept** ([[domains/usa/concepts/second-term-cabinet-formation/_concept]]) documenting the structural differences between first-term and second-term cabinet formation, including: re-election constraint removal (personnel no longer chosen for electoral appeal), the VP-finalist-to-cabinet pipeline (veepstakes finalists become the primary cabinet candidate pool), loyalty tracks by portfolio type (high-credibility roles vs. enforcement roles vs. low-visibility roles), the establishment-credibility threshold for foreign policy portfolios, confirmation dynamics by Senate composition, and historical re-elected president cabinet patterns.

    - **A Trump-specific personnel selection concept** ([[domains/usa/concepts/trump-rival-to-ally-pipeline/_concept]]) documenting the 4-stage sequence (rivalry → endorsement → loyal service → appointment) that governs Trump's personnel decisions, with canonical cases (Rubio, Vance, Haley, Stefanik) and forecasting rules including the VP-finalist-to-cabinet conversion rate.

    - **A cabinet nomination forecasting procedure** ([[domains/usa/procedures/trump-cabinet-selection-patterns]]) formalizing the 6-step assessment: portfolio type identification → personnel pool identification (VP finalists first) → rival-to-ally pipeline stage assessment → Senate confirmation viability → timing cross-check → probability calibration.

    - **Entity stubs for all named cabinet nominees in any forecast question**: For the 2024-2025 transition, the minimum entity stub set includes [[domains/usa/entities/marco-rubio]] (State), [[domains/usa/entities/doug-burgum]] (Interior), [[domains/usa/entities/elise-stefanik]] (UN Ambassador), [[domains/usa/entities/jd-vance]] (VP), and [[domains/usa/entities/tim-scott]] (remaining in Senate).

    **MANDATORY PRE-FORECAST CHECKS for any cabinet nomination question**:

    1. **VP finalist check**: Is the person a VP finalist from the current or previous election cycle? If yes, this is the single most informative variable. Determine the elimination reason: procedural (state-law, timing, optics) → 60-80% cabinet probability; personal (vetting problem, scandal, ideological incompatibility) → <10% cabinet probability.

    2. **Rival-to-ally stage check**: If the person was once a rival of the president, what stage have they reached? Stage 3+ (2+ years of loyal service) required for high-visibility roles; Stage 2+ may suffice for secondary roles.

    3. **Portfolio-type match**: Does the person's background match the portfolio's credibility requirements? High-credibility roles (State, Treasury, Defense) require establishment figures with Senate mana. Enforcement roles (AG, FBI) require loyalty first, competence second.

    4. **Confirmation check**: Is the person a sitting Senator? If yes → 90%+ confirmation probability. What is the Senate composition? President's party majority → smooth path.

    5. **Timing check**: Is this during the post-election transition (Nov-Jan)? If yes, this is the peak cabinet announcement window. The first announced nomination is typically the establishment-signal pick.

    Every contemporary quarter file covering a post-election transition period MUST include a subsection on cabinet formation documenting:
    - The nominee list and their portfolio assignments
    - The Senate confirmation timeline
    - The policy-direction signal of each major pick (what does the pick signal about the president's priorities?)
    - Entity stubs for named nominees who lack them (created before filing the quarter)
    - Links to the [[concepts/second-term-cabinet-formation/_concept]] and relevant personnel selection concepts

    This rule exists because the Marco Rubio Secretary of State question (Question 41 of the PIT blind test, correct YES prediction) was correctly predicted but the vault contributed minimal structural signal. The vault had the VP finalist data (in the 2024 election thread) and the Rubio entity stub but no analytical framework connecting them to cabinet formation. The VP finalist pool — which was the dominant variable making the prediction ~70% probable — was not documented as a cabinet input anywhere in the vault. Every future cabinet nomination question must find vault coverage that explicitly connects the VP finalist pool to cabinet outcomes, with a structured framework for assessing elimination reasons, portfolio fit, and confirmation dynamics.

| ## Frontmatter Audit

After every batch of new or modified files, run a grep-based frontmatter audit to catch drift:

```bash
# Check quarter files use label: format
grep -l "type: quarter" timeline/*.md | xargs grep -L "^label:"

# Check thread files use inception:/conclusion:/status:
grep -rl "type: thread" domains/*/threads/*/_thread.md | xargs grep -L "^status:"

41. **Resolution-criteria precision as mandatory pre-forecast check for all numerical/multi-interpretation questions**: When a forecasting question combines a verb that implies a range or threshold ("control," "wins," "has") with a specific number, the vault MUST check the resolution text for the literal resolution standard before calibrating probability. The most common error is interpreting "will Republicans control 224 seats?" as a threshold question (P(>=224) ≈ 20%) when the resolution text specifies exact count (P(=224) ≈ 5%) — a 4x error factor. The vault MUST maintain:

    - **A resolution-criteria gotchas concept** ([[domains/global/concepts/forecast-resolution-criteria-gotchas]]) documenting all known patterns of question-wording/resolution-text mismatch. This concept MUST include:
      - The "control N" vs. "exactly N" ambiguity (entry #8, added 2026-05-20)
      - The "wins" vs. "assumes office" ambiguity (authoritarian elections)
      - The "announces" vs. "ratifies" ambiguity (multi-step diplomatic processes)
      - The "dips to $X" tick-vs-OHLC ambiguity (price-based questions)
      - The "sentenced to N months" vs. "actually serves" ambiguity (legal questions)
      - The "drop out" vs. "ceases to be a candidate" ambiguity (withdrawal questions)
      - The "ceasefire" temporary-vs-enduring ambiguity (conflict questions)
      - The "banned" legal-status-vs-enforcement ambiguity (regulatory questions)

    - **A pre-forecast resolution-text audit step**: Every forecast must include a verification that the question wording and resolution text are semantically aligned. If they diverge, the resolution text governs — always. Document in the reasoning whether the question title and resolution criteria are aligned or divergent, and what the divergence implies for probability calibration.

    - **A general pattern library**: The resolution-criteria gotchas concept MUST be referenced in every forecast where the question title could be interpreted differently from the resolution text. The minimal failure prevention action is: before finalizing any forecast, read the resolution text aloud and ask "could this mean something different from the question title?" If yes, document the difference and its impact on p_yes.

    **MANDATORY PRE-FORECAST CHECK for any question with ambiguous wording:**

    1. **Extract the resolution criterion** — what exact event, count, or condition must be satisfied? Read the resolution text literally. Do NOT infer meaning from the question title.
    2. **Compare with question title** — does the title suggest a different interpretation? If so, the title is a decoy. The resolution text is the ground truth.
    3. **Classify the ambiguity type** — is it exact-count vs. threshold (this case), temporary vs. enduring, legal vs. enforcement, announcement vs. ratification, or tick vs. close? Each type has a documented pattern in the gotchas concept.
    4. **Apply the appropriate calibration** — exact-count calibration (within-bin distribution) vs. threshold calibration (cumulative distribution) vs. other framework.
    5. **Document both interpretations** — in the reasoning, include what p_yes would be under the misleading interpretation AND the correct interpretation, to make the adjustment visible.
    6. **Check for historical gotchas** — has this specific question type appeared before? The gotchas concept catalogs all known patterns. Load it before forecasting any question where the title could be misleading.

    This rule exists because the "control 224 seats" question (Question 43 of the PIT blind test, correct NO prediction) was correctly predicted, but the vault's contribution was entirely through the exact-count methodology created in a prior reflection. The resolution-criteria ambiguity (question says "control," resolution says "exactly") was not flagged as a separate gotcha — the forecaster happened to read the resolution text, but the vault had no structural check ensuring that future questions with similar phrasing divergences would also be caught. Every future question that combines a verb suggesting threshold/range with a resolution text specifying exact count must be caught by a systematic pre-forecast check, not left to the forecaster's vigilance.

42. **US domestic security incidents and mass sociogenic events as mandatory coverage**: The vault MUST systematically cover major US domestic security incidents — including mystery drone waves, unauthorized military base incursions, domestic terrorism, and nuclear/radiological security events — because (a) these are recurring subjects of prediction market questions that follow distinct structural patterns from foreign policy or military conflict, (b) mass sociogenic events (mystery drone/UFO waves, social panics involving perceived security threats) are an under-recognized forecasting domain where government confirmation requirements create systematically high bars for YES resolution, and (c) genuine security concerns (verified military incursions) can coexist with mass sociogenic events and must be disentangled for accurate forecasting. Every contemporary quarter file (post-2020) MUST include a section on domestic security incidents when relevant events occurred. Additionally:

    - **A mass-sociogenic-event concept** ([[domains/usa/concepts/mass-sociogenic-event/_concept]]) MUST exist documenting the recurring pattern of public concern spirals around mysterious sightings where official investigations find routine explanations, with forecasting implications including the declining probability of confirmation over time and the inverse relationship between social media intensity and confirmation probability.
    
    - **A government-confirmation-requirement concept** ([[domains/global/concepts/government-confirmation-requirement/_concept]]) MUST exist documenting the structural bar for questions requiring official government confirmation — the four confirmation types (vulnerability admission, adversary attribution, technical fact, error admission), their probabilities, and the pre-forecast audit steps.
    
    - **Entity stubs MUST exist** for the lead federal agencies: FBI, DHS, DoD, and any agency specifically named in a forecast question (NNSA, NRC, TSA, FAA, etc.). These agencies are the sources of government confirmation or denial and their institutional incentives determine resolution probability.
    
    - **A thread** ([[domains/usa/threads/us-drone-security-incidents/_thread]]) MUST exist tracking the sequence of verified military incursions, public mystery drone waves, and the evolving counter-drone legislative response. This thread covers the 2023 Langley AFB incursion through the 2024-25 Northeast mystery drone wave and beyond.
    
    - **The mass sociogenic vs. genuine incursion distinction** MUST be documented: genuine military base incursions (350 in 2024, per DoD testimony) are a separate phenomenon from civilian mystery drone waves (thousands of misidentified aircraft). A forecast about one should not conflate data from the other.
    
    - **The FOIA transparency gap** MUST be tracked: when agencies debunk sightings internally but delay public disclosure, the delay prolongs public concern and can influence forecasting markets. This dynamic is documented in [[domains/usa/entities/transportation-security-administration]].

    This rule exists because the Mystery Drones question (Question 44 of the PIT blind test, correct NO) was correctly predicted using general knowledge of the government confirmation standard — but the vault had zero coverage of US domestic security incidents, no concept for mass sociogenic events, no concept for government confirmation requirements, and no entity stubs for any of the federal agencies involved. A correct prediction that relies solely on general knowledge reveals a vault gap as surely as a wrong one. After this rule, any future question about a domestic security incident, mystery drone wave, or government confirmation of a security-related claim will trigger systematic vault analysis using the mass-sociogenic-event and government-confirmation-requirement frameworks.

43. **Public health surveillance and infectious disease outbreak forecasting as mandatory coverage**: The vault MUST systematically cover major infectious disease outbreaks with potential for human spillover and forecasting-relevant case count trajectories, because (a) zoonotic spillover outbreaks (H5N1, mpox, COVID-19 variants) are recurring subjects of prediction market questions, (b) the trajectory of human case counts in an outbreak follows predictable structural dynamics (slow initial detection, rapid growth during agricultural amplification, plateau as mitigation measures take effect) that can be modeled with simple quantitative benchmarks rather than requiring epidemiological expertise, and (c) the CDC and analogous public health agencies maintain publicly accessible case counters that can be tracked across quarters for PIT case count estimates. The vault MUST maintain:

    - **A thread for each major ongoing outbreak** ([[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]]) tracking the outbreak's evolution across quarters: first detection, agricultural/mammalian host expansion, human spillover cases, transmission mechanism confirmation, case count milestones, public health risk assessments, and policy responses. The thread MUST include quarterly case count snapshots from the CDC's public data to build a PIT trajectory that can extrapolate case counts at any future date.

    - **A zoonotic outbreak case count forecasting concept** ([[domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept]]) documenting the structural dynamics of case count trajectories in zoonotic outbreaks that spill over from agricultural hosts: the initial detection delay (cases are undercounted in early months due to surveillance gaps in agricultural worker populations), the agricultural amplification phase (case growth is bounded by the size of the exposed agricultural workforce, not general population transmission), the plateau baseline (cases plateau at tens to low hundreds without human-to-human transmission), the pandemic threshold (human-to-human transmission is the step-change event that would dramatically increase case counts — and this event has NOT occurred in the 2024-2025 H5N1 outbreak), and the CDC risk assessment as a leading indicator (CDC maintained \"low\" risk throughout — a reliable signal that case counts would not explode).

    - **An entity stub for the CDC** ([[domains/global/entities/centers-for-disease-control-and-prevention]]) documenting the agency's role as the authoritative source for US case counts, its institutional incentives (public health reputation, political sensitivity of outbreak declarations), and its public data products (the H5N1 case counter, FluView, situation summaries). This entity MUST exist to ground any forecast that relies on CDC-confirmed case counts.

    - **A procedure for outbreak case threshold forecasting** ([[domains/global/procedures/outbreak-case-threshold-forecast]]) formalizing the assessment: identify the outbreak source and transmission mode (zoonotic spillover from agriculture vs. human-to-human spread), check the CDC risk assessment as a leading indicator (low risk = cases bounded at tens to low hundreds), extrapolate the case count trajectory using the previous quarter's count as a baseline and the known plateau dynamics, check for step-change events (D614G-like mutation, mammalian adaptation, human-to-human cluster) that would invalidate the plateau model, and calibrate probability accordingly.

    **MANDATORY PRE-FORECAST CHECKS for any outbreak case count question**:

    1. **Identify the transmission mode**: Is this a zoonotic spillover (cases from animal contact) or human-to-human transmission? This is the single most important variable. Spillover cases are bounded by the size of the exposed agricultural workforce. Human-to-human transmission can produce exponential growth.

    2. **Check the CDC risk assessment**: What is the official CDC public health risk level? "Low" throughout the H5N1 outbreak was a reliable signal that widespread transmission was not occurring and case counts would remain bounded. A risk level upgrade to "moderate" or "high" would be a leading indicator of accelerated case counts.

    3. **Extract the last known case count from the most recent quarter**: What was the cumulative case count at the last available quarter date? Use this as the baseline for extrapolation. The 2024-Q3 baseline ("fewer than 20") combined with the known plateau dynamics gives a clear upper bound far below 100 by January 31, 2025.

    4. **Check for step-change events**: Has the virus acquired any mammalian adaptation mutations (e.g., PB2 E627K)? Has any human-to-human cluster been documented? If no to both, the plateau model holds and the case count will remain in the tens-to-low-hundreds range.

    5. **Assess the agricultural exposure base**: How many agricultural workers are in contact with infected animals? This is the maximum possible case pool. For the H5N1 dairy cattle outbreak, the exposed population was on the order of ~200,000 dairy workers — but actual infection rates were very low (~0.03% of exposed workers), consistent with inefficient human-to-human transmission and low environmental viral load in most exposure scenarios.

    6. **Check the seasonal effect**: Does the question's deadline cross into a season that increases or decreases exposure risk? Winter may reduce outdoor agricultural work but increase indoor poultry processing — both effects are second-order relative to the transmission mode.

    **The core insight for case-count threshold questions on zoonotic outbreaks**: Without human-to-human transmission, case counts in agricultural spillover outbreaks plateau at tens to low hundreds regardless of how much media attention the outbreak receives. The single event that would break this ceiling is confirmed human-to-human transmission. Until that happens, the default forecast for any threshold above ~200 cases should be NO for any deadline within 12 months of first detection. The default for any threshold up to ~100 cases depends on the time from first detection and the agricultural exposure base — for the 2024 H5N1 outbreak, <20 cases at 9 months from first detection (Q3 2024) made 100+ cases by month 12 (Jan 31, 2025) impossible without a step-change in transmission dynamics, which the CDC risk assessment ruled out.

    This rule exists because the 100+ Bird Flu cases question (Question 47 of the PIT blind test, correct NO prediction) was correctly predicted but the vault contributed minimal structural signal. The vault had a single data point from the 2024-Q3 timeline ("fewer than 20" human cases) but no outbreak thread, no CDC entity, no case-count trajectory framework, and no structural understanding of why agricultural spillover case counts plateau. The correct prediction relied on general knowledge that total human cases in the US H5N1 outbreak were well below 100. Every future outbreak case-count question must find systematic vault coverage with thread data, trajectory modeling, and structural analytical depth — not a single data point from a timeline file.

44. **Policy-domain thread coverage and internal coalition faction analysis as mandatory pre-forecast assessment for any question about executive branch policy action**: The vault MUST systematically maintain active threads for each major policy domain of the current administration, and MUST assess the internal coalition faction dynamics within the governing coalition before forecasting any policy-specific executive action (especially elimination or creation questions). This rule exists because the H-1B elimination question (Question 48, correct NO prediction) exposed a complete absence of US immigration policy coverage in the vault — despite immigration being the central domestic policy focus of Trump's 2024 campaign and the subject of a prediction market question. The vault had entities for Trump's legal cases and linguistic patterns but nothing on his signature substantive policy domain.

    The vault MUST maintain, **for every US presidency (and analogously for any executive with substantial policy discretion)**:

    - **Active threads for each major policy domain**, created BEFORE the quarter files where policy actions occur. The minimum thread set for a US presidency is:
      - Immigration policy (border enforcement AND legal immigration)
      - Trade policy (tariffs, trade agreements, export controls)
      - Energy and climate policy
      - Health policy (Medicare, ACA, FDA, pandemic preparedness)
      - Technology regulation (AI, data privacy, platform governance)
      - Defense and foreign policy (alliances, conflicts, arms control)
      - Budget, debt, and fiscal policy
      - Domestic social policy (education, civil rights, SCOTUS appointments)
      
      Each thread MUST document at minimum:
      - The administration's campaign platform on the issue
      - Key internal coalition factions and their policy preferences (who wants more vs less?)
      - Key personnel with influence (agency heads, advisors, congressional allies, major donors)
      - The factional balance of power (which faction is ascendant at the current moment?)
      - Cross-references to [[concepts/program-restriction-vs-elimination]] and [[concepts/first-100-days-action-horizon]]

    - **Internal coalition faction analysis for every policy-specific question**: When a forecasting question asks whether the executive will take a specific policy action (eliminate X program, impose Y tariff, sign Z bill), the vault MUST identify the competing internal factions within the governing coalition and assess:
      - Which faction would benefit from the action?
      - Which faction would be harmed by the action?
      - What is the relative power/influence of each faction at the decision point?
      - Has the executive previously sided with one faction over the other on this specific issue?
      - Is there a compromise position (restriction vs elimination) that credibly satisfies both factions?
      
      The canonical case is the Trump coalition's split on H-1B: nativist restrictionists (Stephen Miller) want elimination; tech/libertarians (Elon Musk, Vivek Ramaswamy) want preservation or expansion. The factional analysis makes elimination structurally unlikely regardless of campaign rhetoric, because elimination would unite the tech faction in opposition while restriction (tightening wage requirements, narrowing definitions) can satisfy the nativists without triggering the tech faction.

    - **The "ally fallacy" as a structural constraint**: Every federal program has a constituency within the governing coalition. Elimination mobilizes the full constituency against the action. Restriction fragments it. The vault MUST document the ally constituency for any program that is the subject of an elimination question. This is formalized in [[concepts/program-restriction-vs-elimination]].

    - **The 100-day time horizon as a structural constraint**: The first 100 days of any presidency strongly favor actions that can be taken via executive order or agency guidance, and disfavor actions requiring legislation or major rulemaking. Any question about a policy change within the first 100 days MUST apply [[concepts/first-100-days-action-horizon]] to assess feasibility. The budget constraint (CR fights, debt ceiling, nominations) further compresses bandwidth for non-executive-order actions.

    - **Entity stubs for key faction leaders**: For any policy domain with active internal factional conflict, entity stubs MUST exist for the leaders of each faction. For Trump immigration policy, this meant Stephen Miller (nativist wing) and Elon Musk (tech wing) — both of which lacked documented immigration positions in the vault before this reflection.

    **MANDATORY PRE-FORECAST CHECKS for any executive policy action question:**

    1. **Classify the action type**: Executive order? Agency guidance? APA rulemaking? Legislation? Each has a different feasibility profile (see [[concepts/first-100-days-action-horizon]] and [[procedures/executive-action-feasibility-assessment]]).

    2. **Identify the statutory basis**: Is the program statutory (created by Congress) or executive-created? Statutory programs require legislation or legally dubious executive actions to eliminate — this is the highest bar (see [[concepts/program-restriction-vs-elimination]]).

    3. **Map the internal coalition factions**: Who in the governing coalition wants this action? Who opposes it? What is the factional balance of power? Document the specific individuals, their positions, and their leverage.

    4. **Check for a compromise position**: Is there a restriction action that achieves the political goal without triggering elimination-level opposition? If yes, restriction is the dominant probability.

    5. **Assess the time horizon**: Is the deadline within the first 100 days? If yes, apply the first-100-days-action-horizon framework — most large policy changes cannot fit in this window.

    6. **Load the relevant thread and concepts**: Load the policy-domain thread (if it exists), [[concepts/program-restriction-vs-elimination]], [[concepts/first-100-days-action-horizon]], and [[procedures/executive-action-feasibility-assessment]] before calibrating probability. If the thread does not exist, create it as part of the pre-forecast assessment — this is not optional.

    This rule exists because the H-1B elimination question was the 48th question answered, and the vault had zero immigration policy coverage despite immigration being Trump's signature issue. The correct prediction (NO) was supported by general knowledge of Trump's first-term record, the tech coalition's influence, and the 100-day time constraint — but contributed no vault signal. After this rule, any future executive policy action question must trigger systematic vault analysis using dedicated threads, concepts, entity stubs for faction leaders, and a feasibility procedure.

45. **MENA conflict coverage as mandatory domain**: The vault MUST maintain a dedicated Middle East and North Africa (MENA) domain with active threads tracking the Israel-Hamas conflict, Iran's proxy network, and Gulf state dynamics, because (a) MENA conflicts (Israel-Hamas, Iran-Israel, Houthi shipping disruptions, Syria, Lebanon) generate some of the most consequential and heavily traded prediction market questions of the 2020s, (b) the vault had zero MENA coverage — no domain, no thread, no entity stubs, no concept files — at the time of the January 2025 Israel-Hamas ceasefire question, even though the vault's own spec rules (16, 17, 18, 18a) had already established the methodological frameworks for ceasefire forecasting, and (c) the gap was not in analytical methodology but in **execution** — the spec rules existed but no content files operationalized them for the MENA region. The vault MUST maintain:

    - **A MENA domain entry** ([[domains/mena/_domain]]) documenting the region's forecasting-relevant characteristics (US alliance structure, Iran proxy network, mediation-savvy Gulf states (Qatar, Egypt), sectarian and ideological fault lines) and listing active threads and entities.
    
    - **An Israel-Hamas war ceasefire thread** ([[domains/mena/threads/israel-hamas-war-ceasefire/_thread]]) tracking the full chronology from Oct 7, 2023 through the Jan 2025 ceasefire to the Mar 2025 collapse, including: the four negotiation phases, the role of leadership decapitation (Sinwar killing Oct 2024), the US transition-window factor, the regional dimension (Hezbollah ceasefire Nov 2024), and war aims incompatibility analysis.
    
    - **Entity stubs for ALL named actors in any MENA forecast question**: Israel ([[domains/mena/entities/israel]]), Hamas ([[domains/mena/entities/hamas]]), Benjamin Netanyahu ([[domains/mena/entities/benjamin-netanyahu]]), and if the question involves mediation: Qatar ([[domains/mena/entities/qatar]]) and Egypt ([[domains/mena/entities/egypt]]). This is the minimum entity set for Gaza ceasefire questions — analogous to the minimum entity requirements in Rule 10 for US government shutdown questions.
    
    - **A transition-window ceasefire diplomacy concept** ([[domains/mena/concepts/transition-window-ceasefire-diplomacy/_concept]]) documenting the ~10-week period between US presidential election and inauguration when both outgoing and incoming administrations have aligned incentives to push for a ceasefire. This concept formalizes the leading/mid/late indicators, the Bayesian updating framework, false positive risks, and the canonical Israel-Hamas Jan 2025 case. It MUST be cross-referenced with [[domains/usa/concepts/lame-duck-legislative-feasibility/_concept]] because both involve the same structural dynamic (compressed post-election window) applied to different domains (ceasefire vs. legislation).
    
    - **A ceasefire announcement forecast procedure** ([[domains/mena/procedures/ceasefire-announcement-forecast]]) formalizing the 4-phase assessment: pre-flight check (parties, transition, pre-negotiated frameworks, leadership decapitation), structural analysis (incentive matrix, ceasefire equation, dual-frame analysis), calibration (historical analog, market-adjacent info, final probability), and post-forecast reflection.
    
    - **Quarter file coverage**: Every contemporary quarter file covering a period when Israel-Hamas negotiations were active MUST include a subsection on ceasefire negotiations documenting the three key dates (announcement, ratification, effective), the status of negotiations, the mediator role, and all parties' stated positions.

    **MANDATORY PRE-FORECAST CHECKS for any MENA ceasefire question:**

    1. **Load the ceasefire thread** — before any analytical work, load [[domains/mena/threads/israel-hamas-war-ceasefire/_thread]] for temporal context, mechanical patterns, and key variables.
    
    2. **Load the transition-window ceasefire diplomacy concept** — check the US political calendar. Is a US presidential transition in progress? Score the indicators and apply probability multipliers.
    
    3. **Check for leadership decapitation** — has either party's hardline leader been killed or removed? If within 2-4 months, the decapitation effect is active and increases ceasefire probability.
    
    4. **Check for pre-negotiated frameworks** — does a framework exist that one party already accepted and the other rejected? The question is then about political activation, not negotiation from scratch.
    
    5. **Assess war aims compatibility** — using [[domains/mena/threads/israel-hamas-war-ceasefire/_thread]]'s war aims documentation, determine whether the stronger party's stated war aim has been achieved, redefined, or rendered moot. If achieved, this removes a blocking factor.
    
    6. **Apply the ceasefire equation** from the procedure: P(ceasefire) = baseline × transition_multiplier × decapitation_multiplier × alignment_score. Document each variable explicitly.
    
    7. **Create entity stubs for all named actors** — before finalizing the forecast, create stubs for any named party, leader, or mediator that appears in the question but lacks a vault file.

    This rule exists because the Q52 question (Israel announces ceasefire by January 17) was a correct YES prediction, but the vault contributed zero structural signal: no MENA domain, no Israel-Hamas thread, no entity stubs for Israel, Hamas, Netanyahu, Qatar, or Egypt, no transition-window ceasefire diplomacy concept, and no ceasefire forecasting procedure. The correct prediction relied entirely on general knowledge of:
    - Trump's inauguration deadline (Jan 20) creating a forcing function
    - Biden-Trump simultaneous pressure on Israel
    - Sinwar's death (Oct 2024) enabling Hamas flexibility
    - The pre-existing May 2024 framework
    - War exhaustion after 15 months of conflict
    
    None of this information was in the vault, despite the vault's own spec rules (16, 17, 18, 18a) having already established the methodological principles for ceasefire forecasting. This is a violation of Spec Rule 8 ("No freebie predictions") — a correct prediction that relied solely on general knowledge reveals a vault gap as surely as a wrong one. After this rule, every future MENA ceasefire question will trigger systematic vault analysis with the domain, thread, entity stubs, concept, and procedure that were missing for Q52.

46. **"WHO Announces" distinction as mandatory pre-forecast check for all ceasefire-announcement questions**: When a forecast question asks whether a party (Israel, Hamas, Ukraine, Russia, etc.) will announce or agree to a ceasefire, the vault MUST distinguish between: (a) a mediator or third party announcing a framework accepted in principle (e.g., the US president announcing the framework on Jan 15), and (b) the named party formally announcing or ratifying the agreement (e.g., Israel's security cabinet approving on Jan 17). These are separate events that may occur on different dates, and conflating them is the single most common source of error on this question type.

    **Mandatory pre-forecast steps for any "Party X announces ceasefire by date D?" question:**

    1. **Check whether a mediator has already publicly announced a framework** — Search news archives for statements by the US president, UN Secretary-General, Qatar PM, or other lead mediator announcing that a ceasefire "has been agreed" or a framework "has been accepted." If a mediator has announced, the question is about the party's formal ratification, not about a new negotiation.

    2. **If the mediator has announced, apply the commitment-trap framework** — Load [[domains/mena/concepts/public-framework-announcement-commitment/_concept]] and assign P(YES) ~0.90-0.95 for the party's follow-through within 1-3 days. The party cannot reject without damaging its relationship with the superpower patron.

    3. **If the mediator has NOT announced, apply standard ceasefire forecasting** — The probability is MUCH lower (0.01-0.05 for short windows) because the mutual-consent penalty applies and no commitment trap has been activated.

    4. **Track all four dates separately** — Use the four-date framework (framework acceptance, mediator announcement, party announcement, ratification, effective date) from [[domains/global/concepts/ceasefire-announcement-ratification-gap]]. Record all known dates and identify gaps.

    5. **Record which date triggered the resolved outcome** — After resolution, log whether the market resolved on party announcement date, mediator announcement date, or effective date. This calibration data is essential for future questions.

    6. **Create entity stubs for key envoys** — If the question involves a US-brokered transition-window ceasefire, create entity stubs for the key envoys on both sides (Brett McGurk for the outgoing administration, Steve Witkoff for the incoming administration). Their coordination is a structural leading indicator for transition-window breakthroughs.

    7. **Check for dual-presidential endorsement** — If the ceasefire question involves a US presidential transition, check whether the outgoing AND incoming presidents jointly endorsed or announced the framework. A dual endorsement (both presidents publicly supporting the deal before the transition is complete) creates an amplified commitment trap worth an additional ~5-15 percentage points above the standard transition-window probability. The Jan 2025 Biden+Trump joint announcement is the canonical case — it was historically unprecedented and made party ratification near-certain (P ~0.97-0.99). See [[domains/mena/concepts/dual-presidential-endorsement-ceasefire/_concept]] for the full framework. A forecaster who applies only the standard transition-window analysis (P ~0.60-0.80) but misses the dual-endorsement amplification will under-estimate probability by 10-30 percentage points when both incoming and outgoing presidents are actively co-sponsoring the deal.

    This rule exists because the gold_50 question ("Israel announces ceasefire by Sunday?" with window starting Jan 16, 10AM ET) was a wrong NO prediction caused by conflating the mediator announcement (Jan 15, by US/Qatar — before the window) with the party announcement (Jan 17, Israel cabinet approval — within the window). The vault had labeled Jan 15 as "announcement" without the WHO qualifier, leading the forecaster to conclude the announcement was before the window and the answer must be NO. With the WHO-announces distinction and commitment-trap framework, the correct assessment would have been YES at ~0.90-0.95 probability.

47. **Presidential tenure security as a mandatory forecasting domain**: Questions of the form "Will [Leader] cease to be president/prime minister by [Date]?" or "Will [Leader] still be in power on [Date]?" are a distinct question type from election forecasts, requiring fundamentally different analytical frameworks. The vault MUST maintain:

    a. **A concept file for presidential removal risk** documenting:
       - The four removal pathways: impeachment/conviction, resignation, death/incapacity, coup/unconstitutional removal
       - Country-specific thresholds for each pathway (supermajority requirements, court jurisdiction, military posture, etc.)
       - Key forecasting variables: legislative supermajority threshold, economic trajectory, approval trajectory, protest intensity, military loyalty, international support, opposition unity, personal scandal severity
       - Institutional resilience data: years of continuous democracy, coup history, successful impeachment record, civilian control of military
       - Reference class tables for similar leaders (first-term populists, radical reformers, etc.)

    b. **A procedure for presidential tenure risk assessment** with:
       - 8-step structured method: identify leader → assess all 4 removal pathways → check economic trajectory → assess opposition cohesion → check international support → check institutional resilience → synthesize using reference class → produce structured forecast
       - Calibration: if NO pathway's threshold is met AND base rate favors survival, assign <10% removal probability
       - Explicit linkage to the concept file for underlying theory

    c. **Country-specific institutional coverage in the relevant domain or thread**:
       - The leader's entity stub MUST document removal risk factors alongside electoral/policy coverage
       - The domain or thread MUST document the country's democratic continuity and institutional resilience
       - The domain or thread MUST document any removal attempts (impeachment, resignations, coups) that have occurred

    d. **Entity stubs for all named actors** — before finalizing the forecast, create stubs for any named party, leader, or institution that appears in the question but lacks a vault file. For a "Will [Leader] leave office?" question, this includes:
       - The leader themselves (update existing stub if needed)
       - Key opposition figures who could lead impeachment efforts
       - Key institutional actors (Congress/supermajority counts, judiciary, military leadership if relevant)

    This rule exists because the gold_90 question ("Milei out as President of Argentina in 2025?") was a correct NO prediction, but the vault contributed only partial signal. The vault had strong electoral coverage (legislative seats, vote shares, coalition dynamics) but zero coverage of presidential removal mechanisms, Argentina's institutional resilience (zero successful coups since 1976, zero successful impeachments since 1983), the major 2025 scandals that were the most serious removal threats ($LIBRA Cryptogate rug pull, Karina Milei ANDIS corruption scandal), or generalizable concepts about why radical reformers survive their first term despite severe austerity and scandal. The reasoning chain for the correct NO prediction was partly vault-sourced (inflation trajectory, electoral strength) but partly relying on general knowledge (impeachment thresholds, democratic stability). After this rule, every future leader-tenure question will trigger systematic analysis with the removal-risk concept, survival procedure, and institutional coverage that were partially missing for this question.

48. **Question battery detection and saturation shifting**: When multiple questions about the same event or domain appear consecutively, the vault MUST recognize the battery and shift effort from domain coverage to cross-domain abstraction. The saturation cadence is ~3-4 questions per event domain before marginal learning from domain-specific content approaches zero:

    a. **Battery recognition criteria**: A question is part of a battery if it shares ALL of the following with prior questions:
       - Same country/region domain
       - Same year of event
       - Same institution or election
       The Argentina 2025 legislative election is the canonical battery: 5 questions (FIT-U, 3x HNP, LLA) about the same election with different party names.

    b. **Saturation threshold**: A domain is saturated when ALL three conditions are met:
       1. A complete thread documents the event's causal chain with quarter-level granularity
       2. Entity stubs exist for every named actor in the question AND all implicit actors (all major parties, not just the one named)
       3. At least one concept captures WHY the outcome occurred (not just WHAT happened)
       When all three are true, further domain-specific content creation has near-zero marginal forecasting value.

    c. **Saturation response protocol**: When saturation is detected:
       1. Do NOT create new domain-specific threads, entities, or concepts.
       2. Shift effort to cross-domain abstraction: extract patterns from this case that apply to other countries or domains.
       3. Create generalized concepts and procedures in `domains/global/` rather than `domains/[region]/`.
       4. Validate existing concepts by adding this forecast to their "Validated By" tables.
       5. Document the saturation state in the reflection: "Q58 is the Nth question in the Argentina battery. Domain is saturated. Effort shifted to abstraction: [extracted pattern]."

    d. **The abstraction gap**: The vault's learning system has a blind spot for cross-domain abstraction. It excels at learning specific facts about specific countries but does not automatically extract cross-national patterns. Saturation detection is the explicit trigger for abstraction. Every saturated battery MUST produce at least one cross-domain concept or procedure in `domains/global/`. A saturated domain that produces no abstraction is a missed learning opportunity.

    e. **Exception — new data changes the domain**: If new information emerges that invalidates the existing thread or concept (e.g., a previously unknown scandal, a leader's death, a regime change), the domain is no longer saturated. The existing coverage must be updated, not abstracted from. Saturation is a state of knowledge completeness, not a permanent lock.

    This rule exists because the Argentina 2025 legislative election generated 5 separate questions across the seed30 dataset. By the 5th question (Q58), the vault was fully saturated — the thread existed, entity stubs existed for all actors, and multiple concepts captured the dynamics. Further reflection on Argentina-specific content would have had zero marginal value. The vault needed a mechanism to recognize saturation and shift effort to the abstraction that was actually accomplished in Q57 (midterm-referendum-dynamics concept, legislative-plurality-forecast procedure). After this rule, every battery detection will trigger this shift automatically rather than relying on ad-hoc recognition.

27. **Major religious institutions as mandatory coverage**: The vault MUST maintain domain coverage for major religious institutions whose leadership transitions are forecast-relevant, because questions about papal succession, religious leadership changes, and elderly religious leaders appear in prediction markets and require specialized institutional knowledge. The minimum coverage bar for the Catholic Church (1.4B members, sovereign state, recurring forecast questions) is:

    a. **A domain directory** (`domains/religion/`) with a domain file documenting scope, rationale, and key entities/threads/concepts.

    b. **Entity files for the current and preceding pope**: Documenting birth date, health trajectory, key biographical data, and forecasting-relevant characteristics. See [[domains/religion/entities/pope-francis]] and [[domains/religion/entities/pope-leo-xiv]] as templates.

    c. **A papal succession thread** (`domains/religion/threads/papal-succession/`) tracking health decline, death events, conclave processes, and institutional implications. Must document the timeline from death to successor (typically 2-3 weeks).

    d. **An elderly leader mortality risk concept** (`domains/religion/concepts/elderly-leader-mortality-risk/`) providing age-based base rates, comorbidity multipliers, and functional decline signals for calibrating mortality probabilities over defined time horizons.

    e. **An elderly leader mortality assessment procedure** (`domains/religion/procedures/elderly-leader-mortality-assessment/`) with a step-by-step protocol for translating the concept into a calibrated probability when a forecast question arrives.

    The vault's complete absence of any religious institution coverage until Q59 (gold_28, Pope Francis successor question) is a structural gap equivalent to having no US politics coverage. The Catholic Church is a major global institution with 1.4 billion members, a defined succession process, and recurring prediction-market questions about papal health and transitions. A vault that tracks Argentina legislative elections, Turkey monetary policy, and Venezuela authoritarian resilience but lacks entity stubs for the Pope and a thread for papal succession has a blind spot that will produce systematically miscalibrated forecasts on religious leadership questions.

    **Extension to other religious institutions**: When a question arrives involving a non-Catholic religious institution (e.g., the Dalai Lama's succession, the Archbishop of Canterbury, the Aga Khan, the Grand Imam of Al-Azhar), create entity stubs and assess the succession mechanism using the same structural template — birth date, health trajectory, succession rules, institutional authority.

28. **Elderly leader health and mortality as mandatory pre-forecast assessment**: The vault MUST systematically assess the health status and mortality risk of any elderly leader (>75, or >80 depending on context) before forecasting on any question where that leader's continued survival or tenure is a material variable. This applies to popes, presidents, prime ministers, monarchs, party leaders, and any other named actor in a forecast question where the outcome depends on whether the leader survives, remains in office, or is replaced.

    a. **Trigger condition**: Any question where an elderly leader appears as a named actor, where the question's resolution depends on whether the leader remains in office, or where the leader's death/resignation would change the outcome.

    b. **Pre-forecast audit protocol**: Before forecasting, load and apply the [[domains/religion/procedures/elderly-leader-mortality-assessment]] procedure. Document:
        - The leader's age and sex
        - Documented health conditions with a severity classification
        - Observable functional decline signals (cancelled appearances, mobility changes, delegated duties)
        - The base-rate annual mortality for the leader's age
        - The estimated adjusted mortality risk for the relevant time window
        - Whether the "will there NOT be a new [leader]?" framing conflates survival with succession speed

    c. **Default heuristic for "will there be no new [leader] in [year]?" questions**: If the current leader is:
        - Age <75 with no known conditions: P(death in year) <5%, predict YES (90-95%)
        - Age 75-84 with 1-2 conditions: P(death in year) 10-25%, predict YES (70-80%)
        - Age 85+ OR 2+ major conditions: P(death in year) 25-50%, assess carefully
        - Age 85+ AND 2+ major conditions AND 2+ functional decline signals: P(death in year) >40%, lean NO (55-70%)
        
        The gold_28 error (Pope Francis, age 88, respiratory vulnerability, reduced mobility, recurrent hospitalizations, multiple cancelled events) was a miscalibration: the leader fell squarely in the highest risk tier, but the forecast treated the situation as low-risk. The heuristic would have flagged a balanced-to-negative assessment.

    d. **Pope-specific succession node**: In the specific case of a "will there be a new Pope in [year]?" question, note that the papal succession process takes only 2-3 weeks after the pope's death or resignation. This means the question resolves NO almost immediately upon Francis's death — the "new pope" window is not the full year but the entire year. A pope who dies in February produces a new pope by March; a pope who dies in December might not produce a new pope by Dec 31 if the conclave extends into January, but this edge case is rare.

    This rule exists because the gold_28 question (Pope Francis's successor) was answered without any structured health assessment of the leader whose mortality was the determining variable. The vault had extensive coverage of election dynamics, budget processes, and conflict trajectories — but the question's outcome was determined by the health trajectory of an 88-year-old man, and the vault had nothing on that topic. After this rule, every question involving an elderly leader's continuation in office will trigger a systematic health-and-mortality assessment before the probability is calibrated.

49. **US macro-economic indicators as mandatory active thread with explicit GDP threshold forecasting framework**: The vault MUST maintain a dedicated US macro-economic indicators thread (`domains/economics/threads/us-macro-economic-indicators/`) tracking quarterly real GDP, CPI/PCE inflation, Fed policy rate, employment, and trade/tariff impacts. This thread MUST be updated in every contemporary quarter file (post-2020) with the most recent GDP reading, inflation data point, and Fed rate decision. The thread exists in the economics domain ([[domains/economics/threads/us-macro-economic-indicators/_thread]]) and was created in Q60 of the PIT blind test to fill a gap where the spec referenced an "implicit rule" but no file existed.

    Additionally, the vault MUST maintain:

    a. **A GDP tail-risk asymmetry concept** ([[domains/usa/concepts/gdp-tail-risk-asymmetry/_concept]]) documenting the systematic asymmetry in GDP outcome distributions: positive outcomes cluster near trend (1.5-3.0%), while negative tail events require order-of-magnitude larger catalysts. The concept MUST include the historical frequency table for GDP thresholds (below 0%: ~13% of quarters, below -1%: ~6%, below -2%: ~3%, below -5%: ~1%) and the catalyst taxonomy distinguishing tariff-induced statistical contractions from financial-crisis demand collapses.

    b. **A GDP threshold forecasting procedure** ([[domains/usa/procedures/gdp-threshold-forecast]]) formalizing the 7-step assessment: extract question parameters, load current macro state, map severity using the tail-risk concept, check for statistical distortions (import front-loading, inventory swings), check leading indicators, assess advance estimate vintage, and calibrate final probability.

    c. **Explicit distinction between tariff-driven and financial-crisis contractions**: Tariff-driven GDP contractions (Q1 2025, ~-0.6%) are structurally different from financial-crisis contractions (2008, ~-2.1% to -8.5%) in four ways: (1) tariff contractions are driven by import front-loading (a statistical subtraction), not demand collapse; (2) the Fed retains easing capacity; (3) fiscal automatic stabilizers remain active; (4) no balance-sheet contagion channel exists. A question about a -2% threshold during a tariff escalation should default to NO unless a separate financial crisis or pandemic catalyst is also present.

    d. **Every contemporary quarter file must include a GDP data subsection**: The quarter's GDP reading (advance, second, or third estimate), contextualized against the previous quarter's trajectory and any unusual statistical factors. This data feeds directly into the [[domains/economics/threads/us-macro-economic-indicators]] thread.

    **MANDATORY PRE-FORECAST CHECKS for any US GDP threshold question**:

    1. **Load the US macro thread** — Get the most recent quarter's GDP reading and trajectory context.

    2. **Load the GDP tail-risk concept** — Map the threshold to the severity table. If the threshold is -2% or below, assess whether a GFC-class or pandemic-class crisis catalyst is present.

    3. **Check for statistical distortions** — Is there an active import front-loading episode (due to tariff deadlines), a large inventory swing, or one-time factors? Statistical contractions are shallower than genuine demand contractions.

    4. **Check the Fed stance** — Is the Fed easing (supporting GDP) or tightening (restraining)? The Fed's policy stance is a first-order variable for GDP trajectory.

    5. **Check leading indicators** — Are jobless claims, consumer confidence, ISM PMIs, yield curve, credit spreads, and VIX simultaneously flashing red? For a -2% threshold, ALL would typically be in crisis territory simultaneously.

    6. **Default heuristic**: For any negative GDP threshold below -1.5%, absent an identified GFC-class or pandemic-class catalyst, the probability is at the base rate (~3%) and the default forecast is NO.

    This rule exists because the Q60 question (Q1 2025 GDP < -2%) was correctly predicted (NO), but the vault contributed zero structural signal. The vault had no US macro thread (despite one being referenced in the economics domain), no GDP tail-risk concept, no procedure for GDP threshold forecasting, and no quarterly GDP data point in the 2025-Q1 quarter file. The correct NO prediction was supported by general knowledge of the base rate of -2% contractions (extremely rare), the shallow nature of tariff-induced contractions, and the absence of any GFC-class crisis in Q1 2025. The vault must provide structured, non-trivial signal for every future US GDP threshold question.

# Check entity files use title:/slug:
grep -rl "type: entity" domains/*/entities/ domains/*/threads/*/entities/ 2>/dev/null | xargs grep -L "^title:" | grep -v "^$" || true
```

51. **Southeast Asia as mandatory coverage domain**: The vault MUST maintain a dedicated Southeast Asia domain (domains/southeast-asia/) covering the region's 11 states, because (a) Southeast Asian political actors (Philippines, Indonesia, Thailand, Vietnam, Myanmar) generate recurring forecast-relevant questions about elections, elite succession, conflict, and international justice, (b) the region is a frontline state in US-China strategic competition (South China Sea, Taiwan contingency, supply chain relocation) that directly affects global security and economic forecasts, and (c) the vault had zero Southeast Asia coverage at the time of the Duterte ICC arrest question (Q62) — no domain, no entity stubs for any Philippine political actor, no thread tracking the Duterte arrest, no concept for ICC pretrial custody dynamics, even though Duterte had been mentioned in the 2025-Q1 timeline.

    The vault MUST maintain:

    - **A Southeast Asia domain entry** ([[domains/southeast-asia/_domain]]) documenting the region's forecasting-relevant characteristics (US-China competition, elite family dynasties, ASEAN fragmentation, South China Sea disputes, economic transformation) and listing active threads and entities.

    - **Entity stubs for ALL named actors in any Southeast Asia forecast question**: president, vice president, former president, party, key institutions. For the Duterte ICC arrest question, this meant entity stubs for [[domains/southeast-asia/entities/rodrigo-duterte]], [[domains/southeast-asia/entities/bongbong-marcos]], [[domains/southeast-asia/entities/sara-duterte]], and [[domains/southeast-asia/entities/philippines]] — none of which existed at the time of the forecast.

    - **A thread tracking the Duterte ICC arrest** ([[domains/southeast-asia/threads/duterte-icc-arrest/_thread]]) documenting the full timeline from March 2025 arrest through ongoing trial proceedings, including the Marcos-Duterte alliance dynamics and the 2028 election implications.

    - **A concept for ICC pretrial custody duration** ([[domains/global/concepts/icc-pretrial-custody-duration/_concept]]) documenting the structural timeline: arrest → initial appearance (3-7 days) → confirmation of charges hearing (60-90 days) → trial (2-4 years). The key insight for forecasting: release before confirmation of charges hearing is <5% for crimes against humanity charges, making any "released in March" question with a 20-day window a structural P(YES) < 0.05.

    **MANDATORY PRE-FORECAST CHECKS for any "released from custody" or "ICC arrest" question:**

    1. **Identify the arresting body** — Is it the ICC (slowest release mechanism, governed by Rome Statute Article 60), a domestic court, or a hybrid tribunal? The ICC has the highest structural barrier to pretrial release.

    2. **Calculate minimum time-to-release** — For ICC arrests: minimum release time = arrest_to_initial_appearance (3-7 days) + pre_trial_chamber_decision (14-30 days). If the question deadline is <30 days from arrest, P(release) < 0.05 for serious charges.

    3. **Assess charge severity tier** — Crimes against humanity, genocide, war crimes (widespread/systematic): Tier 1 — pretrial release probability < 5%. Isolated war crimes: Tier 2 (~10-20% release probability). Lesser charges: Tier 3.

    4. **Assess flight risk factors** — Former head of state or senior official: +40pp to flight risk assessment. Multiple passports or international property: +30pp. State of nationality hostile to ICC: increases theoretical flight risk (though the suspect is already in ICC custody).

    5. **Check state cooperation** — Did the state of nationality cooperate with the arrest? If yes, the state is unlikely to obstruct the ICC, but the former leader's personal resources are the primary flight risk factor. If no, the arrest was likely extrajudicial (rendition) and the state may actively obstruct — less relevant for release probability.

    6. **Default heuristic**: For any "released from ICC custody by date D" question where D is <60 days from arrest and the charge is Tier 1 (crimes against humanity, genocide, war crimes), the default forecast is P(YES) < 0.05, supported by the structural base rate and Rome Statute release criteria. Load [[domains/global/concepts/icc-pretrial-custody-duration/_concept]] before calibrating.

    **Quarter file coverage mandate**: Every contemporary quarter file must track the Duterte ICC case as a thread entry if active proceedings occurred during that quarter. The Q1 2025 file has the arrest; Q2, Q3, and Q4 2025 files were missing this content — a gap that must be remedied.

    This rule exists because Question 62 of the PIT blind test (Duterte released from custody in March?) was correctly predicted (NO) but the vault contributed ZERO signal: no Southeast Asia domain, no Philippine entity stubs, no Duterte entity, no ICC pretrial custody concept, and no thread tracking the case despite the 2025-Q1 timeline documenting the March 11 arrest. The correct prediction relied entirely on general knowledge of (a) the ICC's slow pretrial process, (b) the absurdity of releasing a former head of state 20 days after arrest on crimes against humanity charges, and (c) the structural base rate of ICC pretrial release. None of this was in the vault — a textbook violation of Spec Rule 8 ("No freebie predictions"). Every future Southeast Asian or international justice forecast question must find a vault with substantive, structured coverage of the region's actors, the ICC's procedural dynamics, and the political consequences of high-profile ICC prosecutions.

52. **US trade policy tariff escalation thread as mandatory for any tariff or auto sector question**: The vault MUST maintain a dedicated US trade policy and tariff escalation thread (`domains/global/threads/us-trade-policy-tariffs/`) covering:

    a. **The full tariff timeline** from the second Trump administration's inauguration (Jan 20, 2025) onward, including: Section 232 steel/aluminum tariff restoration (Feb 11, 2025), Canada/Mexico tariffs (March 2025), Liberation Day IEEPA tariffs (April 2, 2025), CIT vacatur (May 28, 2025), ongoing appeal, and sector-specific tariff extensions (semiconductors, pharmaceuticals, critical minerals).

    b. **The three-layer legal framework** for US tariff authority — Section 232 (national security), Section 301 (unfair trade), IEEPA (emergency) — with each layer's legal vulnerability assessment.

    c. **Trump's tariff escalation-bargaining pattern** captured as a dedicated concept ([[domains/global/concepts/trump-tariff-escalation-bargaining]]): the five-phase cycle of Announcement → Blowback → Tactical Retreat → Negotiation → Repeat at higher scope.

    d. **Entity stubs for key trade actors**: USTR Jamieson Greer, Commerce Secretary Howard Lutnick, the US Court of International Trade, and relevant industry entities (European auto sector representation, major automakers).

    e. **The EU dimension specifically**: EU retaliatory tariffs, US-EU trade negotiation status, European auto sector exposure to US tariffs, German auto sector as the largest European exporter affected.

    f. **Cross-domain connectivity**: The trade policy thread MUST link to the US monetary policy thread (Fed trapped between tariff inflation and tariff growth risk), the US macro-economic indicators thread (tariff-driven GDP contraction dynamics), and the US-China tech decoupling thread (tech-sector tariffs distinct from general trade tariffs).

    **MANDATORY PRE-FORECAST CHECKS for any tariff or auto sector question**:

    1. **Load the US trade policy thread** — Check whether the administration has already taken a tariff action in the questioned sector or on the questioned trading partner.

    2. **Identify the legal authority** — Is the tariff action under Section 232 (lower legal vulnerability), Section 301 (very low vulnerability), or IEEPA (high vulnerability — CIT vacatur precedent)? This determines whether the action is likely to survive legal challenge.

    3. **Assess phase in escalation-bargaining cycle** — Is the administration in Phase 1 (aggressive announcement — high probability), Phase 2 (blowback — can go either way), Phase 3 (tactical retreat — action unlikely to expand), Phase 4 (negotiation — tariffs serve as leverage, may not be durable), or Phase 5 (repeat — new action likely at higher scope)?

    4. **Check EU retaliation capacity** — Has the EU already announced retaliatory tariffs? What specific EU exports to the US are most politically sensitive (bourbon/whiskey, motorcycles, agricultural goods)? EU retaliation increases the political cost of US tariff action.

    5. **Assess auto sector specificity** — Does the questioned tariff target (a) automobiles directly, (b) auto inputs (steel, aluminum, semiconductors), or (c) generic imports that happen to include autos? Direct auto tariffs require additional legal steps (Section 232 auto investigation); input tariffs are already in place.

    6. **Default heuristic**: For any whether-Will-Trump-impose-tariffs-on-X question where X is a sector or trading partner:
        - If Section 232/301 authority exists and campaign rhetoric targeted X: >50% probability within 12 months
        - If only IEEPA authority exists (post-CIT vacatur): <30% until legal workaround is established
        - If X is an allied country (EU, Japan, South Korea): -20pp from above estimates due to intra-coalition political costs
        - If X is China or adversarial: +20pp due to bipartisan support for trade restrictions

    This rule exists because Q61 (US tariffs on European cars before May 2025) was correctly predicted (YES), but the vault had no dedicated trade policy thread connecting the tariff actions scattered across timeline files. The correct prediction relied on general knowledge of Trump's first-100-days tariff trajectory rather than vault-provided structural analysis. The tariff escalation-bargaining pattern was not formalized, the CIT role was not documented, and key trade actors (Greer, Lutnick) had no entity stubs. A future tariff question on a less obvious target or with more complex legal constraints would lack the structured framework needed for a well-calibrated forecast.

53. **Big Tech antitrust enforcement as mandatory coverage**: The vault MUST systematically cover antitrust and competition enforcement against the largest US technology platforms (Google, Meta, Amazon, Apple), because (a) forced divestiture questions via antitrust litigation are distinct from national security divestiture (TikTok) and follow a completely different legal, procedural, and political framework, (b) the multi-year timeline of antitrust litigation (5-10+ years from filing to remedy) makes near-term forced divestiture structurally low-probability, and (c) the 2025 administration shift from enforcement-first to permissive changes remedy posture in ongoing cases. The vault MUST maintain:

    - **A US Big Tech antitrust enforcement thread** ([[domains/usa/threads/us-big-tech-antitrust-enforcement/_thread]]) tracing the full arc from 2020 through 2026: the four major monopolization cases (DOJ v. Google, FTC v. Meta, FTC v. Amazon, DOJ v. Apple), their procedural timelines, the Biden enforcement-first era under Khan, the Trump permissive era under Ferguson, and the key legal and political dynamics affecting enforcement outcomes.

    - **An antitrust divestiture timeline concept** ([[domains/usa/concepts/antitrust-divestiture-timeline/_concept]]) documenting the structural 5-10+ year timeline from complaint filing to forced divestiture, the three-phase process (liability → remedy → appeals), the AT&T and Microsoft precedents, the critical distinction between post-consummation (already owned) and pre-consummation (proposed) mergers, and the comparative framework contrasting antitrust divestiture with national security divestiture.

    - **Entity stubs for the FTC** ([[domains/usa/entities/federal-trade-commission]]), key enforcement chairs including Lina Khan ([[domains/usa/entities/lina-khan]]) and Andrew Ferguson ([[domains/usa/entities/andrew-ferguson]]), the defendant Meta Platforms ([[domains/usa/entities/meta-platforms]]), and any judge presiding over a named antitrust case (e.g., Judge James Boasberg in FTC v. Meta: [[domains/usa/entities/judge-james-boasberg]]).

    - **An antitrust divestiture timeline forecast procedure** ([[domains/usa/procedures/antitrust-divestiture-timeline-forecast]]) formalizing the 6-step assessment: identify legal vehicle (antitrust vs national security) → map case timeline → apply default priors with adjustments for consummated status and political environment → assess remedy-phase barriers (feasibility of unwinding integrated assets) → check political environment (enforcement-first vs permissive administration) → final calibration.

    - **In every contemporary quarter file**: a subsection tracking developments in all active Big Tech antitrust cases — new filings, procedural rulings, discovery milestones, trial dates, settlement negotiations, and changes in agency leadership. This ensures the thread has quarterly data points for timeline estimation.

    **MANDATORY PRE-FORECAST CHECKS when a question asks about forced divestiture of a tech asset:**

    1. **Identify the legal vehicle** — Is the forced sale through (a) antitrust litigation (Sherman Act §2 or Clayton Act §7), (b) national security legislation (TikTok model), (c) executive order, or (d) regulatory action? Each has a fundamentally different timeline and probability profile. **This is the single most important pre-forecast distinction.**

    2. **If antitrust litigation: determine post-consummation vs. pre-consummation** — Is the asset already owned (post-consummation, higher bar, longer timeline) or a proposed acquisition (pre-consummation, lower bar for injunction)? Post-consummation divestiture via antitrust litigation defaults to p_yes < 0.03 for any <2-year horizon.

    3. **Map the case stage** — Has the complaint survived motion to dismiss? Is discovery ongoing? Has summary judgment been briefed? Has trial been scheduled? Each stage provides a floor on the remaining timeline.

    4. **Check the administration's enforcement posture** — Who is the FTC Chair / AAG Antitrust? What is their stated position on structural remedies? Has the agency signaled willingness to settle on weaker terms?

    5. **Assess remedy feasibility** — Can the asset be practically separated? Instagram and WhatsApp are deeply integrated into Meta's infrastructure (backend, user accounts, advertising systems, content moderation). Even if liability is found, feasibility of a clean divestiture is a separate and significant barrier.

    6. **Default heuristic**: For any "Will [Company] be forced to sell [Asset Acquired >5 years ago] via antitrust litigation within N years?" where N < 3: p_yes < 0.03, supported by the structural timeline of US antitrust litigation (5-10+ years from filing to remedy), the post-consummation remedy barrier, and the administration's enforcement posture. Load [[domains/usa/concepts/antitrust-divestiture-timeline/_concept]] and [[domains/usa/procedures/antitrust-divestiture-timeline-forecast]] before calibrating.

    **Quarter file coverage mandate**: Any quarter where antitrust enforcement developments occurred must have a subsection under "Regulation & Antitrust" documenting: (1) all procedural developments in the four major Big Tech cases (Google, Meta, Amazon, Apple), (2) changes in agency leadership or enforcement posture, (3) congressional antitrust legislation activity, and (4) EU DMA enforcement actions against Big Tech (since EU and US enforcement are increasingly cross-referenced).

    This rule exists because Q63 of the PIT blind test (Meta forced to sell Instagram/WhatsApp in 2025) was correctly predicted (NO) but the vault contributed ZERO structural signal — no antitrust enforcement thread, no divestiture timeline concept, no entity stubs for the FTC, Lina Khan, Judge Boasberg, or Meta as defendant. The correct NO prediction relied entirely on general knowledge of (a) how long US antitrust cases take (5-10 years, filed 2020), (b) the 2025 administration shift reducing remedy pressure, and (c) the unprecedented nature of a post-consumption asset divestiture. Every future antitrust divestiture question must find vault coverage with structured, non-trivial analytical signal — the antitrust divestiture timeline concept must provide the default probability framework, the thread must provide the case-specific timeline, and the entities must provide the actor-level context.

11. **Ceasefire questions must be pathway-classified before probability estimation**: A "ceasefire" between two states and a "ceasefire" between a state and a non-state actor resolve to the same Polymarket YES/NO outcome, but the causal pathways differ by orders of magnitude in probability and timeline. Before estimating any ceasefire probability, the vault MUST classify the question into one of three pathways defined in [[domains/global/concepts/ceasefire-pathway-decomposition/_concept]]:

    - **Pathway B (War-Termination)**: State-on-state conflict with identifiable escalation ladder, superpower patron with escalation dominance, adversary without nuclear deterrent against the superpower. P(ceasefire) = P(war in window) × P(termination | war). The escalation-ladder coupling means P(ceasefire) can be structurally high (0.50-0.80) even when no diplomatic framework exists.
    - **Pathway A (Diplomatic/Negotiated)**: All other cases — asymmetric conflicts, state-on-state without superpower termination mechanism. P(ceasefire) follows diplomatic base rates (0.01-0.50 depending on duration, pressure, political deadlines).
    - **Pathway C (None likely)**: Nuclear-armed adversaries, existential war aims, no credible mediator.

    Failure to classify is the single most common source of error on ceasefire forecasting questions. The gold_01 question (Iran-Israel ceasefire before July 2025) is the canonical example: the question was Pathway B (war-termination) but was treated as Pathway A (diplomatic), producing a NO prediction when the correct answer was YES. See [[domains/global/concepts/ceasefire-pathway-decomposition/_concept]] for the full framework and [[domains/global/procedures/state-on-state-ceasefire-decomposition]] for the step-by-step procedure.

11a. **Ceasefire entity completeness — security councils and ratification bodies**: Every state actor with a formal ceasefire ratification body (security cabinet, war cabinet, supreme national security council, Duma, etc.) that appears in a forecasting question must have a vault entity file documenting its composition, decision-making procedure, and crisis-acceleration mechanisms. The Israeli security cabinet ([[domains/mena/entities/israeli-security-cabinet]]) is the canonical case: its crisis-accelerated approval process (0 hours vs 1-2 days standard) is a critical variable for war-termination ceasefire timing. Questions about ceasefires involving any state with a defined ratification body (Iran's SNSC, Russia's Security Council, India's Cabinet Committee on Security, Pakistan's National Security Council) must check for and validate entity coverage of that body.

11b. **Inter-state conventional war ceasefire structural analysis is mandatory coverage**: The vault MUST systematically assess the structural feasibility of inter-state conventional war ceasefires using the protracted-war-stalemate framework, because inter-state ceasefires follow fundamentally different dynamics from asymmetric (state-vs-non-state) or escalation-bargaining (Iran-Israel pattern) ceasefires. The vault MUST maintain:

    - **A dedicated inter-state ceasefire feasibility procedure** ([[domains/global/procedures/inter-state-ceasefire-feasibility]]) formalizing the 8-step checklist: territorial incompatibility → military trajectory → mutually hurting stalemate → credible mediator → external sustainment → political deadline → synthesis → context-specific adjustment. This procedure MUST be loaded before any inter-state ceasefire forecast.

    - **A protracted-war-stalemate concept** ([[domains/global/concepts/protracted-war-stalemate]]) documenting the five-factor mechanism (territorial incompatibility, military balance parity, no mutually hurting stalemate, no credible mediator, external sustainment) and the contrast with fast-ceasefire patterns (escalation-bargaining-termination) and medium-term patterns (diplomatic-pressure-tipping-point). This concept MUST include the validation entry: Russia-Ukraine 2024 ceasefire? → NO ✓.

    - **A Russia-Ukraine war thread** ([[domains/global/threads/russia-ukraine-war/_thread]]) tracking the full conflict timeline with structural ceasefire barrier analysis for each phase. The thread MUST include the six-factor ceasefire forecasting checklist and quarterly event coverage.

    - **Entity stubs for both belligerents and their leadership**: Russia ([[domains/global/entities/russia]]), Ukraine ([[domains/global/entities/ukraine]]), Vladimir Putin ([[domains/global/entities/vladimir-putin]]), Volodymyr Zelenskyy ([[domains/global/entities/volodymyr-zelenskyy]]), and any military leadership named in a forecast question (e.g., Valeriy Zaluzhnyi, Oleksandr Syrskyi).

    - **Quarter file coverage**: Every contemporary quarter file (2022-2025) MUST include a subsection documenting the Ukraine war's trajectory, territorial changes, US/EU aid status, diplomatic activity, and structural ceasefire feasibility assessment at the PIT cutoff. A quarter file covering 2024 with no Ukraine war section is missing the single most consequential ongoing conflict of the period.

    **MANDATORY PRE-FORECAST CHECKS for any inter-state conventional war ceasefire question:**

    1. **Classify the ceasefire pathway** using [[domains/global/concepts/ceasefire-pathway-decomposition]] BEFORE any analysis. Is this a diplomatic ceasefire (Pathway A) or an escalation-bargaining termination (Pathway B)? A nuclear-power conflict defaults to Pathway C (none likely) because the escalation-bargaining mechanism does NOT apply when the adversary has nuclear deterrence against the superpower patron.

    2. **Load the inter-state ceasefire feasibility procedure** — [[domains/global/procedures/inter-state-ceasefire-feasibility]] — and run ALL 8 steps. Skip the procedure only if the pathway classification confirms Pathway C (none likely).

    3. **Load the protracted-war-stalemate concept** — [[domains/global/concepts/protracted-war-stalemate]] — and verify the 5-factor assessment against current conditions.

    4. **Load the Russia-Ukraine war thread** — [[domains/global/threads/russia-ukraine-war/_thread]] — if the question involves that conflict. The thread provides phase-by-phase structural analysis.

    5. **Check for credible mediator emergence**: The single most dynamic variable in inter-state ceasefires is the emergence of a credible mediator with leverage over both parties. For Russia-Ukraine, this was absent in 2024 (Biden refused to pressure Ukraine, China was pro-Russia) and only emerged with the Trump administration in Jan 2025. Did a new leader take office who could fill this role? If no → ceasefire probability remains near-baseline low.

    6. **Check for external sustainment changes**: Has a patron signaled reduced willingness to supply arms or finance? The US aid gap (Oct 2023-Apr 2024) is the canonical case — the delay itself signaled fragility and reduced Russia's incentive to negotiate. If no patron change → continuation is feasible.

    7. **Check domestic political space**: Does either side's leadership have political room to make territorial concessions? The Ukraine law forbidding negotiations with Putin (Sep 2022) and the Russian constitutional incorporation of annexed territories (Sep 2022) created legal barriers that required at least one side to violate its own framework to accept a ceasefire.

    8. **Apply the default heuristic**: If all structural barriers (territorial incompatibility, military parity, no hurting stalemate, no credible mediator, stable external sustainment) are present, the default forecast for a ceasefire within 12 months is P < 0.10. Only a material change in one of these variables (new mediator, patron cutoff, battlefield breakthrough) can elevate probability.

    This rule exists because the Russia-Ukraine ceasefire in 2024 question (Gold Q39) was correctly predicted (NO), but the vault at the time of the initial forecast had a thread that ended at Sep 2022 with a 3+ year gap, no protracted-war-stalemate concept, no inter-state ceasefire feasibility procedure, and 2024 quarter files with minimal war coverage. The correct prediction was supported by general knowledge, not vault content — a violation of Spec Rule 8 (\"No freebie predictions\"). After this rule, every future inter-state conventional war ceasefire question will trigger systematic structural analysis using the procedure, concept, thread, entities, and quarterly coverage that were missing for this question.

54. **Numerical range questions require double-filter analysis (base event + range plausibility)**: When a forecasting question specifies a numerical range as the resolution condition (sentence length, price target, vote share bin, inflation band, time window, age threshold, percentage), the vault MUST apply the [[domains/global/concepts/forecast-range-plausibility-filter]] concept BEFORE any other reasoning. The double-filter framework treats the base event (does X occur at all?) and the range plausibility (is the specified magnitude structurally possible?) as structurally independent assessments and requires both to be documented separately before combination.

    **Rationale**: The single most common forecasting error on range-specified questions is answering "will the base event occur?" when the question is "will the base event occur at the SPECIFIED magnitude?" — and the specified magnitude may be structurally improbable regardless of the base event's direction. The question designer may select the range precisely because it seems "in the middle" (making it a tempting prediction), while structural range analysis shows it is actually a tail outcome. This is amplified when the range spans a structural discontinuity (different felony class, different price regime, different vote-share bin).

    **MANDATORY STEPS for any range-specified question:**

    1. **Parse the resolution text** — extract the exact numerical range from the question. Write it explicitly.

    2. **Identify the structural bounds** — What is the full possible range of outcomes? What is the standard/modal outcome? What distribution does the specified range fall in (normal, tail, off-distribution)?

    3. **Apply Filter A: Base Event** — Does the underlying event occur at all? P(any outcome in this dimension). Use standard forecasting methodology for the specific domain.

    4. **Apply Filter B: Range Plausibility** — Is the specified magnitude structurally plausible for this phenomenon, independent of whether the base event occurs? Consider:
       - Does the range correspond to a different structural category? (e.g., Class E felony → Class C/D range)
       - Is the range narrow relative to the base distribution? 
       - Is the range near the theoretical maximum or minimum?
       - What structural constraints bound the outcome? (sentencing guidelines, market mechanics, electoral system)

    5. **Combine with structural independence**:
       - If Filter B = NO with >95% confidence: Answer is NO regardless of Filter A.
       - If Filter B = NO with moderate confidence: P(overall YES) = P(Filter A YES) × P(Filter B YES | A) — typically very low.
       - If Filter B = YES: P(overall YES) = P(Filter A YES) × P(within-range | event occurs and range is plausible).

    6. **Document both filters separately** in the reasoning, and flag whether the range itself was the trap.

    **Default heuristic**: For any range-specified question where the range corresponds to a different structural category or is more than 1 standard deviation from the expected outcome, the default P(YES) should be < 0.10 regardless of the base event direction, because the range itself filters out nearly all probability mass from the distribution.

    **Validating examples**:
    - "Trump sentenced to 12-23 months?" — Filter A: NO (president-elect, prosecution concession). Filter B: NO (Class E, standard range 0-16mo, 12-23mo exceeds first-offender range). Correct NO.
    - "Trump sentenced to 24-35 months?" — Filter A: NO (same). Filter B: NO (Class E → Class C/D range structurally disproportionate). Correct NO.
    - These examples validate the framework as most valuable not for novel predictions but for confirming that correct forecasts were not lucky — the double-filter provides independent structural backing.

    This rule exists because the Q14 Trump sentencing 12-23 months question was correctly predicted (NO) but the reasoning relied on general knowledge of NY sentencing guidelines and officeholder constraints without the structured double-filter framework. After this rule, any future range-specified question in ANY domain will trigger the double-filter analysis before the domain-specific methodology, ensuring the range itself is not the forecasting trap.

16. **Procedure-referenced entity completeness**: Every named individual, organization, or entity listed in a procedure's `entities:` frontmatter or referenced in the procedure body as "Key Entities to Consult" MUST have an entity stub file in the vault. If a procedure says "consult [[entities/yahya-sinwar]]" and no such file exists, the vault has a procedure-entity completeness gap as severe as a missing question-referenced entity.

    The threshold is met by any of:
    - The entity appears in the procedure's `entities:` frontmatter
    - The procedure body contains a "Key Entities to Consult" section naming the entity
    - The procedure body contains a wikilink to the entity that is used as a knowledge source for the forecast

    **Rationale**: Procedures are the vault's executable knowledge — they are loaded when a forecasting question arrives and their referenced entities are assumed to exist. If a procedure recommends consulting an entity that does not exist, the forecaster either (a) wastes time creating the entity during the forecast (when PIT pressure is highest), or (b) skips the consultation and loses the entity's signal. Both degrade forecast quality.

    **Procedure creation requires entity stub check**: When creating or updating any procedure, the author MUST verify that every entity referenced in the procedure's frontmatter and body has a corresponding entity stub file. If any referenced entity lacks a stub, either (a) create the stub before the procedure is published, or (b) remove the reference and document why the entity is not needed.

    **Canonical case**: The asymmetric ceasefire forecast procedure (`domains/mena/procedures/asymmetric-ceasefire-forecast.md`) listed "[[entities/yahya-sinwar]]", "[[entities/ismail-haniyeh]]", and "[[entities/hassan-nasrallah]]" as entities to consult but none had entity stubs at the time of the procedure's creation. These are three of the most consequential individuals in the entire Israel-Hamas-Hezbollah-Iran conflict. Their absence was a procedure-entity completeness gap that was not remedied until Cycle 16 reflection. After this rule, every procedure undergoes an entity completeness audit before publication or significant update, and any referenced-but-missing entities are stubbed first.

55. **Deadline-constrained analysis as mandatory pre-forecast step for "before [deadline]" withdrawal questions**: When a question asks whether a political leader will withdraw, resign, or step down BEFORE a specified deadline (convention date, filing deadline, nomination vote, leadership ballot, or any fixed calendar date), the vault MUST apply the deadline-constrained withdrawal framework before calibrating probability. The compound probability structure is:

    P(event before deadline) = P(any trigger by effective_trigger_deadline) × P(cascade completes in <remaining_time | trigger)

    The vault MUST document:

    - **The deadline date and its distance from the cutoff**: Record the exact deadline. If the deadline is not explicitly stated in the question, infer it from the resolution text or known institutional calendar (DNC dates, filing deadlines, etc.).

    - **The effective trigger deadline**: Compute as `deadline - cascade_completion_upper_bound` (24 days for debate/performance triggers, 19 days for primary near-loss, 18 days for primary loss). The trigger must occur before this date for the cascade to complete before the deadline.

    - **Constraint binding assessment**: Has a trigger already occurred? If YES and remaining time > cascade_time → constraint is NOT binding. If NO trigger → the effective window for trigger occurrence is reduced from the full forecast horizon to `cutoff_to_effective_trigger_deadline`. Document which scenario applies.

    - **Compound probability calculation**: P(before deadline) = P(trigger by effective_deadline) × cascade_completion_rate (~85%). For post-trigger cutoffs where constraint is not binding, P(before deadline) ≈ P(withdrawal | trigger).

    - **Cascade acceleration factor**: If the deadline itself could accelerate the cascade (because the party faces a binding convention date with a successor ready), document whether acceleration is plausible and adjust the cascade time estimate downward (14-18 days instead of 24).

    This framework is documented in full in [[domains/usa/concepts/incumbent-withdrawal-cascade#Deadline-Constrained Withdrawal Forecasting]].

    The canonical case is the Biden-before-DNC question (gold_18, predicted NO, actual YES). At a pre-debate cutoff, the DNC deadline (Aug 19) minus cascade time (24 days) gave an effective trigger deadline of ~July 26. The scheduled June 27 debate within this window dramatically increased the per-period trigger probability — the debate was itself a trigger risk. A forecaster who computed the compound probability would have arrived at ~45-55%, not a flat NO. At a post-debate cutoff, the deadline constraint was not binding (53 days > 24-day cascade), making the YES prediction the clear default.

    This rule exists because the gold_18 and gold_12 errors share a structural root: answering "will the event occur?" without explicitly modeling the compound probability that includes timing constraints. The vault's procedure step 16 now includes the deadline-constrained sub-step, the concept now includes the three-scenario model, and this rule ensures the framework is MANDATORY for any "before [deadline]" withdrawal question going forward.

56. **Monetary policy developments are mandatory coverage in every contemporary timeline file**: Every contemporary (post-2020) quarter file in `timeline/` MUST include an Economics & Monetary Policy section documenting the major central bank decisions (Fed, ECB, BoJ, BoE) for that quarter. The minimum requirements:
    - **For each FOMC meeting**: Record the rate decision (hike/hold/cut with bps), any forward guidance language shift, and dissenting votes with identities.
    - **For quarterly SEP meetings (Mar, Jun, Sep, Dec)**: Record the dot plot median projection for the federal funds rate at year-end for the current and next year.
    - **For CPI/PCE data**: Record the headline and core readings for the last month of the quarter, noting whether disinflation is accelerating or stalling.
    - **Market pricing shift**: Note how market-implied expectations (CME FedWatch, SOFR futures) changed over the quarter.
    - **Other major central banks**: Record key decisions from the ECB (which sets the tone for European rate forecasts), BoJ (which affects global carry trades), and BoE.
    - **Connections**: Link to the [[domains/economics/threads/us-monetary-policy-cycle-2022-2026]] thread and related entities.

    A quarter file that documents Middle East escalations, European elections, and US budget politics but omits the Fed's dot plot shift from 3 cuts to 1 cut (June 2024 SEP) or the ECB's first cut (June 2024) is materially incomplete for monetary policy forecasting. The standard for "mandatory" is met by a dedicated economics section at the same level of detail as geopolitical sections — not a single bullet point buried in the US politics subsection.

    **Canonical case**: The 2024-Q1 and 2024-Q2 timeline files in the graph-vault (remediated per Q25 reflection) originally had zero monetary policy content despite containing 4 FOMC meetings that were directly relevant to Q25 (Fed rate hike after July 2024) and Q17 (Fed rate cut after July 2024). The June 2024 SEP dot plot shift — which was the single most important structural signal for these forecast questions — was documented only in the pit_blind_test quarters, not in the vault's canonical timeline files. The remediation added full economics sections to both quarter files and added the monetary policy check to the procedure's Phase 1 scoping checklist.
