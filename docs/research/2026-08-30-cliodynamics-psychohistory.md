# Cliodynamics, Asimov’s psychohistory, and this plugin

**Date:** 2026-08-30  
**Status:** research brief (not an ADR; does not change overlay)  
**Books:** seven Foundation-cycle EPUBs in `~/Downloads`, downloaded 2026-08-30. Short quotes only; paraphrase the rest.  
**Question:** what does current cliodynamics actually claim, how does Asimov describe psychohistory, and what methods (not ontologies) are worth stealing for analog cards + dated claims?

This plugin already says it is **not** Seldon mathematics (`CONTEXT.md`). The point of this note is to source the *constraints* Asimov put on the fiction, and the *methods* cliodynamics and related computational fields actually use, so later overlay review is not sci-fi cosplay.

---

## 1. What Asimov actually says

Read from the EPUBs. Prefaces that recount Campbell and 1941 story-planning are author commentary, not in-world science; they are marked as such.

### 1.1 Encyclopaedia Galactica definition (*Foundation*)

The in-world encyclopaedia treats psychohistory as a **statistical science of conglomerates**, not of persons. Two load-bearing conditions appear in the same entry:

- The conglomerate must be **large enough** (Seldon’s First Theorem is named, then elided).
- The conglomerate must be **unaware of psychohistoric analysis**, “in order that its reactions be truly random.”

Hari Seldon, later in the same book, treats inserting “the vagaries of an individual” into the equations as **risky**. Gaal Dornick is told psychohistory “cannot predict the future of a single man with any accuracy.”

### 1.2 Masses, not men (*Foundation and Empire*)

The later encyclopaedia entry is more explicit: psychohistory “dealt not with man, but with man-masses. It was the science of mobs; mobs in their billions.” It claims definite laws for **mass action**, “without pretending to predict the actions of individual humans,” with a billiard-ball metaphor for stimulus → mass reaction.

The same novel states the first proposition of Seldon’s psychohistory as: “the individual does not count, does not make history, and that complex social and economic factors override him.”

### 1.3 Two named axioms (late cycle)

*Foundation and Earth* (Trevize / Pelorat) restates what “everyone knows”:

1. **N large enough** for statistical treatment of randomly interacting individuals.
2. **Humanity must not know the predictions** before they are achieved (bare *existence* of the science is distinguished from knowledge of the *results*).

Trevize then names a **third, unspoken axiom**: the two known axioms assume humans are the only intelligent species whose actions matter. Daneel’s Gaia/Galaxia project is the book’s way of breaking that assumption. That third axiom is late-cycle metaphysics; the first two are the durable method.

### 1.4 Probability, error growth, Seldon crises

Psychohistory predicts **probabilities, not certainties**. *Foundation and Empire*: the margin of error “increases in geometric progression” with time. The Plan therefore uses **Seldon crises**: arranged branch-points that force the Foundation onto the high-probability path. A crisis is a **forcing function with a public answering moment**, not a full-timeline forecast of every event.

### 1.5 The Mule (individual who is not a random draw)

The Mule is the designed counterexample to axiom 1: a mutant whose extra-systemic power lets **one person** override mass dynamics. *Foundation’s Edge* (author preface) calls this “something which Hari Seldon could not have foreseen.” In-world, the Plan is said to have been **broken** until the Second Foundation repaired it. Later, Gendibal invents “micropsychohistory” (prediction of small groups or individuals) as a method the Second Foundation **does not have**; an “Anti-Mule” who *perfects* the Plan rather than disrupting it is the feared alternative.

### 1.6 Observer effect as axiom, not etiquette (*Second Foundation*)

A Speaker states the reason the Plan must stay hidden: the laws “are statistical in nature and are rendered invalid if the actions of individual men are not random.” If a sizable group learns key details, “their actions would be governed by that knowledge and would no longer be random in the meaning of the axioms.” This is an **observer effect**, not a secrecy fetish.

The Second Foundation exists to **fine-tune** when currents leave the plotted path (*Foundation’s Edge*). First Foundation operates openly along the Plan; Second Foundation keeps the mathematics and adjusts.

### 1.7 How the science is supposed to be *built* (*Prelude*, *Forward*)

*Prelude to Foundation* is the origin story. Useful method, not plot:

- Seldon starts with elegant equations in “innumerable unknowns” and **no practical technique**.
- He cannot ingest all of history; Dors tells him that if he must know *all* of it, he will never formulate the laws.
- He looks for a **simpler society** as a testbed (legendary single world; then Trantor as the one world under his feet).
- Hummin (Demerzel) blocks a fake test: if you already know the year-1000 outcome, you will tune the equations to match. That is **not a fair test**.
- Rashelle can **fake** psychohistory: people will act on rumoured predictions even if Seldon says nothing. The *belief* that a science exists already changes behavior.

*Forward the Foundation*: the Project is a large political machine; psychohistory still cannot speak of the future “with certainty” even while the Empire’s disintegration is visible without it. Mentalics (Wanda) appear as a **different** instrument from the mathematics.

### 1.8 What the fiction is *not*

Asimov (1941 preface, *Foundation*): Campbell and he “thrashed out” psychohistory as a narrative device to illuminate a thousand-year interregnum. It is not a research program. The later books add robots, Gaia, and a missing Earth; those are not load-bearing for a forecasting plugin.

---

## 2. What cliodynamics actually is

**Primary self-description.** Turchin, *Nature* 454:34–35 (2008), “Arise ‘cliodynamics’”: history should become an “analytical, and even a predictive, science”; long-timescale processes affect societal health; competing untested explanations of the same collapse (Rome) are “as risible as if, in physics, phlogiston theory and thermodynamics coexisted on equal terms.” PDF mirror: [peterturchin.com Nature 2008](https://peterturchin.com/publications/arise-cliodynamics/). DOI: [10.1038/454034a](https://doi.org/10.1038/454034a).

He coined the name in *Historical Dynamics* (Princeton, 2003): combine **mathematical models** with **statistical tests on historical data**, starting from territorial rise/fall, collective solidarity, and population–instability feedbacks.

**Journal.** *Cliodynamics: The Journal of Quantitative History and Cultural Evolution* (UC Riverside eScholarship, 2010–). Scope (journal site): transdisciplinary mix of historical macrosociology, cultural evolution, cliometrics, mathematical models of long-term social processes, and historical databases. Mature theory “integrates models with data.” Open access, CC BY. Latest volumes: 15 (2024), 16 (2025). [escholarship.org/uc/cliodynamics](https://escholarship.org/uc/cliodynamics).

Turchin’s 2010 launch column: search for general principles, build models, discover empirical patterns, test assumptions, test predictions on actual societies. [Launching the Journal](https://escholarship.org/uc/item/70p271c9).

This is **not** “predict every election.” It is: treat societies as dynamical systems; prefer mechanism + data over narrative uniqueness; use **retrodiction** and rare **live forecasts** as theory tests.

Turchin, “Toward Cliodynamics,” *Cliodynamics* (2011): scientific prediction is “also (and much more usefully) employed in empirical tests of scientific theories”; forecasts of the future are optional; self-defeating prophecy and chaos are named reasons accurate *future* social forecasts often fail. Analogy: weather (days) vs climate (forced, slower). [doi:10.21237/C7clio21210](https://doi.org/10.21237/C7clio21210), [PDF](https://escholarship.org/content/qt82s3p5hj/qt82s3p5hj.pdf).

### 2.1 Structural-demographic theory (SDT)

**Origin.** Jack A. Goldstone, *Revolution and Rebellion in the Early Modern World* (1991). Revolutions compared to earthquakes: **pressures** (slow structural conditions) vs **triggers** (sudden releasing events). Triggers are hard or impossible to predict; pressures build slowly.

**Developed as equations.** Turchin and others (Nefedov; Korotayev et al.). Three compartments — **population, elites, state** — plus **instability**, linked by nonlinear feedbacks. Three named pressures:

| Pressure | Typical observables | Role |
| -------- | ------------------- | ---- |
| Popular immiseration | real/relative wages, access to land, youth bulge, urbanization | Mass Mobilization Potential (MMP) |
| Elite overproduction | elite numbers vs offices, elite incomes, intra-elite conflict | Elite Mobilization Potential (EMP) |
| State fiscal distress | debt/GDP, trust/legitimacy | State Fiscal Distress (SFD) |

**Political Stress Indicator (PSI):** product of MMP × EMP × SFD. Rising PSI ⇒ rising *risk* of unrest, **not** a named event. Formulas and Antebellum US illustration: Turchin, “Modeling Social Pressures Toward Political Instability,” *Cliodynamics* 4 (2013). [doi:10.21237/C7clio4221333](https://doi.org/10.21237/C7clio4221333). Decade scoring: Turchin & Korotayev, *PLOS ONE* 15(8): e0237458 (2020). [doi:10.1371/journal.pone.0237458](https://doi.org/10.1371/journal.pone.0237458). Turchin & Hoyer 2023 (OSF `yrqw5`) call for **pre-registering** contemporary PSI forecasts and admit one 2010 hit “may have been a result of luck.”

**Secular cycles** (~2–3 centuries) vs **fathers-and-sons** ~50-year spikes (US peaks ~1870, 1920, 1970). Turchin, *Journal of Peace Research* (2012): 1,590 US political-violence events, 1780–2010. [doi:10.1177/0022343312442078](https://doi.org/10.1177/0022343312442078). Book-length US application: *Ages of Discord* (2016); popular restatement: *End Times* (2023).

**Five-stage schematic** (Hoyer et al. 2025, *Cliodynamics*): growth → immiseration + elite golden age → elite overproduction → state starvation of revenue/legitimacy → trigger. Idealized; not a law of every polity. [doi:10.21237/c7clio.38365](https://doi.org/10.21237/c7clio.38365).

**Gingko-leaf (critical for this plugin).** Same paper: pressures in the run-up are comparatively **narrow** (the stem); **outcomes fan** (the leaf). SDT was “not designed to explain how different societies would respond” once stress is high. Crisis *onset pressure* ≠ crisis *shape*.

### 2.2 Seshat: Global History Databank

Founded 2011 (Evolution Institute; Turchin, Whitehouse, François, Currie, Feeney et al.). Codes polities on comparable variables so hypotheses can be tested rather than illustrated. Site: [seshat-db.com](https://www.seshat-db.com/).

Landmark paper: Turchin, Currie, Whitehouse, François et al., *PNAS* 115(2): E144–E151 (2018). 414 societies, 30 NGAs, ~10,000 years, 51 variables / 9 characteristics. Result: a **single principal component** explains **77.2 ± 0.4%** of variance in those complexity traits (they coevolve). [doi:10.1073/pnas.1708800115](https://doi.org/10.1073/pnas.1708800115) (PMC: [PMC5777031](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5777031/)).

Methods paper: Turchin, “Fitting Dynamic Regression Models to Seshat Data,” *Cliodynamics* 9 (2018): 25–58. Time-series cross-cultural models that control for spatial diffusion and Galton’s problem (shared ancestry). [doi:10.21237/C7clio9137696](https://escholarship.org/uc/item/99x6r11m).

**Coding rule that analog cards should copy:** unknown is first-class (`absent` / `present` / `inferred` / `unknown`), not a zero. Whitehouse et al. 2019 *Nature* letter on moralizing gods was **retracted** (2021) after Beheim et al. showed missing values recoded as absences reversed the finding. Do not cite that letter. [retraction](https://doi.org/10.1038/s41586-021-03656-3); [Beheim](https://doi.org/10.1038/s41586-021-03655-4). Founding databank paper: Turchin et al. 2015, *Cliodynamics* 6. [doi:10.21237/C7clio6127917](https://doi.org/10.21237/C7clio6127917).

### 2.3 CrisisDB

Project page: [peterturchin.com/research/current-research/crisis-databank/](https://peterturchin.com/research/current-research/crisis-databank/). Aim: ~300 historical crisis cases; contemporary extension (“Our Ages of Discord”) with 100+ annual variables since 1900 for a growing country set.

**Hoyer, Turchin, et al., *Social Science History* (published 29 Dec 2025).** CrisisDB: **168** uniformly coded cases. Consequences (war, epidemic, reform, fragmentation…) are **highly variable, uncorrelated with each other, and largely unpredictable** from polity size, religion, administrative scale, or recency. “There is no ‘typical’ societal crisis of the past.” [doi:10.1017/ssh.2025.10113](https://doi.org/10.1017/ssh.2025.10113).

**Hoyer et al., “Crises Averted,” *Cliodynamics* (2025).** Four outlier cases (early Republican Rome; Chartist England; Alexander II Russia; US Progressive Era) where SDT pressures rose but adaptations limited the worst outcomes. Pressures can be forecast-ish; **severity is not predetermined**. [doi:10.21237/c7clio.38365](https://doi.org/10.21237/c7clio.38365).

**Multipath forecasting (program, not a scored archive).** Turchin, Witoszek, Thurner et al., “A History of Possible Futures,” *Cliodynamics* 9 (2018): ensemble stats on a crisis database, then retrodict held-out upheavals. [doi:10.21237/C7clio9242078](https://doi.org/10.21237/C7clio9242078).

### 2.4 Cultural evolution / warfare ABMs

Turchin, “Warfare and the Evolution of Social Complexity: A Multilevel-Selection Approach,” *Cliodynamics* (2008). Intergroup competition as a driver of ultrasocial institutions. [doi:10.5070/sd943003313](https://doi.org/10.5070/sd943003313).

Turchin, Currie, Turner & Gavrilets, “War, space, and the evolution of Old World complex societies,” *PNAS* (2013). Afroeurasian ABM 1500 BCE–1500 CE: warfare + military-tech diffusion explained **65%** of variance in large-society locations; ablating mil-tech dropped that to **16%**. [doi:10.1073/pnas.1308825110](https://doi.org/10.1073/pnas.1308825110).

Turchin et al., “Disentangling the evolutionary drivers of social complexity,” *Science Advances* (2022). Best-supported among 17 predictors: agricultural productivity + military technology; moralizing “Big Gods” not the initial driver. [doi:10.1126/sciadv.abn3517](https://doi.org/10.1126/sciadv.abn3517).

Kondor & Turchin (CrisisDB page): agent-based models of Neolithic European farmers vs archaeological population series; **density-dependent conflict** outperforms climate-only drivers for boom–bust.

These are **mechanism tests on the past**, not a live ticker. Do not import Seshat’s nine complexity characteristics as overlay fields.

---

## 3. Empirical track record (honest)

| Claim | Kind | Result | Source |
| ----- | ---- | ------ | ------ |
| “Next decade likely growing instability in US and Western Europe” | **Live forecast** (2010, before the decade) | CNTS riots/demonstrations rose after 2010; US 1960s spike was violence-heavy, 2010s spike demonstration-heavy | Turchin, *Nature* 463 correspondence (3 Feb 2010), [PDF](https://peterturchin.com/wp-content/uploads/2023/07/Nature2020letter.pdf); retrospective Turchin & Korotayev 2020 |
| English Civil War, French 1789, 19th-c. France/Germany PSI | **Retrodiction** (theory built on some of these) | Goldstone: PSI leading indicator in his detailed cases | Goldstone 1991; summarized in Turchin & Korotayev 2020 |
| US 1780–2010 violence oscillations | **In-sample / historical pattern** | Secular wave + 50-year spikes | Turchin 2012 *JPR* |
| Seshat complexity PC1 | **Comparative pattern**, not a dated event | One dimension structures complexity variables | Turchin et al. 2018 *PNAS* |
| Crisis *consequences* | **Negative result** | No typical crisis outcome | Hoyer et al. 2025 *SSH* |
| SDT labor-oversupply → wages → elite overproduction → PSI in industrial US | **Critique** | Georgescu 2023: automation/globalization explain wage variance better; elite-income path does not match hump-shaped model; PSI rise tracks inequality, not the SDT mechanism | Georgescu, *PLOS ONE* (2023), [doi:10.1371/journal.pone.0287912](https://doi.org/10.1371/journal.pone.0287912) |

Turchin’s own 2010 framing (later restated): he did **not** claim the future can be predicted “with any accuracy”; the letter was a **theory test**. If it failed, the theory should change. That is closer to this plugin’s reflect loop than to a Seldon Plan.

The *Nature* note also **stacks clocks** (PSI-style indicators + 50-year spike due ~2020 + Kondratiev dip + youth bulge). A decade-scale rise in protest frequency is a coarse hit; it did not pre-register named events. Turchin & Hoyer 2023: one success is not enough and “may have been a result of luck.”

**Adjacent decay.** Bowlsby, Chenoweth, Hendrix, Moyer, “The future is a moving target,” *BJPS* (2019): PITF-style accuracy **peaks in the original validation window and decays later**. [doi:10.1017/s0007123418000443](https://doi.org/10.1017/s0007123418000443). A card that worked in one decade can die.

**Circularity risk:** early SDT tests used the same cases that inspired the theory. Later Seshat/CrisisDB work is the attempt to escape that. Georgescu is the live reminder that **industrial mechanisms may not be agrarian ones**. Effective N for *live industrial* PSI forecasts is still tiny (US letter; US/UK/W. Europe protest counts).

---

## 4. Adjacent computational and agent-driven fields

Not cliodynamics, but the real computational neighbors.

### 4.1 Generative social science / classical ABM

- Epstein & Axtell, *Growing Artificial Societies* (Brookings/MIT, 1996). Sugarscape: grow inequality, trade, disease from local rules.
- Epstein, *Generative Social Science* (Princeton, 2006). Explanation = **sufficient generative mechanism**, not a fitted curve.

Historical ABMs that *earned* the mechanism first: Axtell et al., Kayenta Anasazi collapse, *PNAS* (2002) [doi:10.1073/pnas.092080799](https://doi.org/10.1073/pnas.092080799); Brughmans & Poblome, MERCURY Roman tableware, *Antiquity* (2016) [doi:10.15184/aqy.2016.35](https://doi.org/10.15184/aqy.2016.35). Retrodiction against archaeology/ceramics, not a ticker.

**Steal:** a simulator is a *mechanism claim*, earned when a class of cases needs one. Matches this repo’s “simulators only when a grade earns them.”

### 4.2 Conflict early-warning (theory-light vs theory-heavy)

- Goldstone, Bates, Epstein, Gurr, Lustik, Marshall, Ulfelder, Woodward, “A Global Model for Forecasting Political Instability,” *AJPS* 54(1) (2010). Country-year, ~2-year lead, **few variables**; **regime type** (nonlinear Polity categories) dominates. >80% in-sample accuracy on 1955–2003 onsets. [doi:10.1111/j.1540-5907.2009.00426.x](https://doi.org/10.1111/j.1540-5907.2009.00426.x). This is **PITF**, not this plugin’s retired PIT vault.
- Hegre et al., “ViEWS: A political violence early-warning system,” *Journal of Peace Research* (2019); Hegre et al. 2020 revision. Ensembles, conflict history, grid-month Africa, public scoring. [doi:10.1177/0022343319823860](https://doi.org/10.1177/0022343319823860).
- Cederman, Wimmer & Min, “Why Do Ethnic Groups Rebel?” *World Politics* (2010). Exclusion from state power (especially recent downgrade). A **different analog class** from elite overproduction. [doi:10.1017/S0043887109990219](https://doi.org/10.1017/S0043887109990219).
- Cederman & Weidmann, “Predicting armed conflict: Time to adjust our expectations?” *Science* (2017). Brute-force big data will not yield valid political-violence forecasts. [doi:10.1126/science.aal4483](https://doi.org/10.1126/science.aal4483).

**Contrast with SDT:** PITF/ViEWS forecast **near-term event onset** with parsimonious or ML ensembles. SDT forecasts **slow pressure**. This plugin’s analog cards sit closer to SDT (mechanism, clock, base rate); its **resolution day** sits closer to PITF/ViEWS (a named answering date). Do not collapse them. Do not borrow last-month conflict-history features for analog-regime problems.

### 4.3 Threshold / micromotives (mass vs individual)

- Schelling, *Micromotives and Macrobehavior* (1978).
- Granovetter, “Threshold Models of Collective Behavior,” *AJS* 83(6) (1978).

These are the non-fictional version of “the individual does not count”: **distribution of thresholds** produces riots, not a representative agent.

### 4.4 LLM agents (emerging, not proven at historical scale)

- Park, O’Brien, Cai, Morris, Liang, Bernstein, “Generative Agents: Interactive Simulacra of Human Behavior,” UIST 2023. 25 agents, memory–reflection–planning; emergent diffusion (a party). [arXiv:2304.03442](https://arxiv.org/abs/2304.03442), [doi:10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763).
- Park, Zou, Shaw, Hill, Cai, Morris, Willer, Liang, Bernstein, “Generative Agent Simulations of 1,000 People,” [arXiv:2411.10109](https://arxiv.org/abs/2411.10109) (v3 2026). 1,052 interview-grounded agents; GSS replication ~85% of test–retest. **Individual attitude simulation**, not secular cycles.
- Horton, “Large Language Models as Simulated Economic Agents,” [arXiv:2301.07543](https://arxiv.org/abs/2301.07543) (2023). Lab-task *homo silicus*, not history.
- Törnberg et al., “Do Large Language Models Solve the Problems of Agent-Based Modeling?” [arXiv:2504.03274](https://arxiv.org/abs/2504.03274) (2025). LLM-ABMs inherit calibration problems and add new ones.

**Do not** treat Smallville or interview-twins as a Seldon engine. They are a possible later **tool class** if a graded series shows that a mechanism needs interacting agents rather than a base-rate card. That is exactly the overlay-growth rule already in ADR 0007 / 0013.

### 4.5 Cultural evolution (Boyd/Richerson, Henrich)

Formal dual-inheritance models; Seshat is the empirical arm cliodynamics actually uses. Steal **multilevel selection / institution as evolving trait** only when a live series needs it — not as a global ontology (user preference 2026-08-19; parked “hegemony as overlay fields”).

---

## 5. Mapping: fiction → field → this plugin

The plugin already has the right *shape*: analog cards (class, mechanism, base rate, disanalogy, falsifiers) + dated claims + reflection cull. Cliodynamics and Asimov mostly tell you **what not to flatten**.

| Asimov | Cliodynamics / neighbors | This plugin |
| ------ | ------------------------ | ----------- |
| N large; individuals are noise | Goldstone: pressures vs triggers; Granovetter thresholds | Analog card = **class of mechanisms**, not a person. Structure block before news. |
| Subjects must not know the *predictions* | Scientific prediction as **theory test**; public forecast can change behavior | **Openings stay in chat** (ADR 0014). Scored claims are anonymized public events, not tenant playbooks. Do not publish a Plan that the actors then perform. |
| Probability; error grows with horizon | PSI = risk, not event; ViEWS = short-horizon onset | Horizon mix (ADR 0012). Near clocks so reflection can score; far analog as minority. |
| Seldon crisis = arranged answering moment | Named public date vs slow pressure | **Resolution day** on the problem; **forecast day** on the claim (ADR 0010). Slide Resolution if the public date moves. |
| Retrodiction with known answers is invalid (*Prelude*, Hummin) | Turchin: retrodiction is valid *if* data unused in theory construction; still weaker than live tests | **Forward loop only** (ADR 0003). Do not restore PIT/gold vault as the epoch. |
| Fake psychohistory still moves people (Rashelle) | Markets, polls, “the model says” | Motivation is not a card. Don’t treat overlay language as if it were a scored forecast. |
| Mule / micropsychohistory | Individual outliers; LLM agents of 1,000 people | Falsifier: extra-systemic actor, one-person override, unpublished instrument. Do not pretend analog cards cover mutants. |
| Second Foundation fine-tunes when off-path | Crisis averted via reform (Hoyer 2025) | **Typical openings** earned on reflection, anonymized. Not a playbook; not graded as the claim. |
| Simpler society as testbed (*Prelude*) | Seshat: many small coded polities beat one Rome essay | While overlay is thin: overweight **short resolution days**; K=15; anti-cluster. |
| Equations in innumerable unknowns, no technique yet | Mature theory = model **plus** data | **No graph-vault restore.** Simulator only after a grade earns it. |
| No “typical” path after the Mule | CrisisDB: consequences uncorrelated | **Disanalogy + falsifiers** are not optional. Do not mint a “crisis” mega-card. |
| Plan is not a playbook for a party | SDT silent on *how* elites respond | Tenants see **openings**, not campaigns. |

### Concrete inspirations to steal as method

1. **Score structural drivers and clocks, not vibes.** PSI-style thinking: name immiseration / elite congestion / fiscal-legitimacy *if the class needs them*. Do not import PSI as a global field.
2. **Separate pressure from outcome.** Analog card holds the stem; the dated claim holds one leaf. CrisisDB 2025 is the citation for why a single “what crises do” card will fail.
3. **CrisisDB-shaped case cards.** Uniform envelope, many instantiations, consequences listed without forcing correlation. That is already ADR 0013’s envelope.
4. **Live forecast as theory test.** Turchin 2010 is the template: a dated, embarrassing, checkable sentence. That is `ledger.md`, not a Nature letter.
5. **Parsimonious event models when the clock is short.** Goldstone 2010: few variables, regime/institution specified *nonlinearly*. For news-now problems, do not drown Structure in Seshat-scale coding.
6. **ABM/LLM simulators are generative explanations**, earned after a miss or a hit that a base rate cannot carry. Epstein’s criterion, Asimov’s “no technique yet,” ADR 0007.
7. **Galton’s problem / shared ancestry.** When two instantiations are the same empire twice, they are not two independent base-rate draws. Cite it on the card.
8. **Industrial ≠ agrarian.** Georgescu 2023: do not copy labor-oversupply cards onto automation-driven wage series without a disanalogy line.
9. **Observer effect is operational.** If a tenant acts on an opening, the world is no longer a random draw from the analog. That is why openings are not scored. Turchin 2011 names the same thing as self-defeating prophecy.
10. **Gingko-leaf as overlay review prompt.** After a graded series: did we forecast pressure, event, or both? Did we punish the card for a leaf it never claimed?
11. **Unknown is not absent.** Seshat codebook + Beheim 2021. A missing caption or uncoded mechanism is not a negative observation and is not a base-rate draw.
12. **Accuracy decay is a falsifier.** Bowlsby: a model can be 80% in its validation decade and poor in the next. Cards need a *when this analog died* line (regime change, industrial translation). Georgescu is that line for agrarian labor-oversupply → US wages.
13. **Ablate rivals on the card.** 2013 PNAS (65% vs 16%) is the method: name the competing class and what evidence would prefer it. Disanalogy/falsifiers already exist; use them as ablation rows, not hedges.
14. **Do not stack every clock that points at the same year.** The 2010 *Nature* note layered PSI + 50-year spike + Kondratiev + youth bulge. One named resolution day per problem.

---

## 6. Paper table

| Year | Authors | Title | Venue | Link | Relevance |
| ---- | ------- | ----- | ----- | ---- | --------- |
| 1978 | Granovetter | Threshold models of collective behavior | *AJS* | [doi:10.1086/226707](https://doi.org/10.1086/226707) | Mass action from threshold distributions |
| 1991 | Goldstone | *Revolution and Rebellion in the Early Modern World* | book | — | Pressures vs triggers |
| 1996 | Epstein & Axtell | *Growing Artificial Societies* | book | — | Generative ABM |
| 2003 | Turchin | *Historical Dynamics* | Princeton | — | Name and method of cliodynamics |
| 2008 | Turchin | Arise ‘cliodynamics’ | *Nature* | [10.1038/454034a](https://doi.org/10.1038/454034a) | Program statement |
| 2009 | Turchin & Nefedov | *Secular Cycles* | Princeton | — | Agrarian cycle evidence |
| 2010 | Turchin | Political instability may be a contributor… | *Nature* corr. | [10.1038/463608a](https://doi.org/10.1038/463608a); [PDF](https://peterturchin.com/wp-content/uploads/2023/07/Nature2020letter.pdf) | Rare live forecast |
| 2010 | Goldstone et al. | A global model for forecasting political instability | *AJPS* | [10.1111/j.1540-5907.2009.00426.x](https://doi.org/10.1111/j.1540-5907.2009.00426.x) | Short-horizon, few variables |
| 2011 | Turchin | Toward cliodynamics | *Cliodynamics* | [10.21237/C7clio21210](https://doi.org/10.21237/C7clio21210) | Retrodiction ≠ live forecast; self-defeating prophecy |
| 2012 | Turchin | Dynamics of political instability in the US, 1780–2010 | *JPR* | [10.1177/0022343312442078](https://doi.org/10.1177/0022343312442078) | Secular + 50-year waves |
| 2013 | Turchin | Modeling social pressures toward political instability | *Cliodynamics* | [10.21237/C7clio4221333](https://doi.org/10.21237/C7clio4221333) | PSI formulas |
| 2013 | Turchin et al. | War, space, and Old World complex societies | *PNAS* | [10.1073/pnas.1308825110](https://doi.org/10.1073/pnas.1308825110) | ABM after a named mechanism; ablation |
| 2018 | Turchin et al. | Single dimension of social complexity | *PNAS* | [10.1073/pnas.1708800115](https://doi.org/10.1073/pnas.1708800115) | Seshat pattern, not a ticker |
| 2019 | Bowlsby et al. | The future is a moving target | *BJPS* | [10.1017/s0007123418000443](https://doi.org/10.1017/s0007123418000443) | Out-of-sample accuracy decay |
| 2019/21 | Whitehouse / Beheim | Moralizing gods letter, retracted | *Nature* | [10.1038/s41586-021-03655-4](https://doi.org/10.1038/s41586-021-03655-4) | Unknown ≠ absent |
| 2018 | Turchin | Fitting dynamic regression models to Seshat data | *Cliodynamics* | [escholarship](https://escholarship.org/uc/item/99x6r11m) | Galton’s problem controls |
| 2019 | Hegre et al. | ViEWS | *JPR* | [10.1177/0022343319823860](https://doi.org/10.1177/0022343319823860) | Public conflict early warning |
| 2020 | Turchin & Korotayev | 2010 SD forecast: retrospective | *PLOS ONE* | [10.1371/journal.pone.0237458](https://doi.org/10.1371/journal.pone.0237458) | How a decade forecast is scored |
| 2023 | Georgescu | SDT revisited (industrialized societies) | *PLOS ONE* | [10.1371/journal.pone.0287912](https://doi.org/10.1371/journal.pone.0287912) | Mechanism critique |
| 2023 | Turchin & Hoyer | Empirically testing and refining SDT | OSF | [osf.io/yrqw5](https://osf.io/yrqw5) | Methodological guide; forecast-to-refine |
| 2023 | Park et al. | Generative agents | UIST / arXiv | [2304.03442](https://arxiv.org/abs/2304.03442) | LLM-ABM, small N |
| 2024–26 | Park et al. | Generative agent simulations of 1,000 people | arXiv | [2411.10109](https://arxiv.org/abs/2411.10109) | Individual twins, not empires |
| 2025 | Hoyer et al. | Crises averted | *Cliodynamics* | [10.21237/c7clio.38365](https://doi.org/10.21237/c7clio.38365) | Reform as falsifier of “collapse follows PSI” |
| 2025 | Hoyer, Turchin, et al. | All crises are unhappy in their own way | *SSH* | [10.1017/ssh.2025.10113](https://doi.org/10.1017/ssh.2025.10113) | Why analog cards need disanalogy |
| 2025 | Törnberg et al. | Do LLMs solve the problems of ABM? | arXiv | [2504.03274](https://arxiv.org/abs/2504.03274) | Why not to grow an LLM town because the name is cool |

---

## 7. What these fields do **not** claim (and what we should not import)

- **A closed-form Seldon Plan.** No one has functions “congruent to social and economic forces” that output a thousand-year path.
- **Trigger prediction.** Goldstone and Turchin both punt on the spark. Resolution day is a *public answering date we chose*, not a derived singularity.
- **Typical crisis outcomes.** CrisisDB 2025 is a published negative result.
- **That knowing the theory is harmless.** Asimov’s second axiom is the one empirical social science keeps rediscovering as performativity.
- **That LLM towns are historical dynamics.** Different scale, different validation (believability / survey replication vs CNTS / Seshat).
- **A global ontology of society.** Seshat’s PC1 is an empirical compression of *coded past polities*, not a schema for `references/`. User lock: no a-priori taxonomy; class names from use (ADR 0013).
- **Restoring `graph-vault/` or a GNN as the product.** That is parked. If a grade ever earns a simulator, it should look like a **small, class-specific generative model** (Epstein), not a France-warehouse graph.

---

## 8. Limits of this note

- EPUBs were searched for definitional passages; plot-heavy hits were dropped. Page numbers are unavailable (reflowable text).
- eScholarship HTML often 403s from this environment; journal claims use Turchin’s site, PLOS, PNAS/PMC, OSF, and DOI pages.
- *Foundation and Earth*’s third axiom and Daneel are not recommended as overlay content.
- No overlay files were changed. If a later reflect wants CrisisDB-shaped deepening of analog cards, that is a Parent write after a graded series — not this brief.
- Paper-side detail (PSI formulas, Seshat retraction, Bowlsby decay, historical ABMs) was merged from a parallel literature pass; the Asimov extracts are from the EPUBs in this session.

## Sources of the books

| In-world order | File |
| -------------- | ---- |
| Prelude to Foundation | `Foundation_1_Asimov_Isaac_-_Prelude_to_Foundation.epub` |
| Forward the Foundation | `Foundation_2_Asimov_Isaac_-_Forward_the_Foundation.epub` |
| Foundation | `Foundation_3_Asimov_Isaac_-_Foundation.epub` |
| Foundation and Empire | `Foundation_4_Asimov_Isaac_-_Foundation_and_Empire.epub` |
| Second Foundation | `Foundation_5_Asimov_Isaac_-_Second_Foundation.epub` |
| Foundation’s Edge | `Foundation_6_Asimov_Isaac_-_Foundation_39_s_Edge.epub` |
| Foundation and Earth | `Foundation_7_Asimov_Isaac_-_Foundation_and_Earth.epub` |
