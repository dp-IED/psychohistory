# Scale forecast roadmap: what to model, what plugin runs actually buy

**Status:** proposed (not an ADR). Lock the table at the end before growing overlay into a world-model.  
**Date:** 2026-08-19  
**Fits:** ADR 0002–0003 (forward epochs), 0007 (reflection grows overlay), 0011–0013 (mix + analog cards).  
**Does not:** restore `graph-vault/`, France GNN, PIT/gold scoring, or a global Gramsci ontology.

Psychohistory here means: **longer-horizon event forecasts from overlay that survived live scoring**, eventually with simulators only when a grade earned them. The long-term intellectual target (infrastructure, superstructure, mental structures) is a **use of those results**, not a second product on this branch.

## The mismatch you have to stare at

The ledger scores **dated events** (who wins, does a tariff fire, what a jury picks). Gramsci’s objects are **slow relations** (how a historic bloc holds, how consent is organized, how production constrains politics). Mental structures are **representations** (what people treat as common sense, what a cabinet can publicly justify).

If you treat those as three databases you “also model,” you will either:

- mint unscoreable metaphysics and never cull, or
- keep scoring primaries and call the analog cards “superstructure” after the fact.

The only path that stays honest with this plugin: **events remain the scoring surface; structures are loadable mechanisms that must pay rent on those events.** Cards are the intermediate representation (ADR 0013). Plugin runs produce the only training signal you trust: series after resolution day, plus whether the Structure block transferred.

---

## Grill (answer these before you scale)

These are not brainstorming prompts. Each one, unanswered, will fork the overlay into sludge.

### 1. What is the predictand at scale?

Are you trying to predict **named events** (elections, wars, rate decisions, strikes), **state variables** (who holds hegemony in X, tightness of a labor market), or **trajectories** (a historic bloc decaying over years)?

If the predictand is events, infrastructure/superstructure/mentality are **features and mechanisms**, not extra Y’s. If the predictand is “the structure itself,” you need an observation rule that a stranger could apply in 2028 without you in the room. Write that rule or drop the claim.

**Hard version:** name ten events you would bet the overlay on in 2028 that are *not* national elections. If you cannot, you are building an election-scraper with Marxist vocabulary.

### 2. Which clock is the object?

Gramsci’s organic crisis is years; a conjunctural fight is months; a ledger problem is a **resolution day**. ADR 0010 already collapsed Due/Y into one event clock. Do not smuggle a second “structural time” into scoring.

You *may* keep a **tempo tag on the card** (organic / conjunctural / event) as mechanism text. You may not grade “did hegemony weaken this quarter” unless that sentence is operationalized as a public observable with a date.

**Hard version:** when a far analog problem misses, will you blame the card’s mechanism or the fact that the resolution day was the wrong grain for that mechanism? If you cannot say in advance, the card is unfalsifiable.

### 3. Whose mental structure?

Three different objects, currently collapsed into one Justification:

| Layer | Whose mind | Where it should live | How it dies |
| --- | --- | --- | --- |
| Forecaster IR | the plugin / worker | Structure block + analog card | reflection culls the card |
| Actor justification | a named elite, party, court, firm | cited speech/text in the claim, later maybe a card field | contradicted by what they do or say next |
| Collective common sense | a public that can withhold consent | harder; needs a public trace (polls, strikes, turnout, boycotts, church, fandom) | base-rate of that trace fails on the next instance |

If you mix (1) with (2), you will “learn” the model’s prose. If you mix (2) with (3), you will treat a press secretary as the masses.

**Hard version:** after `P-usca-338` and the Knesset series resolve, what sentence about *Canadian or Israeli public reason* will you allow into overlay, and what would delete it?

### 4. Infrastructure is not “economics domain.”

ADR 0011’s bucket `economics/markets/trade` is a **discover mix constraint**, not a model of the base. A Section 338 tariff row is a legal-political instrument sitting on a trade flow. A US Open final is not a superstructure of tennis capital.

Infrastructure worth modeling (only as cards that must forecast something dated):

- energy and feedstock constraints
- logistics and chokepoints
- labor supply and strike capacity
- credit / fiscal space / who can roll debt
- production capacity and lead times (chips, munitions, housing, ships)
- platform and attention infrastructure (who can broadcast, who can shut it off)

If discover never opens live problems whose **mechanism is a material constraint**, reflection cannot grow base cards. You will have a superstructure plugin that names “incentives” and never names barrels, berths, or payrolls.

**Hard version:** pick one material constraint that could have changed `P-usca-338` or could change `P-il-knesset-26`. If none, those problems cannot train infrastructure. Open different problems.

### 5. Superstructure is not “all politics.”

Useful superstructure objects (again: cards, not a taxonomy):

- the **rule that can actually bind** (statute, court, central bank mandate, party selectorate, junta decree)
- the **personnel machine** (lists, primaries, appointments, officer corps)
- the **consent machines** (media, schools, unions, churches, fandoms, NGOs)
- the **coercion machines** (police, army, prosecutors, platforms’ takedown)

Gramsci’s point is the **relation** between political society and civil society, not a pile of election markets. `P-gw-const-ref` (junta constitution) and `P-is-eu-ref` (accession talks) are closer to that than another US Senate runoff — if the Structure block names the consent/coercion mix, not just “polls plus incumbency.”

**Hard version:** will you allow a card that says “this junta referendum’s base rate is yes” without stating *who cannot campaign*? If yes, you are fitting outcomes, not superstructure.

### 6. Hegemony, historic bloc, war of position — operationalize or park.

These terms are load-bearing in your long-term goal and **absent from the analog envelope** (class, instantiations, mechanism, base rate, disanalogy, falsifiers). That is correct for v1: ADR 0013 forbids growing a taxonomy in advance.

They become legal overlay only when a live series **needs** them to explain an early hit or a miss, and the next transfer reopen can kill them.

Minimum operational meanings if they ever earn a field:

- **Hegemony:** a measurable ability of a bloc to make its justification the default (vote share + strike absence + elite defection rate — pick observables).
- **Historic bloc:** a named coalition of organizations that must stay aligned for the predicted event. Name the organizations.
- **War of position:** slow institution-building that changes the base rate of a class; not a vibe on a news-now primary.

Until then they are **banned in Motivation as decoration**.

### 7. Mediation is the fourth object (the one you asked “what else”).

Base, superstructure, and mental maps do not sit in parallel. Events happen when a **coupling** fires: a constraint hits an institution that can only speak in certain justifications, which then licenses or blocks an act.

What else to model, in priority order, **as card mechanism text and later as earned tools**:

1. **Coupling / transmission** — the path from material fact → institutional bottleneck → public justification → dated act. If the path has a missing step, the analog is a slogan.
2. **Capability** — who can actually do the thing (votes, guns, liquidity, hulls, hashes, jurisdiction). Outcome forecasts without capability are fanfic.
3. **Information regime** — lag, secrecy, propaganda, what predict is allowed to scrape vs what actors know. News-now vs analog (ADR 0012) is a thin version of this.
4. **Selectorates and veto points** — not “politics,” a map of who can say no.
5. **Coordination and defection** — blocs fail by split, not by ideology drifting in the abstract.
6. **Legibility / scoring surface** — which world-events even have a public resolution day. Scale prediction that cannot name a resolution rule is not this plugin’s product.
7. **Disanalogy as first-class** — already in the envelope. At scale this is *the* skill: most structural analogies are false.
8. **Composition** — using two cards on one problem (logistics + selectorate). This is how you get to “events at scale” without a GNN. Do not compose until single-class cards survive transfer reopen.
9. **Failure / organic crisis** — when the usual card’s base rate breaks because consent collapsed. That is a **new class**, opened as a live analog problem, not a switch flipped on all cards.

Park until a grade earns them: graph simulators, learned embeddings of the overlay, global type systems, “R(q,h_t)” schemas beyond the audit envelope.

### 8. Scale is not more K.

`K=15` already overfits if discover clusters (ADR 0011–0012). Scale prediction means:

- **more survived cards** that transfer across domains,
- **longer horizons** on problems those cards actually govern,
- **composition of a few cards**, not 10⁴ news-now rows.

A thousand scored US primaries will not yield infrastructure. Ten scored chokepoint / labor / credit / coercion-mix series might.

### 9. The plugin is not a world-model. The overlay is a cullable library.

Weights stay frozen (ADR 0002). Memory is overlay. Justifications are **traces**, not memory (`CONTEXT.md`). If you dump every Structure block into `references/`, you recreate graph-vault under a nicer name.

Rule: a card exists only if reflection can point to a series and a proposed transfer problem. Raw ledger is the log. Overlay is the theory that still has a job.

### 10. Normative Gramsci vs predictive psychohistory.

Gramsci is also a project of *changing* the historic bloc. This plugin scores **forecasts**. If you want intervention (what we should organize), that is a different tick and a different grade. Mixing them will make reflection keep morally pleasing mechanisms that miss.

Lock: predict remains p(event). Strategy memos, if any, are not claims.

---

## What plugin runs actually produce (use this, not a parallel science)

```text
discover  →  problems + resolution day + Motivation (class name if analog)
predict   →  claim series + Justification + Structure block  (forecaster IR)
reflect   →  grade series + grow/cull overlay
              skills/methods, analog cards, later scripts/simulators
```

| Run artifact | Use for the long-term goal | Do not use as |
| --- | --- | --- |
| Dated claims | Calibration, horizon (forecast day vs resolution day), revision discipline | A map of the base |
| Structure blocks | Audit of which mechanisms workers actually used; seed for cards | Durable ontology |
| Analog cards | The only structural memory; library of mechanisms | A GNN, a taxonomy, a vault |
| Skills / agents | Transferable *methods* (how to fetch, how to disanalogize) | Domain facts |
| Domain mix of problems | Coverage of scoring surfaces | Proof you modeled infrastructure |
| Misses on analog problems | The training signal for cull/rewrite | An excuse to add more fields |
| Transfer reopen (ADR 0012) | Falsification of a card on a new live instance | Gold replay |

**Harvest loop (after the host ticker is armed, not instead of it):**

1. Let series resolve. Do not design Gramsci fields while overlay is empty of cards.
2. After each reflection, classify the **winning and losing Structure blocks** into: capability, constraint, selectorate, consent/coercion, actor-speech, none. No new file unless a class transferred.
3. If a layer never appears in winning blocks, **open analog-regime problems whose Motivation names that layer’s class** (material constraint, junta consent, labor stoppage, credit rollover, platform shutdown). Still no quality gate (ADR 0005); reflection culls.
4. When two cards both survive, allow predict to **compose** them in the Structure block (two class names, one coupling sentence). Grade the coupling.
5. Simulator / script: only if a survived class is repeatedly dynamic (queues, force ratios, runoff math) and a worker without the tool cannot beat the card. That is ADR 0007. It is not “now we need the France GNN.”

---

## Roadmap (plugin-native, no calendar)

**Now (this branch, already in `next_steps.md`):** arm `/predict` `/reflect` `/discover`. Keep scoring the live inventory. Reflection writes overlay only. Do not restore parked trees.

**After first analog cards exist:** audit whether cards are election-shaped. If they are, discover must feed material-constraint and consent/coercion classes or you are done pretending this is Gramsci.

**After first transfer reopen of a non-election class:** optionally add *one* sentence to the card envelope — coupling or capability — only if the miss/hit required it. Still not a global schema.

**After composition works on live problems:** that is the first “events at scale” prototype: one resolution day, several mechanisms, no graph store.

**Only then:** plugin-grown simulators as `scripts/` for a specific class; Polymarket as a discovery/scoring surface; any later graph analysis of *cards that already exist*, not a restored vault.

---

## Locked / proposed decisions

Decisions are not locked until copied here or into an ADR. Proposed defaults:

| # | Decision | Proposed lock |
| --- | --- | --- |
| L1 | Scoring surface stays dated public events | Yes. Structures are mechanisms that must forecast those events. |
| L2 | No Gramsci taxonomy in overlay a priori | Yes. Class names from use; merge later (ADR 0013). |
| L3 | Split mental layers in justifications | Forecaster Structure vs cited actor speech vs collective traces. Do not store (1) as world-facts. |
| L4 | Discover must be able to train the base | When mix allows, include analog problems whose class is a material constraint, not only “economics” elections. |
| L5 | Hegemony / historic bloc | Parked as decoration until a series earns the words with observables. |
| L6 | Fourth object is coupling | Mechanism text should state the transmission path; extra envelope fields only if reflection needed them. |
| L7 | Scale = survived transferable cards + composition + longer horizons | Not more K, not a GNN, not gold replay. |
| L8 | Simulators | Overlay tools when a grade earns them (ADR 0007). France/warehouse tree stays parked. |
| L9 | Predictand of claims | p(event), not what ought to be organized. |
| L10 | Harvest is a reflection habit, not a host job | Do not add a fourth ticker. Parent already grades Structure. |

---

## What this document is not

It is not permission to implement a world-model, a graph, or new discover fields. It is the grill and the use-path for plugin results. If a line here fights an ADR, the ADR wins until superseded.
