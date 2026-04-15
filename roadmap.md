Good. Then treat this as a **program roadmap**, not an engineering scaffold.

The right way to do this is to de-risk in this order:

1. **Can you define a forecastable ontology?**
2. **Can you build a temporally clean graph from real sources?**
3. **Can simple baselines forecast anything useful?**
4. **Does a graph model add signal beyond recurrence?**
5. **Can the system explain forecasts with historical analogs?**
6. **Can it generalize across domains?**

That sequence matters because the data layer and evaluation layer decide whether
the rest is science or just an impressive demo. The France protest benchmark has
now served its purpose as the first validation domain: the temporal tape,
snapshot exporter, recurrence baselines, XGBoost baseline, and heterogeneous GNN
all run on the same historical replay setup, and the GNN improves the main
calibration/error metrics on the holdout period.

The roadmap therefore moves from "prove a GNN is worth trying" to "expand the
graph while preserving the benchmark discipline." France protest forecasting is
the regression benchmark, not the final product definition.

Wikidata remains a viable backbone for canonical entities and relationships, with multiple access methods and weekly dumps; ACLED exposes a documented API; Polymarket currently documents market data and WebSocket access; and GDELT still provides its event/GKG streams and documentation. X/Twitter may later provide narrative, attention, or realtime discussion signals, but it is not part of the first event-source tape or baseline benchmark. That means your proposed stack is implementable now, at least for an MVP built around those sources. ([Wikidata][1])

## Stage 0 — Frame the problem like a forecasting program, not a philosophy project

Timebox this stage tightly. The output is not code. The output is a **forecast charter**.

You decide three things:

- the first domain
- the first forecast primitive
- the first evaluation horizon

For the first domain, I would use **geopolitical contention / protests / conflict escalation**, because ACLED, ICEWS, GDELT, and Wikidata line up naturally there. ACLED is particularly well-suited because it is explicitly structured around political violence, protest, actors, and locations through its API-accessible event data. ([ACLED][2])

For the first forecast primitive, choose only one:

- next event edge
- next-week event intensity
- next-week directional probability shift in a market
- future-answer QA

The cleanest starting point is:

**“Given date t, what actor–relation–target events are most likely in the next 7 days in region R?”**

Your success criterion here should be brutally narrow: better calibration and ranking than naive recurrence.

The deliverable at the end of Stage 0 is a one-page spec with:

- target variable
- forecast horizon
- unit of prediction
- sources allowed at inference time
- metrics
- non-goals

Do not proceed before this is frozen.

## Stage 1 — Build the temporal ontology and make it boring

This stage is where most ambitious projects overreach. Your ontology should feel disappointingly small at first.

You need four classes only:

- actors
- ideas / narratives
- events
- markets

And maybe one auxiliary class:

- locations

The real task is not philosophical correctness. It is **operationalization**.

An “idea” only exists if it can be attached to observable evidence such as:

- repeated topic clusters
- slogan/phrase clusters
- article embedding clusters
- persistent co-movement with actor behavior
- market belief movement around the same issue

A “trend” only exists if you can define its state over time:

- rising
- decaying
- stable
- fragmenting
- spreading geographically

Your ontology is ready when two different people can code the same raw evidence into roughly the same internal structure.

The gate for finishing this stage is simple:

- 100 sampled events from at least 2 sources
- hand-mapped into your ontology
- acceptable inter-annotator consistency on types and relations

If that fails, your ontology is too abstract.

## Stage 2 — Build a historical world model that is temporally clean

This is the first truly hard stage.

You are not “collecting data.” You are building an **immutable historical simulation substrate**. Every later forecast must be reconstructible as of date `t` with only information available at `t`.

Use the sources in separated roles.

Wikidata should serve as the **canonical identity layer**, because it supports multiple access methods and downloadable dumps, which is what you need for controlled snapshots and canonical entity resolution. ([Wikidata][1])

ACLED, ICEWS, and GDELT should serve as the **event evidence layers**. GDELT’s event and GKG feeds are still documented as frequent, rolling updates, which makes them useful for live-ish event flow and narrative extraction; ACLED provides a cleaner, analyst-grade event layer for conflict and protest settings. ([GDELT Project][3])

Polymarket should be treated as a **belief layer**, not ground truth. Its current documentation confirms structured market/event concepts and API access patterns, including streaming interfaces. ([Polymarket Documentation][4])

Your goal here is to produce, for any historical day:

- graph snapshot as of that day
- event tape visible by that day
- market state visible by that day
- narrative states inferred only from prior windows

The critical output of Stage 2 is not a model. It is a **time-travel-safe replay engine**.

The gate:

- select 5 arbitrary historical dates
- reconstruct the graph visible on each date
- prove that no later facts leaked in

If you cannot do that, stop everything else.

## Stage 3 — Establish non-neural baselines before trusting a GNN

This stage is non-negotiable, and the first version is now in place.

A lot of temporal KG work looks strong until you compare it to recurrence-heavy baselines. A 2024 IJCAI paper found that a simple recurrence-based baseline ranked first or third on three of five temporal KG benchmarks, which is exactly why you need this stage before you trust any graph model gains. ([Wikidata][1])

So your first forecasting stack should be intentionally plain:

- recurrence counts
- recent-window frequency
- seasonal frequency
- actor-pair memory
- Hawkes-style self-excitation
- tabular boosted trees using graph-derived features

Feature families should include:

- recent event counts
- rate of change
- relation entropy
- neighborhood churn
- coalition turnover
- location spillover
- narrative pressure
- divergence from market belief where applicable

You want to answer one question:
**Is there any real predictive structure in the data after temporal hygiene is enforced?**

The initial France protest gate:

- rolling backtests over multiple years
- Brier score and calibration materially better than naive baseline
- at least one prediction slice where simple engineered features work

Status: passed for the first benchmark. Recurrence and XGBoost provide usable
comparison points, and the tabular model improves over recurrence on the 2025
holdout. Keep this stage alive as a regression suite whenever a new source,
entity type, or target is added.

## Stage 4 — Expand the dynamic graph forecaster

The first graph model is now trained. The result supports the central modeling
bet: graph message passing can improve the main forecast quality metrics beyond
recurrence and tabular engineered features on the controlled benchmark.

Use the graph model for what it is good at:

- typed relational structure
- temporal dependency
- neighborhood context
- generalization across sparse actor combinations

Do not ask it to infer everything.

Architecturally, the next version should combine:

- temporal event encoder
- heterogeneous graph structure
- autoregressive or continuous-time update mechanism
- optional narrative-state side channel

This is where you now test the stronger version of the project hypothesis:

**Which additional graph sources and relations improve forecasts beyond the
validated event-location GNN?**

The evaluation should be adversarial. Slice performance by:

- frequent vs rare actors
- stable vs regime-shift periods
- conflict vs cooperation
- local vs cross-border spillovers

You are done with Stage 4 only if the expanded graph model delivers one of two things:

- better overall predictive performance
- same predictive performance, but much better transfer in sparse / hard regimes

If it does neither, then the graph is not yet earning its complexity.

Immediate Stage 4 work:

- freeze the France protest benchmark as the regression harness
- add deterministic GNN seeds and multi-seed audits
- add ablations for location-only, event-edge, actor-edge, source-edge, and
  narrative features
- add ACLED as a second event evidence layer
- add point-in-time Wikidata grounding for actors and locations
- promote only the node and edge types that survive backtest comparison

## Stage 5 — Add the “lame de fond” layer as explicit latent macro-dynamics

This is where your idea becomes distinctive.

But it must be done after you already have a working forecast engine.

At this stage, create a layer for macro-forces such as:

- elite fragmentation
- repression cycle
- protest contagion
- ideological radicalization
- geopolitical alignment shift
- inflation/grievance pressure

Do not inject these as hand-wavy priors. Infer them from evidence.

A good implementation pattern is:

- discover candidate latent states from text/topic/event co-movement
- test whether they improve forecast performance
- test whether they improve analog retrieval
- keep only the states that are measurable and forecast-relevant

The output should be something like:

- a small set of macro-state variables per region or actor-cluster
- transition probabilities over time
- interpretable links to observed events and narratives

The gate is not elegance. The gate is utility:

- does adding latent macro-states improve calibration, analog quality, or early-warning lead time?

If not, the macro-layer stays out.

## Stage 6 — Build analog retrieval before open-ended Q&A

Before you let an LLM answer anything, the system needs a way to retrieve:

- similar graph neighborhoods
- similar event trajectories
- similar market divergences
- similar narrative configurations

This matters because the final product should not merely say:
“Escalation risk is 0.68.”

It should say:
“Escalation risk is 0.68 because the current pattern resembles episodes A, B, and C, where these precursor relations appeared in similar order.”

The analog retrieval engine should work on:

- local subgraph topology
- event-sequence similarity
- narrative-state similarity
- market disagreement patterns

The gate:

- analysts can inspect a forecast and agree that the retrieved historical analogs are genuinely comparable more often than chance

This is the first point where the system starts to feel like an analyst machine rather than a classifier.

## Stage 7 — Add the Q&A layer as a constrained forecasting interface

Now you can add natural-language interaction.

The LLM should not forecast directly. It should only:

- interpret the question
- call the structured forecast and retrieval stack
- synthesize a grounded answer
- state uncertainty and disagreement

This produces two classes of QA:

First, **forecast QA**:

- What is likely to happen next in country X?
- Who is most likely to initiate hostile action toward Y in the next week?
- What is the probability unrest intensifies in capital Z?

Second, **causal-analytic QA**:

- Which ideas or narratives are most associated with the current escalation?
- Which past episodes most resemble this configuration?
- What changed over the last 14 days?

The rule is strict: every answer must be reducible to

- retrieved graph evidence
- forecast model outputs
- retrieved analogs
- cited source traces

No free-form punditry.

The gate:

- answer faithfulness audits
- answer correctness on held-out historical questions
- no systematic hallucinated evidence chains

## Stage 8 — Integrate markets as a disagreement and calibration instrument

Polymarket should not be the center of the system at first. It becomes extremely valuable later.

Once the base forecaster works, use market data in three ways.

First, as a feature:

- current implied probability
- recent move
- velocity
- spread/liquidity proxies where available

Second, as an evaluation target:

- does your model beat the market on some class of questions?

Third, as a disagreement detector:

- where does model probability materially diverge from market probability?

That third use case is especially strong. Often the most interesting output is not the raw forecast but:

- “the model is more bearish than the market because it sees structural precursor patterns that the crowd may be underweighting”

Polymarket’s current docs support both market/event structure and API access patterns for that layer. ([Polymarket Documentation][4])

The gate:

- clear evaluation of whether market features improve calibration
- at least a few historically interesting disagreement case studies

## Stage 9 — Stress-test across one related and one unrelated domain

Only after the expanded graph works in the first benchmark should you test your
"deep currents" thesis across domains.

A good second domain should first be adjacent, then unrelated.

Adjacent examples:

- protests in another country
- conflict escalation in ACLED
- election-related unrest

Unrelated examples:

- labor unrest
- public-health panic dynamics
- sanctions and trade retaliation
- technology/narrative adoption contests

The point is not scale. The point is abstraction:

- do the same latent dynamics, graph motifs, and forecasting interfaces transfer?

Your thesis becomes much stronger if the same architecture works in two structurally similar but substantively different domains.

The gate:

- one domain-local retraining
- one shared ontology subset
- evidence that at least part of the framework transfers

## Stage 10 — Decide what this project actually is

At that point you will know whether you are building:

- a forecasting engine
- an analyst copilot
- a market-disagreement engine
- an early-warning system
- a historical analog machine

It may be all of them eventually, but not as a first product identity.

That decision should happen after evidence, not before.

## Suggested timing and milestone logic

I would sequence the program like this:

**Completed:** freeze forecast charter, build the replayable France protest
event tape, export weekly snapshots, run recurrence/XGBoost/GNN backtests.

**Next:** freeze comparison audits, add GNN reproducibility, run ablations, and
ingest a second event source.

**After that:** add point-in-time Wikidata grounding, actor/source/narrative
node types, and historical analog retrieval.

**Later:** constrained Q&A, market integration, disagreement analytics, and
second-domain transfer.

The important thing is not the calendar duration. It is that each stage has a hard go/no-go test.

## The decision gates that matter most

There are only five that really matter:

**Gate 1:** Can you reconstruct the world as of historical date `t` without leakage?

**Gate 2:** Can simple baselines beat naive recurrence strongly enough to justify the task?

**Gate 3:** Does the graph model add predictive value or transfer robustness?

**Gate 4:** Can you retrieve historical analogs that humans find genuinely informative?

**Gate 5:** Can the Q&A layer stay grounded and temporally faithful?

If you pass those five, the project is real.

## What to avoid

Avoid building a giant universal ontology first.

Avoid letting “ideas” and “trends” remain metaphorical.

Avoid measuring success with only ranking metrics and no calibration.

Avoid using the LLM as the forecaster.

Avoid mixing canonical entity truth with historically visible truth.

Avoid moving to a second domain before the first one has a live decision workflow.

## My blunt recommendation

Your implementation roadmap should be centered on one sentence:

**Now that a temporally clean heterogeneous graph has beaten the main baselines
on the first benchmark, expand sources and node types under ablation, then prove
that the model can explain forecasts with valid historical analogs.**

Everything else is expansion.

[1]: https://www.wikidata.org/wiki/Wikidata%3AData_access?utm_source=chatgpt.com "Wikidata:Data access"
[2]: https://acleddata.com/acled-api-documentation?utm_source=chatgpt.com "API documentation"
[3]: https://www.gdeltproject.org/data.html?utm_source=chatgpt.com "Data: Querying, Analyzing and Downloading"
[4]: https://docs.polymarket.com/concepts/markets-events?utm_source=chatgpt.com "Markets & Events"
