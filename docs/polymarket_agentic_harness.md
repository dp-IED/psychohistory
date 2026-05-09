# Polymarket agentic harness v1

This note integrates the pivot from a deterministic `subgraph builder -> GNN -> forecast head` pipeline to an agentic Polymarket prediction harness.

## Building philosophy

Use the LLM agent for recall, text understanding, research planning, and candidate world framing. Use the graph/GNN layer for structural reweighting: macro dependencies, underweighted remote influences, branch sensitivity, and missingness risk.

The GNN should not invent a whole world from a sparse seed. V1 makes the harness propose bounded expansions and asks the graph/GNN layer to evaluate reachability, fragility, and missingness over those candidate portfolios.

## Core objects

Implemented in `schemas/polymarket_agentic.py`:

- `MarketFrame`: normalized market question, resolution criteria, close/resolution dates, binary outcomes, and resolved outcome when known.
- `OutcomeHypothesis`: an agent-authored Yes-world or No-world.
- `SubgraphPortfolio`: a bounded portfolio of branches for the hypothesis.
- `Branch`: one of `local`, `analogue`, `disruptor`, or `counterworld`.
- `PortfolioElement`: an admissible graph node/factor with `role` and `direction` relative to the hypothesis.
- `Prerequisite`: a binding gate that must hold for the outcome-world to remain reachable.
- `RequirementStressTest`: graph/GNN output: probability, uncertainty, branch contributions, missingness risk, surfaced prerequisites, fragile elements, disagreement, and macro correction.

## V1 branch policy

The harness validates portfolios against a family-specific policy before scoring.

| Family | Required branches | Main blind-spot checks |
| --- | --- | --- |
| `institutional_process` | local + disruptor | agenda disruption, coalition fracture, legal/scandal branch |
| `event_negotiation` | local + analogue + disruptor | spoiler actor, unrelated escalation, sponsor/domestic constraints |
| `macro_policy_print` | local + analogue | methodology changes, exogenous policy shocks, liquidity stress |

The local branch must include both `for` and `against` elements. A Yes-world local branch is not an advocacy case; it is the contested local world the Yes path must overcome.

## Detail admissibility

Represent a detail as a node/factor only if it is an actor, institution, event, condition, coalition block, measurable indicator, or persistent constraint that can plausibly move forecast probability or alter branch reachability.

Use:

- node/factor: standalone causal role;
- edge attribute: describes a relationship between factors;
- trace text: interpretive commentary without independent causal role.

Practical v1 budgets:

- local branch: up to 20-25 elements;
- prerequisite cluster: 3-10 gates depending on family;
- analogue branch: up to 15 elements;
- disruptor branch: up to 8-12 elements.

## Resolved market metadata

`ingest/polymarket_resolved.py` fetches resolved binary Yes/No market metadata from the public Polymarket Gamma API. It stores terminal resolution metadata for benchmark construction, not PIT forecast evidence. Downstream forecast runs must attach evidence with cutoff-safe source refs.

Run:

```bash
python scripts/fetch_polymarket_resolved.py --limit 100
```

Outputs:

- `data/polymarket/resolved_binary_markets.json`
- `data/polymarket/resolved_binary_markets.csv`

## Training labels enabled by this contract

Resolved Polymarket questions can supervise:

1. final outcome probability;
2. branch usefulness via branch ablations and Brier/log-loss deltas;
3. missingness risk when a late-found branch materially improves or flips a forecast;
4. branch disagreement as an abstention/error-risk signal;
5. prerequisite validity when a proposed gate turns out binding rather than decorative.

This directly targets the known failure mode: conservative subgraph selection that looks locally reasonable but misses remote, outcome-flipping structure.
