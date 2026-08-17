---
type: concept
tags: [concept, methodology, calibration, polymarket, structural-reasoning]
title: "Sequential Hurdle Divergence"
slug: sequential-hurdle-divergence
domain: global
first_observed: 2026-06-21
canonical_case: "Andy Burnham UK PM (PM=0.953 → vault=0.35)"
status: active
related_concepts:
  - market-vault-structural-divergence
  - structural-improbability-check
  - short-horizon-procedural-certainty
---

# Sequential Hurdle Divergence

## Definition

A calibration pattern where a prediction market prices a compound event as if it were a single outcome, but structural analysis decomposes it into an AND-chain of N independent sequential hurdles whose joint probability is far below the market price.

The market error is not about hidden information but about **ignoring structural process**: traders focus on the endpoint (person X becomes PM / bill becomes law / deal is signed) without modeling the procedural steps required to get there.

## Mathematical Structure

For a compound event requiring N sequential hurdles:

$$P(compound) = \prod_{i=1}^{N} P(hurdle_i | hurdles_{<i})$$

Even if each hurdle is individually plausible (0.40-0.50), three hurdles produce joint probability of 0.40 × 0.50 × 0.40 = 0.08 = 8%.

The market implicitly assumes the hurdles **collapse into one step** — i.e., P(compound) ≈ the most probable single hurdle. The sequential hurdle check reveals the missing multiplicative factor.

## Diagnostic Criteria

Apply the sequential hurdle check when:

1. **The market price >0.80 for a compound event** — any outcome requiring 2+ independent procedural steps with the market at 80%+
2. **The steps are sequential, not parallel** — parallel paths are additive, NOT multiplicative. "Candidate A wins OR Candidate B wins" is a union, not an AND-chain. Sequential hurdles are AND-chains.
3. **Time pressure compounds the chain** — each step takes minimum time (election, vote count, transition, appointment). Total time must fit within the deadline.
4. **The steps are procedurally verifiable** — each step's probability can be estimated from institutional base rates (how often do UK PMs resign in a given window? how often do by-elections open in safe seats?).

## High-Frequency Applications

| Market Type | Typical Hurdles | Common Error |
|-------------|----------------|--------------|
| **"Next PM"** (parliamentary systems) | ① Win/by-election seat → ② Leadership contest → ③ PM appointment | Market prices endpoint without modeling seat requirement |
| **"Bill becomes law"** | ① Pass House → ② Pass Senate → ③ Signed by president → ④ Not vetoed/overturned | Market collapses bicameral passage + executive action |
| **"Candidate wins nomination"** | ① Qualify for ballot → ② Survive early rounds → ③ Win majority at convention | Market prices name recognition without delegate math |
| **"International deal signed"** | ① Negotiators agree text → ② Domestic ratification (legislative) → ③ Implementation legislation | Market collapses negotiation + ratification into one event |
| **"CEO fired / executive ousted"** | ① Board builds majority → ② Vote called → ③ Vote passes → ④ Replacement identified | Market prices "unrest" as "removal" without board mechanics |

## Relationship to Other Concepts

- **[[domains/global/concepts/market-vault-structural-divergence]]** — Sequential hurdle divergence is a **subtype** of market-vault structural divergence. The Burnham case is Canonical Case 8 in that concept.
- **[[domains/global/concepts/structural-improbability-check/_concept]]** — Related but distinct: structural improbability checks whether YES requires a cascade of unlikely events; sequential hurdle checks whether the market is ignoring AND-chain structure.
- **[[domains/global/concepts/short-horizon-procedural-certainty/_concept]]** — Time-pressure analysis is shared (both assess whether deadlines constrain procedural steps).
- **[[domains/global/concepts/pre-negotiated-framework-activation/_concept]]** — The inverse: where a deal is pre-negotiated and ratification is near-automatic (AND-chain with near-1.0 final steps).

## Forecasting Rule

When a prediction market price implies >80% probability for a multi-step sequential process (AND-chain with 3+ independent hurdles), **always** decompose the hurdles explicitly. Document:
- What specific hurdles must clear
- The institutional base rate for each (how often does this step occur in this time window?)
- The sequential dependency (can steps run in parallel, or must they be sequential?)
- The deadline constraint (can all steps fit within the remaining time?)

If the decomposition produces compound probability <40% while the market is at >80%, the vault should diverge regardless of market volume (the Burnham case was $991K — high liquidity doesn't prevent this error mode).
