## Summary

The vault already had no relevant content for this question. No existing concept, entity, or thread tracked Biden's reinstatement possibility.

## What I Changed

### 1. New concept: `domains/usa/concepts/nominee-withdrawal-irreversibility/_concept.md`

**Problem**: The vault had concepts for how a nominee withdraws (incumbent-withdrawal-cascade Stages 0-7, post-nomination-persistence-baseline for non-incumbents who don't withdraw), but nothing covering what happens AFTER withdrawal — specifically, the structural mechanisms that make reinstatement impossible.

**What the concept covers**:
- The four irreversibility locks: delegate release (procedural), ballot access transfer (legal), party consolidation (political), convention ratification (institutional)
- Canonical test: the Biden 2024 reinstatement question — all four locks were engaged after July 21
- Zero historical precedent in US presidential history for a withdrawn candidate being reinstated
- Distinction from the withdrawal cascade concept (cascade = how withdrawal happens; irreversibility = why it can't be reversed)
- Cross-country applicability for other party-based nomination systems
- Calibration table: <0.1% reinstatement probability when successor is formally nominated (virtual roll call complete); <1% when candidate has withdrawn and endorsed but no roll call yet; <5% when withdrawn with no endorsement

**Forecasting value**: The next time someone asks "Can [withdrawn candidate] be reinstated?" or "Will [party] reverse its nominee replacement?", the vault has a structured framework showing why the answer is structurally NO, with specific mechanism-level reasoning.

### 2. New entity: `domains/usa/entities/democratic-national-committee.md`

**Problem**: The DNC was a named institution in the question ("reinstated at DNC") but had no vault entity stub. The DNC's role as a ratifying body vs. decision-making body, its virtual roll call authority, and its convention procedures are all forecasting-relevant.

**What the entity covers**:
- Institutional role: the DNC manages the nominating process but its convention function is ratification, not selection
- 2024-specific: virtual roll call authority (August 5, 2024) procedurally locked in Harris's nomination before the in-person convention
- Key figures: DNC Chair, Rules and Bylaws Committee
- Forecasting relevance: ballot access deadlines, delegate allocation rules, convention timing

### 3. Updated `_spec.md` — added Rule 37 (Withdrawal Irreversibility)

**Problem**: The spec had Rules 30-36 covering pre-withdrawal and withdrawal dynamics (aging-incumbent vulnerability, post-nomination baseline, candidate-type differentiation, impeachment, DOJ policies) but nothing about the post-withdrawal irreversibility dynamic. The Biden reinstatement question exposed this gap.

**Rule 37 mandates**:
- Apply the nominee-withdrawal-irreversibility framework for any reinstatement question
- Entity stub for the convention body (DNC or equivalent)
- Procedural-stage gate: is successor formally nominated? If yes, <0.1% reinstatement probability
- Historical zero-base-rate: zero precedents for reinstatement in US history
- Prohibition on treating reinstatement as a generic "political possibility" — it's procedural, not political

## Vault Contribution Score

**0% (freebie)** — The correct NO prediction came entirely from general knowledge ("once you drop out, you don't come back"). The vault had no concept, no entity for the DNC, and no framework for analyzing reinstatement questions. The three additions above are the minimum coverage bar for any future reinstatement question.

## Why This Gap Mattered

Although this particular question was correctly predicted, a vault that cannot answer "why can't a withdrawn candidate be reinstated?" with structural reasoning is missing a fundamental piece of US election mechanics. The failure mode would be: a future question asks about reinstatement in a non-US context (e.g., "Will [UK party leader who resigned] be reinstated before the leadership election?") and the vault has neither the conceptual framework nor the institutional entity stubs to support the reasoning. The concept's cross-country applicability section addresses this directly.
