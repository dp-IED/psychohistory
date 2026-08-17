---
type: concept
tags: [concept, forecasting, methodology, calibration, meta]
status: seed
created: 2026-06-17
pit_cutoff: 2026-06-17
purpose: "Formalizes the pattern where the same forecasting question, answered by the same agent in different sessions, produces diverging probability estimates. The divergence range is itself a useful uncertainty signal."
---

# Sibling-Run Calibration Divergence

**Pattern**: When the same forecasting question is answered in independent agent sessions ("sibling runs"), the resulting probability estimates often diverge. The divergence interval provides a self-calibrating uncertainty estimate — the true probability is likely within the range spanned by the siblings.

## The MV Hondius Canonical Case

| Run | p_yes | Source Platform | Date |
|-----|-------|----------------|------|
| [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius-h\|MV Hondius (cup)]] | 0.20 | Metaculus Cup | 2026-05-21 |
| [[runs/20260521-184716-will-at-least-5-non-passengers-be-linked-to-the-mv-hondius-h\|MV Hondius (tournament)]] | 0.24 | Metaculus Tournament | 2026-05-21 |

**Same question**: Will at least 5 non-passengers be linked to the MV Hondius hantavirus outbreak?
**Same deadline**: August 1, 2026
**Same underlying mechanism**: Andes virus H2H transmission is inefficient; surveillance found zero secondary cases after 17+ days; tail risk from Saint Helena disembarkees and long incubation (1-7 weeks).
**Divergence gap**: 0.04 (0.20 vs 0.24) — approximately 20% relative difference.

### Root Cause of Divergence

- **Cup run (0.20)**: Leaned more heavily on the "zero confirmed secondary cases after 17+ days" surveillance signal — a negative evidence signal that reduces probability.
- **Tournament run (0.24)**: Gave more weight to the historical base rate (36% for "will it reach N cases" category questions) — a statistical prior that increases probability.

Both agents had access to the same vault content. The divergence is not a defect but a feature: different reasoning frameworks (empirical-observational vs. prior-driven) produced different weights on the same evidence.

## Pattern Mechanics

1. **Sibling discovery**: The same question appears on two platforms, or the agent answers it twice in different sessions
2. **Probability extraction**: Both runs produce p_yes estimates
3. **Divergence measurement**: Calculate absolute and relative difference
4. **Interval construction**: Treat the spread [min(siblings), max(siblings)] as a probabilistic confidence interval
5. **Resolution tracking**: On resolution, check whether the actual outcome falls within the sibling range

### Calibration Value

The sibling range is a **free uncertainty estimate** — it costs nothing beyond the second run. Key rules:
- Narrow divergence (<0.05 absolute, <25% relative) → high confidence that vault evidence is producing consistent outputs
- Wide divergence (>0.10 absolute, >50% relative) → evidence may be ambiguous, agent reasoning is unreliable on this topic
- Divergence that brackets the market price → vault is not confidently disagreeing with the market
- Divergence that excludes the market price → vault confidently disagrees (structural divergence signal)

## Distinction from Related Patterns

| Pattern | Relationship |
|---------|-------------|
| [[domains/global/concepts/paired-questions-calibration/_concept]] | Different questions, same structure. Sibling runs are the same question — maximum possible similarity. |
| [[domains/global/concepts/market-vault-structural-divergence/_concept]] | Vault estimate vs market price. Sibling divergence is vault-vs-vault, measuring internal consistency. |
| [[domains/global/concepts/short-horizon-momentum-check/_concept]] | Single-run methodology. Sibling comparison is a meta-pattern. |

## Known Sibling Pairs in Vault

| Pair | p_yes Range | Spread | Observation |
|------|-------------|--------|-------------|
| MV Hondius cup vs tournament | 0.20 - 0.24 | 0.04 (20%) | Narrow — good consistency. Both correctly bracketed by Andes H2H ceiling. |
| Colombia first-round (pre- vs post-first-round) | 0.08 → 0.65-0.70 | 0.57 | Not siblings (different phase of same event). The jump reflects new information (first-round results), not reasoning divergence. |

## Forecasting Decision Procedure

When a sibling pair exists:
1. Check if the sibling range brackets the market price
2. If yes: vault is not confidently disagreeing; use the sibling midpoint as the adjusted estimate
3. If no: vault diverges from market; structural-divergence analysis should be applied to the sibling that had stronger reasoning
4. On resolution: add the sibling pair to the calibration log — does the actual outcome fall inside or outside the range?

## Wikilinks

- [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius-h]] — Cup run (p_yes=0.20)
- [[runs/20260521-184716-will-at-least-5-non-passengers-be-linked-to-the-mv-hondius-h]] — Tournament run (p_yes=0.24)
- [[domains/global/concepts/paired-questions-calibration/_concept]] — Sibling-adjacent concept
- [[domains/global/concepts/market-vault-structural-divergence/_concept]] — Vault-vs-market divergence
- [[events/mv-hondius-hantavirus-outbreak-2026]] — Canonical event
- [[domains/health/entities/hantavirus]] — Pathogen entity
