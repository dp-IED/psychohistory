---
type: reflection
tags: [meta, curation]
date: 2026-05-20
trigger: "Graph vault reflection after 2 recent forecast runs — curation pass 3"
---

# Graph Vault Curation — May 20, 2026 (Pass 3)

## Recent Runs Re-assessed

| Run | p_yes | Brier | Vault Contribution | Verdict |
|-----|-------|-------|-------------------|---------|
| Bitcoin >$72K by Feb 13 | 0.25 | 0.0625 | Strong: crypto-macro-linkages, SBR, halving-cycle concepts + Bitcoin event | Vault-driven |
| Cervia: Marmousez vs Bondioli | 0.01 | 0.0001 | Freebie (zero signal — sports domain was empty shell) | ❌ Freebie (now partially remediated) |

## Structural Fixes

### 1. _macro_gaps.md frontmatter bug
- **What**: Line 8 had `tags: [meta]e` — trailing 'e' broke YAML parsing integrity
- **Fix**: Changed to `tags: [meta]`
- **Why**: Structural integrity fix. A file that tracks gaps shouldn't have its own gap.

### 2. Broken wikilinks repaired (bitcoin-feb-2026-drawdown.md)
- **What**: Two thread wikilinks referenced `[[...threads/us-crypto-regulation]]` without the `/_thread` suffix, pointing to non-existent files
- **Fix**: `[[domains/economics/threads/us-crypto-regulation]]` → `[[domains/economics/threads/us-crypto-regulation/_thread]]` (same for `us-monetary-policy-cycle-2022-2026`)
- **Why**: Spec Rule 36 — every wikilink must resolve. Dangling references create illusion of coverage.

### 3. fed-chair-jerome-powell redirect standardized
- **What**: Used `redirect_to: "jerome-powell"` (relative path, no namespace) instead of full wikilink format matching the gary-gensler redirect convention
- **Fix**: Standardized to `type: entity-redirect` with `redirect_target: "[[domains/economics/entities/jerome-powell]]"` and added `canonical` field
- **Why**: Consistency across duplicate entity redirects prevents parser confusion

## Sports Domain — First Content (Priority 2 Gap Filled)

### Why This Pass
The sports domain was flagged as a freebie gap in **two prior curation passes** (Pass 1 and Pass 2 of May 20) but remained an empty shell — zero entities, zero concepts, zero procedures. The Cervia run was identified as a "freebie" in both passes but no content was created. This pass breaks that pattern.

### What Was Created

| File | Rationale |
|------|-----------|
| `domains/sports/entities/jannik-sinner.md` | ATP No. 1, most prominent Italian tennis player. The "Sinner effect" (Italian Challenger proliferation driven by his rise) was mentioned in the Cervia PIT context but no entity existed. Essential anchor entity for tennis forecasting. |
| `domains/sports/entities/cervia-challenger.md` | The specific tournament from the freebie run. Documents that it's an ATP Challenger 125 on clay, and that its match-level Polymarket questions typically have sub-$1K liquidity. |
| `domains/sports/concepts/sports-market-liquidity-signal/_concept.md` | The most important generalizable insight from the Cervia freebie: sports markets with <$1K USDC liquidity are noise, not signal. Formalizes the four-tier liquidity framework ($1K → $10K → $100K thresholds) with appropriate forecasting approaches for each tier. |

### Gaps Remaining in Sports Domain
- ATP/WTA ranking snapshots for draw-verification
- Tournament-draw-verification procedure
- Surface-specific dynamics (clay vs hard vs grass for forecasting)
- Italian tennis market proliferation concept (Sinner effect)
- Additional ATP player entities (Alcaraz, Djokovic, Musetti)

## Economics Domain — No New Gaps Identified

The Bitcoin run (p_yes=0.25, Brier=0.0625) was vault-driven — the crypto-market thread, bitcoin-feb-2026-drawdown event, and existing concepts (crypto-macro-linkages, SBR, halving cycle) supported the forecast. No new gaps identified. The two thread wikilink fixes in the bitcoin event were the only economics-domain work needed.

## Vault Health Summary

| Metric | Value | Change |
|--------|-------|--------|
| Entity files | ~200 | +3 (Sinner, Cervia, +1 concept) |
| Events | 1 event file | +0 |
| Concepts | 52 | +1 (sports liquidity) |
| Domains with content | 6/6 | USA/Global/LA/EA/Economics/MENA — all populated; Sports now has first 3 files |
| Broken wikilinks found & fixed | 3 | -3 structural |
| Frontmatter bugs fixed | 1 | -1 structural |

## Cross-References

- [[runs/20260520-065636-will-the-price-of-bitcoin-be-above-72-000-on-february-13]] — Bitcoin run
- [[runs/20260520-065750-cervia-completed-match-lilian-marmousez-vs-federico-bondioli]] — Cervia freebie
- [[events/bitcoin-feb-2026-drawdown]] — Fixed thread wikilinks
- [[domains/sports/_domain]] — Sports domain (updated)
- [[_macro_gaps]] — Gap tracker (updated)
