---
type: reflection
tags: [meta, curation]
date: 2026-05-20
trigger: "Graph vault reflection after 2 recent forecast runs — curation pass 2"
---
# Graph Vault Curation — May 20, 2026 (Pass 2)

## Recent Runs Re-assessed

| Run | p_yes | Brier | Vault Contribution | Verdict |
|-----|-------|-------|-------------------|---------|
| Bitcoin >$72K by Feb 13 | 0.25 | 0.0625 | Strong: crypto-macro-linkages, SBR, halving-cycle concepts + Bitcoin event | ✅ Proper vault-driven forecast |
| Cervia: Marmousez vs Bondioli | 0.01 | 0.0001 | Zero: no sports domain, no entities, no procedure | ❌ Freebie (Spec Rule 8 violation) |

## Curations Performed

### 1. Crypto-Market Thread Created (Priority 1 Gap Filled)
- **File**: `domains/economics/threads/crypto-market/_thread.md`
- **Rationale**: The Bitcoin drawdown run confirmed the vault needed a market dynamics thread distinct from the regulation-focused `us-crypto-regulation` thread. The new thread tracks price action, ETF flows, stablecoin dynamics, and most importantly the **macro-crypto linkage regime** — which macro factors dominate crypto at which phase. The Feb 2026 drawdown is documented as the canonical case where tariff uncertainty > monetary policy signal.
- **Links to**: 6 concepts, 3 threads, 6+ entities, the Bitcoin drawdown event

### 2. Duplicate Entity Merged: gary-gensler
- **Economics version** (`domains/economics/entities/gary-gensler.md`) → redirect stub pointing to canonical `domains/usa/entities/gary-gensler.md`
- **Rationale**: Gensler was US SEC Chair first, economics actor second. The USA version had richer content (full timeline, enforcement analysis, forecasting significance).

### 3. Sports Domain Stub Created
- **File**: `domains/sports/_domain.md` (status: nascent)
- **Rationale**: The Cervia tennis run was a freebie — correct prediction (Brier 0.0001) but zero vault signal. The sports domain currently has zero entities, concepts, procedures, or threads. Documented as Priority 2 gap.

### 4. Broken Wikilinks Fixed
- Bitcoin event file: `[[domains/economics/runs/...]]` → `[[runs/...]]` (runs are at vault root, not under domain)
- Quoted descriptions removed for cleaner cross-references

### 5. Gap Tracking Updated
- `_macro_gaps.md`: New "Pass 2" section with 3 filled gaps. Priority 1 cleared (crypto-market thread created). Priorities re-ranked with sports gap as new Priority 2 #1.
- `runs/_index.md`: Added freebie detection section and explicit Cervia freebie violation documentation.
- `events/_index.md`: Marked Bitcoin Feb 2026 drawdown as filled.

### 6. Economics Domain Updated
- Added `crypto-market` thread to frontmatter and thread listing

## Gaps Remaining for Future Passes

1. **Sports domain build-out** (Priority 2): Jannik Sinner entity, draw-verification procedure, liquidity-signal concept
2. **ETF flow tracking** (Priority 2): Dedicated thread or component within crypto-market thread
3. **Bitcoin ETF issuer entities** (Priority 2): Fidelity (FBTC), ARK 21Shares, Bitwise stubs
4. **Macro concept files** (Priority 3): yield-curve-dynamics, debt-sustainability-framework, dollar-smile-theory, taylor-rule-calibration
5. **Broken wikilink cleanup**: Pre-existing broken wikilinks in us-crypto-regulation thread (`[[regulatory-precedent-cascade]]`, `[[bitcoin-halving-cycle]]`, `[[dc-circuit-court-of-appeals]]` — all missing full path)
6. **Thread continuity**: All 5+ active threads should be verified for content through 2026-Q1 at minimum
