---
type: domain
tags: [domain]
title: "Sports"
slug: sports
status: nascent
created: 2026-05-20
coverage: "10 files — 5 entities (jannik-sinner, cervia-challenger, alexa-schull, ava-cavataio, ppa-tour), 1 concept (sports-market-liquidity-signal), 1 procedure (tournament-draw-verification), 2 threads (tennis-challenger-forecasting, ppa-pickleball-tour), 1 domain index"
trigger: "Cervia tennis forecast (Brier 0.0001) was correct on domain reasoning alone — vault contributed zero signal. The PIT context referenced a 'Jannik Sinner entity stub' and 'Sinner effect on Italian Challenger proliferation' but no such file existed at the time. Remediation in May 20 pass 3: created Jannik Sinner entity, Cervia Challenger entity, and sports-market-liquidity-signal concept."
---
# Sports

## Status: Nascent (Expanding)

This domain was a placeholder until the May 20, 2026 Pass 3 curation, which added the first 3 files:
- [[domains/sports/entities/jannik-sinner]] — ATP No. 1, face of Italian tennis
- [[domains/sports/entities/cervia-challenger]] — Italian Challenger tournament
- [[domains/sports/concepts/sports-market-liquidity-signal]] — liquidity threshold for sports markets

## Cervia Freebie (May 20, 2026)

The forecast "Cervia: Completed Match: Lilian Marmousez vs Federico Bondioli" was resolved NO with p_yes=0.01 and Brier=0.0001. This was a correct prediction but **the vault contributed zero structured signal** — the correct prediction came from checking the tournament draw directly, not from vault content (violation of Spec Rule 8).

**Key vault gaps exposed:**
1. No tennis player rankings or tournament data stored on disk
2. No sports domain directory or _domain.md
3. No sports-forecasting methodology or procedure
4. No entity stubs for Lilian Marmousez, Federico Bondioli, or any tennis players
5. PIT context mentioned a "Jannik Sinner entity stub" that does not exist in the vault's file tree
6. No concept for sports betting liquidity signals (the market had $100 USDC volume, an obvious scam/low-liquidity market flag)

**What's needed for future sports questions:**
- A sports-market-liquidity concept: markets with <$1K USDC volume are structurally unreliable — the probability estimate should be based on draw verification, not market price
- At minimum, entity stubs for the Jannik Sinner "Sinner effect" mentioned in prior vault context
- A tennis-specific procedure for checking tournament draws, player rankings, and match schedules

## Forecasting Significance

Sports markets are common on Polymarket but often have thin liquidity. The dominant variable for most sports question accuracy is **draw verification** — checking whether the named players/teams are actually in the tournament. The second-order variable is market liquidity: sub-$1K USDC markets have no efficient pricing signal.

## Gaps (Remaining)

- No ATP/WTA ranking snapshot data for draw-verification
- No tennis surface-specific forecasting dynamics concept
- No concept for Italian tennis market proliferation (Sinner effect on Challenger expansion)
- No ATP player entities beyond Sinner (Alcaraz, Djokovic, Musetti, etc.)

## Recent Additions (May 21, 2026 — Graph Vault Reflection Pass)

This pass was triggered by two recent runs (Cervia tennis p_yes=0.01, Bitcoin >$72K p_yes=0.25) both sharing the "negative-space" pattern: correct NO predictions driven by structural constraints rather than deep domain knowledge.

| What | File | Type |
|------|------|------|
| Tennis Challenger Forecasting thread | `domains/sports/threads/tennis-challenger-forecasting/_thread.md` | Thread |
| PPA Pickleball Tour thread | `domains/sports/threads/ppa-pickleball-tour/_thread.md` | Thread |
| Alexa Schull entity stub | `domains/sports/entities/alexa-schull.md` | Entity |
| Ava Cavataio entity stub | `domains/sports/entities/ava-cavataio.md` | Entity |
| PPA Tour entity stub | `domains/sports/entities/ppa-tour.md` | Entity |

### Key Insight: The Negative-Space Pattern

Both recent runs share a meta-pattern: **correct predictions based on the absence of conditions for YES, not the presence of conditions for NO**. This is already formalized in [[domains/global/concepts/short-horizon-momentum-check/_concept]] but the sports domain provides its clearest expression: a Challenger tennis market with $100 USDC volume and absent players is resolved by draw verification (structural constraint), not tennis expertise.

### Filled (This Pass — May 21, 2026)

- **Cross-referenced**: [[domains/global/concepts/short-horizon-momentum-check/_concept]] — The Cervia case is a canonical exemplar of the existence-failure variant of this global pre-filter. The sports-market-liquidity-signal and short-horizon-momentum-check concepts should be applied together for future sports markets: liquidity filter first, then momentum/existence check.

### Filled (This Pass — May 20, 2026)

| Gap | Resolution |
|-----|-----------|
| No tournament draw verification procedure | Created: `domains/sports/procedures/tournament-draw-verification.md` — 7-step procedure covering ATP/Challenger/WTA draw sources, round compatibility, liquidity thresholds, probability calibration. Canonical example: Cervia Challenger run. **Status: Active and referenced from sports-market-liquidity-signal concept (stale TBD note fixed this pass).** |
