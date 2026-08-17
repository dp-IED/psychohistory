---
type: procedure
tags: [procedure, sports]
domain: sports
title: "Tournament Draw Verification"
slug: tournament-draw-verification
purpose: "Systematic verification of whether a named player/team appears in a tournament draw before forecasting match existence or outcome markets"
trigger: "ANY Polymarket question about a match result, completed match, or player performance in a specific tournament"
status: active
created: 2026-05-20
---

# Tournament Draw Verification Procedure

## Why This Exists

The Cervia Challenger run (`20260520-065750`, p_yes=0.01, Brier=0.0001) was resolved correctly by checking the tournament draw — but the vault contributed zero signal. This procedure codifies the method so future runs don't rediscover it from scratch.

**Core insight**: For low-volume Polymarket sports markets (<$1K USDC), the single highest-leverage question is not "how good are these players?" but "are these players actually in the tournament?" Many scam/error markets are created for matchups that never appear in the draw.

## Step 1: Identify the Tournament

Extract from the question:
- Tournament name (e.g., "Challenger Città di Cervia", "Internationaux de Strasbourg")
- Tournament tier (ATP Challenger/WTA 250/Grand Slam/etc.)
- Location and dates
- Surface type (clay, hard, grass, carpet)

Reference: [[domains/sports/entities/cervia-challenger]] for Challenger tournament context.

## Step 2: Locate the Official Draw

ATP Challenger draws are published on:
- **ATP Tour website** (atptour.com) — main and qualifying draws for ATP/Challenger events
- **ITF website** (itftennis.com) — ITF World Tennis Tour events
- **Flashscore / Tennis Explorer** — aggregated third-party sources

For WTA events: WTA website (wtatennis.com).

**Key date check**: The draw is typically published 24-48 hours before the tournament starts. If the question's cutoff is before the draw release, the match existence is inherently uncertain but highly unlikely (default priors: >95% NO for specific named matchups in Challenger-tier events).

## Step 3: Verify Player Presence in Draw

For each named player in the question:

1. **Check main draw entry list**: Is the player listed in the main draw? ATP rankings determine direct acceptance.
2. **Check qualifying draw**: Is the player listed in qualifying? Some players enter through qualifying.
3. **Check alternate list / withdrawals**: A player may have withdrawn before the draw was finalized.
4. **Check protected ranking / wild card**: Lower-ranked players may enter via wild card or protected ranking.

**Decision rules**:
- Both players in the main draw → Match possible; proceed to Step 4
- One or both players NOT in draw → Match impossible; p_yes ≈ 0.01 (edge case: qualifying-to-main-draw promotion, but this is ~5-10% probability for any given qualifier)
- Draw not yet published → p_yes = (prior of both being in draw) × (base rate of random matchups resolving). Default prior: 0.02-0.05 for specific named matchups in Challenger/ITF events.

## Step 4: Verify Round Compatibility

Even if both players are in the draw, they must be in the same half/quarter to meet:

1. **Check draw bracket**: Are they in the same half of the draw?
2. **If same half**: Can they meet in the current or a future round?
3. **If opposite halves**: They could only meet in the final.
4. **If one is in qualifying and one in main draw**: They cannot meet in the main draw unless the qualifier qualifies.

**Common error**: A market may name two players who are both in the tournament but in opposite halves or already eliminated. The resolution condition "a completed match between Player A and Player B" requires they actually played each other.

## Step 5: Check Market Liquidity

Polymarket liquidity is a sanity check on market legitimacy:

| Volume | Signal | Action |
|--------|--------|--------|
| <$100 USDC | Scam/error market | Verify draw independently; market price is noise |
| $100-$1K USDC | Thin, unreliable | Weight draw verification >> market price |
| $1K-$10K USDC | Moderate | Draw still dominates for existence markets |
| >$10K USDC | Material | Market may have efficient signal; cross-check with draw |

Reference: [[domains/sports/concepts/sports-market-liquidity-signal/_concept]]

## Step 6: Set Probability

Aggregate findings:

- **Draw confirms both players in same half**: p_yes = base rate of that match occurring (considering injury, withdrawal, retirement). Default: 0.85-0.95.
- **Draw confirms both players in tournament but opposite halves**: p_yes = (base rate of final meeting) ~0.10-0.20 for Challenger events where upsets are common.
- **Draw confirms at least one player NOT in tournament**: p_yes = 0.01 (allow residual for late wild card, alternate entry).
- **Draw not published / insufficient data**: p_yes = 0.02-0.05 (matchup existence prior).

## Step 7: Document in Run File

Record in the run file:
- Which draw source was checked
- Each player's draw status (main draw, qualifying, or absent)
- Round compatibility assessment
- Liquidity assessment and its influence on the final probability

## Example: Cervia Challenger (May 2026)

- Tournament: Challenger Città di Cervia (ATP Challenger 75, clay)
- Players: Lilian Marmousez (FRA) vs Federico Bondioli (ITA)
- Draw check: Marmousez NOT in main draw or qualifying; Bondioli drew Petr Nesterov in R1
- Verdict: Match impossible → p_yes = 0.01
- Liquidity: $100 USDC → noise market confirmed
- Outcome: Resolution = NO, Brier = 0.0001

## Cross-References

- [[domains/sports/entities/cervia-challenger]]
- [[domains/sports/concepts/sports-market-liquidity-signal/_concept]]
- [[domains/sports/entities/jannik-sinner]] — Sinner effect on Italian Challenger proliferation
- [[domains/sports/_domain]] — Sports domain index
- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — Resolution criteria gotchas (match "completed" ≠ match "scheduled")
