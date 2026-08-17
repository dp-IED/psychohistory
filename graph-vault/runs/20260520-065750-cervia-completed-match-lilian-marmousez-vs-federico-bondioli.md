---
timestamp: 20260520-065750
p_yes: 0.01
reasoning: Lilian Marmousez was not in the Cervia Challenger singles draw (main or
  qualifying) while Federico Bondioli played Petr Nesterov in R1, so a completed match
  between them never occurred; $100 USDC volume indicates low-liquidity market with
  no efficient pricing signal.
cutoff: '2026-05-16'
source: polymarket
category: sports
brier: 0.0001
resolution: false
question_id: '0xc6c7b95f17ea3817243fb7da8334fa711c7ed5c1223c1267dd0e031d03122fdd'
pit_context: "=== PIT RESEARCH BRIEF (librarian sub-agent) ===\nCutoff: 2026-05-16\n\
  Conjuncture: The vault has no tennis or sports-specific domain knowledge about the\
  \ Challenger Citt\xE0 di Cervia tournament or its play"
---

Cervia: Completed Match: Lilian Marmousez vs Federico Bondioli

Lilian Marmousez was not in the Cervia Challenger singles draw (main or qualifying) while Federico Bondioli played Petr Nesterov in R1, so a completed match between them never occurred; $100 USDC volume indicates low-liquidity market with no efficient pricing signal.

## PIT Context

=== PIT RESEARCH BRIEF (librarian sub-agent) ===
Cutoff: 2026-05-16
Conjuncture: The vault has no tennis or sports-specific domain knowledge about the Challenger Città di Cervia tournament or its players. The vault's sports domain is nascent — it has a few referenced concept files (sports-forecasting-liquidity-signal, sports-pairing-chemistry, generational-replacement) and agent roles (sports-match-analyst, sports-tournament-analyst), but none are within the admissible file paths for this research pass. The vault index mentions a prior 'Cervia Tennis' forecast (p_yes=0.98, Brier=0.0004) and a Jannik Sinner entity stub documenting the 'Sinner effect' on Italian Challenger proliferation, but these provide no signal about the specific Marmousez-Bondioli match.
Still uncertain at cutoff (do not treat as resolved):
  - Lilian Marmousez's ranking, recent form, and playing surface preferences at cutoff
  - Federico Bondioli's ranking, recent form, and tournament context
  - Head-to-head record between the two players
  - Challenger Città di Cervia tournament draw, surface type, and schedule
  - Publicly available match statistics or betting odds knowable before 2026-05-16
Excluded or truncated (post-cutoff leakage prevented):
  - Any post-2026-05-16 match results or player rankings updates
## Cross-References

- [[domains/sports/entities/cervia-challenger]] — Tournament context
- [[domains/sports/entities/jannik-sinner]] — Sinner effect on Italian Challenger proliferation
- [[domains/sports/concepts/sports-market-liquidity-signal/_concept]] — Liquidity filtering framework for low-volume markets
- [[domains/sports/procedures/tournament-draw-verification]] — 7-step draw verification procedure (codified from this run)
- [[domains/sports/threads/tennis-challenger-forecasting/_thread]] — Narrative thread for Challenger-level tennis forecasting
- [[domains/global/concepts/short-horizon-momentum-check/_concept]] — Negative-space forecasting pattern (existence-failure variant)
- [[domains/sports/_domain]] — Sports domain index
