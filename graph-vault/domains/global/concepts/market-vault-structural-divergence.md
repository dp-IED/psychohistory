---
type: concept
tags: [concept, methodology, calibration, polymarket]
title: "Market-Vault Structural Divergence"
slug: market-vault-structural-divergence
domain: global
first_observed: 2026-05-21
canonical_cases:
  - "SCOTUS TikTok delay (gold_22): PM=0.24 → vault=0.07"
  - "Venezuela Maduro wins (gold_20): PM=0.655 → vault=0.12"
  - "Raúl Castro US custody (pm-cuba): PM=0.1235 → vault=0.005"
  - "US strike on Colombia (pm-strike-colombia): PM=0.27 → vault=0.12"
  - "Cloobeck CA Governor (pm-cloobeck): vault=0.002, PM=0.0015 — VAULT-MARKET CONVERGENCE (inverse pattern)"
  - "Colombia first-round Cepeda (colombia-2026): PM=0.825 → actual 40.9% second place — polling-error propagation type"
status: active
related_concepts:
  - polymarket-residual-uncertainty-after-public-event
  - structural-improbability-check
  - forecast-resolution-criteria-gotchas
  - scotus-procedural-signals
  - authoritarian-electoral-facade
---

# Market-Vault Structural Divergence

## Definition

A calibration pattern where the vault's procedural or structural reasoning
produces a probability **significantly lower** than the Polymarket price
(divergence >0.10), and the divergence is justified by vault knowledge
that is structured, procedural, and knowable at cutoff — not post-hoc
narrative or vault hindsight.

This is the **inverse** of the [[polymarket-residual-uncertainty-after-public-event]]
pattern. In that pattern, the market stays below 1.0 after an event has
occurred; here, the vault goes below the market because procedural/
structural reasoning shows the market is overpricing YES.

## Distinctive Features

This is NOT the same as:
- **"I'm more pessimistic than the market"** — generic uncertainty
  reduction. Structural divergence requires a specific procedural or
  structural mechanism the market is failing to price.
- **"The market is wrong"** — the market can be "right" in its pricing
  of noise/recency effects while the vault's structural signal diverges.
- **Post-hoc certainty** — correctly calling a low-probability outcome
  NO because you know it didn't happen. Structural divergence must be
  justified by pre-cutoff knowledge.

## Canonical Case 1: SCOTUS TikTok Delay (gold_22)

| Parameter | Value |
|-----------|-------|
| Question | Will SCOTUS delay the TikTok ban? |
| Cutoff | 2025-01-12 |
| Polymarket YES | 0.2400 |
| Vault p_yes | 0.07 |
| Divergence | -0.17 (vault lower) |
| Type | **Procedural determinism** |

**Why the vault was lower**: The vault's SCOTUS procedural signals
framework showed four unmistakable signals that NO was near-deterministic:
(1) cert before judgment granted Dec 18 — extremely rare, used only when
a question is clear-cut; (2) compressed 3-week schedule from cert to
argument — Court expedited merits review to rule before the Jan 19
deadline, not to delay; (3) zero stays at any stage — strongest single
signal the Court finds the challenge unlikely to succeed; (4) oral
arguments held without any interim relief.

**Why the market was higher**: The 0.24 price reflected retail recency
noise from Jan 10 oral arguments (polarized reactions) and a functionally
irrelevant Trump hush-money stay denial (5-4) on the same day. Retail
traders conflated the two cases despite them being procedurally distinct.

**Key insight**: The market was wrong NOT because of information
asymmetry (the vault had no secret data) but because it failed to
integrate **procedural sequencing** — the Court's own procedural choices
had already locked in the outcome direction. The vault's SCOTUS signals
framework provided this integration step that the market missed.

## Canonical Case 2: Venezuela Maduro Wins (gold_20)

| Parameter | Value |
|-----------|-------|
| Question | Will Nicolás Maduro win the 2024 Venezuela presidential election? |
| Cutoff | 2024-06-30 |
| Polymarket YES | 0.6550 |
| Vault p_yes | 0.12 |
| Divergence | -0.535 (vault far lower) |
| Type | **Structural knowledge override** |

**Why the vault was lower**: The vault's authoritarian-electoral-facade
concept and late-candidate-substitution framework showed: (1) all 4
independent pollsters gave Gonzalez 30-40pt lead (Gonzalez 52-59%,
Maduro 17-22%); (2) ConVzla parallel vote tabulation was operational
across >80% of polling centers; (3) the late-candidate substitution from
Machado to Gonzalez succeeded on all 5 framework dimensions; (4) the
resolution criteria for Polymarket's "win" language pointed to vote
outcome, not power transition — meaning the regime's ability to
fabricate results was irrelevant if the question resolved on who got
more votes.

**Why the market was higher**: The 0.655 price reflected the intuitive
heuristic "Maduro controls the CNE, TSJ, and military — he can't lose."
This heuristic was correct about regime institutional control but
wrong about multiple compensating factors: (a) opposition PVT
infrastructure made outright fabrication documentable; (b) international
pressure (US oil sanctions reinstated Apr 2024) constrained the regime;
(c) the resolution criteria decoupled "regime claims victory" from
"market resolves YES."

**Key insight**: The market priced the **surface-level structural
variable** (regime institutional control → Maduro can't lose) but failed
to price the **deeper structural variables** (PVT documentation,
resolution criteria, late-candidate-substitution viability). The vault
integrated all five levels of structural analysis.

## Diagnostic Questions

Before diverging below the Polymarket YES price by more than ±0.05:

1. **Is the divergence based on procedural determinism?** Do procedural
   rules (SCOTUS process, legislative calendar, regulatory timeline)
   lock in the outcome direction regardless of surface-level noise?
   → Strong justification for divergence.

2. **Is the divergence based on structural knowledge the market lacks?**
   Does the vault have a framework (PVT infrastructure, late-candidate
   substitution, electoral-facade resolution criteria) that reveals a
   factor the market is systematically ignoring?
   → Justification depends on framework maturity and pre-cutoff
   verifiability.

3. **Is it based on post-hoc narrative?** Did you learn the outcome
   from reading a vault file that contains post-resolution facts?
   → **Do not diverge.** Apply Rule 11: output within ±0.05 of PM.

4. **Is the divergence supported by multiple independent frameworks?**
   The Maduro case used: authoritarian-electoral-facade + late-candidate-
   substitution + win-vote-vs-take-office + poll aggregation. Multiple
   frameworks converging on the same direction is much stronger than
   relying on a single framework.

5. **Can you articulate why the market is wrong?** Not "the market is
   stupid" but a specific mechanism: the market is pricing retail
   recency noise (SCOTUS) or surface-level heuristic (Maduro). If you
   cannot articulate the market's error mechanism, diverge at your
   peril.

## Canonical Case 3: Raúl Castro US Custody (2026-05-22)

| Parameter | Value |
|-----------|-------|
| Question | Ex-Cuba leader Raul Castro in US custody by June 30? |
| Cutoff | 2026-05-22 |
| Polymarket YES | 0.1235 |
| Vault p_yes | 0.005 |
| Divergence | -0.1185 (vault lower) |
| Type | **Structural impossibility check** |

**Why the vault was lower**: The vault's 5-mechanism checklist (extradition, abduction, voluntary surrender, regime-change→arrest, medical-evacuation-as-custody) returned zero active paths for any mechanism within the 39-day window. Each mechanism was independently evaluated and ruled out on structural grounds: no US-Cuba extradition treaty, no US criminal indictment against Raúl, no Interpol notice, no active extraction operation, zero incentive for voluntary surrender by a 95-year-old retired former leader living in his home country. The structural-improbability-check decision tree was applied step-by-step and produced p_yes < 0.01.

**Why the market was higher**: The 12.35% Polymarket YES price was an anomalous speculative position in a $99K liquidity pool — likely driven by (a) retail confusion between "Raúl Castro" and unrelated Cuba news, (b) gamblers pricing a black-swan tail (e.g., Raúl dying abroad with US claiming custody of remains — not what "custody" means in context), or (c) low-liquidity noise where a single large position can set the price. Unlike the SCOTUS and Maduro cases, there was no plausible market-priced mechanism to override — just noise.

**Key insight**: This is the vault's first **zero-mechanism** divergence case. The SCOTUS and Maduro cases involved the vault identifying a mechanism the market was mispricing (procedural determinism, structural knowledge override). The Raúl Castro case involves the vault identifying that **no mechanism exists at all**. This changes the divergence taxonomy: vault can diverge not only by (a) better pricing an existing mechanism, but also by (b) recognizing the absence of any mechanism.

**Cross-reference**: [[domains/global/concepts/structural-improbability-check/_concept]] — The structural improbability check formalizes the mechanism-checklist approach used here. The Raúl Castro case is the canonical example of Step 1→5 producing p < 0.01 via joint probability of independent failures.

## Canonical Case 4: US Strike on Colombia by Dec 31 (2026-05-22)

| Parameter | Value |
|-----------|-------|
| Question | US-initiated drone/missile/air strike on Colombian soil before Dec 31, 2026 |
| Cutoff | 2026-05-22 |
| Polymarket YES | 0.2700 |
| Vault p_yes | 0.12 |
| Divergence | -0.15 (vault lower) |
| Type | **Theater-specific base rate override** |

**Why the vault was lower**: The vault's theater-specific-strike-base-rates framework classifies Latin America as an **inactive strike theater** — zero US kinetic strikes in 35+ years. The forecast applied a 4-factor multiplicative chain (trigger × authorization × operational × non-suppression), each factor <0.3 for any plausible scenario. This contrasts with the market's implicit linear trigger→strike model.

**Why the market was higher**: The 27% price reflected: (a) Soleimani recency bias — treating a single 2020 precedent as evidence that unilateral strike authority is elastic; (b) conflation of advisory/training presence (continuous) with kinetic strikes (zero); (c) general "Trump will do something crazy" sentiment that priced constant per-month probability instead of structural barrier analysis.

**Key insight**: This is the vault's first **theater-level base rate** divergence case. Previous canonical cases diverged on procedural determinism (SCOTUS), structural knowledge override (Maduro), or zero-mechanism recognition (Raúl Castro). This case diverges on a different dimension: the **geographic distribution** of a military action type. The market treated strike probability as uniformly responsive to triggers; the vault showed that theater classification creates structural floors that even plausible triggers cannot easily overcome.

---

## Vault-Market Convergence (Inverse Pattern)

Not all structural probability estimates diverge from market prices. The Cloobeck CA Governor case demonstrates the **inverse pattern**: vault and market converging on a structural near-zero outcome, which validates both the market's liquidity-adequate pricing and the vault's structural-improbability-check framework.

### Canonical Case 5: Cloobeck CA Governor (Vault-Market Convergence)

|| Parameter | Value |
||---|---------|-------|
|| Question | Will Stephen Cloobeck win the California Governor Election in 2026? |
|| Cutoff | 2026-05-22 |
|| Polymarket YES | 0.0015 |
|| Vault p_yes | 0.002 |
|| Discrepancy | +0.0005 (within ±0.001 — effectively identical) |
|| Volume | $994K (10x the Raúl Castro market) |

**Why vault and market converged**: Both applied a structural-blockers checklist to Cloobeck's candidacy:
1. Xavier Becerra is prohibitive frontrunner (67.65% YES, statewide name recognition, institutional support)
2. Tom Steyer has $250M+ self-funding capacity (21.05% YES)
3. Cloobeck has never held elected office and lacks celebrity status of CA's last non-politician governor
4. Diamond Resorts' aggressive sales practices provide devastating attack-ad material in a Democratic primary
5. With 15+ Democratic candidates in a top-two primary, Cloobeck must first survive to the general — Hilton (85% to advance) and Becerra (83% to advance) are the prohibitive top-two

**Why this case matters for the divergence framework**:

| Dimension | Divergence Cases (1-4) | Convergence Case (5) |
|-----------|----------------------|---------------------|
| Vault-Market gap | >0.10 | <0.001 |
| Market liquidity | $50-100K range | $994K (high) |
| Market error mechanism | Retail noise, recency bias, thin liquidity | None — market correctly priced structural near-zero |
| Vault contribution | Identifying the error | Validating the market's pricing |
| Calibration signal | Vault adds value over market | Vault confirms market is functioning |

**Paired insight with Case 3 (Raúl Castro)**: Both Cloobeck and Raúl Castro are structural near-zero forecasts using the same structural-improbability-check framework. The Raúl Castro case produced a 0.1185 divergence (vault 0.005 vs PM 0.1235), while the Cloobeck case produced convergence (vault 0.002 vs PM 0.0015). The key difference is **market liquidity**: the $99K Raúl Castro market was 10x thinner than the $994K Cloobeck market. This pair demonstrates that:

- When liquidity is adequate ($1M+), Polymarket correctly prices structural near-zero outcomes
- When liquidity is thin (<$100K), Polymarket can diverge wildly from structural reality
- The structural-improbability-check framework is NOT systematically biased toward underestimation — it converges when the market is rational

**Forecasting rule distilled**: Before diverging from a Polymarket YES price on structural grounds, check market liquidity. If volume is <$100K, skepticism is warranted. If volume is >$500K, the vault should treat market price as a strong prior and diverge only when multiple independent frameworks converge on a divergence direction.

### Canonical Case 6: AI Safety Bill (Dynamic Vault-Market Convergence)

| Parameter | Value |
|-----------|-------|
| Question | US enacts AI safety bill before 2027? |
| Cutoff | 2026-05-21 (initial), 2026-05-23 (swing) |
| Polymarket YES (May 21) | 0.495 |
| Vault p_yes (May 21) | 0.38 |
| Initial gap | -0.115 (vault lower — divergence type) |
| Polymarket YES (May 23) | 0.415 |
| Updated gap | -0.035 (within ±0.05 — convergence) |
| Volume | $98.8K |
| Type | **Dynamic convergence** (market corrects toward vault over time) |

**What happened**: On May 21, the vault assigned p_yes=0.38 to the AI safety bill passing before 2027 — well below the Polymarket price of 49.5%. The vault's reasoning was based on six structural barriers: Trump administration deregulatory stance, midterm-year bandwidth constraints, partisan gridlock on liability frameworks, industry opposition (only Anthropic supported mandates), preemption deadlock with states, and compressed legislative calendar post-election (~40 working days).

By May 23, the Polymarket price had dropped to 41.5% — an 8.5pp swing toward the vault's estimate, with the gap narrowing to 0.035 (within the ±0.05 convergence zone).

**Why this matters for the convergence framework**:

| Dimension | Static Convergence (Cloobeck) | Dynamic Convergence (AI Safety Bill) |
|-----------|------|-------------------------------------|
| Vault-Market gap at forecast | <0.001 | -0.115 (initially diverged) |
| Gap after correction | <0.001 | -0.035 (converged) |
| Market movement | None (both already aligned) | 8.5pp correction over 2 days |
| What the vault contributed | Validated market pricing | Predicted correction direction and magnitude |
| What caused the market movement | N/A — no correction needed | Market absorbed structural gridlock analysis that vault had already priced |
| Vault role | Verification | Leading indicator |

**Why the market moved**: The drop from 49.5% to 41.5% was likely driven by:
1. **News absorption**: May 21-23 saw no single dramatic event but a cumulative realization that the Trump administration's deregulatory posture was hardening
2. **Midterm calendar crystallizing**: The Senate's legislative calendar for June-July took shape, revealing limited floor time for AI legislation
3. **Stakeholder positioning**: Industry lobbying against mandatory safety testing became more visible, reinforcing the gridlock narrative
4. **Market self-correction**: The initial 49.5% price reflected residual hope from bipartisan AI concern (2023-2024 Schumer framework era). As the structural barriers became clearer, the market re-rated downward

**Key insight for forecasters**: When the vault diverges from the market on structural grounds, the right forecast is not to bet against the market immediately but to (a) document the structural divergence with specific mechanisms, (b) note the market price as a lagging indicator, and (c) anticipate convergence as the market absorbs the structural analysis. The AI safety bill case shows that convergence can happen within 2 days when the structural factors are unambiguous.

**Relationship to the four divergence types**: This case is unique because it starts as a divergence (structural knowledge type — the vault had a comprehensive legislative history the market was ignoring) and transitions to convergence. It demonstrates that the divergence/convergence binary is not permanent — some questions shift from divergence to convergence as the market recalibrates.

**Forecasting rule distilled**: When vault and market diverge on structural grounds, do not treat the divergence as permanent. Track the market price over time. If the market converges toward the vault within 48-72 hours without a news catalyst, the vault's structural analysis was likely correct and the market was simply slow to re-price.

---

## Canonical Case 7: Colombia First-Round Winner — Polling Error Propagation (2026-05-21)

| Parameter | Value |
|-----------|-------|
| Question | Who wins Colombia presidential election first round — Cepeda or a specific right candidate? |
| Cutoff | 2026-05-21 |
| Polymarket Cepeda YES | **0.825** (82.5% to win first round) |
| Actual first-round result | Cepeda 40.9% — **second place** behind de la Espriella 43.7% |
| Vault p_yes (Cepeda first-round) | 0.08 (indirect — vault forecast no first-round winner outright) |
| Type | **Polling error propagation** |

**What happened**: The Polymarket priced Iván Cepeda at 82.5% to win the May 31 first round, reflecting pre-election polling that showed Cepeda at ~35-38% and de la Espriella at ~15-18%. The actual result was reversed: de la Espriella 43.7%, Cepeda 40.9%. The market was profoundly wrong — one of the largest documented prediction market pricing failures relative to structural reality.

**Why the vault did NOT diverge on this specific metric (first-round winner)**: The vault's fragmented-right-wing-field concept correctly identified that no candidate would reach 50%+1 in the first round (p_yes=0.08 for first-round outright win). However, the vault did NOT explicitly model the scenario where de la Espriella — not Cepeda — would be the first-round frontrunner. The vault's analytical framework correctly predicted a runoff but misidentified which candidate would lead into it.

**Why the market was wrong**: The Polymarket price embedded the polling error without correction. Pre-election polls systematically underestimated right-wing consolidation by ~25pp, likely because:
1. **Polling mode effects**: Polls failed to capture the intensity of anti-Petro sentiment among right-wing voters
2. **Name recognition lag**: De la Espriella's populist campaign gained momentum faster than pollsters' field windows captured
3. **Social desirability bias**: Voters reluctant to express support for de la Espriella's combative style to pollsters
4. **Late-breaker dynamics**: The final polling week saw significant movement toward de la Espriella that static polling averages couldn't capture

**Why this case is different from Cases 1-6**:

| Dimension | Previous Cases | Case 7 (Colombia) |
|-----------|---------------|-------------------|
| Market error type | Vault structural reasoning overrides market | Market priced flawed polling data without correction |
| Vault role | Identifying mechanism the market missed | Correctly predicting structural outcome (runoff) but for wrong candidate |
| Resolution predictability | Vault predicted NO direction correctly | Vault predicted runoff (correct) but market's 82.5% on Cepeda was wrong — vault didn't explicitly target this sub-question |
| Timeframe | Vault p_yes stable across cases | Vault p_first_round_win=0.08 was correct; p_Cepeda_wins_first_round was not explicitly forecast |
| Post-mortem value | Vault's structural frameworks validated | Vault's frameworks partially validated but had a blind spot on which right candidate would consolidate |

**Key insight**: This case establishes a **new divergence type — polling-error propagation** — where the market's error originates not from pricing noise or structural naivety, but from embedding flawed polling data into prices without correction. Unlike the Maduro case (where the market ignored PVT data and resolution criteria) or the SCOTUS case (where the market ignored procedural determinism), this case involves the market treating polls as truth when the polls themselves were structurally wrong.

**Forecasting rule distilled**: When a prediction market price for an electoral outcome is closely aligned with published polling averages (>0.80), verify the polling methodology independently before trusting the market price. Markets can be profoundly wrong when the polls feeding them are wrong. Look for:
- Large polling errors in recent elections in the same country/region (even if polls were previously reliable)
- A populist/anti-establishment candidate whose support polls may undercount (name recognition lag, social desirability bias)
- Late-breaking momentum that static polling averages do not capture
- The market pricing at >0.75 for a candidate whose polling is <0.40 with the margin of error overlapping second place

**Cross-reference**: [[domains/latin-america/concepts/fragmented-right-wing-field/_concept]] — The concept that partially captured this dynamic but missed the direction of right-wing consolidation. [[domains/latin-america/threads/colombia-2026-presidential-election/_thread]] — Full thread with pre/post-first-round analysis.

---

## Canonical Case 8: Andy Burnham Next UK PM — Sequential Hurdle Divergence (2026-06-21)

| Parameter | Value |
|-----------|-------|
| Question | Will Andy Burnham be the next Prime Minister of the UK in 2026? |
| Cutoff | 2026-06-21 |
| Polymarket YES | **0.953** (95.3%) |
| Vault p_yes | **0.35** |
| Divergence | **-0.603** (vault far lower — largest absolute divergence in vault history) |
| Market volume | $991K (high liquidity — not a thin-market artifact) |
| Type | **Sequential hurdle chain** |

**Why the vault was lower**: The vault decomposed the Burnham path into THREE independent sequential hurdles, each with moderate-to-low probability, producing a compound probability far below the market's implicit single-event pricing:

**Hurdle 1 — Parliamentary seat**: Burnham is Mayor of Greater Manchester, NOT an MP. He must win a by-election in a safe Labour seat. Requires: (a) an MP in a safe seat resigning or dying, (b) Burnham being selected as Labour candidate for that seat, (c) winning the by-election (usually safe but takes 6-12 weeks). Estimated probability: ~40%.

**Hurdle 2 — PM resignation/removal**: The current Labour PM must resign or be removed. No challenge had been announced as of June 21. Labour leadership requires: (a) nomination by 20+ Labour MPs, (b) ballot of members/affiliates, (c) several weeks for the contest. Estimated probability: ~30-50% if polling declines; ~10-20% if stable.

**Hurdle 3 — Burnham wins leadership**: He has strong name recognition ("King of the North") but faces Cabinet ministers with more recent national platforms. Estimated probability: ~30-40%.

**Compound probability**: 0.40 × 0.50 × 0.40 = **8%** under optimistic assumptions. Even with generous adjustments, vault could not justify above 35%.

**Why the market was higher at 95.3%**: This is extraordinary for any G7 "next PM" market without an announced resignation or obvious succession crisis. Potential explanations: (a) insider information — market participants know of an imminent PM resignation announcement; (b) resolution criteria quirk — "next PM" interpreted as next person to hold the office even briefly; (c) concentrated position — a small number of informed traders dominating a multi-candidate contract; (d) Contract structure — 95.3% may be Burnham's share against other named candidates (e.g., Streeting, Rayner), not against "status quo/no change."

**Key insight — new divergence type: Sequential Hurdle Chain**: This case establishes a **fifth divergence type**. Previous types were:

| Type | Case | Mechanism |
|------|------|-----------|
| Procedural Determinism | SCOTUS TikTok | Procedural rules lock in outcome |
| Structural Knowledge Override | Maduro win | Vault has framework market lacks |
| Zero Mechanism | Raúl Castro | No active path exists at all |
| Theater Base Rate | Colombia strike | Geographic distribution creates floor |
| **Sequential Hurdle Chain** | **Burnham PM** | **Market prices single event; vault decomposes into AND-chain** |

The sequential hurdle chain differs from previous types because:
1. **It's not about hidden knowledge** — the hurdles are publicly visible (Burnham is not an MP, no PM resignation announced). The market is ignoring structural process, not lacking information.
2. **The compound probability is multiplicative** — even if each hurdle is plausible (40-50%), the joint probability is far below the market price. The market implicitly assumes the hurdles collapse into one correlated step.
3. **Time pressure compounds the chain** — 6.5 months remaining in 2026 means each hurdle must clear sequentially in ~2-month windows. Hurdles don't just need to clear — they need to clear in sequence before deadline.
4. **Insider-knowledge caveat** — Unlike Cases 1-4 where the vault's advantage was structural knowledge, the Burnham case has a real insider-information risk. The market at 95.3% may be pricing an unannounced PM resignation that the vault cannot know. This is why vault p_yes (0.35) is above the pure structural calculation (~0.08) — it partially incorporates insider-knowledge risk but still diverges far from the market.

**Cross-reference**: [[domains/global/concepts/sequential-hurdle-divergence/_concept]] — The formalized concept for this pattern: any forecast where multi-step AND-chains determine the outcome and the market prices as a single event. Candidate markets, veto-process markets, and multi-body approval markets are high-frequency applications.

**Forecasting rule distilled**: When a prediction market price implies a >80% probability for a multi-step sequential process (AND-chain with 3+ independent hurdles), apply the sequential hurdle chain decomposition even against high-volume markets. The Burnham case shows that Polymarket can price at 95%+ for a structural ~8-35% event when traders focus on the endpoint ("Burnham will be PM") rather than the process ("how does a mayor with no seat become PM in 6.5 months?"). Document the specific hurdles and their sequential dependency; if the market cannot explain how each step clears in time, the vault should diverge regardless of volume.

## Relationship to Other Concepts

- [[polymarket-residual-uncertainty-after-public-event]] — The inverse
  pattern (market below vault after event occurs)
- [[structural-improbability-check]] — Overlapping but distinct: the
  improbability check is about whether YES requires a cascade of
  unlikely events; divergence is about whether the market is
  systematically overpricing YES
- [[forecast-resolution-criteria-gotchas]] — The Maduro case's
  divergence depended critically on resolution criteria analysis
  (win-vote vs take-office)
- [[scotus-procedural-signals]] — Framework that powered the SCOTUS
  TikTok divergence; its value is in providing the procedural
  determinism mechanism
- [[authoritarian-electoral-facade]] — Framework that powered the
  Maduro divergence; its value is in decoupling regime claims from
  vote outcome
- [[short-horizon-momentum-check]] — A simpler pre-filter that also
  can justify divergence (short window + strong trend + no catalyst),
  but relies on trend analysis rather than procedural/structural
  knowledge

## When NOT to Use

This concept is **easily abused** as post-hoc overconfidence. Default
to Rule 9 (output within ±0.05 of PM). Only diverge when:

1. You can cite a **specific procedural or structural framework** from
   the vault (not general intuition).
2. The framework was **verifiable at cutoff** (not discovered after
   resolution).
3. You can articulate **why the market is pricing the wrong signal**
   (not just that the market disagrees with you).
4. At least two independent framework strands support the divergence.
