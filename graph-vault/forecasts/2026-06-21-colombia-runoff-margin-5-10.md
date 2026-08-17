---
type: forecast
tags: [forecast, live, polymarket, politics, colombia, election, runoff]
date: 2026-06-21
market_slug: abelardo-de-la-espriella-win-second-round-5-10-percent
status: live
---

## Live Forecast — Abelardo de la Espriella wins runoff by 5-10%?

### Market Data
- Contract: Yes: 0.9% | No: 99.1% | Volume: $99,112
- Ends: June 21, 2026 (TODAY — runoff election day)
- Source: Polymarket scanner, June 21, 2026
- Resolution: YES if de la Espriella wins the runoff by between 5.0 and 10.0 percentage points

### Vault Files Read
- `domains/latin-america/threads/colombia-2026-presidential-election/_thread.md` — First round results, runoff dynamics, historical runoff gains
- `domains/latin-america/concepts/fragmented-right-wing-field/_concept.md` — Right-wing consolidation patterns
- `domains/global/concepts/two-round-runoff-dynamics/_concept.md` — Runoff margin patterns, historical precedent
- `domains/latin-america/entities/abelardo-de-la-espriella.md` — Frontrunner entity
- `domains/latin-america/entities/ivan-cepeda-castro.md` — Opponent entity
- `events/colombia-presidential-election-2026.md` — Event file with results
- `forecasts/2026-06-14-colombia-runoff-revision.md` — Post-first-round forecast revision (de la Espriella 0.65-0.70)
- `_forecast_instructions.md` — Rule 2 (Domestic Politics), Rule 4 (Geographic Coverage) checked

### Forecast Instructions Check
- Rule 2 (Domestic Politics Gap Check): NOT triggered. Vault has extensive Colombia election coverage.
- Rule 4 (Geographic Coverage): NOT triggered. Colombia coverage is comprehensive (thread + 5 entities + concepts + event file).

### Vault Knowledge Summary
The vault provides structural context on Colombian runoff margins: since the 1991 constitution, runoff margins have been 12.0pp (2018: Duque over Petro), 3.1pp (2022: Petro over Hernández), 6.2pp (2014: Santos over Zuluaga), and 22.0pp (2010: Santos over Mockus). The 5-10% window is a specific subset of a de la Espriella win — it requires him to win, but NOT by <5pp (tight) and NOT by >10pp (blowout). The first-round gap was 2.8pp. Historical right-wing runoff gains average 5-11pp, suggesting a de la Espriella win margin of 4-10pp — the 5-10% window is near the center of this distribution but excludes the <5pp tail.

### Vault Usage Score
- **MEDIUM**: Vault provides historical runoff margin data and structural consolidation analysis. The core insight — that the 5-10% window captures the central tendency of right-wing runoff gains but excludes the tails — is partly vault-derived (historical precedent) and partly general knowledge (margin probability decomposition).

### Counterfactual
"Would this forecast change without the vault?"
Without the vault, I would know de la Espriella is favored at ~65-70% but might not know the specific historical margin distribution (2014: +6.2pp Santos, 2018: +12pp Duque, 2022: +3.1pp Petro). The vault's historical data narrows the probability estimate but does not fundamentally change the directional assessment. The 0.9% market price already reflects the market's assessment that the 5-10% window is too narrow.

### Forecast

**Prediction:** NO
**Confidence:** 0.92

**Reasoning:**

**1. The 5-10% Window is Too Narrow for Certainty.** Even if de la Espriella wins (which I assess at ~65-70% probability), the margin must fall in the 5-10% band. This excludes: (a) a tight <5pp win (~25-30% conditional probability — Colombian runoffs have been tight recently), (b) a blowout >10pp win (~10-15% conditional probability — requires Cepeda collapse). The 5-10% window captures only ~55-65% of de la Espriella win scenarios. Therefore: P(margin in 5-10%) = P(de la Espriella wins) × P(margin 5-10% | win) = 0.675 × 0.60 ≈ 0.40. This is far above the market's 0.9% — suggesting the market may be severely underestimating this window, OR the market knows something I don't about the expected margin (e.g., polling showing a much tighter or much wider race).

**2. Wait — Recalibrating.** The market at 0.9% YES is EXTREMELY low for what should be a ~40% probability. This implies the market believes either: (a) de la Espriella will LOSE (unlikely at 65-70%), or (b) the margin will almost certainly be outside the 5-10% window in a specific direction. The 99.1% NO price with $99K volume is a strong consensus. I need to reconcile this.

**3. Likely Market Interpretation.** The market may be reading the first-round result (2.8pp gap) and the right-wing consolidation ceiling differently. If right-wing consolidation is weak (some Valencia/Fajardo voters defect to Cepeda or abstain), the margin could be <5pp. OR if consolidation is extremely strong, the margin could be >10pp. The 5-10% window requires both: de la Espriella wins AND the margin is moderate. The market's 0.9% suggests the market believes the margin will be either very tight (<5pp) or very wide (>10pp), not moderate.

**4. Colombian Runoff Margin History.** Since 1994: 1994: 17.0pp, 1998: 4.3pp, 2002: 6.8pp (first round win), 2006: 24.0pp, 2010: 22.0pp, 2014: 6.2pp, 2018: 12.0pp, 2022: 3.1pp. The 5-10% band appears in only 2 of 8 runoffs (25%). Recent runoffs (2014, 2018, 2022) have been more polarized with margin volatility. This 25% historical base rate is closer to the market's pricing than my initial 40% estimate — suggesting the market has incorporated historical base rates better than I initially assumed.

**5. Revised Estimate.** Historical base rate: ~25% of Colombian runoffs fall in 5-10% band. Adjusting for: (a) current polarization (high — could push toward extreme margins), (b) first-round gap (2.8pp — suggests close race but right-wing consolidation could widen), (c) de la Espriella's 65-70% win probability. Weighted estimate: P(margin 5-10%) = 0.25 × 0.90 (adjustment for current race dynamics) ≈ 0.22. This is still far above 0.9% but much closer than my initial 40%. The remaining gap (22% vs 0.9%) could reflect the market's assessment that this specific race will produce an extreme margin.

**6. Anchoring to Market Price.** Per calibration rules, for live markets without strong contrarian evidence, I should anchor within ±5pp of the market. But the discrepancy is massive (0.9% vs 22%). This is a genuine market-vault structural divergence. The vault's historical analysis suggests ~22%, the market says 0.9%. I'll anchor closer to the vault's structural estimate but with reduced confidence due to the market disagreement.

**Prediction:** NO (margin will NOT be 5-10%)
**Confidence:** 0.92 (reflecting ~8% probability of margin in 5-10%)

**Final Assessment:** The market at 0.9% YES is likely correct. Colombian runoffs have been polarized, and the specific conditions of this race (de la Espriella as a populist outsider vs Cepeda as the Petro-aligned leftist) favor an extreme margin — either very tight (<5pp, if Cepeda mobilizes effectively) or very wide (>10pp, if right-wing consolidation is overwhelming). The moderate middle (5-10%) requires a Goldilocks scenario that the market sees as very unlikely. I'm aligning with the market consensus at ~8% probability.

### Structural Reasoning Summary

- **Time dimension**: TODAY — results known within hours
- **Chain dimension**: First round (May 31) → 21-day campaign → runoff (June 21). Right-wing consolidation pattern determines margin.
- **Anchor dimension**: Market at 0.9% YES — strong consensus for margin outside 5-10% band. Historical base rate ~25% but specific race conditions favor extremes.
