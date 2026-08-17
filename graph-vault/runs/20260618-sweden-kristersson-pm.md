---
type: forecast
market: "Will Ulf Kristersson be the next Prime Minister of Sweden?"
slug: will-ulf-kristersson-be-the-next-prime-minister-of-sweden
source: polymarket
cutoff: 2026-06-18
end_date: 2026-09-13
condition_id: "0x8a42bb4cb9b9f157b539611f6a8c122388f652570cfc3c07d538e7df2bb78894"
p_yes: 0.22
market_yes: 0.220
market_no: 0.780
volume: 99581.91
domain: politics
tags: [sweden, election, kristersson, scandinavia, eu]
vault_usage_score: MEDIUM
---

# Live Forecast — Will Ulf Kristersson be the next Prime Minister of Sweden?

## Market Data
- Contract: Yes: 22.0% | No: 78.0% | Volume: $99,582
- Ends: 2026-09-13 (Swedish general election)
- Source: Polymarket — slug `will-ulf-kristersson-be-the-next-prime-minister-of-sweden`

## Vault Files Read
- `domains/europe/entities/sweden.md` — Entity file with structural position, political landscape, key actors, and forecasting notes. Notes explicitly state right-bloc is trailing in polls and SD kingmaker dynamics create coalition fragility. The market entry in the file says 17.5% YES ($97K vol) — this has since moved to 22.0%.
- `_forecast_instructions.md` — Rules 1-16 checked. Rule 4 (Geographic Coverage Gap) triggered but vault HAS Sweden coverage. Rule 2 (Domestic Politics) not triggered (non-US).
- `procedures/structural-reasoning.md` — Applied: Time dimension (87 days to election), Chain dimension (single-step ballot, but coalition formation is multi-step), Anchor dimension (Polymarket price at 22% is the primary calibration signal).

## Forecast Instructions Check
- Rule 1 (Central Bank): Not triggered — this is an electoral question.
- Rule 2 (Domestic Politics - US): Not triggered — Sweden is not US domestic politics.
- Rule 4 (Geographic Coverage Gap): Triggered but satisfied — vault has `domains/europe/entities/sweden.md` with structural coverage of Kristersson, polling, coalition dynamics, and election calendar. Gap noted: no entity stubs for individual candidates (Kristersson, Andersson, Åkesson) and no Sweden 2026 election thread.
- Rule 16 (Mechanism Calibration): No calibration table for Swedish proportional-representation coalition dynamics — fall back to standard forecast methodology.

## Vault Knowledge Summary
The vault's Sweden entity file provides structural context on the Swedish political system (proportional representation, 4% threshold, blocs, SD kingmaker role) and explicitly notes right-bloc polling weakness. This reinforces the market's 78% NO price. Without the vault, the forecaster might default to incumbency-advantage heuristics which would overestimate Kristersson's chances. The vault correctly identifies that Kristersson's 22% reflects (a) right-bloc trailing in polls and (b) SD coalition fragility — two structural factors that general knowledge alone might miss or underweight.

## Vault Usage Score: MEDIUM
The vault provides useful contextual background (Swedish electoral system structure, current polling dynamics, SD kingmaker fragility) that enriches the forecast. However, the core signal (Kristersson's party bloc is trailing) is accessible from general knowledge. The vault adds structural precision — proportional representation mechanics, 4% threshold effects, and SD dynamics — but a forecast without the vault would still reach a similar conclusion (NO, ~20-25% YES). Missing: individual entity stubs for candidates and a dedicated election thread would elevate this to HIGH usage.

## Counterfactual
"Would this forecast change without the vault?"
Without the vault: the forecaster would still know Swedish right-bloc is polling behind and would likely assign 25-30% YES (higher due to incumbency bias). The vault's specific structural analysis (PR with 4% threshold fragmenting seats, SD kingmaker creating coalition fragility) anchors the estimate closer to the market's 22% — a modest but real contribution. The forecast would not fundamentally change, but confidence intervals would be wider.

## Structural Reasoning

### Time
87 days to the September 13 election. Swedish elections have fixed 4-year terms — no snap election risk. Probability is not flat: it concentrates around polling trends and campaign dynamics. Time elapsed without a polling shift toward the right bloc is mild negative evidence for Kristersson. The summer campaign period (June-August) typically sees lower voter engagement, with most movement in the final 3-4 weeks.

### Chain
Two-step: (1) Election produces seat distribution, (2) Coalition negotiations produce a PM. Step 1 is the bottleneck — if the right bloc wins a majority (or plurality with SD support), Kristersson continues. If the left bloc wins, Andersson becomes PM. Coalition negotiation (Step 2) adds modest uncertainty but Swedish coalition formation is typically predictable based on seat math.

### Anchor
Polymarket price at 22% YES. This aligns with polling data showing left bloc ahead. No reason to diverge significantly — the market is pricing the structural dynamics correctly. Historical base rate: Swedish PMs have served an average of ~6 years since 1970, but the current coalition is fragile and the election is competitive.

## Forecast

**Prediction:** NO (Kristersson will NOT be the next PM)
**Confidence:** 0.78 (aligned with market)
**Reasoning:**

1. **Polling deficit**: Current polls consistently show the left bloc (S+V+MP+C) ahead of the right bloc (M+KD+L+SD). While within margin of error, the persistent deficit means Kristersson starts from behind.

2. **Coalition fragility**: The right-bloc government depends on Sweden Democrats (SD) confidence-and-supply. SD's far-right positioning creates friction with the Liberal Party and Christian Democrats. These internal tensions reduce the coalition's campaign effectiveness and could depress right-bloc turnout.

3. **SD kingmaker paradox**: SD is polling as the largest right-bloc party (~20%) but is toxic to the centre-right parties in the coalition. If SD gains seats at the expense of Moderates, the right bloc's parliamentary math becomes harder, not easier — Kristersson would need SD support while maintaining Liberal and Christian Democrat participation.

4. **Incumbency without advantage**: Kristersson's government has faced headwinds on crime (gang violence continues despite tough rhetoric), energy prices, and NATO integration costs. The traditional incumbency advantage in Swedish politics is weak — voters frequently rotate governments.

5. **Proportional representation mechanics**: The 4% threshold means small parties near the threshold (Liberals, Christian Democrats, Greens) are at risk of falling out of parliament. If one right-bloc party falls below 4%, the bloc loses those seats entirely, making a parliamentary majority nearly impossible even with SD support.

6. **Market alignment**: The 22% YES price reflects these structural dynamics accurately. The residual 22% accounts for: (a) polling error systematically underestimating right-wing parties (common in European elections), (b) a campaign surge, (c) left-bloc coalition negotiation failure post-election. These are real but collectively unlikely scenarios.

**Vault gap note (RESOLVED)**: The vault gap noted at forecast time (entity stubs for Kristersson, Andersson, Åkesson; Sweden election thread) was filled in Pass 42 (June 18) and Pass 43 (June 19). Entities: [[domains/europe/entities/ulf-kristersson]], [[domains/europe/entities/magdalena-andersson]], [[domains/europe/entities/jimmie-akesson]]. Thread: [[domains/europe/threads/sweden-2026-election/_thread]]. All now cross-referenced from the run index.
