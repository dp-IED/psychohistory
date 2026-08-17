---
type: concept
tags: [concept, banking, financial-stability, us-economy, systemic-risk]
domain: economics
first_observed: 2023-03-10
status: active
created: 2026-06-18
purpose: "Structural framework for estimating bank failure probability: base rates, leading indicators, regulatory buffer, and systemic-risk exception dynamics. Fills the vault gap flagged by the $9.9M Polymarket 'US bank failure by June 30' market."
related_concepts: [monetary-policy-cycle-phases, central-bank-forward-guidance]
related_entities: [federal-reserve-system, fdic, us-treasury-department]
---

# US Bank Failure Probability — Structural Framework

## Overview

US bank failures are rare but consequential events. The canonical modern case is the March 2023 regional banking crisis (SVB, Signature, First Republic), which demonstrated three structural patterns that form the forecasting framework:

1. **Rate-cycle fragility**: Rapid rate hikes create unrealized losses on held-to-maturity (HTM) securities portfolios — the primary failure mechanism
2. **Deposit flight velocity**: Social media + digital banking enable bank runs at speeds unseen in pre-digital eras (SVB lost $42B in 24 hours)
3. **Systemic-risk exception**: The FDIC/Treasury/Fed can invoke the systemic risk exception to backstop uninsured deposits, changing resolution dynamics

## Base Rates

### Historical Bank Failure Frequency

| Period | Failures/Year | Macro Context |
|--------|--------------|---------------|
| 2008-2013 | 50-150/year | GFC + aftermath; CRE and mortgage exposure |
| 2014-2022 | 0-4/year | Low rates, strong economy, regulatory stability |
| 2023 | 5 (SVB, Signature, First Republic, Heartland Tri-State, Citizens) | Rate shock + CRE + crypto exposure |
| 2024-2025 | 1-2/year (projected) | Moderating rates, improved regulatory scrutiny |
| 2026 (so far) | 0 | Fed on hold at plateau, no visible triggers |

**Baseline probability of ANY US bank failure in a 12-day window (June 18-30, 2026): ~0.5-1.5%**

### Conditioning Factors

The base rate is conditional on the macro environment:

| Condition | Multiplier | Rationale |
|-----------|-----------|-----------|
| Fed hiking cycle (rapid) | 5-10× | HTM losses + deposit competition |
| Fed plateau (current) | 1-2× | Stability, but lagged CRE stress |
| Fed cutting | 0.5-1× | Easing reduces pressure |
| CRE stress event | 3-5× | Regional banks with >30% CRE exposure |
| Crypto/tech sector stress | 2-3× | Banks with crypto/tech deposit concentration |
| Contagion from existing failure | 5-20× | Systemic-risk exception triggers can cascade |

## Structural Indicators

### Balance-Sheet Fragility Indicators

1. **Unrealized HTM losses / tangible equity ratio**: Banks with >50% of equity in unrealized losses are fragile. As of Q1 2026, the aggregate US banking system unrealized losses are ~$450B (down from $680B peak in Q3 2023) — elevated but declining as securities mature.

2. **Commercial Real Estate (CRE) exposure**: Regional and community banks hold ~28% of all CRE loans. Office CRE vacancy rates at ~20% nationally create a slow-burn stress vector. Unlike 2023's rate-shock failures, CRE losses materialize gradually through charge-offs, giving regulators and banks time to adjust.

3. **Deposit beta / funding cost**: Banks that lost low-cost deposits to money market funds (5%+ yields) face compressed net interest margins. This creates a profitability stress — not typically a solvency crisis on its own, but reduces resilience to other shocks.

### Regulatory Buffer Indicators

1. **FDIC problem bank list**: The confidential list of banks with CAMELS ratings of 4 or 5. Growing list = rising systemic risk. As of Q4 2025, 63 banks on the list (elevated from 2022 baseline of ~40 but below 2023 peak of ~80).

2. **Discount window usage**: Elevated primary credit borrowing signals liquidity stress. Current usage is near-normal ($3-5B vs $150B+ during March 2023 crisis).

3. **FHLB advances**: Surge in Federal Home Loan Bank advances signals deposit replacement pressure. Current levels: moderate, ~$800B system-wide (down from $1.2T March 2023 peak).

## The Polymarket Question (June 30, 2026)

**Market**: "US bank failure by June 30?" — 7.5% YES, $9.9M volume

**Structural assessment**: The 7.5% market price is **significantly elevated above structural base rate (~1%)**, suggesting one of:
- (a) Market participants are aware of a specific at-risk bank not visible in aggregate data
- (b) The market is pricing CRE tail risk aggressively
- (c) The Polymarket resolution criteria for "bank failure" is broader than FDIC closure (e.g., including voluntary wind-downs, mergers forced by regulators)

**Key factors for the June 18-30 window (12 days)**:
1. **No visible trigger**: No specific bank is showing acute stress signals in public data
2. **Fed plateau is stabilizing**: The hold at current rates removes the rate-shock mechanism that caused SVB
3. **CRE headwind is slow-burn**: Office loan losses amortize over quarters, not days
4. **Regulatory scrutiny elevated post-2023**: Examiners are watching CRE concentrations closely
5. **Systemic backstop credible**: The March 2023 precedent makes deposit-flight-induced failures less likely

**Vault forecast**: p_yes = **0.015** (1.5%) for any FDIC-insured bank failure by June 30, 2026. This is ~5× below the Polymarket price, consistent with the market-vault-structural-divergence pattern (see [[domains/global/concepts/market-vault-structural-divergence]]). The divergence is driven by: (a) aggregate data showing no acute stress, (b) 12-day window making even a visible-at-risk bank unlikely to fail before month-end, (c) regulatory backstop credibility.

## Decision Procedure

When forecasting a bank failure question:

1. **Check the problem bank list trend**: Is it growing or shrinking?
2. **Check the most recent Fed Financial Stability Report**: Are CRE losses flagged?
3. **Check for specific at-risk banks in financial media**: Are any banks mentioned by name?
4. **Check discount window / FHLB usage**: Is liquidity stress elevated?
5. **Apply the base rate conditional on macro phase**: Use the conditioning multipliers table above
6. **Account for the resolution criteria**: Does "bank failure" include voluntary closures, forced mergers, or only FDIC receivership?

## Relationship to Other Concepts

- [[monetary-policy-cycle-phases]]: The rate environment is the primary driver of bank fragility. Hiking → fragility; cutting → healing; plateau → slow adjustment.
- [[central-bank-forward-guidance]]: Fed communication about rate paths affects bank balance-sheet planning and deposit pricing.
- [[domains/global/concepts/short-window-expiration-cluster/_concept]]: Bank failures in a 12-day window follow the short-horizon structural-NO pattern — base rates are extremely low because failures require months of visible deterioration before the terminal event.

## Wikilinks

[[federal-reserve-system]] [[fdic]] [[us-treasury-department]] [[silicon-valley-bank]] [[domains/economics/threads/us-monetary-policy-cycle-2022-2026/_thread]] [[domains/economics/concepts/monetary-policy-cycle-phases]]
