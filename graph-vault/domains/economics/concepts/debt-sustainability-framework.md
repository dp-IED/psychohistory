---
type: concept
tags: [economics, fiscal-policy, macro, debt, forecasting]
domain: economics
status: active
created: 2026-05-21
updated: 2026-05-21
pit_cutoff: 2026-12-31
related_concepts:
  - monetary-policy-cycle-phases
  - yield-curve-dynamics
  - central-bank-forward-guidance
  - debt-ceiling-mechanics
related_threads:
  - us-debt-ceiling-crises
  - us-macro-economic-indicators
  - us-monetary-policy-cycle-2022-2026
---

# Debt Sustainability Framework

## Core Definition

Debt sustainability assesses whether a sovereign can service its debt without an implausibly large adjustment to its primary balance. The framework evaluates whether current fiscal trajectories are consistent with stable or declining debt-to-GDP ratios under plausible macroeconomic scenarios.

## Key Variables

1. **Debt-to-GDP ratio** (D/Y) — the stock metric. Level matters less than trajectory.
2. **Primary balance** (PB) — fiscal balance excluding interest payments. The operational lever.
3. **Interest rate-growth differential** (r − g) — the single most important determinant of debt dynamics. When r < g, the economy "grows out of" debt naturally. When r > g, debt compounds faster than output.
4. **Marginal borrowing rate** — the rate at which additional debt is issued, which may diverge from average rate.
5. **Currency composition** — foreign-currency-denominated debt creates exchange rate vulnerability absent in domestic-currency debt.

## Debt Dynamics Equation

The change in debt-to-GDP ratio is governed by:

```
Δ(D/Y) = (r − g) × (D/Y)₋₁ − PB + Stock-Flow Adjustment
```

Where:
- The `(r − g)` term is the snowball effect (automatic debt dynamics)
- The `-PB` term is the fiscal effort channel
- Stock-flow adjustments capture below-the-line items (asset sales, valuation effects, statistical discrepancies)

## Forecasting Application

### 1. Baseline Trajectory Assessment

For a given sovereign, estimate the 5-year debt trajectory under a baseline scenario:

| Variable | Source | Signal Strength |
|----------|--------|-----------------|
| r (effective interest rate) | Treasury yield curve, auction data, CBO projections | High |
| g (nominal GDP growth) | CBO, IMF WEO, consensus forecasts | High |
| Primary balance | CBO baseline, administration budget, legislative proposals | Medium-High |
| Stock-flow adjustments | Historical average (typically 0.5-1.5% of GDP for US) | Low |

### 2. Threshold Analysis

**Advanced economies (own currency):**
- D/Y < 60%: Low concern — wide policy space
- D/Y 60-100%: Monitor trajectory — direction matters more than level
- D/Y 100-150%: Requires primary surplus or r < g condition to stabilize
- D/Y > 150%: High vulnerability to interest rate shocks; fiscal dominance risk becomes material

**Emerging markets (external debt):**
- External debt/GDP > 50%: Vulnerability threshold (IMF early warning)
- External debt service/exports > 20%: Liquidity stress threshold
- FX reserves/short-term external debt < 1.0: Rollover risk acute

**US-Specific Thresholds (as of May 2026):**
- US debt-to-GDP: ~123% (2025), projected ~130% by 2030 (CBO baseline)
- r − g differential: Currently favorable (~1-2% spread) but sensitive to Fed rate path
- Primary deficit: ~5-6% of GDP (2025), no consolidation path legislated
- **Assessment**: Sustainable in the medium term due to reserve currency privilege and r ≈ g condition, but trajectory is deteriorating. The US is not at immediate risk of a fiscal crisis, but its fiscal space is diminishing — each 100bp increase in the effective interest rate adds ~1.2% of GDP to interest costs.

### 3. Trigger Events for Fiscal Sustainability Crises

Fiscal crises typically require a **conjunction** of:
1. **Deteriorating r − g** (rising rates + slowing growth)
2. **Political inability to adjust primary balance** (divided government, fiscal rules broken)
3. **Loss of market confidence** (auction failures, widening spreads, CDS repricing)
4. **External shock** (contagion, commodity price collapse, banking crisis)

Single-factor deterioration (e.g., rising rates with strong growth) rarely triggers a crisis. The conjunctions are what matter.

### 4. Probability Calibration Table

| Scenario | Probability (5-year horizon) | Indicators to Watch |
|----------|------------------------------|---------------------|
| Debt trajectory improves (PB consolidation + favorable r-g) | 10-15% | Bipartisan fiscal commission, entitlement reform, strong growth |
| Debt trajectory stable (r ≈ g, no consolidation) | 40-50% | Steady rates, moderate growth, political gridlock preventing both consolidation AND expansion |
| Debt trajectory deteriorates gradually (r > g, deficit persists) | 25-35% | Rising rates, recession, unfunded tax cuts or spending, no offsetting consolidation |
| Fiscal crisis (auction failure, forced austerity, monetization) | 5-10% | Auction tailing, CDS widening, dollar decline, loss of reserve confidence, political dysfunction |

### 5. Cross-Domain Linkages

**Monetary Policy Channel:**
- Higher debt → fiscal dominance risk → constrains Fed independence → higher inflation expectations → higher term premium → higher borrowing costs → self-reinforcing loop.
- This is the "fiscal dominance spiral" — the most dangerous debt sustainability dynamic because it removes the independent central bank as a stabilization mechanism.

**Currency Channel:**
- Reserve currency status provides an "exorbitant privilege" — the US can borrow in its own currency at rates below what its fundamentals would otherwise justify.
- A loss of reserve currency confidence would raise US borrowing costs by an estimated 50-150bp, adding $150-450B/year to interest costs at current debt levels.
- Probability of rapid reserve currency displacement in the next 5 years: <5% (no viable alternative, network effects, institutional depth).

**Political Economy Channel:**
- Entitlement spending (Social Security, Medicare, Medicaid) is the primary long-run fiscal driver, not discretionary spending.
- Trust fund depletion dates: Social Security OASI 2033, Medicare HI 2036 (per 2025 Trustees Report).
- Political constraint: both parties have campaigned against entitlement reform; the electoral penalty for touching these programs is high.

## Canonical Cases

### Japan (High Debt, Low Rates — Sustainable)
- D/Y: ~260% of GDP (highest among advanced economies)
- r − g: Consistently negative (near-zero rates, modest nominal growth)
- Domestic ownership: >90% of JGBs held domestically (BoJ, banks, pension funds)
- **Lesson**: High D/Y alone does not imply crisis. The r − g differential and ownership structure matter more than the level.

### Greece (Moderate Debt, Sudden Stop — Crisis)
- D/Y: ~130% (2009) — lower than current US
- r − g: Turned sharply positive (rates spiked + GDP collapsed)
- Foreign ownership: ~70% of debt held externally
- No independent monetary policy (Eurozone member)
- **Lesson**: Currency sovereignty and the r − g differential dominate the debt level as crisis predictors.

### UK Gilt Crisis (September 2022)
- D/Y: ~100% — moderate by advanced economy standards
- Unfunded tax cuts (Truss/Kwarteng "mini-budget") triggered a confidence shock
- Pension fund LDI collateral spiral amplified the selloff
- BoE forced to reverse course (emergency gilt purchases) despite inflation-fighting mandate
- **Lesson**: Market confidence can change rapidly. The UK crisis was triggered by fiscal policy signaling, not debt level. Duration: ~3 weeks from announcement to policy reversal.

## Forecasting Checklist

For any question about sovereign debt sustainability, fiscal crisis risk, or bond market stress:

1. **Calculate r − g**: Current effective borrowing rate vs nominal GDP growth.
2. **Project 5-year trajectory**: Under CBO/IMF/OECD baseline, no new policy.
3. **Identify political constraints on adjustment**: Can the primary balance be improved without electoral penalty?
4. **Assess ownership structure**: Who holds the debt? Domestic vs foreign, price-sensitive vs captive.
5. **Check for shock vulnerabilities**: Currency mismatch, rollover concentration, contingent liabilities (banking sector, state-owned enterprises).
6. **Monitor market signals**: Auction bid-to-cover ratios, CDS spreads, yield curve shape, breakeven inflation.
7. **Calibrate probability**: Use the conjunction model — single-factor stress rarely crystallizes into crisis.

## Relationship to Existing Vault Concepts

- [[monetary-policy-cycle-phases]] — Determines the r (policy rate) input
- [[yield-curve-dynamics]] — Term premium affects marginal borrowing cost
- [[central-bank-forward-guidance]] — Forward guidance shapes the expected r path
- [[debt-ceiling-mechanics]] — Procedural dimension of US fiscal governance (distinct from sustainability)
- [[policy-expectation-without-delivery]] — Pattern applicable to fiscal consolidation promises that don't materialize

## Validation History

| Question | Outcome | Date | Notes |
|----------|---------|------|-------|
| (new concept — no validation data yet) | — | 2026-05-21 | This framework has not yet been tested against a resolved forecasting question. The next US fiscal/debt question should apply it. |

## Key Forecasting Principle

**Debt sustainability crises are politically triggered, not mechanically triggered.** The math of debt dynamics can deteriorate for years without crisis — it takes a political event (unfunded fiscal expansion, government collapse, loss of market confidence) to crystallize. Forecast the political economy, not just the debt equation.
