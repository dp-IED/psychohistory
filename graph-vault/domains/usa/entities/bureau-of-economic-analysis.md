---
type: entity
tags: [entity, economics, gdp, data-source]
kind: organization
title: "Bureau of Economic Analysis"
slug: bureau-of-economic-analysis
domain: usa
parent_entity: "[[domains/usa/entities/us-department-of-commerce]]"
date_start: 1953-12-01
pit_cutoff: 2026-05-20
---

# Bureau of Economic Analysis (BEA)

## Summary

The Bureau of Economic Analysis is an agency within the US Department of Commerce that produces the US national accounts — including GDP (Gross Domestic Product), GDI (Gross Domestic Income), personal income and outlays, and international trade statistics. The BEA is the authoritative source for the GDP data that nearly all prediction market GDP questions resolve against.

The BEA releases three GDP estimates per quarter:
1. **Advance Estimate** (~30 days after quarter end) — the most commonly used resolution metric for prediction markets
2. **Second Estimate** (~60 days after quarter end)
3. **Third Estimate** (~90 days after quarter end)

## Significance for Forecasting

The BEA's data release calendar is the single most important schedule for any question involving US GDP. Key release dates are published a year in advance and can be found at: bea.gov/schedule

The advance estimate is the modal resolution metric for GDP prediction markets. It is based on approximately 75-80% of source data and has a standard error of ~0.5-1.0pp relative to the final estimate. The advance estimate tends to be slightly less accurate in quarters with extreme volatility (late-quarter shocks, unusual seasonal patterns) but is generally reliable for mid-to-large magnitude GDP readings.

## Key Series

- **Gross Domestic Product (GDP)**: GDPA (current dollar), GDPC1 (real chained 2017 dollars), A191RL1Q225SBEA (real GDP SAAR — the standard publication metric)
- **Final Sales to Private Domestic Purchasers (FSDPDP)**: Series ID A262RX1Q020SBEA — the best measure of domestic demand, used to diagnose whether a GDP contraction is a statistical artifact (see [[domains/economics/concepts/gdp-contraction-signal-vs-noise/_concept]])
- **Personal Consumption Expenditures (PCE)**: The consumer spending component of GDP
- **Gross Private Domestic Investment**: Business investment + residential investment + inventories

## Release Schedule (2025 Relevant Dates)

- Q1 2025 Advance Estimate: April 30, 2025
- Q2 2025 Advance Estimate: July 30, 2025
- Q3 2025 Advance Estimate: October 30, 2025
- Q4 2025 Advance Estimate: January 29, 2026

## Appears In

- [[domains/economics/threads/us-macro-economic-indicators/_thread]]
- [[domains/usa/procedures/gdp-threshold-forecast]]
- [[domains/economics/concepts/gdp-contraction-signal-vs-noise/_concept]]
- [[domains/usa/concepts/gdp-tail-risk-asymmetry/_concept]]

## Wikilinks

[[domains/economics/threads/us-macro-economic-indicators/_thread]] [[domains/usa/procedures/gdp-threshold-forecast]] [[domains/economics/concepts/gdp-contraction-signal-vs-noise/_concept]] [[domains/usa/entities/us-department-of-commerce]]
