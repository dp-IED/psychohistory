---
type: procedure
tags: [procedure]
title: "Oil Price Forecast Procedure"
slug: oil-price-forecast
---

# Oil Price Forecast Procedure

## When to Use

Any question asking whether crude oil (Brent or WTI) will reach a specific price level on a specific date.

## Steps

### Step 1: Identify the benchmark
- Does the question say "a barrel of crude oil," "Brent," "WTI," or specify a benchmark?
- Default: "crude oil" = Brent (global benchmark). "West Texas Intermediate" = WTI.
- Check entity: [[domains/economics/entities/brent-crude-oil]] or [[domains/economics/entities/wti-crude-oil]]

### Step 2: Map the price context
- Check the question date against known oil price events:
  - Is there a geopolitical shock within the 8 weeks before the target date?
  - What was the price 1 month, 1 week, and 1 day before the target date?
  - What was the peak price during any recent spike?

### Step 3: Assess the spike-reversion pattern
- If a geopolitical shock occurred recently (within 4 weeks):
  - What phase is the market in: Fear Spike (Phase 2), Reality Calibration (Phase 3), or New Equilibrium (Phase 4)?
  - Days since peak vs target date: if target date is <2 weeks after peak, calibration is still active
  - Apply [[domains/economics/concepts/geopolitical-commodity-spike-reversion]] framework
  - Default: if the target price is near the peak level from a spike within the past 2-4 weeks, forecast NO (the market will have partially reverted)

### Step 4: Check policy responses
- Are SPR releases active, announced, or telegraphed?
- Is the IEA coordinating a release?
- Are alternative buyers absorbing displaced supply?
- Are there price caps or sanctions on the affected producer?
- Check [[domains/economics/concepts/strategic-petroleum-reserve]]

### Step 5: Check physical supply
- Is actual production disrupted? (Pipeline damage, port closure, mine shutdown)
- Or is the disruption in routing only? (Sanctions redirecting flows but not reducing total supply)
- If physical supply is NOT disrupted and the disruption is routing/financial, the reversion is faster and more complete

### Step 6: Check demand-side factors
- Is the target price level high enough to trigger demand destruction (~$120+ Brent)?
- Are there recession signals (inverted yield curve, rate hikes, PMI contraction)?
- Is China (largest oil importer) experiencing lockdowns or economic weakness?

### Step 7: Write reasoned forecast
- State the baseline price before the shock
- Describe the calibration timeline
- Note the policy response and its anticipated effect
- Conclude with probability estimate

## Validated By

| Date | Question | Framework Suggestion | Actual | Correct? |
|------|----------|---------------------|--------|----------|
| 2022-03-15 | Crude oil $115? | NO — spike peak was Mar 8 at ~$128, 7 days of calibration, SPR telegraphed | ~$99, NO | YES |
