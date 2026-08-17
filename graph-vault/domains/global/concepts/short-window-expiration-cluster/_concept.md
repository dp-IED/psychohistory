---
type: concept
tags: [concept, forecasting, methodology, structural]
status: active
updated: 2026-06-21
pit_cutoff: 2026-06-21
---

# Short-Window Expiration Cluster

**Pattern**: Multiple independent forecasting questions expire on the same date, creating a cluster of structurally similar NO predictions driven by timeline arithmetic rather than domain-specific knowledge.

## The June 30, 2026 Canonical Case

On June 16, 2026 (14 days before expiry), five vault-registered forecasts shared the June 30 deadline. By June 19, a sixth joined: Iran enrichment agreement ($9.9M, vault p_yes=0.09). The cluster now spans 6 domains:

| Forecast | p_yes | Market Volume | Primary NO Mechanism |
|----------|-------|---------------|---------------------|
| [[runs/20260615-israel-lebanon-withdrawal\|Israel withdraws from Lebanon]] | 0.07 | $994K | 14 days << 4-8 week withdrawal cycle |
| [[runs/20260616-diaz-canel-cuba\|Díaz-Canel out as Cuba leader]] | 0.07 | $992K | Base rate ~0.2% for 14-day window; no mechanism |
| [[runs/20260615-kharg-island\|Kharg Island not under Iran control]] | 0.01 | $995K | Military seizure impossible under ceasefire; IRGC revolt near-zero |
| [[runs/20260522-raul-castro-us-custody\|Raúl Castro US custody]] | 0.005 | $99K | Zero-mechanism structural impossibility |
| [[runs/20260616-trump-al-sharaa-meeting\|Trump speaks to al-Sharaa]] | 0.15 | $99K | 14-day window + terrorist designation barrier |
| [[runs/20260619-iran-enrichment-june30\|Iran ends enrichment by June 30]] | 0.09 | $9.9M | 11-day window + enrichment-as-right doctrine + post-Khamenei vacuum |

All six converged on NO (structural or near-structural) despite different domains (MENA, Latin America, US foreign policy, Iran energy, nuclear nonproliferation). The common thread: **short remaining window relative to required action timeline** created timeline-arithmetic constraints that dominated all other factors.

## Pattern Mechanics

1. **Discovery**: Multiple markets share a common expiry (often end-of-quarter or end-of-month)
2. **Window assessment**: Calculate remaining days vs. minimum time needed for the action
3. **Structural override**: When window < required timeline, timeline arithmetic dominates even without deep domain knowledge
4. **Calibration check**: Market consensus often already reflects the window constraint, but confidence can be high regardless

## When to Apply

Use this concept when:
- Multiple forecasts share an expiry date cluster (typically end of quarter)
- At least one has a short remaining window (<30 days)
- The action requires institutional/government decision-making (not individual action)
- Pre-announcement signals are absent

## Contrast Patterns

| Pattern | When to Use |
|---------|-------------|
| [[domains/global/concepts/short-horizon-momentum-check/_concept]] | Single forecast, short horizon, trend/catalyst analysis needed |
| **Short-window expiration cluster** | Multiple forecasts, same expiry, timeline arithmetic dominates |
| [[domains/global/concepts/structural-improbability-check/_concept]] | Zero-mechanism analysis for near-impossible events |

## Post-Resolution Analysis Framework (To Be Filled After June 30, 2026)

### Pre-Resolution Status (June 21, 2026 — T-9 Days)

**Summary**: All 6 forecasts remain structurally NO. No mechanism has emerged for any event. Colombia runoff resolves today — freeing vault attention for the June 30 cluster watch.

| Forecast | p_yes | Market Vol | Market Moves Since Jun 16 | Mechanism Status |
|----------|-------|-----------|--------------------------|-----------------|
| [[runs/20260615-israel-lebanon-withdrawal\|Israel withdraws from Lebanon]] | 0.07 | $994K | 94.5% NO holds steady | No breakthrough. Israel maintains security rationale. |
| [[runs/20260616-diaz-canel-cuba\|Díaz-Canel out as Cuba leader]] | 0.07 | $992K | 92.5% NO, stable | No health/coup/resignation indicators. 9-day base rate ~0.13%. |
| [[runs/20260615-kharg-island\|Kharg Island not under Iran control]] | 0.01 | $995K | 99% NO, near-structural | Ceasefire holds. Military seizure functionally impossible. |
| [[runs/20260522-raul-castro-us-custody\|Raúl Castro US custody]] | 0.005 | $99K | 99.5% NO, structural | Zero-mechanism structural impossibility confirmed. 5 mechanisms all dead. |
| [[runs/20260616-trump-al-sharaa-meeting\|Trump speaks to al-Sharaa]] | 0.15 | $99K | NO↑ from 67.5% to ~72% | Market converging toward vault 85% NO. HTS designation barrier + bandwidth constraint. |
| [[runs/20260619-iran-enrichment-june30\|Iran ends enrichment by June 30]] | 0.09 | $9.9M | 95.5% NO, stable | Post-Khamenei political vacuum prevents enrichment renunciation. 11-day window insufficient. |

**Key observation**: The June 30 cluster is passing its most important pre-resolution test: **no false positive mechanism has appeared in any domain**. If a mechanism were going to emerge, the 9-day window should show early signals. Their absence strengthens the structural-NO thesis across all 6 forecasts.

**Market movement signal**: Trump-al-Sharaa has moved ~5pp toward NO since Jun 16 (67.5%→72% NO). Iran enrichment holds steady at 95.5% NO despite $9.9M volume — a high-liquidity consensus. The enrichment market's addition to the cluster adds a nuclear-policy dimension that was previously unrepresented.

**Resolution countdown**: 
- June 21 (T-9): Colombia runoff resolves TODAY — frees vault attention
- June 24 (T-6): Last realistic window for pre-weekend announcement
- June 28 (T-2): Last chance for weekend news cycle to affect markets
- June 30 (Mon): All 6 resolve simultaneously

**Pooled calibration observation**: The 6 forecasts collectively assign an average probability of ~0.064 to any event resolving YES. The market-implied collective probability is ~0.055-0.065 — consistent but slightly higher for Trump-al-Sharaa and Iran enrichment. If all 6 resolve NO, the pooled Brier ≈ 0.064 — a validation of timeline-arithmetic methodology at scale.

The six June 30 forecasts form a pooled calibration cohort. Their collective resolution provides a unique calibration signal: they share no common catalyst (covering MENA, Latin America, US foreign policy, Iran energy, AND nuclear nonproliferation) but all converged on structural-NO via timeline arithmetic.

| Forecast | p_yes | Resolution | Brier | Primary NO Mechanism |
|----------|-------|-----------|-------|---------------------|
| [[runs/20260615-israel-lebanon-withdrawal\|Israel withdraws from Lebanon]] | 0.07 | [TBD Jun 30] | [TBD] | 14 days << 4-8 week withdrawal cycle |
| [[runs/20260616-diaz-canel-cuba\|Díaz-Canel out as Cuba leader]] | 0.07 | [TBD Jun 30] | [TBD] | Base rate ~0.2% for 14-day window |
| [[runs/20260615-kharg-island\|Kharg Island not under Iran control]] | 0.01 | [TBD Jun 30] | [TBD] | Ceasefire makes military seizure impossible |
| [[runs/20260522-raul-castro-us-custody\|Raúl Castro US custody]] | 0.005 | [TBD Jun 30] | [TBD] | Zero-mechanism structural impossibility |
| [[runs/20260616-trump-al-sharaa-meeting\|Trump speaks to al-Sharaa]] | 0.15 | [TBD Jun 30] | [TBD] | 14-day window + terrorist designation barrier |
| [[runs/20260619-iran-enrichment-june30\|Iran ends enrichment by June 30]] | 0.09 | [TBD Jun 30] | [TBD] | Post-Khamenei political vacuum + enrichment doctrine |

**Calibration cohort utility**: If all 6 resolve NO, pooled Brier = avg(p_yes²) ≈ avg(0.0049, 0.0049, 0.0001, 0.000025, 0.0225, 0.0081) ≈ 0.0068 — near-perfect calibration for structural-NO predictions. The enrichment forecast (0.09) adds the highest variance member: if it resolves YES, Brier = 0.8281, a significant calibration failure. If it resolves NO, Brier = 0.0081 — well-calibrated. The cluster provides a high-signal calibration test of the short-window structural-NO methodology across 6 independent domains.

**To be completed after June 30, 2026:**
1. Fill resolution column for each forecast
2. Calculate individual and pooled Brier scores
3. Assess whether the timeline-arithmetic mechanism was validated or refuted
4. If any forecast resolved YES: analyze what mechanism the concept missed
5. Update the concept with post-resolution calibration data
6. Archive the resolved cluster data to [[meta/changelog/_changelog-june-30-cluster-resolution]]

## Forecasting Rule
When ≥3 vault-managed forecasts share a common expiry date within 30 days, and all require multi-step institutional action, the cluster warrants a coordination check: are the forecasts structurally independent or linked via a common catalyst? Independence strengthens confidence in each individual prediction; a common catalyst (e.g., a peace deal unlocking multiple expiries) could create correlated tail risk.

## Wikilinks
- [[runs/20260615-israel-lebanon-withdrawal]]
- [[runs/20260616-diaz-canel-cuba]]
- [[runs/20260615-kharg-island]]
- [[runs/20260522-raul-castro-us-custody]]
- [[runs/20260616-trump-al-sharaa-meeting]]
- [[runs/20260619-iran-enrichment-june30]]
- [[domains/iran/concepts/iran-nuclear-enrichment-post-war/_concept]]
- [[timeline/2026-Q2]]
- [[domains/global/concepts/short-horizon-momentum-check/_concept]]
- [[domains/global/concepts/structural-improbability-check/_concept]]
