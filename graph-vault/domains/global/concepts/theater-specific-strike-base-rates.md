---
type: concept
tags: [concept, methodology, us-military, base-rates, kinetic-strikes]
title: "Theater-Specific Strike Base Rates"
slug: theater-specific-strike-base-rates
domain: global
first_observed: 2026-05-22
canonical_cases:
  - "US strike on Colombia by Dec 31 (pm-us-strike-colombia-dec31): vault=0.12 vs market=0.27"
status: active
related_concepts:
  - market-vault-structural-divergence
  - structural-improbability-check
  - short-horizon-momentum-check
  - rare-event-base-rate
---

# Theater-Specific Strike Base Rates

## Definition

US kinetic strike operations (drone strikes, air strikes, missile strikes, special operations raids) are **not uniformly distributed** across global theaters. Their base rates vary by orders of magnitude depending on the theater, driven by persistent structural factors (legal authorities, diplomatic relationships, military infrastructure, historical precedent, public tolerance) rather than transient political conditions.

The common forecasting error is to use the **global strike base rate** (which is dominated by MENA/South Asia operations) as a proxy for any specific theater, or to treat a one-off strike in one theater as a precedent that generalizes to other theaters.

## Current Strike Base Rates by Theater

### MENA / South Asia (High — >90% of all US strikes since 2001)

| Sub-region | Annual strikes (2020-2025 avg) | Last known strike | Authorization |
|---|---|---|---|
| Yemen (AQAP, Houthis) | 30-50 | Ongoing | AUMF 2001 + self-defense |
| Somalia (Al-Shabaab) | 10-25 | Ongoing | AUMF 2001, AFRICOM self-defense |
| Syria (ISIS remnants) | 5-15 | Ongoing (transition) | AUMF 2001 |
| Iraq (ISIS remnants) | 5-10 | 2024-2025 drawdown | AUMF 2001 / Iraqi consent |
| Afghanistan (post-withdrawal) | 0 (over-the-horizon) | Occasional | AUMF 2001 (contested) |
| Pakistan (North Waziristan) | 0-5 | ~2022 (tribal areas) | CIA paramilitary (covert) |

**Structural drivers**: AUMF 2001 legal cover, established basing infrastructure (Qatar, UAE, Djibouti, Kuwait), local government consent/tolerance, operational precedent, media indifference to collateral damage in these theaters.

### Horn of Africa (Medium-Low)

| Sub-region | Annual strikes | Notes |
|---|---|---|
| Somalia | 10-25 | AFRICOM active, but permissive consent from Federal Government |
| Djibouti | 0 | Hosts Camp Lemonnier; strikes would be political crisis for host |

### Latin America (Near-Zero)

| Sub-region | US strikes (post-1990) | Notes |
|---|---|---|
| Colombia | 0 | Closest US ally in the region. Zero kinetic strikes in 35+ years. Joint operations (counternarcotics training, advisory) are extensive — but no unilateral US bombs/missiles on Colombian soil. |
| Mexico | 0 | Cartel FTO designation rhetoric has not translated to strikes. US operations remain DEA/ICE-led. |
| Venezuela | 0 | Heavily sanctioned, hostile regime — but no strikes despite regime-change pressure (2019 coup attempt failed using economic/sanctions tools) |
| Other LAC | 0 | Panama invasion (1989) was full-scale intervention, not strike. 1983 Grenada invasion same category. |

**Structural drivers**: No AUMF applicability, strong sovereignty norms in Western Hemisphere, diplomatic blowback through OAS/CELAC, economic interdependence (trade relationships with multiple countries), SOUTHCOM lowest-priority combatant command, no basing infrastructure for strike operations, Congressional notification requirements under War Powers Resolution create friction in democracies' alliances.

### Europe (Near-Zero)

| Sub-region | US strikes (post-1990) | Notes |
|---|---|---|
| Balkans | ~0 | 1999 NATO bombing of Serbia was alliance operation, not US-only. Since 2000: zero. |
| Ukraine | 0 | US supplies weapons, intel, training — but no US strikes. Russia would treat a unilateral US strike as direct NATO intervention. |

### East Asia / Pacific (Near-Zero)

| Sub-region | US strikes | Notes |
|---|---|---|
| Philippines | 0 | Joint counterterrorism operations (against ISIS-linked groups) are Philippine-led with US advisory support |
| Taiwan | 0 | Would mean war with China |
| China/NK | 0 | NK: occasional naval exercises, shows of force. No kinetic strikes. |

## Forecasting Application

### Step 1 — Identify the Theater

Before estimating strike probability, classify the proposed strike's theater:
- **Active theater** (MENA, Horn of Africa): Base rate is non-trivial (annual strike count >0). Focus on trigger assessment, authorization status, and operational feasibility.
- **Inactive theater** (Latin America, Europe, East Asia): Base rate is near-zero (rare event). Focus on structural barriers first — only if barriers are overcome should trigger analysis begin.

### Step 2 — Check for Precedent

Zero strikes in 35+ years in a theater means the theater is **structurally non-conductive**, not that it is "overdue" for a strike. The absence of strikes is the equilibrium — positive reasons must exist for why a first strike would occur, not just "it's been a while since the Soleimani strike."

### Step 3 — Apply Rare-Event Decomposition

For inactive theaters, decompose the probability into:
1. **Trigger probability**: P(a specific trigger event that would justify a strike occurs)
2. **Authorization probability**: P(trigger → authorization decision given legal/political constraints)
3. **Operational probability**: P(authorization → successful strike given military readiness)
4. **Non-suppression probability**: P(strike is not prevented by last-minute diplomatic/political intervention)

Each factor for an inactive theater is typically <0.1, and the product of 4 factors is <10⁻⁴ before any positive evidence.

## Canonical Case: US Strike on Colombia by December 31, 2026

| Parameter | Value |
|-----------|-------|
| Question | US-initiated drone/missile/air strike on Colombian soil before Dec 31, 2026 |
| Cutoff | 2026-05-22 |
| Polymarket YES | 0.27 |
| Vault p_yes | 0.12 |
| Market volume | $992,691 |
| Theater class | Inactive (Latin America — zero strikes in 35+ years) |

**Theater analysis**: Latin America is an inactive strike theater. Six structural barriers identified: no precedent, ally sovereignty, diplomatic blowback ($40B+ trade at risk), lowest-priority combatant command (SOUTHCOM), Congressional notification friction, and no operational basing for unilateral strikes.

**Why the vault was lower**: The vault applied Steps 1-3 above, identifying Latin America as an inactive theater (Step 1) with zero precedent (Step 2). The trigger scenarios were then assessed (Step 3): cartel attack on US citizens, Venezuela-linked cross-border incident, hostage rescue, ELN/FARC attack on US personnel. Each trigger had P < 0.05 across the 7-month window. Authorization and operational probabilities were each <0.3 given Trump administration's non-expansionist kinetic posture. Product: ~0.04 before upward adjustment for black swan, yielding 0.12.

**Why the market was higher**: The 27% price appears to reflect (a) recency bias from the Soleimani killing (2020), treated as a precedent for unilateral strike authority anywhere; (b) conflation of US military presence in Colombia (advisory, training, joint operations — continuous) with kinetic strikes; (c) general "Trump will do something crazy" sentiment; (d) the December 31, 2026 window is 7 months — long enough for the market to price a constant per-month probability of ~4-5%, which ignores the structural barrier analysis showing near-zero per-month probability.

**Key insight**: The forecasting error is treating the **inactive theater** as elastically responsive to trigger conditions, when in fact the structural barriers create a near-absolute floor even with plausible triggers. A mass-casualty cartel attack on US citizens would increase trigger probability but would not automatically overcome sovereignty barriers, diplomatic costs, Congressional friction, and basing limitations. The market implicitly assumed a linear trigger→strike relationship; the vault modeled it as a 4-factor multiplicative chain.

## When NOT to Use

This concept describes **persistent structural constraints**, not transient political conditions. It is valid for forecasting horizons up to ~2 years (the duration within which theater classification is stable). Do not use for:

1. **Regime-change scenarios** where a theater's alliance structure changes fundamentally (e.g., Colombia leaves US alliance and hosts Chinese military base)
2. **Wartime emergencies** where the theater becomes an active conflict zone (e.g., full-scale Venezuela-Colombia war)
3. **Longer horizons** (>2 years) where basing, legal authorization, and alliance structures may shift
4. **Non-kinetic operations** (cyber operations, sanctions, training missions, intelligence sharing — these are not "strikes" and have different base rates)
