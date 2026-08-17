---
type: concept
tags: [concept, methodology, procedure, certainty]
title: "Short-Horizon Procedural Certainty"
slug: short-horizon-procedural-certainty
domain: global
first_observed: 2024-07-19
canonical_cases:
  - "Ethereum ETF begins trading by July 26, 2024 — p_yes=0.9, S-1 approval was the only remaining procedural step after 19b-4 approval"
  - "Israel announces ceasefire by Jan 17, 2025 — p_yes=0.91, Jan 15 framework announcement activated commitment trap, cabinet vote was scheduled Jan 17"
  - "Israel announces ceasefire by Sunday (Jan 19), 2025 — p_yes=0.9, same mechanism, slightly longer window"
  - "SCOTUS does NOT delay TikTok ban, Jan 2025 — p_yes=0.93 (inverse: p_no_delay=0.93), cert before judgment + compressed schedule + zero stays = procedural determinism"
status: active
related_concepts:
  - short-horizon-momentum-check
  - structural-improbability-check
  - public-framework-announcement-commitment
  - scotus-procedural-signals
  - regulatory-precedent-cascade
  - forecast-resolution-criteria-gotchas
---

# Short-Horizon Procedural Certainty

## Definition

A pre-forecast diagnostic for recognizing when a question's YES outcome is **procedurally locked in** within a short window: the necessary bureaucratic/legal/institutional steps are known, the remaining steps are ministerial (not discretionary), and the timeline is compressed enough that no intervening shock can realistically disrupt the process.

This is the **YES-side complement** to [[domains/global/concepts/short-horizon-momentum-check/_concept]], which covers structural NO cases. Together they form a two-sided pre-filter:

- **Momentum check** (NO side): Strong trend away from YES + no catalyst → P(YES) < 0.10
- **Procedural certainty** (YES side): Procedural steps complete or locked in + remaining steps ministerial → P(YES) > 0.90

## Distinction From Related Concepts

| Concept | When to Use | Canonical Example |
|---------|-------------|-------------------|
| **short-horizon-procedural-certainty** *(this)* | Procedural steps are known and nearly complete; remaining steps are ministerial; short window makes disruption unlikely | Ethereum ETF: 19b-4 approved, S-1 was the only remaining step, issuers submitting final amendments → P(trading by July 26) = 0.90 |
| **short-horizon-momentum-check** | Short window, strong trend away from YES, no known catalyst → near-zero probability | Bitcoin >$72K: -38% trend, 7 days, no catalyst → P(YES) = 0.08 |
| **structural-improbability-check** | YES requires 2+ independent unlikely events → multi-collapse filter | Far-left party winning Latin American presidency |
| **public-framework-announcement-commitment** | Superpower patron publicly announces framework before local party ratification → commitment trap | Gaza ceasefire Jan 2025: Biden+Trump+Qatar announce → Israel cannot reject |
| **scotus-procedural-signals** | Court's procedural choices (cert before judgment, no stay, compressed schedule) imply outcome direction | TikTok ban: 9-0 cert grant, 3-week schedule, zero stays → near-certain no delay |

## Key Question: Is the Remaining Step Ministerial or Discretionary?

The entire diagnostic hinges on one distinction:

**Ministerial step** — A decision whose outcome is predetermined (legal obligation, no realistic refusal path, or top-level approval already signaled). Examples:
- Issuer submits final S-1 amendments → SEC staff processes them and declares effective (Ethereum ETF)
- Security cabinet votes on framework that PM already accepted under dual-presidential endorsement (Israel ceasefire)
- Court that expedited merits review to beat a statutory deadline issues its ruling (SCOTUS TikTok)

**Discretionary step** — A decision that could realistically go either way even after preconditions are met. Examples:
- Senate vote on a SCOTUS nominee when confirmations are split party-line
- Fed rate decision when data is ambiguous and FOMC is divided
- Corporate board approval of a merger under regulatory uncertainty

### Procedural Lock-In Checklist

Apply before every short-horizon forecast:

```
Step 1: Identify the remaining procedural step(s)
  → What must happen for YES to occur?
  → List each step in order

Step 2: Classify each step
  → Ministerial (90-99% probability given preconditions)
  → Discretionary (50-90% probability)
  → Purely uncertain (<50% probability even with preconditions)

Step 3: Check for disruption risk
  → Can a third party block the process? (Court, regulator, opposition)
  → Can the deciding party change their mind without severe cost?
  → Is there a hard deadline that creates a failure mode?

Step 4: Anchor the probability
  → All remaining steps ministerial + no disruption path: P(YES) = 0.90-0.95
  → Mixed ministerial/discretionary: P(YES) = 0.60-0.85
  → All remaining steps discretionary: standard forecast (no shortcut)
```

## Canonical Case 1: Ethereum ETF Begins Trading by July 26, 2024

| Parameter | Value |
|-----------|-------|
| Indicator Combination | Mechanism | YES | NO | Hit Rate | N | Updated |
|----------------------|-----------|-----|----|--------- |----|---------|
| All remaining steps ministerial + <2 week window + no disruption path | Ministerial lock-in | 4 | 0 | 100% | 4 | 2026-05-21 |
| All remaining steps ministerial + 2-4 week window + no disruption path | Ministerial lock-in (longer) | 2 | 0 | 100% | 2 | 2026-05-21 |
| Court cert before judgment + compressed schedule + zero stays | Procedural inverse (NO lock-in) | 0 | 1 | 0% | 1 | 2026-05-21 |
| Remaining steps discretionary + >1 month horizon | Standard structural (not short-horizon) | 1 | 0 | 100% | 1 | 2026-05-21 |

**Diagnosis**: All procedural conditions for a high-confidence YES forecast were met. The 19b-4 approval was the substantive hurdle; S-1 was ministerial paperwork. Multiple independent sources confirmed issuers had submitted final amendments. No plausible disruption path existed in the remaining 7 days. Both the vault forecast (0.90) and the market (0.9010) correctly identified the near-certainty.

**Distinction from short-horizon-momentum-check**: This is a YES case, not a NO case. The momentum check would not apply here (no trend to reverse). The procedural-certainty framework correctly identifies the high-confidence YES scenario.

**Vault files**: [[runs/20260521-030532-question-ethereum-etf-begins-trading-by-july-26-forecast-usi]]

## Canonical Case 2: Israel Announces Ceasefire by Jan 17, 2025

| Parameter | Value |
|-----------|-------|
| Indicator Combination | Mechanism | YES | NO | Hit Rate | N | Updated |
|----------------------|-----------|-----|----|--------- |----|---------|
| All remaining steps ministerial + <2 week window + no disruption path | Ministerial lock-in | 4 | 0 | 100% | 4 | 2026-05-21 |
| All remaining steps ministerial + 2-4 week window + no disruption path | Ministerial lock-in (longer) | 2 | 0 | 100% | 2 | 2026-05-21 |
| Court cert before judgment + compressed schedule + zero stays | Procedural inverse (NO lock-in) | 0 | 1 | 0% | 1 | 2026-05-21 |
| Remaining steps discretionary + >1 month horizon | Standard structural (not short-horizon) | 1 | 0 | 100% | 1 | 2026-05-21 |

**Diagnosis**: The public framework announcement on Jan 15 by Biden, Trump, and Qatar PM was the substantive decision point. Israel's formal cabinet vote was the ministerial follow-through. The dual-presidential endorsement activated the [[domains/mena/concepts/public-framework-announcement-commitment/_concept]] at maximum strength, making rejection functionally impossible for a US-dependent ally. The 1-day window was the only residual uncertainty.

**Sibling run**: [[runs/20260521-033557-israel-announces-ceasefire-by-sunday-forecast-using-only-the]] — Same event, slightly longer window (Jan 16→Jan 19), same mechanism, p_yes=0.90.

**Vault files**: [[runs/20260521-033327-israel-announces-ceasefire-by-january-17-forecast-using-only]]

## Canonical Case 3 (Inverse): SCOTUS Does NOT Delay TikTok Ban, Jan 2025

| Parameter | Value |
|-----------|-------|
| Indicator Combination | Mechanism | YES | NO | Hit Rate | N | Updated |
|----------------------|-----------|-----|----|--------- |----|---------|
| All remaining steps ministerial + <2 week window + no disruption path | Ministerial lock-in | 4 | 0 | 100% | 4 | 2026-05-21 |
| All remaining steps ministerial + 2-4 week window + no disruption path | Ministerial lock-in (longer) | 2 | 0 | 100% | 2 | 2026-05-21 |
| Court cert before judgment + compressed schedule + zero stays | Procedural inverse (NO lock-in) | 0 | 1 | 0% | 1 | 2026-05-21 |
| Remaining steps discretionary + >1 month horizon | Standard structural (not short-horizon) | 1 | 0 | 100% | 1 | 2026-05-21 |

**Diagnosis**: The question frames a YES for delay, but the procedural certainty framework correctly identifies the near-deterministic NO. The four procedural signals (cert before judgment, compressed schedule, no stays, oral arguments without interim relief) formed a unanimous pattern: the Court expedited to rule, not to delay. The market at 0.24 was structurally mispriced due to recency noise from oral arguments and the unrelated Trump stay denial.

**Key insight**: The procedural-certainty framework applies to both YES outcomes (Ethereum ETF) and NO outcomes (TikTok delay) — the critical variable is whether the procedural trajectory locks in a specific resolution, regardless of direction.

**Vault files**: [[runs/20260521-033054-will-supreme-court-delay-the-tiktok-ban-forecast-using-only-]], [[events/tiktok-scotus-ruling-jan-2025]]

## Canonical Case 4 (Structural Variation): Argentina LLA Most Seats

| Parameter | Value |
|-----------|-------|
| Indicator Combination | Mechanism | YES | NO | Hit Rate | N | Updated |
|----------------------|-----------|-----|----|--------- |----|---------|
| All remaining steps ministerial + <2 week window + no disruption path | Ministerial lock-in | 4 | 0 | 100% | 4 | 2026-05-21 |
| All remaining steps ministerial + 2-4 week window + no disruption path | Ministerial lock-in (longer) | 2 | 0 | 100% | 2 | 2026-05-21 |
| Court cert before judgment + compressed schedule + zero stays | Procedural inverse (NO lock-in) | 0 | 1 | 0% | 1 | 2026-05-21 |
| Remaining steps discretionary + >1 month horizon | Standard structural (not short-horizon) | 1 | 0 | 100% | 1 | 2026-05-21 |

**Diagnosis**: This is a **negative case** — the procedural-certainty framework correctly identifies that 7 months is too long for a "short horizon" lock-in. The remaining steps are discretionary (campaign dynamics, scandal evolution), not ministerial. The forecast correctly fell back to standard structural analysis (4/5 favorable factors but long horizon). The framework prevents over-application to genuinely uncertain long-horizon questions.

## Empirical Calibration

This concept's hit rates are maintained by the **tag-based calibration model**
in `harness/tag_calibration.py`.  Query at forecast time:

```python
cal = TagCalibration()
cal.load_jsonl("data/polymarket/resolved_markets.jsonl")
r = cal.query(["Geopolitics", "Politics"])  # for procedural certainty questions
```

PIT cutoffs enforced automatically via `end_date`.  The tag model pools across
1277+ resolved markets (not the 8 samples formerly listed here).

**Previous tables (removed 2026-05-22):**  4Y/0N ministerial lock-in, 2Y/0N
longer-horizon, 0Y/1N procedural inverse, 1Y/0N standard — all superseded.

## Cross-Run Pattern Annotation

Four gold set runs share the procedural-certainty pattern:

| Run | p_yes | Mechanism | Step Classification | Brier Status |
|-----|-------|-----------|--------------------|-------------|
| [[runs/20260521-030532-question-ethereum-etf-begins-trading-by-july-26-forecast-usi]] | 0.90 | S-1 after 19b-4 approval | Ministerial | Unresolved |
| [[runs/20260521-033327-israel-announces-ceasefire-by-january-17-forecast-using-only]] | 0.91 | Cabinet vote after framework announcement | Ministerial | Unresolved |
| [[runs/20260521-033557-israel-announces-ceasefire-by-sunday-forecast-using-only-the]] | 0.90 | Same, slightly longer window | Ministerial | Unresolved |
| [[runs/20260521-033054-will-supreme-court-delay-the-tiktok-ban-forecast-using-only-]] | 0.07 (inverse) | Cert + schedule + no stays = no delay | Ministerial inverse | Unresolved |

These four runs form a calibration cluster that will validate the procedural-certainty framework when their Brier scores are recorded.

## Relationship to Short-Horizon Momentum Check

The two concepts form complementary sides of the same meta-framework:

```
Before any short-horizon forecast:
  1. Apply momentum check (NO side)
     → Is the trend strongly against YES with no catalyst?
     → If yes: P(YES) < 0.10, no further analysis needed

  2. Apply procedural certainty check (YES side)
     → Are remaining steps ministerial with no disruption path?
     → If yes: P(YES) > 0.90, no further analysis needed

  3. If neither applies
     → Proceed with standard forecasting methodology
```

Applying both filters in sequence ensures the forecaster doesn't miss either the structural-NO or structural-YES pattern. The two concepts should be loaded together for any short-horizon question (calendar window < 4 weeks or existence condition).

## Pitfalls

1. **Confusing "likely" with "procedurally locked"**: A 70% probability is not procedural certainty. Only apply the framework when the remaining step is genuinely ministerial (90%+ confidence). If there's any realistic path to rejection, use standard forecasting.

2. **Disruption blindness**: Even ministerials steps can be disrupted by exogenous events (health emergency, natural disaster, terror attack). The residual 5-10% accounts for this. Never set P(YES) = 1.0.

3. **False certainty in multi-step processes**: If 2+ steps remain and either could block YES, the probability is the joint probability (e.g., 0.90 × 0.90 = 0.81), even if both steps appear ministerial individually.

4. **Over-extension to long horizons**: The framework ONLY applies to short windows (< 1 month). Over 1 month, disruption risk accumulates and steps become discretionary as contexts shift. The Argentina LLA case (7-month horizon) is the counterexample showing when NOT to apply this framework.

5. **Mirror confusion with momentum check**: A high-confidence YES is not the same as "no momentum against YES." The Ethereum ETF case had no strong trend either way — the certainty came from procedural mechanics, not trend analysis. Always check the correct framework.

## Cross-References

- [[runs/_index]] — Runs index identifying this pattern cluster
- [[domains/global/concepts/short-horizon-momentum-check/_concept]] — Complementary structural-NO filter
- [[domains/global/concepts/structural-improbability-check/_concept]] — Related multi-collapse pre-filter
- [[domains/mena/concepts/public-framework-announcement-commitment/_concept]] — Specific mechanism for Israel ceasefire certainty
- [[domains/mena/concepts/scotus-procedural-signals/_concept]] — Specific mechanism for TikTok no-delay certainty
- [[domains/global/concepts/regulatory-precedent-cascade/_concept]] — Mechanism for Ethereum ETF certainty (spot Bitcoin ETF precedent forced SEC's hand on Ethereum)
- [[domains/global/concepts/forecast-range-plausibility-filter]] — Related pre-filter for numerical questions
- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — Prevents criteria misreading that could misclassify a step as ministerial vs discretionary
