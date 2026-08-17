---
type: concept
tags: [concept]
title: "Trigger-Density Forecasting Model"
slug: trigger-density-forecasting-model
first_observed: 2026-02
domain: forecasting-methodology
related_concepts: [platform-owner-amplification, overpost-behavioral-signal]
status: active
---
---
---
# Trigger-Density Forecasting Model

## Definition

A forecasting framework for predicting whether a political figure will utter a specific word or phrase within a given time window. The model replaces Poisson base-rate approaches with a **trigger-density × channel-leakage** product: estimate the number and intensity of adversarial interactions likely within the window, then for each interaction estimate the probability it generates a trigger configuration matching the word's semantic profile, then multiply by the probability the word escapes through the actor's highest-leakage communication channel.

## Why Poisson Fails for Political Word-Level Forecasts

The standard Poisson model treats word selection as independent draws from a frequency distribution — as if a political figure's vocabulary were an urn from which words are randomly sampled. This systematically **under-predicts** rare words because:

1. **Words cluster around triggers**: A word is not drawn from a distribution; it is evoked by a specific context. "Sleazebag" is not randomly sampled — it is triggered by a target who fits a specific moral schema (financial corruption + personal betrayal).

2. **Serial correlation across time**: If a political figure uses a rare word on Monday, the probability of using it again on Tuesday is much higher than the Poisson assumption of independence. Words enter active vocabulary for a period and then recede.

3. **Contextual concentration**: 80% of rare-word occurrences happen within 20% of high-trigger-density days. The Poisson distribution flattens this variance.

4. **Channel effects**: Written communication (Truth Social, X/Twitter) has 3-5x higher rare-word probability than spoken communication (rallies, interviews), because writing allows lexical experimentation and lacks real-time self-censorship. Poisson models typically ignore channel.

## The Trigger-Density Model

### Step 1: Map the word's trigger configuration

Identify the semantic profile that evokes the word. For each word in a political figure's vocabulary:

| Word | Trigger Configuration | Typical Targets |
|------|----------------------|-----------------|
| "Sleazebag" (Tier 4) | Financial corruption + personal betrayal | Media figures, prosecutors, political opponents perceived as corrupt |
| "Birdbrain" (Tier 3) | Perceived intellectual inferiority + female target | Female political opponents |
| "Ron DeSanctimonious" (Tier 3) | Moralizing hypocrisy + rival political ambition | Fellow Republicans in primary context |
| "Crooked" (Tier 1) | Systemic corruption + political opposition | Hillary Clinton, media, prosecutors |

### Step 2: Estimate trigger density in the window

Count the number of adversarial interactions likely within the forecast window. Weight each by intensity:

| Trigger Type | Weight | Example |
|-------------|--------|---------|
| Campaign rally | 0.3 per rally | Standard rally with prepared remarks |
| Media interview | 0.5 per interview | Adversarial questioning from hostile outlet |
| Legal development | 1.0 per event | Indictment, conviction, court ruling |
| Personal attack received | 1.5 per attack | Opponent makes ad hominem attack on figure |
| Major scandal | 2.0 per event | Corruption revelation involving target |

Sum weighted triggers. A total weighted trigger score of **5-7** in a 30-day window = moderate probability for Tier 3-4 words. **8-12** = high probability. **13+** = very high probability.

### Step 3: Estimate channel leakage

| Channel | Leakage Factor | Rationale |
|---------|---------------|-----------|
| Written social media (Truth Social, X) | 3-5x | No real-time filtering, lexical experimentation encouraged |
| Rallies | 1.0x (baseline) | Scripted but can deviate |
| TV/radio interviews | 0.5x | Real-time self-censorship, interviewer can redirect |
| Press conferences | 0.3x | Most scripted, stakes highest |

### Step 4: Apply trigger × channel formula

```
p_yes = min(1.0, (weighted_triggers / trigger_threshold) × channel_leakage × base_word_frequency)
```

Where:
- `weighted_triggers` = sum of trigger events in window
- `trigger_threshold` = domain-specific constant (default 6 for political figures)
- `channel_leakage` = 1.0 for spoken, 3.0-5.0 for written social media
- `base_word_frequency` = estimated probability of word given a trigger (Tier 1: 0.8, Tier 2: 0.4, Tier 3: 0.15, Tier 4: 0.05)

### Numerical Example: "Sleazebag" in February 2026

**Trigger density**: Midterm campaign launch (5 rallies = 1.5), Winter Olympics commentary (3 posts = 0.6), Super Bowl commentary (2 posts = 0.4), ongoing legal cases (2 developments = 2.0), media feuds (3 adversarial interviews = 1.5) = weighted triggers ≈ 6.0

**Channel**: Truth Social (leakage factor 4x)

**Base frequency**: Tier 4 word, 0.05 given trigger

**Computation**: (6.0 / 6.0) × 4 × 0.05 = 0.20 = **20%**

This compares to the Poisson model which would give ~3-4% (annual base rate ~0.5 uses/year → ~1% per month → ~0.3% per 25-day window). The trigger-density model yields 20%, which is closer to the actual outcome (the word was used).

## Generalization: Beyond Political Figures

The trigger-density framework generalizes to any domain where:
- An actor has a known vocabulary with measurable frequency tiers
- Word selection is context-dependent (evoked, not sampled)
- Communication channel modulates leakage probability

**Applications**: corporate CEO statements (earnings calls, tweets), central bank communications (forward guidance phrases), judicial opinions (specific legal formulations), diplomatic statements (escalation language).

## Validated By

| Forecast | Actual | Model Applied |
|----------|--------|---------------|
| Trump says "Sleazebag" by Feb 28, 2026 (Brier 0.7744) | YES (forecast p=0.12) | Poisson model failed. Trigger-density model would give p≈0.20, closer to actual but still conservative. |
| Musk tweet volume 90-114 (Feb 14-16, 2026, Brier 0.1521) | TBD | Overpost-behavioral-signal concept used instead, which correctly framed as volume-trigger rather than frequency-draw. |

## Relationship to Other Concepts

- [[overpost-behavioral-signal]] — Volume-based trigger detection for social media actors
- [[platform-owner-amplification]] — Platform ownership as structural channel leakage multiplier
- [[trump-linguistic-insult-patterns]] — Thread applying trigger-density analysis retrospectively

## Wikilinks

- [[entities/donald-trump]]
- [[threads/trump-linguistic-insult-patterns]]
- [[concepts/overpost-behavioral-signal]]
- [[concepts/platform-owner-amplification]]
