---
type: procedure
tags: [procedure, forecasting, resolution]
title: "Ban Resolution Terminology Checklist"
slug: ban-resolution-checklist
domain: forecasting-methodology
version: 1.0
date: 2026-05-20
parent_concept: [[domains/global/concepts/national-security-tech-ban]]
---

# Ban Resolution Terminology Checklist

## Purpose

A structured method for assessing what "banned" means in a prediction market resolution context. The TikTok divest-or-ban case proved that the everyday meaning of "banned" (ongoing, persistent prohibition) differs from the market resolution meaning (legal prohibition that took effect, even if enforcement is later delayed or reversed).

## When to Activate

This procedure MUST be loaded BEFORE forecasting any question with the word "banned," "prohibited," "blocked," "restricted," "outlawed," or similar language about a government restriction on technology, products, services, or individuals.

## Procedure Steps

### Step 1: Read the Full Resolution Text

Find the exact resolution criteria. Identify key terms:

- "Banned for download and/or use" — the ban is satisfied by removal from app stores, not just legislative action
- "Banned from operating" — may require ongoing enforcement
- "Banned with no exemptions" — a higher bar (ban must be complete)
- "Banned or restricted" — lower bar (restriction alone may suffice)

Document the exact phrasing used.

### Step 2: Distinguish Legal Status from Practical Enforcement

For every "banned" question, answer these three sub-questions separately:

| Sub-question | Assessment | Data Sources |
|-------------|------------|-------------|
| Did the law/order legally take effect? | Did it pass Congress? Was it signed? Did it survive legal challenge? Was the deadline reached? | Congress.gov, court rulings, executive orders |
| Did enforcement actions occur? | Was the product removed from app stores? Was service suspended? Were fines imposed? Were companies ordered to comply? | News reports, company statements, app store listings |
| Is enforcement persisting? | Is the ban still in effect? Has enforcement been delayed? Has the product been reinstated? Has the executive issued a non-enforcement order? | Executive orders, agency guidance, news reports |

**Resolution rule**: Prediction markets typically resolve based on (1) AND (2) — the legal effect PLUS enforcement actions. Condition (3) is almost never required unless the resolution text explicitly says "banned and remains banned" or "permanently banned."

### Step 3: Check for Executive Enforcement Delay

If the law legally took effect and enforcement actions occurred, check whether any of these delay mechanisms are active:

- **Formal executive order delaying enforcement**: Does the president or governor have the authority to delay? For what duration? (TikTok: 75-day delay via executive order)
- **Administrative non-enforcement**: Has the relevant agency issued guidance that it will not enforce the ban?
- **Regulatory inaction**: Does the law require implementing regulations that have not been issued?
- **Prosecutorial discretion**: Has the DOJ or equivalent announced it will not prosecute violations?

If any delay mechanism is active, note that the ban is legally in effect but practically suspended. Continue to Step 4.

### Step 4: Map to the Resolution Timeline

Plot the key dates:

| Date | Event | Resolution Significance |
|------|-------|----------------------|
| [Date] | Law passed/signed | Legal framework established |
| [Date] | Court decision upholding law | Legal challenge resolved |
| [Date] | Deadline for compliance | Ban legally takes effect |
| [Date] | Enforcement actions (app removed, service dark) | Ban enforced — key resolution event |
| [Date] | Executive delay order | Ban suspended — no effect on resolution |
| [Date] | Service restored | No effect on resolution (post-enforcement) |

**Key insight**: The resolution date is typically the deadline/compliance date. Events after that date (including enforcement delays) do not retroactively change whether the ban occurred by the deadline.

### Step 5: Cross-Reference with the National Security Tech Ban Lifecycle

Apply the lifecycle from [[concepts/national-security-tech-ban]]:

1. Which stage is the ban in? (Threat framing → Political mobilization → Legislative/executive action → Legal challenge → Implementation → Alliance pressure → Adaptation)
2. Has the ban reached the Implementation stage? If yes, legal effect has occurred.
3. Is the Implementation bifurcated (legal effect vs practical enforcement)? If yes, the ban likely resolved YES for market purposes.

### Step 6: Apply Dual-Frame Analysis

Document BOTH frames:

- **Affirmative case (ban legally took effect)**: Law passed, deadline reached, enforcement actions occurred → "banned" criteria satisfied
- **Countervailing case (ban didn't really happen)**: Enforcement was delayed, service was restored, the executive branch refused to enforce → "banned" may feel incomplete

**Decision heuristic**: If the law legally took effect AND enforcement actions occurred (even briefly), predict YES on a "banned" question, even if enforcement was subsequently delayed. The countervailing case reflects the everyday meaning, not the market resolution meaning.

### Step 7: Document the Reasoning

Record in the forecast entry:
- The exact resolution text
- The legal status at the deadline (YES/NO)
- Whether enforcement actions occurred (YES/NO)
- Whether enforcement persistence is required by the resolution text (YES/NO → if NO, persistence is irrelevant)
- Whether any executive enforcement delay is active
- The final forecast with rationale mapping to the ban-resolution framework

## Validation

This procedure was developed based on the TikTok divest-or-ban case (2024-2025):
- The law passed with veto-proof majorities → legal status clear
- The deadline passed Jan 19, 2025 → legal effect triggered
- App stores removed TikTok, service went dark → enforcement actions occurred
- Trump issued executive order delaying enforcement Jan 20 → enforcement suspended
- Service restored → practical reinstatement
- Market resolved YES because resolution criteria required only legal effect + enforcement actions, not persistence

## Dependencies

This procedure references:
- [[domains/global/concepts/national-security-tech-ban]] — the lifecycle framework for tech bans
- [[domains/global/concepts/executive-enforcement-delay]] — the legal-vs-practical enforcement distinction
- [[domains/global/concepts/forecast-resolution-criteria-gotchas]] — the broader gotcha pattern
- [[domains/global/entities/tiktok]] — TikTok entity stub
- [[domains/global/entities/bytedance]] — ByteDance entity stub
