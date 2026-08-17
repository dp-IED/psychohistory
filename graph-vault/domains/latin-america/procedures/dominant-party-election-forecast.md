---
type: procedure
title: "Dominant-Party Election Forecast"
slug: dominant-party-election-forecast
tags: [procedure, latin-america, elections, forecast]
---

# Procedure: Dominant-Party Election Forecast

## When to Use
Use this procedure when a forecasting question asks whether a candidate from a dominant party (a party that has won 3+ consecutive presidential elections or holds >45% party identification) will win an upcoming election. This procedure is applicable in Mexico (MORENA), Venezuela (PSUV), Nicaragua (FSLN), Bolivia (MAS), and similar contexts.

This procedure is **complementary to** (not a replacement for) the [[domains/east-asia/procedures/taiwan-election-forecast]] and [[domains/latin-america/procedures/dominant-party-election-forecast]] procedures. Use the general [[domains/east-asia/procedures/third-party-candidate-viability-check]] for fragmentation assessment.

## Step-by-Step

### Phase 1: System Identification
1. **Confirm dominant party status**: Has the incumbent party won 3+ consecutive presidential elections? Is its party identification >45%? If yes, dominant-party framework applies.
2. **Check incumbent eligibility**: Can the incumbent president run for re-election? If term-limited, the successor dynamic is active (see [[domains/latin-america/concepts/incumbent-successor-dominant-party/_concept]]). If eligible and running, use different framework (incumbent re-election).
3. **Identify electoral system**: Single-round plurality, two-round runoff, or proportional representation? Single-round plurality makes dominant-party candidates especially hard to dislodge even with a fragmented opposition.

### Phase 2: Approval Transfer Assessment
4. **Measure incumbent approval**: What is the outgoing (or running) incumbent's approval rating? Thresholds:
   - >55%: Strong successor dominance — forecast the successor win at >95%
   - 40-55%: Competitive — forecast depends on opposition quality and economic conditions
   - <40%: Successor penalty active — opposition likely wins unless fragmented
5. **Calculate approval-to-vote efficiency**: In Latin American dominant-party systems, there is typically 5-15% "wastage" between incumbent approval and successor vote share (some approval voters do not transfer). In Mexico 2024, wastage was near zero (60% approval → 60% vote). In Venezuela 2013, wastage from Chávez (55%) to Maduro (50.6%) was ~4.5%.
6. **Assess the successor's independent appeal**: Does the successor have their own public profile and political identity, or is their identity entirely derivative of the incumbent? Sheinbaum had both (own profile as Mexico City mayor + AMLO endorsement). Derivative-only successors underperform the approval-transfer model.

### Phase 3: Opposition Assessment
7. **Count opposition candidates**: How many opposition candidates are running? In single-round plurality systems, each additional opposition candidate splits the anti-dominant-party vote.
8. **Assess opposition coalition coherence**: Is there a unified opposition candidate (e.g., "Fuerza y Corazón por México") or multiple competing candidates? An incoherent coalition (ideologically diverse) is less effective than a coherent one — voters face a choice between the known dominant-party candidate and an unknown coalition compromise.
9. **Check opposition candidate quality**: Does the opposition have a candidate with high personal approval independent of party brand? Xóchitl Gálvez had a strong personal story but was dragged down by the PAN-PRI-PRD coalition's negative brand associations.
10. **Apply the [[domains/east-asia/concepts/divided-opposition-plurality-win/_concept]] framework**: If 3+ candidates are viable, the dominant-party win probability is further elevated by fragmentation — even if the dominant party's vote share is modest (35-48%).

### Phase 4: Structural Factors
11. **Assess economic conditions**: In dominant-party systems, the economy matters less for incumbent-party successor elections than in competitive systems. Voters attribute economic outcomes to the specific incumbent, not the party — so a successor candidate can maintain support despite mediocre economic growth if the incumbent is personally popular.
12. **Assess social program entrenchment (see [[domains/latin-america/concepts/social-program-approval-sustainability/_concept]])**: In dominant-party systems where the ruling party has expanded social programs to cover >40% of households:
   - The approval floor is 5-15% higher than raw incumbent approval would predict
   - Economic perception is decoupled from GDP growth — voters evaluate performance through program continuity, exchange rate visibility, and personal transfer receipt
   - Track three metrics: real value of transfers, exchange rate trend, inflation relative to benefit growth — NOT GDP growth
   - If social program coverage exceeds 40% of households AND the party is maintaining or expanding programs, add +5-10% to the expected successor vote share
13. **Check for electoral manipulation infrastructure**: Does the dominant party control the electoral commission, judiciary, or security forces? If yes, even an opposition win at the ballot box may not result in the opposition assuming office (Venezuela 2024 pattern).
14. **Assess media environment**: Does the dominant party have a media advantage (TV coverage, social media, press conferences)? State media bias is common in Latin American dominant-party systems.

### Phase 5: Probability Calibration
15. **Start from dominant-party baseline**: In a dominant-party system with a popular incumbent, start from >90% probability of the successor winning. Adjust down only for specific identified risks.
16. **Adjust for identified risks**:
    - Successor scandal or major gaffe: -5 to -15%
    - Economic crisis in last 12 months: -10 to -20%
    - Incumbent approval below 50%: -20 to -40%
    - Unified opposition with strong candidate: -10 to -20%
    - Electoral manipulation/uncertainty: adjust credibility interval, not point estimate
17. **Document both the structural case and the risk case**: The reasoning MUST include both why the dominant party is expected to win (the structural mechanics) AND what could plausibly cause an upset (specific risk scenarios with estimated probabilities).

## Probability Calibration Table

|| Configuration | Baseline Probability | Range |
||--------------|---------------------|-------|
|| Incumbent approval >55%, successor dynamic, single-round plurality, fragmented opposition, SOCIAL PROGRAM FLOOR >40% HH | 95-99% | Very high |
|| Incumbent approval >55%, successor dynamic, single-round plurality, fragmented opposition, no social program floor | 90-95% | Very high |
|| Incumbent approval >55%, successor dynamic, single-round plurality, unified opposition | 80-90% | High |
|| Incumbent approval 40-55%, competitive succession with social program floor | 65-85% | Moderate-high |
|| Incumbent approval 40-55%, competitive succession | 55-75% | Moderate-high |
| Incumbent approval 40-55%, incumbent running for re-election | 55-75% | Competitive |
| Incumbent approval <40%, successor dynamic | 20-40% | Unlikely |
| Incumbent approval <40%, incumbent running for re-election | 25-50% | Underdog |

## Verification
After forecasting, verify:
- [ ] Did you check the electoral system type?
- [ ] Did you measure incumbent approval rating from PIT-compatible polling?
- [ ] Did you count all viable opposition candidates?
- [ ] Did you assess opposition coalition coherence?
- [ ] Did you distinguish successor dominance from fragmentation dynamics?
- [ ] Did you assess social program entrenchment coverage and its approval floor effect?
- [ ] Did you check whether the successor was selected via dedazo or competitive primary?
- [ ] Did you assess economic perception decoupling (exchange rate, transfers, inflation)?
- [ ] Did you document both the structural case and the risk case?

## Wikilinks
- [[domains/latin-america/concepts/incumbent-successor-dominant-party/_concept]]
- [[domains/latin-america/concepts/social-program-approval-sustainability/_concept]]
- [[domains/latin-america/entities/claudia-sheinbaum]]
- [[domains/latin-america/entities/andres-manuel-lopez-obrador]]
- [[domains/latin-america/entities/morena]]
- [[domains/latin-america/threads/mexican-politics/_thread]]
- [[domains/east-asia/concepts/divided-opposition-plurality-win/_concept]]
