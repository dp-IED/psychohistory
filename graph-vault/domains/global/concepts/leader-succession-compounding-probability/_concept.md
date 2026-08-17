---
type: concept
tags: [concept, forecasting, methodology, structural, political-succession]
status: seed
updated: 2026-06-17
pit_cutoff: 2026-06-17
---

# Leader-Succession Compounding Probability

**Pattern**: Forecasting a specific individual becoming a national leader requires multiplying the probabilities of multiple independent gates (party winning + incumbent departing + candidate selected + coalition accepting). The compounding product is typically much lower than intuition suggests.

## The Israeli PM Canonical Case

Two June 2026 forecasts demonstrate identical structural mechanics:

| Forecast | p_yes | Compounding Chain | Product |
|----------|-------|-------------------|---------|
| [[runs/20260616-yair-golan-pm-israel|Yair Golan next PM]] | 0.004 | (a) Center-left wins enough seats (~10%) × (b) Golan leads coalition (~30%) × (c) Coalition accepts (~30%) × (d) Netanyahu not blocking (~30%) ≈ 0.27% |
| [[runs/20260616-amir-ohana-pm-israel|Amir Ohana next PM]] | 0.005 | (a) Likud wins (~37.5%) × (b) Netanyahu steps down (~17.5%) × (c) Ohana selected over Levin/Katz/Barkat (~7.5%) × (d) Coalition accepts (~35%) ≈ 0.17% |

Both estimates are within 3× of their respective market prices (0.4% and 0.5%), demonstrating the framework's calibration power.

## Expanded "Lottery Ticket" Leadership Cluster (June 2026)

After the June 15-16 forecast batch, the vault now catalogs **5 structurally-impossible leadership outcomes** with p<0.01, forming a "lottery ticket" cross-domain cluster:

| Forecast | p_yes | Domain | Gates | Primary Barrier |
|----------|-------|--------|-------|-----------------|
| [[runs/20260616-yair-golan-pm-israel|Yair Golan next PM]] | 0.004 | Israel | 4-gate chain | Center-left win + Golan leads coalition |
| [[runs/20260616-amir-ohana-pm-israel|Amir Ohana next PM]] | 0.005 | Israel | 4-gate chain | Netanyahu departure + Ohana selected over rivals |
| [[runs/20260615-shaked-israel-pm|Ayelet Shaked next PM]] | 0.002 | Israel | 4-gate chain | No Knesset seat + collapsed political base |
| [[runs/20260615-haley-2028-nomination|Nikki Haley 2028 GOP nomination]] | 0.01 | USA | 4-gate chain | Trump-MAGA consolidation + Vance heir apparent |
| [[runs/20260615-mekonnen-ethiopia|Demeke Mekonnen next PM of Ethiopia]] | 0.01 | Ethiopia | 3-gate chain | Abiy's parliamentary lock + removed as Deputy PM |

### Cluster Insights
1. **Cross-domain consistency**: All 5 have p<0.01 despite different political systems (parliamentary, presidential, dominant-party) — the compounding effect is a universal structural constraint, not domain-specific
2. **Compounding as calibration tool**: The ratio of market price to compounded-gate product reveals whether the market is pricing genuine probability or lottery-ticket behavior
3. **3 Israel PM bets in one cluster**: Golan (0.004), Ohana (0.005), Shaked (0.002) — three separate markets for "next PM of Israel" with different candidates, all structurally near-zero. If ANY resolves YES, it's a major political disruption. The cluster provides a sanity check: if the sum of all three probabilities (~1.1%) seems too high, consider correlated collapse scenarios
4. **Non-Israel complement**: Haley (USA) and Mekonnen (Ethiopia) extend the pattern outside Israel's hyper-fragmented politics, confirming it's a universal forecasting heuristic

### Calibration Value
- These 5 runs will provide a pooled calibration signal: if all 5 resolve NO (the structural prediction), the concept's framework is validated
- If any resolve YES, it's a structural failure requiring concept revision — but the nature of the YES (which candidate, which gates opened) will teach which failure modes the concept missed

## Pattern Mechanics

1. **Gate identification**: List all sequential events required for the individual to become leader
2. **Probability per gate**: Estimate each independently (base rates, institutional constraints, polling)
3. **Multiplication**: Multiply all gate probabilities (assuming independence)
4. **Correlation adjustment**: Adjust upward if gates are positively correlated (e.g., a national crisis that simultaneously weakens the incumbent and boosts the opposition)
5. **Market comparison**: Compare compounded product to market price; divergences indicate either market mispricing or missing gates

## When to Apply

- Any "will X become leader of Y" forecast
- Especially useful when X is not the obvious successor (deputy, heir apparent, frontrunner)
- Critical when the chain has 4+ gates — intuition systematically overestimates products
- Most applicable in parliamentary systems (coalition required) and authoritarian systems (controlled succession)

## Three Subtypes

1. **Outsider succession** (Golan type): Candidate from minor party/opposition. Requires party breakthrough + coalition magic. Typical range: 0.001–0.01
2. **Internal succession** (Ohana type): Candidate from dominant party but not heir apparent. Requires party retaining power + incumbent departing + candidate out-competing rivals. Typical range: 0.001–0.05
3. **Heir apparent succession**: Deputy/deputy leader succeeds predictable departure. Fewer gates (2-3). Typical range: 0.1–0.5
4. **Collapsed-base resurgence** (Shaked type): Former leader with no current parliamentary seat or party infrastructure. Requires rebuilding from zero. Typical range: 0.001–0.005
5. **Ex-faction outsider** (Mekonnen type): Former insider who left the dominant coalition. Requires defection cascade or external shock. Typical range: 0.005–0.02

## Relationship to Other Concepts

| Concept | Relationship |
|---------|-------------|
| [[domains/global/concepts/structural-improbability-check/_concept]] | Broader method for estimating near-zero probabilities across any domain |
| **Leader-succession-compounding-probability** | Specific application to political succession chains |
| [[domains/global/concepts/short-horizon-momentum-check/_concept]] | For short-window succession scenarios |

## Wikilinks
- [[forecasts/2026-06-16-yair-golan-pm-israel]]
- [[forecasts/2026-06-15-amir-ohana-pm]]
- [[runs/20260615-shaked-israel-pm]]
- [[runs/20260615-haley-2028-nomination]]
- [[runs/20260615-mekonnen-ethiopia]]
- [[domains/mena/entities/yair-golan]]
- [[domains/mena/entities/amir-ohana]]
- [[domains/mena/entities/ayelet-shaked]]
- [[domains/usa/entities/nikki-haley]]
- [[domains/africa/entities/demeke-mekonnen]]
- [[domains/mena/threads/israeli-domestic-politics/_thread]]
- [[domains/global/concepts/structural-improbability-check/_concept]]
