## Report: Q17 Reflection (Fed July 2024 Rate Decision)

**Prediction: NO → Correct**

### Diagnosis

This was correct but mostly a "freebie" — the vault contributed ~30% signal. The 2024-Q3 timeline documented the July hold and September 50bp cut *post-hoc*, and 2024-Q1 had macro context (rates steady at 5.25-5.50%, cuts expected "later in 2024"). But the vault lacked the key analytical framework that makes Fed rate decisions forecastable: the forward guidance pipeline.

**Why the prediction worked (general knowledge, not vault):**
- The June 2024 dot plot shifted from 3 cuts to 1 cut projected for 2024
- Powell's June press conference said "not confident enough to cut yet"
- Markets understood July as a "skip" meeting and September as "live"
- The Fed telegraphs rate moves 1-2 meetings in advance — this pattern was in my training data but not in the vault

**Vault structural gaps identified:**
1. No entity files for the Federal Reserve, Jerome Powell, or FOMC (despite being among the most consequential actors in global macro)
2. No concept file for central bank forward guidance (the pattern that makes rate decisions forecastable)
3. No thread file for the post-COVID monetary policy cycle (Fed entries scattered across 7 quarter files)
4. No spec principle mandating monetary policy coverage (unlike US budget dynamics and legal proceedings which have principles)
5. No procedure steps for assessing rate decision questions in the pre-forecast audit

### Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `concepts/central-bank-forward-guidance.md` | CREATE | Core concept: how Fed telegraphs rate moves through statement, dot plot, press conference |
| `entities/federal-reserve-system.md` | CREATE | Entity stub for US central bank |
| `entities/jerome-powell.md` | CREATE | Entity stub for Fed Chair |
| `entities/federal-open-market-committee.md` | CREATE | Entity stub for FOMC |
| `threads/us-monetary-policy-cycle-2022-2026.md` | CREATE | Unified thread tracking hiking→hold→cutting cycle across 2022-2026 |
| `forecasts/2026-05-18-fed-july-2024-rate.md` | CREATE | Forecast entry documenting this question |
| `_spec.md` | UPDATE | Added principle #14 mandating central bank/monetary policy coverage |
| `_procedure.md` | UPDATE | Added step #16 to Pre-Forecast Audit for central bank rate decision assessment |
| `timeline/2024-Q3.md` | UPDATE | Added wikilinks to new entity files in Fed section + cross-domain threads |
| `timeline/2024-Q1.md` | UPDATE | Added wikilink to Federal Reserve entity |

### Key lesson

The Federal Reserve is arguably the single most important institution for global macro forecasting, yet had zero vault coverage until now. The Fed's structured forward guidance makes its decisions *more* forecastable than most geopolitical questions (where signals are noisy and intentions unreliable). This is an asymmetric opportunity — rate questions have a defined methodology that can produce consistently above-chance forecasts. The new concept file and procedure step formalize this.