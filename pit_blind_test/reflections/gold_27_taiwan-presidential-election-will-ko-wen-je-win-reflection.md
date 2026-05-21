## Report: Q27 Reflection — Ko Wen-je Taiwan Election

Prediction was correct (NO), and the vault contributed ~80% of the reasoning via the existing third-party-ceiling-fptp and divided-opposition-plurality-win concepts. However, three gaps prevented the vault from contributing 100%.

### Diagnosis

The vault had strong structural coverage but missed **amplifying dynamics** — factors that don't change the binary outcome but improve probability calibration and provide deeper explanatory power:

1. **Cross-strait thread was structurally broken** — marked `status: fading` with a deprecation note, making it look abandoned. But it contains essential analysis of the "China threat effect" that amplifies DPP's electoral advantage. A forecaster loading it would see a dead file and potentially miss the cross-reference.

2. **No formalized concept for external-threat-incumbency-boost** — The third-party-ceiling concept explains why Ko couldn't win. The divided-opposition concept explains why Lai won. But neither explains *why the DPP's margin was 40.05% rather than a bare plurality of ~36%*. China's pressure campaign against Lai is a known amplifier, but it wasn't abstracted into a reusable forecasting tool.

3. **Late-campaign collapse was under-documented** — The third-party-ceiling concept mentioned "late-campaign collapse effect" in one sentence. The collapse trajectory (peak at 6+ months out, accelerated collapse in final 4 weeks) is a predictable, non-linear pattern that should be formalized for calibration.

### Files Changed

**Updated — `domains/east-asia/threads/taiwan-cross-strait-relations/_thread.md`**
- Status changed from `fading` to `active`. Removed the deprecation note that told readers the file was stale.
- Added "Relationship to Taiwan Presidential Election Thread" section with an integration table showing how the two threads are complementary (tension level vs fragmentation) and must be loaded together.
- Added wikilink to new external-threat-incumbency-boost concept.

**Created — `domains/global/concepts/external-threat-incumbency-boost/_concept.md`** (13KB)
- New cross-domain concept: external threats from an adversary boost the incumbent party's electoral standing through a *partisan advantage* channel, distinct from the temporary rally-around-the-flag effect.
- 5 canonical examples: Taiwan (PRC pressure boosts DPP), Georgia (Russia boosts nationalism), Israel (rocket attacks boost right-wing), Pakistan (India tensions boost military-aligned gov), US 2020 as negative counterexample.
- 4 mechanisms: issue salience shift, incumbent-as-defender framing, opposition delegitimization, nationalist mobilization.
- 7-step forecasting application with calibrated magnitude table (strong 3-8pp, moderate 1-3pp, neutral/reversed).

**Updated — `domains/east-asia/concepts/third-party-ceiling-fptp/_concept.md`**
- Expanded late-campaign collapse pattern from 1 sentence to a 4-phase trajectory model (6+ months out, 2-4 months out, 0-4 weeks out, election day).
- Added 3 canonical data points: Ko Wen-je (~30% to 26.46%), James Soong (42% to 36.84%), Ross Perot (39% to 18.9%).
- Added forecasting implication: assume 3-8 point collapse from peak, non-linear, concentrated in final 4 weeks.
- Updated Validated By entry to reference the collapse pattern.

**Updated — `_procedure.md`**
- Added step 8c to the Pre-Forecast Audit: "Check for external threat/interference effects on elections" with 7 sub-steps (identify adversary, assess partisan alignment, check timing/blowback/economic pain, apply concept, document explicitly).
- Renamed old step 8c (shutdown) to 8d to maintain sequential numbering.

**Updated — `meta/reflections/_reflection-2026-05-20-per-q27-v2.md`**
- Updated to v3 with new score trend line (90% amplified) and remediation table.
- Added insight: progression across cycles = coverage -> structure -> amplification. After ensuring structural coverage, check for amplifying dynamics that affect the probability window.

### What Changed Strategically

The vault went from having strong structural mechanics (3-way FPTP -> opposition fragmentation -> DPP wins) to also having the amplifying dynamics (external pressure -> incumbent boost, late-campaign collapse trajectory). These don't change the binary prediction (still NO) but improve probability calibration and give the next forecaster more precise tools for reasoning about margins, timing, and secondary variables.