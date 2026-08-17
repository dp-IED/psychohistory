---
type: procedure
tags: [procedure]
title: "Central Bank Rate Decision"
domain: "[[domains/economics]]"
concept: 
  - "[[domains/economics/concepts/central-bank-forward-guidance]]"
  - "[[domains/mena/concepts/em-central-bank-credibility-normalization]]"
functions:
  - "[[functions/run_structured().md|run_structured]]"
  - "[[functions/pit_search().md|pit_search]]"
---
---
---

# Central Bank Rate Decision

Estimate whether a central bank will change interest rates at a specific meeting or within a given timeframe. Supports both advanced-economy central banks (Fed, ECB, BoE, BoJ) and EM central banks (TCMB, RBI, BCB, CBR, SARB, etc.).

## When

Questions about rate hikes, cuts, or holds at specific FOMC/ECB/BoJ/TCMB meetings, or within a given timeframe.

## Approach

### For All Advanced-Economy Central Banks

1. **Identify the central bank and load its thread**: 
   - **Fed**: Load [[domains/economics/threads/us-monetary-policy-cycle-2022-2026]]
   - **ECB**: Load [[domains/economics/threads/ecb-monetary-policy-cycle-2022-current]]
   - **BoE/BoJ**: Check whether a dedicated thread exists; if not, search temporal quarter files for recent policy actions
2. **Check forward guidance**: Central bank statement language, press conference signals, meeting minutes. For the Fed, check the dot plot and Powell's press conference. For the ECB, check Lagarde's press conference language and the Governing Council consensus tone.
3. **Identify the monetary policy cycle phase**: Apply [[domains/economics/concepts/monetary-policy-cycle-phases]] to determine structural baseline (tightening, plateau, easing).
4. **Check whether easing-cycle sequential momentum applies**: If the central bank is in an active easing cycle (2+ cuts already delivered), load [[domains/economics/concepts/easing-cycle-sequential-momentum]] — the default expectation is continued cuts at the next 1-2 meetings absent a data shock or guidance shift.
5. **Read market pricing**: CME FedWatch (Fed), €STR / OIS rates (ECB), SONIA futures (BoE), OIS rates (BoJ).
6. **Apply [[domains/economics/concepts/central-bank-forward-guidance]]** for telegraphing analysis.
7. **Check the meeting calendar**: Verify the meeting is actually scheduled. Central banks publish their full-year meeting calendars in advance. For ECB, meetings are approximately every 6 weeks; for Fed, 8 fixed meetings per year.
8. **Calibrate**: Apply the calibration heuristic from [[domains/economics/concepts/easing-cycle-sequential-momentum]] if applicable, or the general calibration below.

### For EM Central Banks (TCMB, RBI, BCB, CBR, SARB, etc.)

EM central banks operate under different constraints than advanced-economy central banks. Apply additional considerations:

1. **Identify the political context**: Is the central bank independent or politically constrained? Check for recent governor turnover, presidential pressure, or finance minister changes.

2. **Read the entity stub**: Each EM central bank should have a domain-specific entity stub documenting its governance structure, political context, and historical policy patterns. See [[domains/mena/entities/turkish-central-bank-tcmb]] as a template.

3. **Determine the regime phase**: Is the central bank in:
   - **Normal tightening/easing cycle** — responding to inflation like a conventional EM central bank
   - **Credibility normalization** — reversing a period of unorthodox policy (see [[domains/mena/concepts/em-central-bank-credibility-normalization]])
   - **Political capture** — rates set based on political preferences rather than inflation

4. **Check the most recent policy statement**: EM central banks typically release shorter statements than the Fed. Key phrases to parse:
   - "Decisive tightening" = more hikes coming
   - "Maintain tight stance" = hold likely
   - "Data dependent" = ambiguous, check inflation dynamics
   - "Careful monetary policy" = approaching a pivot

5. **Track the political shield**: For EM central banks in normalization (like TCMB), identify the senior official providing political cover (e.g., Finance Minister Simsek). The normalization remains credible only as long as this shield is in place.

6. **Monitor currency and reserves**: EM rate decisions are heavily influenced by currency depreciation and FX reserve adequacy. A 5%+ monthly depreciation forces action. Depleted reserves reduce policy space.

7. **Check market pricing**: Use EM-specific market indicators where available (5-year CDS, NDF rates, local currency bond yields). For Turkey, check USD/TRY NDFs and CDS.

8. **Apply the appropriate concept framework**:
   - [[domains/economics/concepts/central-bank-forward-guidance]] for advanced-economy central banks
   - [[domains/mena/concepts/em-central-bank-credibility-normalization]] for EM central banks in normalization phases

## Calibration

### Advanced Economy
- Market-implied P < 40% one week before meeting → almost certainly no move
- Market-implied P > 80% → almost certainly moves (direction, not magnitude)
- Chair telegraphing a move → P(move) > 0.9
- No telegraphing + market-implied ~50% → ambiguous, deeper analysis needed
- First cut of a new cycle → often larger than expected (e.g., 50bp vs 25bp)

### EM Central Banks
- Inflation accelerating + above target → P(hike) > 0.7 (but dependent on political context)
- Inflation declining + expectations converging → P(hold) > 0.6 if credibility is intact
- Political shield removed (finance minister fired / governor replaced) → P(policy reversal) > 0.8 within 6 months
- Currency depreciating 5%+ monthly → P(emergency hike) > 0.5 within 2 weeks
- Positive real rates achieved → central bank typically holds until real rates decline through inflation
