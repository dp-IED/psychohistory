---
type: reflection
tags: [meta, curation]
date: 2026-05-22
trigger: "Graph vault reflection after recent forecast runs — curation pass 11"
---

# Graph Vault Curation — May 22, 2026 (Pass 11)

## Trigger

Standard curation sweep after pit blind test forecast (Cloobeck CA Governor, fc-20260522-001) and review of vault integration gaps.

## Vault State Before

- 26 run files (25 in runs index, 1 pit blind test orphan)
- 22 event files
- 306 entities + 2 broken wikilinks (katie-porter, chad-bianco referenced but nonexistent)
- No formal event for California 2026 gubernatorial election ($24M Polymarket domain)

## Actions Taken

### 1. Fixed 2 Broken Wikilinks

| File Created | References From | Rationale |
|-------------|-----------------|-----------|
| `domains/usa/entities/katie-porter.md` | CA governor thread wikilinks, active 2024 Senate candidate with progressive base and possible 2026 CA governor entry | Porter's 0.6% PM win odds and progressive lane matter for fragmented-Democratic-field analysis |
| `domains/usa/entities/chad-bianco.md` | CA governor thread wikilinks, Steve Hilton entity | Bianco (1.75% PM) is the right-flank Republican alternative; his primary advancement viability affects Hilton's consolidation ceiling |

### 2. Created Event for California 2026 Gubernatorial Election

Created `events/california-2026-gubernatorial-election.md` — formal event file capturing the $24M Polymarket election with structural context, candidates table, key dates, and vault cross-references. The event file bridges the detailed thread analysis (candidate-level) with broader forecasting questions.

### 3. Integrated Orphan Pit Blind Test Forecast

The Cloobeck CA Governor forecast (p_yes=0.002) existed only as a JSON file in `pit_blind_test/forecasts/` with no vault-managed run file. Created `runs/20260522-002-cloobeck-california-governor.md` and added it as entry #26 in `runs/_index.md`.

Key analytical contribution: The Cloobeck run is the vault's first **vault-market convergence** case on a structural near-zero question (vault 0.002 vs PM 0.0015). This contrasts with the Raúl Castro divergence case (0.005 vs 0.1235) and demonstrates that the market can correctly price structural near-zero outcomes when liquidity is adequate ($994K vs $99K).

### 4. Enriched 2026-Q2 Timeline

- Added `[[events/california-2026-gubernatorial-election]]` to Wikilinks Created section
- Added Cloobeck run to Q2-linked active forecasts table with convergence analysis

## Vault State After

| Metric | Before | After |
|--------|--------|-------|
| Entities | 306 | 308 (+2) |
| Events | 22 | 23 (+1) |
| Run files (vault-managed) | 23 | 24 (+1) |
| Runs index entries | 25 | 26 (+1) |
| Broken wikilinks in CA domain | 2 | 0 (fixed) |
| Version | 2.3 | 2.4 |

## Decisions

### Why create Katie Porter entity instead of removing the wikilink?

Could have removed the broken wikilink from the CA governor thread (Porter is only 0.6% in Polymarket). Decided to create because: (1) Porter is a real political figure with forecasting-relevant base, (2) her decision to enter or not enter the CA race is a structural uncertainty, (3) the CA governor thread already contained substantive analysis of her candidacy — the entity stub formalizes that analysis rather than deleting it.

### Why an event file for CA governor when a thread already exists?

The thread provides live-candidate-level analysis. The event file provides structural context, key dates, market structure, and platform-agnostic framing. Having both is not redundant — the thread tracks the race as it evolves; the event is a referenceable node for cross-linking from timeline entries and other events.

### Why flag the Cloobeck convergence as analytically significant?

The Raúl Castro divergence (vault 0.005 vs market 0.1235) and Cloobeck convergence (vault 0.002 vs market 0.0015) form a paired calibration signal. They demonstrate that the structural-improbability-check mechanism is NOT systematically biased toward underestimating market prices — it converges when the market is liquid and informed, and diverges when the market is thin or confused. This distinction is itself a useful forecasting insight.

## Lessons for Future Curation

1. **Check ALL wikilinks in modified files**: The CA governor thread referenced `katie-porter` and `chad-bianco` in its wikilinks section — these were created by the pit blind test forecast but never verified until this curation pass. Any file creation during a forecast run should trigger a wikilink integrity check.

2. **Pit blind test forecasts need explicit vault integration**: The `pit_blind_test/` directory is outside the vault's awareness. Forecasts created there create vault files but leave no trace in the runs index. The curation workflow should include scanning `pit_blind_test/forecasts/` for unvaulted runs.

3. **The _macro_gaps.md is stabilizing**: This pass identified zero new open gaps — the first time in the vault's history. Previous passes consistently found 2-5 new gaps. This may indicate the vault is approaching coverage saturation for the current forecasting domains.
