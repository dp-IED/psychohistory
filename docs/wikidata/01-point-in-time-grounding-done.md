# Plan: Point-in-time Wikidata grounding (reproducibility)

**Goal:** Grounding resolves surface strings to Wikidata QIDs **without leaking future Wikidata facts** and with **reproducible** artifacts for benchmark runs. This plan does **not** require the GNN to consume QIDs; it hardens the graph IR and audits.

**Prerequisite:** Current pipeline: [`evals/wikidata_grounding.py`](../../evals/wikidata_grounding.py), [`evals/wikidata_linking.py`](../../evals/wikidata_linking.py), optional `--grounding-cache` on [`baselines/backtest`](../../baselines/backtest.py) `source-experiments`.

---

## Problem statement

- **Live API** (`wbsearchentities`) is non-deterministic over time: merges, redirects, and ranking changes shift QIDs for the same label.
- **“Point-in-time”** for forecasting means: features available **as of** snapshot origin `t` must not depend on Wikidata edits after `t` (charter-aligned).

Two layers:

1. **Operational reproducibility:** Same frozen cache file → same QIDs in snapshots.
2. **Semantic point-in-time:** QIDs and labels resolved from a **snapshot of Wikidata** valid for date `t` (dump slice or query with date filter), not “whatever the API returns today.”

---

## Design options

### A. Frozen API cache (current MVP)

- JSON cache keyed by `(node_type, normalized_label)`; negative cache with TTL.
- **Reproducibility:** Commit or archive `wikidata_grounding_cache.json` per benchmark run; document `grounded_at_utc` wall time.

### B. Weekly dump slices (target state)

- Ingest a **Wikidata JSON dump** (or entity subset) into DuckDB or sqlite; index by label / alias / QID.
- For origin `t`, select the **latest available dump with `dump_date <= t`** from a pre-materialized dump manifest.
- **Reproducibility:** Run is fully determined by `dump_id` + query code.

### C. Hybrid

- Resolve from **dump first**; API only for misses; cache misses with explicit `method: dump_slice | search_api`.

---

## Ablations (benchmark design)

These are **not** model ablations; they are **grounding-method** comparisons on the **same** forecast task and model. Hold model and training fixed; only change grounding source.

| Run ID | Grounding source | What varies |
|--------|------------------|-------------|
| **G0** | None | Baseline: no `external_ids.wikidata` on nodes |
| **G1** | Frozen API cache (single committed file) | Same cache across all origins |
| **G2** | Fresh API (no cache) | Stochastic; use only for sensitivity analysis, not headline numbers |
| **G3** | Dump slice v1 (date D1) | PIT-aligned |
| **G4** | Dump slice v2 (date D2) | Sensitivity to dump date |

**Success criteria (grounding layer):**

- **Stability:** G1 vs G2 on a **small** eval window: distribution of QIDs per label (not forecast metrics); expect high agreement for G1.
- **PIT audit:** For G3/G4, log `dump_date` and entity revision ids in snapshot `metadata.wikidata_grounding`.
- **Disclosure:** `resolved` / `attempted` / unresolved labels reported (analogous to partial GDELT fetch in the benchmark note).

**Forecast metrics:** Expect **no change** in GNN metrics until [`02-qid-input-features.md`](02-qid-input-features.md) lands; if G0 vs G1 differ materially, that indicates a **bug** (e.g. grounding mutates non-QID fields).

---

## Implementation sketch (high level)

1. **Dump ingest module** (new): download, decompress, stream entities; table `(dump_id, qid, label_en, aliases, …)`.
2. **Resolver API:** `resolve_qid(label, as_of_date, country_hint) -> (qid, confidence, method)`.
3. **Wire** into [`apply_wikidata_grounding`](../../evals/wikidata_grounding.py) with `method` in cache entries.
4. **Persist mapping:** write `origin_date -> dump_id` selection to run metadata and artifact bundle.
5. **Tests:** golden-file tests for a tiny fixture dump and fixed `as_of_date`.

---

## Risks

- Dump size and ingest time; start with **France-relevant** entity subset (admin regions, frequent actor labels).
- API rate limits during hybrid fallback.

---

## Dependencies

- None for G0/G1 (existing code).
- **02** depends on stable QIDs from **01** for meaningful embedding tables.
