# Point-in-Time Wikidata Grounding — Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add point-in-time Wikidata grounding so Actor and Location nodes carry stable canonical QIDs verifiable as of each snapshot origin, unifying identity across GDELT and ACLED after source-layer experiments.

**Architecture:** New `evals/wikidata_grounding.py` orchestrates `search_wikidata_entity` from `evals/wikidata_linking.py`, persists a shared cache, and optionally plugs into `build_snapshot_payload()` after nodes are assembled. Grounding is opt-in via `grounding_cache` / `--grounding-cache` so existing runs stay reproducible. MVP uses API search + frozen cache; later phases use weekly dump slices for true point-in-time reproducibility.

**Tech Stack:** Wikidata `wbsearchentities` API, JSON cache, optional DuckDB for dump slices later; integration in `ingest/snapshot_export.py` and `baselines/source_experiments.py`.

---

**Date:** 2026-04-17  
**Roadmap:** Stage 4 — expand graph forecaster; see `docs/2026-04-16-france-gdelt-benchmark-note.md` step 4, `forecast_charter.md`.

## File Map

| Path | Role |
|------|------|
| `evals/wikidata_linking.py` | Extend search with country hint; reuse cache/TTL |
| `evals/wikidata_grounding.py` | **New:** `ground_snapshot_nodes()`, stats, optional `__main__` |
| `ingest/snapshot_export.py` | Optional `grounding_cache` on `build_snapshot_payload()` |
| `baselines/source_experiments.py` | Thread cache; audit `wikidata_grounding` |
| `baselines/backtest.py` | `--grounding-cache` on `source-experiments` |
| `tests/test_wikidata_grounding.py` | **New** |
| `tests/test_snapshot_export.py`, `tests/test_source_experiments.py` | Extend |

## MVP vs Later

**MVP:** API search + persistent cache; `grounded_at_utc` and `method: search_api` per entry; rate limiting; partial-resolution disclosure in audit.

**Later:** Weekly Wikidata JSON dump slices in DuckDB; `dump_date` per entry; SPARQL fallback; cross-origin QID consistency checks; GNN actor merge by shared QID (separate ablation).

## Risks

- **Rate limits:** throttle, batch runs, cache before experiments.
- **Reproducibility:** live API drifts; freeze committed cache for benchmarks.
- **Partial grounding:** disclose unresolved labels (analogous to GDELT partial fetch in benchmark note).
- **Disambiguation:** short GDELT codes may not resolve; country hints; do not merge on low confidence until validated.

## Tasks (checkboxes)

### Phase 1 — Core module

- [ ] Read `evals/wikidata_linking.py` and confirm reuse (`search_wikidata_entity`, negative cache, `wikidata_qid_for_node`).
- [ ] Add `evals/wikidata_grounding.py` with `ground_snapshot_nodes(nodes, *, cache_path, entity_types, request_delay_s) -> (nodes, stats)`.
- [ ] Optional: extend `search_wikidata_entity` with country-code hint for disambiguation.
- [ ] Add `tests/test_wikidata_grounding.py` (monkeypatch API): Actor/Location QIDs, skip existing QIDs, skip other types, stats keys, negative cache TTL.
- [ ] Commit: `add wikidata grounding module and unit tests`

### Phase 2 — Snapshot export

- [ ] Extend `build_snapshot_payload(..., grounding_cache: Path | None = None)`; call grounding after node assembly; add `metadata.wikidata_grounding`.
- [ ] Extend snapshot tests; ensure `GraphArtifactV1` still validates (`docs/gnn_graph_artifact_contract.md`).
- [ ] Commit: `extend snapshot_export for optional wikidata grounding`

### Phase 3 — Source experiments

- [ ] Thread `grounding_cache` through `run_source_layer_experiments` → `_build_origin_inputs`.
- [ ] Add `--grounding-cache` to `source-experiments` CLI (default off).
- [ ] Extend `tests/test_source_experiments.py` for audit keys.
- [ ] Commit: `wire grounding into source-experiments`

### Phase 4 — CLI audit

- [ ] Optional `__main__` on `wikidata_grounding.py` for one-shot pass from warehouse labels.
- [ ] Document cache location / partial disclosure in run notes.
- [ ] Commit: `add grounding CLI and audit trail`

### Phase 5 — Validation

- [ ] Run `gdelt_only` with `--grounding-cache`; check no serialization regression; Brier neutral until GNN uses QIDs.
- [ ] Update benchmark note with grounded counts and unresolved disclosure.

### Phase 6 (later) — Dumps + GNN merge

- [ ] Ingest dump slice → DuckDB; prefer dump lookup; SPARQL fallback; QID drift checks; GNN `same_as` / merge ablation.

## Verification

- [ ] `pytest tests/test_wikidata_grounding.py tests/test_snapshot_export.py tests/test_contracts.py tests/test_source_experiments.py`
- [ ] Grounded snapshots include `metadata.wikidata_grounding` with resolved/attempted counts.
