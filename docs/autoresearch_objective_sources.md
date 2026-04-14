# Autoresearch Objective Sources

## Purpose

The next autoresearch phase should optimize against a stricter objective surface, not just saturated structural, functional, and constraint gates. The intended long-term composite is:

```text
S = 0.15V + 0.20G + 0.20I + 0.30F + 0.15C
```

Where:

- `V`: temporal validity. Entities, events, and claims appear in valid time windows.
- `G`: grounding coverage. Graph nodes are anchored to stable external identifiers.
- `I`: ideology and narrative retrieval. Claims, frames, counterclaims, and ideological transmission remain queryable.
- `F`: forecast skill. Graph-derived signals improve Brier score or log loss against forecast baselines.
- `C`: counterweight sensitivity. Removing opposition, protest, conflict, repression, or counterweight layers degrades relevant tasks.

## Source Roles

- Wikidata / Wikipedia: entity spine for actors, places, organizations, events, and historical chains. This should be the first real external grounding adapter because it benefits all probe families.
- Seshat Global History Databank: slow-moving historical and social-complexity variables such as institutions, governance, religion, inequality, polity scale, and continuity. This is the best source for long-run civilization probes.
- GDELT: broad, high-volume modern event stream from news, useful for heterogeneous actor-action-target coverage in contemporary probes.
- ICEWS: CAMEO-coded socio-political event stream, useful for diplomacy, repression, protest, and crisis interaction probes where access and coverage are adequate.
- ACLED: conflict and political violence events with local precision, useful for conflict, protest-to-conflict, and sparse local detail.
- Polymarket: resolved forecast market probabilities and price history, useful for backtesting graph-derived forecasts with Brier score or log loss.

## Probe Family Mapping

| Probe family | Primary sources |
| --- | --- |
| Long-run civilization, such as Roman or Abbasid probes | Seshat + Wikidata |
| Financial ideology, such as 2008 crisis or eurozone probes | Wikidata + GDELT |
| Electoral realignment, such as Brexit or US election probes | Wikidata + GDELT/ICEWS + Polymarket |
| Protest to conflict, such as Arab Spring or Sudan probes | ICEWS + ACLED |
| Policy under uncertainty, such as COVID or inflation probes | GDELT + Polymarket |
| Forecast backtesting | Polymarket |

## Implementation Order

1. Add strict objective fields to the pilot probes: `golden_tasks`, `designated_ablations`, and `complexity_budget`.
2. Make strict eval pass on the pilot subset using synthetic or precomputed graph artifacts first.
3. Add the `V/G/I/F/C` report buckets, with explicit stubs where adapters are not implemented yet.
4. Implement `G` first using Wikidata grounding coverage.
5. Implement `V` next for historical probes using Seshat and Wikidata snapshots.
6. Add event-stream adapters after entity grounding exists: GDELT for broad modern coverage, ACLED for conflict, and ICEWS where access and coverage are suitable.
7. Add `F` with Polymarket only after market mappings, resolution dates, graph cutoff dates, and entity mappings are explicit.

## Autoresearch Timing

Do not run another broad schema-search phase before the strict-objective pilot exists. Current gates are mostly saturated, so additional schema search is likely to produce metadata churn rather than meaningful improvements.

The next useful autoresearch phase is immediately after the strict pilot passes. It should run against the pilot subset and accept candidates only when they improve traversal, discipline, ablation behavior, or failure diagnostics without broadening the schema unnecessarily.

## Expected End State Of The Pilot

- Strict eval passes on the pilot probes.
- The eval report includes `V/G/I/F/C`.
- `G` produces real signal from Wikidata coverage.
- `V`, `I`, `F`, and `C` are present with clear stub or unavailable status until their adapters are implemented.
- The remaining source adapters have documented artifact contracts and can be implemented incrementally.

## Wikidata Grounding Contract

Graph artifacts can now contribute to the `G` bucket by attaching Wikidata QIDs
to graph nodes. The evaluator recognizes QIDs in common fields such as
`wikidata_id`, `wikidata_qid`, `external_ids.wikidata`, `identifiers`,
`same_as`, `provenance`, `references`, or canonical node IDs such as `wd:Q42`
and `https://www.wikidata.org/wiki/Q42`.

The Wikidata score reports both graph-node grounding coverage and
`seed_entities` label coverage. A seed entity counts as linked when a grounded
graph node exposes the same normalized `label`, `name`, `title`,
`canonical_label`, `aliases`, or `alt_labels` value.

To populate these fields from Wikidata during artifact refresh, run eval with:

```bash
python -m evals.run_eval \
  --schema schemas.base_schema \
  --probe-dir probes \
  --probe-id roman_family_1A \
  --wikidata-enrich-graph-artifacts
```

The enrichment step uses Wikidata's `wbsearchentities` Action API, writes QIDs
into `external_ids.wikidata`, and stores a local ignored search cache at
`cache/wikidata_search_cache.json` by default. Override it with
`--wikidata-cache-path` when running isolated experiments.

When enrichment produces a non-zero `G` score, the eval composite includes a
`source_G` component. This is intentionally narrow while the remaining probes
are still being annotated for strict objective mode and seed-entity coverage.
First-pass Wikidata lookups are serialized with retry/backoff. Unresolved or
rate-limited labels are cached as negative entries for a short TTL so repeated
runs avoid hammering the API without treating failures as permanent.
