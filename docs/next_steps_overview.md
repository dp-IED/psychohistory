# Next Steps Overview

## Goal

Move from a strict eval harness with synthetic artifacts toward a grounded graph
representation that can support first-pass GNN training.

## Immediate Sequence

1. Stabilize `graph_artifact_v1`.
2. Export artifact tables for GNN ingestion.
3. Train a minimal baseline on strict-pilot graph artifacts.
4. Use baseline failures to refine graph representation, not to broaden schema
   vocabulary prematurely.
5. Expand strict-objective probe coverage only after the export/training loop is
   working on the pilot set.

## Current Useful Pilot Set

- `roman_family_1A`
- `roman_family_1B`
- `6C-3`
- `probe-8a-ukraine-war-sanctions-external-arming-sovereignty-full-scale-invasion`
- `probe-8b-sudan-war-fragmented-sovereignty-regionalized-armed-competition`
- `probe-7a-protestant-reformation-print-publics-confessional-state-formation`

## Acceptance Checks

- `pytest -q` passes.
- Strict pilot eval has `failure_class: null`.
- `source_G` is non-zero with `--wikidata-enrich-graph-artifacts`.
- Every emitted artifact validates as `graph_artifact_v1`.
- Exported training tables contain no dangling edge references.

## Do Not Do Yet

- Do not expand all probes before the GNN export path exists.
- Do not optimize for only higher composite score if `task_path_grounding_rate`
  worsens.
- Do not train a full forecasting model before edge reconstruction and
  path-present baselines are working.
