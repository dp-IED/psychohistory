# GNN Graph Artifact Contract

Graph builders should emit `graph_artifact_v1` JSON. This is the boundary
between probe-specific graph construction and downstream GNN projection/training.

## Top-Level Shape

Required:

- `artifact_format`: `graph_artifact_v1`
- `probe_id`: stable probe identifier
- `schema_version`: graph IR schema version used by the builder
- `nodes`: list of node records
- `edges`: list of edge records

Recommended for training:

- `task_labels`: supervised path/task labels
- `target_table`: slice-level or node-level targets
- `metadata`: builder and provenance metadata

## Node Record

Required:

- `id`: stable artifact-local node ID
- `type`: schema node type

Recommended:

- `layer`: semantic layer such as `material`, `institutional`, `epistemic`, or `persistence`
- `label`: human-readable label
- `external_ids.wikidata`: Wikidata QID when available
- `time.start`, `time.end`, `time.granularity`
- `slice_ids`: one or more train/eval slice IDs
- `train_eval_split`: `train`, `validation`, `test`, or `eval`
- `provenance.sources`: source identifiers or URLs
- `attributes`: builder-specific structured fields

## Edge Record

Required:

- `source`: node ID
- `target`: node ID
- `type`: schema edge type

Recommended:

- `confidence`: float in `[0, 1]`
- `provenance.sources`: source identifiers or URLs
- `time.start`, `time.end`, `time.granularity`
- `slice_ids`: one or more train/eval slice IDs
- `train_eval_split`: `train`, `validation`, `test`, or `eval`
- `task_ids`: golden-task IDs this edge supports
- `attributes`: builder-specific structured fields

## Training Labels

Use `task_labels` for path-level or retrieval labels:

- `task_id`
- `label`
- `node_ids`
- `edge_indices`
- `split`

Use `target_table` for downstream supervised targets:

- `target_id`
- `name`
- `value`
- `split`
- `slice_id`
- `node_ids`
- `metadata`

## Current Builder Behavior

The built-in graph precompute command emits `graph_artifact_v1` and keeps legacy
`nodes` / `edges` fields compatible with existing evals. It creates seed-entity
nodes first, uses them in golden-task paths when types match, and Wikidata
enrichment attaches QIDs to `external_ids.wikidata`.
