# GNN Export Next Steps

## Purpose

Convert `graph_artifact_v1` JSON into stable training inputs. This should be a
separate exporter, not baked into eval scoring.

## Export Files

Recommended first format: JSONL. Parquet can come later once the row contract is
stable.

- `nodes.jsonl`
- `edges.jsonl`
- `task_labels.jsonl`
- `target_table.jsonl`
- `node_type_vocab.json`
- `edge_type_vocab.json`
- `layer_vocab.json`
- `manifest.json`

## Node Rows

Minimum row fields:

- `global_node_id`: `{probe_id}:{node.id}`
- `probe_id`
- `node_id`
- `type_id`
- `type`
- `layer_id`
- `layer`
- `label`
- `wikidata_qid`
- `time_start`
- `time_end`
- `slice_ids`
- `split`
- `attributes`

## Edge Rows

Minimum row fields:

- `global_edge_id`: `{probe_id}:edge:{index}`
- `probe_id`
- `source_global_node_id`
- `target_global_node_id`
- `edge_type_id`
- `edge_type`
- `confidence`
- `time_start`
- `time_end`
- `slice_ids`
- `split`
- `task_ids`
- `attributes`

## Baseline Features

Start with structural features only:

- node type ID
- layer ID
- Wikidata-grounded flag
- time-start/time-end normalized where available
- seed-entity flag
- task-path participation flag

Do not add text embeddings until the table contract and baseline losses are
stable.

## First Baselines

1. Masked edge reconstruction:
   - hide a fraction of edges
   - predict edge existence/type between candidate node pairs
2. Path-present prediction:
   - use `target_table.name == "path_present"`
   - predict whether a golden-task path exists from node/edge features
3. Ablation sensitivity:
   - remove persistence/epistemic/counterweight edges and verify target quality
     degrades for relevant tasks

## Acceptance Checks

- Exporter is deterministic.
- Vocab files are stable under repeated runs on the same artifact set.
- Edge rows reference existing node rows.
- Splits are explicit; no row silently defaults to train.
- A trivial baseline can overfit a tiny pilot subset before generalization is
  attempted.
