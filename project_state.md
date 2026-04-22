# Project state (engineering ground truth)

Rolling snapshot of what exists in **this repo** for the graph-builder / France harness / Stage 2 stack. Update this when you land or remove major pieces so planning sessions start from facts, not stale “not implemented” lists.

---

## Graph builder — implemented

| Area | Location | Notes |
|------|----------|--------|
| CAMEO root severity (v0 tiers) | [`schemas/cameo_escalation_v0.py`](schemas/cameo_escalation_v0.py) | `cameo_tier()`; bands `01–09` / `10–17` / `18–20`. |
| Gate-aware offline labels | [`baselines/graph_builder_probe_labels.py`](baselines/graph_builder_probe_labels.py) | Per-gate tape rules; `SUPPRESSION_BASELINE_DAYS = 30`; JSONL via `write_probe_labels_jsonl`. **Persistence:** if pre-t baseline mean is 0 (no actor matches in window), label is **unanswerable** (`None` / skipped in harness stats), not `True`. |
| France harness heuristic sidecars | [`france_plumbing_sidecar_v0()`](baselines/graph_builder_probe_labels.py) | **Heuristic only** — `FR11`, hint-derived actors; Precursor anchor root **`01`** (tier 0) so tier-1 protest can count as escalation vs verbal. Propagation / Suppression / Coordination still need real geography + actors for meaningful France rates. |
| France harness label stats CLI | same module `__main__` | `--france-harness-stats` loads DuckDB + corpus, prints JSON distribution. |
| BagEncoder (mean-pool) | [`baselines/graph_builder_bag_encoder.py`](baselines/graph_builder_bag_encoder.py) | `RetrievedGraphBatch` → `[B, 128]`. |
| GateMLP + ForecastHead + loss | [`baselines/graph_builder_forecast_stack.py`](baselines/graph_builder_forecast_stack.py) | Path A `head_input` 133-dim; `forecast_brier_log_loss`. |
| Legacy GNN embeddings (Path B) | [`baselines/gnn.py`](baselines/gnn.py) | `forward_location_embeddings`, `legacy_graph_embedding` (mean-pool). |
| Stage 2 training (Path A) | [`baselines/graph_builder_stage2_train.py`](baselines/graph_builder_stage2_train.py) | Loads Stage 1 `QueryEncoder` ckpt; forecast + Δ-Brier-gated InfoNCE; `train_state_stage2.json`. |
| Dual-path primitives | [`baselines/graph_builder_dual_path_ablation.py`](baselines/graph_builder_dual_path_ablation.py) | `forward_path_a` / `forward_path_b`, shared `ForecastHead`; **Path B does not use GateMLP** (see file header). |
| Path B wall-clock helper | same module `__main__` | `--bench-path-b SNAPSHOT.json` times repeated Path B forwards (random-init GNN unless you extend). |

---

## Not implemented yet (honest gaps)

- **Single entrypoint for “alternating epochs”** (Path A train one epoch, Path B the next, shared head) on the real France loop — Stage 2 today is **Path A retrieval stack only**; dual-path is **library + bench**, not a fused trainer.
- **Trained** `HeteroGNNModel` checkpoint wired into Path B training (bench uses **random** weights unless you load weights in a small wrapper script).
- **Hand-authored probe label sidecars** at scale (production labels should not rely on `france_plumbing_sidecar_v0` alone).

---

## Deferred tech debt (don’t lose track)

- **Optimizer carry-over:** Stage 2 has an explicit `# TODO(optimizer)` next to `Adam` — Stage 1 checkpoints still omit Adam state; extending Stage 1 `torch.save` to persist optimizer state is what makes “carry over” literal.
- **`brute_topk` / ANN `k=100`:** still a magic constant shared by Stage 1 / Stage 2 pipelines; easy to miss during refactors.

---

## Overnight / batch commands

From repo root (`porto`). Uses default [`ingest/paths.py`](ingest/paths.py) warehouse resolution unless you override.

### 1) France corpus — gate label distribution (DuckDB warehouse required)

Inspect `positives_by_gate` / `warning` for **Suppression** and **Coordination** (and others) before Arab Spring-scale work.

```bash
cd /Users/darenpalmer/conductor/workspaces/psychohistory-v2/porto
export PYTHONPATH=.

# Default: PSYCHOHISTORY_DATA_ROOT or built-in default → warehouse/events.duckdb
python -m baselines.graph_builder_probe_labels --france-harness-stats

# Explicit warehouse:
# python -m baselines.graph_builder_probe_labels --france-harness-stats --warehouse /path/to/events.duckdb

# Explicit data root (warehouse = <root>/warehouse/events.duckdb):
# python -m baselines.graph_builder_probe_labels --france-harness-stats --data-root /path/to/psychohistory-v2
```

Redirect output if you want a log: `... | tee france_harness_label_stats.json`.

### 2) Path B wall-clock (one snapshot file)

Measures **Path B forward** cost (`legacy_graph_embedding` → `path_proj` → `ForecastHead`) on CPU by default. Uses **untrained** GNN weights — good for **relative** cost vs snapshot size; load a trained `state_dict` separately if you need production-like timing.

```bash
cd /Users/darenpalmer/conductor/workspaces/psychohistory-v2/porto
export PYTHONPATH=.

SNAP=/path/to/your/snapshots/as_of_2019-06-01.json
python -m baselines.graph_builder_dual_path_ablation --bench-path-b "$SNAP" --repeats 500 --device cpu
```

If Path B `ms_per_forward` is large vs your Path A epoch budget, decide now whether Path B belongs in **eval-only** checkpoints rather than every epoch (see plan “not decided”).

### 3) Full regression (after changes)

```bash
pytest -q
```

---

## Related docs

- Locked product contract: [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md)
- Program narrative: [`project.md`](project.md), [`roadmap.md`](roadmap.md)
