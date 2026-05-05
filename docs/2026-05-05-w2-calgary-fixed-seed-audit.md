# W2 Calgary 2/4 checkpoint — fixed-seed reproducible PIT audit

Motion task: `[W2 Calgary 2/4] Run fixed-seed reproducible audit`.

## Command

Two identical audit passes were run with fixed Python hash seed and identical inputs:

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python scripts/pit_subgraph_audit.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --mmap artifacts/warehouse_validation/node_warehouse_v0.mmap \
  --checkpoint /Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt \
  --output docs/pit_subgraph_audit_2026-05-05.fixed_seed_run_a.json \
  --k 10

PYTHONHASHSEED=0 PYTHONPATH=. python scripts/pit_subgraph_audit.py \
  --manifest artifacts/warehouse_validation/node_warehouse_v0_manifest.json \
  --mmap artifacts/warehouse_validation/node_warehouse_v0.mmap \
  --checkpoint /Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt \
  --output docs/pit_subgraph_audit_2026-05-05.fixed_seed_run_b.json \
  --k 10
```

## Reproducibility check

Both runs were byte-identical and canonically identical.

- run A raw SHA256: `c52cbb7278e3b0b6be1ca55e7fd2f9d413f1f878ca0554b79373920c03a1e0a7`
- run B raw SHA256: `c52cbb7278e3b0b6be1ca55e7fd2f9d413f1f878ca0554b79373920c03a1e0a7`
- canonical SHA256: `0db40ab98188f67f87c583ab0c69c64abbfdc793b52a11f9e7939a3d00797322`
- canonical equality: true
- output bytes each: 187,418

Raw JSON outputs are ignored by git per `docs/artifact_policy.md` and can be regenerated from the command above.

## Summary metrics

Checkpoint audited:

`/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt`

Global summary:

- probe_count: 243
- top1_time_ok_rate: 1.0
- topk_time_ok_ratio_mean: 1.0
- top1_country_ok_rate: 0.4074074074
- topk_country_ok_ratio_mean: 0.3160493827
- topk_hint_hit_ratio_mean: 0.0
- precursor_hit_rate: 0.0
- future_rank_lift_mean: 0.0
- horizon_consistency_rate: 1.0

Top-1 country rate by geography:

- Egypt: 0.4320987654
- Libya: 0.2962962963
- Tunisia: 0.4938271605

## Decision

Reproducibility gate: PASS.

Scientific promotion gate: NO-GO for now.

Reason: PIT time/horizon behavior is stable and reproducible, but retrieval utility remains weak: country consistency is modest, Libya is thin, and hint/precursor diagnostics are still zero. This should be treated as a deterministic audit baseline, not a promoted retrieval checkpoint.
