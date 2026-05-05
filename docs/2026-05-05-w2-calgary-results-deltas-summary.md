# W2 Calgary 3/4 checkpoint — reproducible audit results + deltas

Motion task: `[W2 Calgary 3/4] Write reproducible results+deltas summary`.

## Inputs compared

Current reproducible run:

- manifest: `artifacts/warehouse_validation/node_warehouse_v0_manifest.json`
- mmap: `artifacts/warehouse_validation/node_warehouse_v0.mmap`
- checkpoint: `/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt`
- k: 10
- fixed seed environment: `PYTHONHASHSEED=0`
- raw SHA256: `c52cbb7278e3b0b6be1ca55e7fd2f9d413f1f878ca0554b79373920c03a1e0a7`

Reference run for deltas:

- report: `docs/pit_subgraph_audit_2026-04-28.phase4_gates_geo_fixed.json`
- manifest: `/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/node_warehouse_v1_fixed_manifest.json`
- same checkpoint path for the first checkpoint row.

Important caveat: these deltas are not a pure model delta because the warehouse artifact changed (`v1_fixed` shared-data manifest vs current local `v0` validation manifest). Treat them as an audit/gate-path stabilization delta, not a scientific checkpoint improvement claim.

## Current results

| metric | current |
|---|---:|
| probe_count | 243 |
| top1_time_ok_rate | 1.0000 |
| topk_time_ok_ratio_mean | 1.0000 |
| top1_country_ok_rate | 0.4074 |
| topk_country_ok_ratio_mean | 0.3160 |
| topk_hint_hit_ratio_mean | 0.0000 |
| precursor_hit_rate | 0.0000 |
| future_rank_lift_mean | 0.0000 |
| horizon_consistency_rate | 1.0000 |

Top-1 country rate by geography:

| geo | current |
|---|---:|
| Egypt | 0.4321 |
| Libya | 0.2963 |
| Tunisia | 0.4938 |

## Deltas vs 2026-04-28 geo-fixed audit

| metric | previous | current | delta |
|---|---:|---:|---:|
| top1_time_ok_rate | 0.1564 | 1.0000 | +0.8436 |
| topk_time_ok_ratio_mean | 0.1722 | 1.0000 | +0.8278 |
| top1_country_ok_rate | 0.2963 | 0.4074 | +0.1111 |
| topk_country_ok_ratio_mean | 0.2963 | 0.3160 | +0.0198 |
| topk_hint_hit_ratio_mean | 0.0000 | 0.0000 | +0.0000 |
| precursor_hit_rate | 0.0000 | 0.0000 | +0.0000 |
| future_rank_lift_mean | 0.0000 | 0.0000 | +0.0000 |
| horizon_consistency_rate | 0.2140 | 1.0000 | +0.7860 |

## Warehouse quality-gate delta

Reference quality report:

- report: `docs/warehouse_quality_gate_2026-04-28.v1_fixed.json`
- passed: false
- missing_entity_hint_rows: 183
- mixed_format_rows: 507
- mixed_format_rate: 0.0051316828
- invalid_country_rows: 0
- missing_first_seen_rows: 0
- unmapped_admin1_rows: 0

Current quality report:

- report: `docs/warehouse_quality_gate_2026-05-05.current.json`
- passed: true
- missing_entity_hint_rows: 0
- mixed_format_rows: 0
- mixed_format_rate: 0.0
- invalid_country_rows: 0
- missing_first_seen_rows: 0
- unmapped_admin1_rows: 0

## Interpretation

What improved:

- The local validation artifact now passes the warehouse quality contract.
- Fixed-seed audit outputs are byte-identical across reruns.
- The audit path no longer presents a hard time/horizon failure on the local validation artifact.
- Country metrics are non-zero and non-collapsed, with a modest global lift vs the old geo-fixed report.

What did not improve:

- `topk_hint_hit_ratio_mean`, `precursor_hit_rate`, and `future_rank_lift_mean` remain exactly zero.
- Libya remains the weakest geography bucket.
- Because the artifact changed, this is not evidence that the checkpoint got scientifically better.

Decision:

- Reproducibility: PASS.
- Warehouse quality gate on local validation artifact: PASS.
- Retrieval checkpoint promotion: NO-GO.
- Next action: enforce the quality gate in the default run flow, then only run longer or broader training/eval after hint/precursor diagnostics are made informative.
