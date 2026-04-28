"""Build/load Stage1 predictive-coding targets from warehouse manifest rows.

This implementation computes per-slice weighted-degree centrality over a
co-activation graph derived from existing warehouse rows (no warehouse rebuild),
then uses within-slice future rank improvement at fixed horizons.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from schemas.graph_builder_warehouse import NodeWarehouseManifest
from schemas.stage1_predictive_targets import Stage1PredictiveTargetsManifest

DEFAULT_HORIZONS_DAYS = [30, 180, 730, 3650, 10950]
DEFAULT_HORIZON_WEIGHTS = [1.0, 0.6, 0.35, 0.2, 0.1]
TARGET_MANIFEST_BASENAME = "stage1_predictive_targets.rankchange_v1.meta.json"


def _parse_actor(node_id: str) -> str:
    parts = node_id.split("|")
    if len(parts) >= 4:
        return parts[1].strip().lower()
    return ""


def build_predictive_targets(
    manifest: NodeWarehouseManifest,
    output_dir: Path,
    *,
    source_manifest_path: str,
    horizons_days: list[int] | None = None,
    horizon_weights: list[float] | None = None,
) -> Path:
    rows = manifest.rows
    if rows is None:
        raise ValueError("manifest.rows must be present to build predictive targets")
    if len(rows) != manifest.row_count:
        raise ValueError(f"rows length {len(rows)} != row_count {manifest.row_count}")

    hs = list(DEFAULT_HORIZONS_DAYS if horizons_days is None else horizons_days)
    ws = list(DEFAULT_HORIZON_WEIGHTS if horizon_weights is None else horizon_weights)
    if len(hs) == 0:
        raise ValueError("horizons_days must be non-empty")
    if len(ws) == 0:
        raise ValueError("horizon_weights must be non-empty")
    if len(hs) != len(ws):
        raise ValueError("horizons_days and horizon_weights length mismatch")
    if any((not isinstance(h, int)) or h <= 0 for h in hs):
        raise ValueError("horizons_days must be positive integers")
    if any((not isinstance(w, (int, float))) or w <= 0 for w in ws):
        raise ValueError("horizon_weights must be positive numbers")
    if any(hs[i] <= hs[i - 1] for i in range(1, len(hs))):
        raise ValueError("horizons_days must be strictly increasing with no duplicates")

    actor_missing_rows = 0
    rows_without_first_seen = 0

    # Actor activity counts per (admin1, date). Co-activation remains pair-based,
    # but edge weights now reflect multiplicity: w(i,j)=count_i*count_j.
    # This avoids all-equal centrality for dense slices while staying deterministic.
    actor_counts_by_slice: dict[tuple[str, date], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        admin1 = (row.admin1_code or "").strip()
        if not admin1:
            continue
        if row.first_seen is None:
            rows_without_first_seen += 1
            continue
        actor = _parse_actor(row.node_id)
        if not actor:
            actor_missing_rows += 1
            continue
        actor_counts_by_slice[(admin1, row.first_seen)][actor] += 1

    # Centrality rank maps per (admin1, date): actor -> rank (1=highest score)
    # Weighted-degree with multiplicity edges:
    #   C_i = sum_{j!=i} count_i*count_j = count_i*(sum_counts-count_i).
    # This is O(k) per slice (k actors), replacing naive O(k^2) pair iteration.
    rank_maps: dict[tuple[str, date], dict[str, int]] = {}
    admin1_dates: dict[str, list[date]] = defaultdict(list)
    for (admin1, d), actor_counts in actor_counts_by_slice.items():
        total_count = sum(actor_counts.values())
        centrality: dict[str, int] = {
            actor: count * (total_count - count) for actor, count in actor_counts.items()
        }
        ranked = sorted(centrality.items(), key=lambda kv: (-kv[1], kv[0]))
        rank_maps[(admin1, d)] = {actor: i + 1 for i, (actor, _score) in enumerate(ranked)}
        admin1_dates[admin1].append(d)
    for admin1 in list(admin1_dates.keys()):
        admin1_dates[admin1] = sorted(set(admin1_dates[admin1]))

    def nearest_date_at_or_after(admin1: str, target_date: date) -> date | None:
        dates = admin1_dates.get(admin1)
        if not dates:
            return None
        lo, hi = 0, len(dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if dates[mid] < target_date:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(dates):
            return None
        return dates[lo]

    horizon_targets: dict[int, np.ndarray] = {
        h: np.zeros((manifest.row_count,), dtype=np.float32) for h in hs
    }

    for idx, row in enumerate(rows):
        admin1 = (row.admin1_code or "").strip()
        d0 = row.first_seen
        if not admin1 or d0 is None:
            continue
        actor = _parse_actor(row.node_id)
        if not actor:
            continue

        now_rank_map = rank_maps.get((admin1, d0))
        if not now_rank_map:
            continue
        r0 = now_rank_map.get(actor)
        if r0 is None:
            continue

        for h in hs:
            future_date = nearest_date_at_or_after(admin1, d0 + timedelta(days=h))
            if future_date is None:
                continue
            fut_rank_map = rank_maps.get((admin1, future_date))
            if not fut_rank_map:
                continue
            r1 = fut_rank_map.get(actor)
            if r1 is None:
                continue
            delta = r0 - r1
            if delta <= 0:
                continue
            horizon_targets[h][idx] = np.float32(delta / max(1, r0))

    output_dir.mkdir(parents=True, exist_ok=True)
    target_files: dict[str, str] = {}
    for h in hs:
        name = f"rankchange_h{h}d.npy"
        np.save(output_dir / name, horizon_targets[h])
        target_files[str(h)] = name

    m = Stage1PredictiveTargetsManifest(
        embedding_version=manifest.embedding_version,
        source_manifest_path=source_manifest_path,
        row_count=manifest.row_count,
        horizons_days=hs,
        horizon_weights=ws,
        target_files=target_files,
        actor_missing_rows=actor_missing_rows,
        rows_without_first_seen=rows_without_first_seen,
    )
    meta_path = output_dir / TARGET_MANIFEST_BASENAME
    meta_path.write_text(json.dumps(m.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def load_predictive_targets(meta_path: Path, manifest: NodeWarehouseManifest) -> tuple[Stage1PredictiveTargetsManifest, dict[int, np.ndarray]]:
    meta = Stage1PredictiveTargetsManifest.model_validate_json(meta_path.read_text(encoding="utf-8"))
    if meta.embedding_version != manifest.embedding_version:
        raise ValueError(
            f"embedding_version mismatch targets={meta.embedding_version!r} manifest={manifest.embedding_version!r}"
        )
    if meta.row_count != manifest.row_count:
        raise ValueError(f"row_count mismatch targets={meta.row_count} manifest={manifest.row_count}")

    out: dict[int, np.ndarray] = {}
    for h in meta.horizons_days:
        fname = meta.target_files[str(h)]
        arr = np.load(meta_path.parent / fname, allow_pickle=False)
        if arr.dtype != np.float32:
            arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != manifest.row_count:
            raise ValueError(f"target horizon {h} has invalid shape {arr.shape}")
        out[h] = arr
    return meta, out


__all__ = [
    "DEFAULT_HORIZONS_DAYS",
    "DEFAULT_HORIZON_WEIGHTS",
    "TARGET_MANIFEST_BASENAME",
    "build_predictive_targets",
    "load_predictive_targets",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Stage1 predictive-coding rank-change targets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-path", type=str, default="")
    args = parser.parse_args()

    manifest = NodeWarehouseManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    src = args.source_manifest_path or str(args.manifest)
    out = build_predictive_targets(
        manifest=manifest,
        output_dir=args.output_dir,
        source_manifest_path=src,
    )
    print(str(out))
