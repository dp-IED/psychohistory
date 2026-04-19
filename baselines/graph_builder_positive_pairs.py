"""Precomputed SSL positive pairs for graph-builder training (admin1 + 14-day rule).

Pairs are mmap **row indices** (not ``node_id`` strings). The v0 rule uses each
row's ``first_seen`` as the activity date proxy: two rows form a positive pair
when they share the same stripped ``admin1_code`` and calendar-day distance
between ``first_seen`` values is at most 14. When ``manifest.as_of`` is set,
rows whose ``first_seen`` falls outside the inclusive PIT window
``[as_of - (window_days - 1), as_of]`` are excluded (same convention as
``node_warehouse_build_v0``).

No torch dependency — numpy and stdlib only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta

POSITIVE_PAIR_VERSION = "admin1_14day_v0"
MAX_EVENT_GAP_DAYS = 14
PAIRS_ARRAY_BASENAME = "positive_pairs.admin1_14day_v0.npy"
META_JSON_BASENAME = "positive_pairs.admin1_14day_v0.meta.json"


def _is_basename_only(name: str) -> bool:
    p = Path(name)
    return len(p.parts) == 1 and p.name == name and not p.is_absolute()


def _first_seen_in_pit_window(
    first_seen: date,
    *,
    as_of: date | None,
    window_days: int,
) -> bool:
    if as_of is None:
        return True
    start = as_of - timedelta(days=window_days - 1)
    return start <= first_seen <= as_of


def _eligible_row_indices(
    rows: Sequence[NodeWarehouseRowMeta],
    *,
    as_of: date | None,
    window_days: int,
) -> list[tuple[int, date, str]]:
    out: list[tuple[int, date, str]] = []
    for idx, meta in enumerate(rows):
        admin1 = (meta.admin1_code or "").strip()
        if not admin1 or meta.first_seen is None:
            continue
        if not _first_seen_in_pit_window(
            meta.first_seen,
            as_of=as_of,
            window_days=window_days,
        ):
            continue
        out.append((idx, meta.first_seen, admin1))
    return out


def _pairs_from_buckets(entries: Iterable[tuple[int, date, str]]) -> list[tuple[int, int]]:
    bucket: dict[str, list[tuple[int, date]]] = defaultdict(list)
    for idx, first_seen, admin1 in entries:
        bucket[admin1].append((idx, first_seen))

    found: set[tuple[int, int]] = set()
    for _admin, members in bucket.items():
        n = len(members)
        for a in range(n):
            i, di = members[a]
            for b in range(a + 1, n):
                j, dj = members[b]
                lo, hi = (i, j) if i < j else (j, i)
                gap = abs((di - dj).days)
                if gap <= MAX_EVENT_GAP_DAYS:
                    found.add((lo, hi))
    return sorted(found)


def build_positive_pairs(
    manifest: NodeWarehouseManifest,
    mmap_path: Path,
    output_dir: Path,
) -> Path:
    rows = manifest.rows
    if rows is None:
        raise ValueError("manifest.rows must be present to build positive pairs")
    if len(rows) != manifest.row_count:
        raise ValueError(
            f"manifest.rows length ({len(rows)}) must equal row_count ({manifest.row_count})",
        )

    entries = _eligible_row_indices(
        rows,
        as_of=manifest.as_of,
        window_days=manifest.window_days,
    )
    pair_tuples = _pairs_from_buckets(entries)
    pairs = np.asarray(pair_tuples, dtype=np.int32).reshape(-1, 2) if pair_tuples else np.zeros((0, 2), dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / PAIRS_ARRAY_BASENAME
    meta_path = output_dir / META_JSON_BASENAME

    np.save(array_path, pairs)

    # Record path as given (no resolve) so metadata stays portable across machines.
    mmap_str = str(Path(mmap_path))
    as_of_val = manifest.as_of.isoformat() if manifest.as_of is not None else None
    meta: dict[str, object] = {
        "embedding_version": manifest.embedding_version,
        "recipe_id": manifest.recipe_id,
        "window_days": manifest.window_days,
        "pair_count": int(pairs.shape[0]),
        "mmap_path": mmap_str,
        "as_of": as_of_val,
        "positive_pair_version": POSITIVE_PAIR_VERSION,
        "pairs_path": PAIRS_ARRAY_BASENAME,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def load_positive_pairs(
    metadata_path: Path,
    manifest: NodeWarehouseManifest,
) -> tuple[np.ndarray, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta_ev = metadata.get("embedding_version")
    if meta_ev != manifest.embedding_version:
        raise ValueError(
            "embedding_version mismatch between positive-pairs metadata and manifest: "
            f"metadata={meta_ev!r}, manifest={manifest.embedding_version!r}",
        )

    pairs_name = metadata.get("pairs_path")
    if not isinstance(pairs_name, str) or not pairs_name:
        raise ValueError("metadata pairs_path must be a non-empty string basename")
    if not _is_basename_only(pairs_name):
        raise ValueError(
            f"metadata pairs_path must be a basename only (no directories), got {pairs_name!r}",
        )

    array_path = metadata_path.parent / pairs_name
    loaded = np.load(array_path, allow_pickle=False, mmap_mode="r")
    if loaded.ndim != 2 or loaded.shape[1] != 2:
        raise ValueError(f"pairs array must have shape (P, 2), got {loaded.shape}")
    if loaded.dtype != np.int32:
        raise ValueError(f"pairs array must be int32, got {loaded.dtype}")
    pairs = np.asarray(loaded, dtype=np.int32)

    expected_count = metadata.get("pair_count")
    if isinstance(expected_count, int) and expected_count != pairs.shape[0]:
        raise ValueError(
            f"pair_count in metadata ({expected_count}) does not match array rows ({pairs.shape[0]})",
        )

    if pairs.shape[0] > 0:
        if int(np.max(pairs)) >= manifest.row_count or int(np.min(pairs)) < 0:
            raise ValueError("pairs array contains row indices out of range for manifest.row_count")
        if not np.all(pairs[:, 0] < pairs[:, 1]):
            raise ValueError("pairs array must have i < j for every row")

    row_tuples = list(map(tuple, pairs.tolist()))
    if row_tuples != sorted(set(row_tuples)):
        raise ValueError("pairs must be sorted lexicographically with no duplicate rows")

    return pairs, metadata


__all__ = [
    "MAX_EVENT_GAP_DAYS",
    "META_JSON_BASENAME",
    "PAIRS_ARRAY_BASENAME",
    "POSITIVE_PAIR_VERSION",
    "build_positive_pairs",
    "load_positive_pairs",
]
