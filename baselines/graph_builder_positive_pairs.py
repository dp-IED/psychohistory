"""Precomputed SSL positive pairs for graph-builder training (admin1 lead-lag rule).

Pairs are mmap **row indices** (not ``node_id`` strings). ``admin1_lead_lag_v0`` uses each
row's ``first_seen`` as the activity date proxy: two rows form a positive pair when they
share the same stripped ``admin1_code`` and the calendar-day distance between
``first_seen`` values is in **[32, 90]** inclusive. The minimum excludes same-calendar-month
pairs on typical monthly snapshots (gap 32 forces a month boundary); the maximum keeps
pairs within roughly one quarter for regional temporal persistence.

When ``manifest.as_of`` is set, rows whose ``first_seen`` falls outside the inclusive PIT
window ``[as_of - (window_days - 1), as_of]`` are excluded (same convention as
``node_warehouse_build_v0``). With a finite window, no pair can have a calendar gap larger
than ``window_days - 1`` between eligible rows; in particular ``window_days < 33`` forces
zero pairs under this rule because the min lead-lag is 32 days.

Uses numpy and stdlib; optional ``tqdm`` progress bars when ``show_progress=True`` on
``build_positive_pairs`` (dependency listed in project ``pyproject.toml``).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta

# Inner tqdm only for large buckets so tiny regions do not flash a second bar.
_INNER_TQDM_MIN_BUCKET = 4000


def _progress_note(message: str, *, show_progress: bool) -> None:
    """Log to stderr without breaking tqdm; visible in IDE terminals that buffer stdout."""
    if show_progress:
        tqdm.write(message, file=sys.stderr)

POSITIVE_PAIR_VERSION = "admin1_lead_lag_v0"
LEAD_LAG_MIN_DAYS = 32
LEAD_LAG_MAX_DAYS = 90
PAIRS_ARRAY_BASENAME = "positive_pairs.admin1_lead_lag_v0.npy"
META_JSON_BASENAME = "positive_pairs.admin1_lead_lag_v0.meta.json"
_PAIR_DTYPE = np.dtype([("i", np.int32), ("j", np.int32)])


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


def _pairs_from_buckets(
    entries: Iterable[tuple[int, date, str]],
    *,
    show_progress: bool = False,
) -> list[tuple[int, int]]:
    """All pairs (row i, row j) with i < j lexicographically on row index.

    Per admin1 bucket, sort by ``(first_seen, row_index)`` and scan with two pointers so
    each candidate second endpoint lies in a ``first_seen`` window of width
    ``[LEAD_LAG_MIN_DAYS, LEAD_LAG_MAX_DAYS]`` days. This is **O(n)** per bucket in the
    length of the bucket plus the number of pairs emitted, instead of **O(n²)** nested
    loops over all unordered date pairs (which can stall for hours on large regions).
    """
    bucket: dict[str, list[tuple[int, date]]] = defaultdict(list)
    for idx, first_seen, admin1 in entries:
        bucket[admin1].append((idx, first_seen))

    bucket_items = list(bucket.items())
    outer = tqdm(
        bucket_items,
        desc="Positive pairs (admin1)",
        unit="region",
        dynamic_ncols=True,
        disable=not show_progress,
        file=sys.stderr,
    )

    found: list[tuple[int, int]] = []
    for admin1, members in outer:
        if show_progress:
            outer.set_postfix_str(admin1[:40] + ("…" if len(admin1) > 40 else ""), refresh=False)
        n = len(members)
        if n < 2:
            continue
        members = sorted(members, key=lambda t: (t[1], t[0]))
        ordinals = [d.toordinal() for _, d in members]
        j_lo = 0
        j_hi = 0
        a_iter: Iterable[int] = range(n - 1)
        if show_progress and n >= _INNER_TQDM_MIN_BUCKET:
            a_iter = tqdm(
                range(n - 1),
                desc="  lead-lag scan",
                leave=False,
                unit="i",
                dynamic_ncols=True,
                total=n - 1,
                file=sys.stderr,
            )
        for a in a_iter:
            o_a = ordinals[a]
            if j_lo < a + 1:
                j_lo = a + 1
            while j_lo < n and ordinals[j_lo] - o_a < LEAD_LAG_MIN_DAYS:
                j_lo += 1
            if j_hi < j_lo:
                j_hi = j_lo
            while j_hi < n and ordinals[j_hi] - o_a <= LEAD_LAG_MAX_DAYS:
                j_hi += 1
            idx_a = members[a][0]
            for b in range(j_lo, j_hi):
                idx_b = members[b][0]
                if idx_a < idx_b:
                    found.append((idx_a, idx_b))
                else:
                    found.append((idx_b, idx_a))
    return found


def build_positive_pairs(
    manifest: NodeWarehouseManifest,
    mmap_path: Path,
    output_dir: Path,
    *,
    show_progress: bool = False,
) -> Path:
    rows = manifest.rows
    if rows is None:
        raise ValueError("manifest.rows must be present to build positive pairs")
    if len(rows) != manifest.row_count:
        raise ValueError(
            f"manifest.rows length ({len(rows)}) must equal row_count ({manifest.row_count})",
        )

    rows_for_eligible = rows
    if show_progress:
        rows_for_eligible = tqdm(
            rows,
            desc="Eligible rows (PIT filter)",
            unit="row",
            total=len(rows),
            dynamic_ncols=True,
            file=sys.stderr,
        )
    entries = _eligible_row_indices(
        rows_for_eligible,
        as_of=manifest.as_of,
        window_days=manifest.window_days,
    )
    pair_tuples = _pairs_from_buckets(entries, show_progress=show_progress)
    n_pairs = len(pair_tuples)
    if not pair_tuples:
        pairs = np.zeros((0, 2), dtype=np.int32)
    else:
        _progress_note(
            f"[positive_pairs] Collected {n_pairs:,} pair rows; converting to array and sorting…",
            show_progress=show_progress,
        )
        arr = np.asarray(pair_tuples, dtype=np.int32).reshape(-1, 2)
        order = np.lexsort((arr[:, 1], arr[:, 0]))
        pairs = np.ascontiguousarray(arr[order])

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / PAIRS_ARRAY_BASENAME
    meta_path = output_dir / META_JSON_BASENAME

    _progress_note(
        f"[positive_pairs] Writing {array_path.name} ({pairs.nbytes / (1024 * 1024):.1f} MiB)…",
        show_progress=show_progress,
    )
    np.save(array_path, pairs)
    _progress_note(
        f"[positive_pairs] Wrote {int(pairs.shape[0]):,} pairs and metadata to {output_dir}",
        show_progress=show_progress,
    )

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
        # Vectorized sorted-unique check (tolist() on 10^8 rows is unusable in Python).
        p0, p1 = pairs[:-1, 0], pairs[1:, 0]
        p0p, p1p = pairs[:-1, 1], pairs[1:, 1]
        lex_strict_inc = (p0 < p0p) | ((p0 == p0p) & (p1 < p1p))
        if not bool(np.all(lex_strict_inc)):
            raise ValueError("pairs must be sorted lexicographically with no duplicate rows")

    return pairs, metadata


class PositivePairLookup:
    """Fast membership checks over lexicographically sorted ``(i, j)`` pair rows."""

    def __init__(self, pairs: np.ndarray) -> None:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs array must have shape (P, 2), got {pairs.shape}")
        if pairs.dtype != np.int32:
            raise ValueError(f"pairs array must be int32, got {pairs.dtype}")
        if not pairs.flags.c_contiguous:
            pairs = np.ascontiguousarray(pairs)
        self._pairs = pairs
        self._keys = pairs.reshape(-1).view(_PAIR_DTYPE).reshape(-1)

    def contains_many(self, pairs: np.ndarray) -> np.ndarray:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"query pairs must have shape (P, 2), got {pairs.shape}")
        if pairs.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        query = np.asarray(pairs, dtype=np.int32)
        if not query.flags.c_contiguous:
            query = np.ascontiguousarray(query)
        keys = query.reshape(-1).view(_PAIR_DTYPE).reshape(-1)
        pos = np.searchsorted(self._keys, keys)
        in_bounds = pos < self._keys.shape[0]
        out = np.zeros(keys.shape[0], dtype=bool)
        valid_pos = pos[in_bounds]
        out[in_bounds] = self._keys[valid_pos] == keys[in_bounds]
        return out


__all__ = [
    "LEAD_LAG_MAX_DAYS",
    "LEAD_LAG_MIN_DAYS",
    "META_JSON_BASENAME",
    "PAIRS_ARRAY_BASENAME",
    "POSITIVE_PAIR_VERSION",
    "PositivePairLookup",
    "build_positive_pairs",
    "load_positive_pairs",
]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build positive pairs for SSL training")
    parser.add_argument("--manifest", required=True, help="Path to warehouse manifest JSON")
    parser.add_argument("--mmap", required=True, help="Path to warehouse mmap file")
    parser.add_argument("--output-dir", required=True, help="Output directory for pairs")
    parser.add_argument("--show-progress", action="store_true", help="Show progress bars")
    
    args = parser.parse_args()
    
    # Load manifest
    with open(args.manifest) as f:
        manifest_dict = json.load(f)
    manifest = NodeWarehouseManifest.model_validate(manifest_dict)
    
    # Build positive pairs (always uses admin1_lead_lag_v0 recipe)
    # Function writes files and returns output directory path
    output_path = build_positive_pairs(
        manifest=manifest,
        mmap_path=Path(args.mmap),
        output_dir=Path(args.output_dir),
        show_progress=args.show_progress,
    )
    
    if args.show_progress:
        print(f"[✓] Positive pairs built", file=sys.stderr)
        print(f"    Output dir: {output_path}", file=sys.stderr)
