"""France (FRA) node warehouse v0: locked 128-d features from ``EventTapeRecord``.

Point-in-time window (inclusive on both ends): for ``as_of`` and ``window_days``,
events satisfy ``event_date <= as_of`` and
``event_date >= as_of - timedelta(days=window_days - 1)``. That is exactly
``window_days`` calendar days ending on ``as_of``.

**128-dimensional layout** (built deterministically, then each full row is L2-normalized;
if the L2 norm is 0, the row stays all zeros):

1. **Dims 0–63**: Take ``event_root_code`` stripped; pad or truncate to **two**
   characters (pad the second with ASCII space if length is 1; empty becomes two
   spaces). Map each event to a bin ``(ord(c0) % 8) * 8 + (ord(c1) % 8)`` in
   ``[0, 63]``, accumulate counts per node, then **L1-normalize** the 64-bin
   histogram (all zeros when there are no events).

2. **Dims 64–71**: **8-way one-hot** for ``actor_slot`` (see below).

3. **Dims 72–103**: **32-bin** deterministic hash of ``admin1_code``:
   ``idx = zlib.crc32(admin1_code.encode('utf-8')) & 0xFFFFFFFF`` then ``idx % 32``;
   a single **1.0** at that index (spike), others 0.

4. **Dims 104–127**: Summary scalars then zero padding — in order:
   ``log1p(event_count) / 10.0``,
   mean ``goldstein_scale`` over events with non-null values (0 if none),
   mean of ``quad_class`` values mapped linearly from ``{1,2,3,4}`` to ``[-1, 1]``
   (only events with non-null ``quad_class``; 0 if none),
   mean ``avg_tone`` over events with non-null ``avg_tone`` (0 if none);
   remaining entries through dim 127 are **0**.

**Node identity**: one row per ``(admin1_code, actor_slot)`` for France rows with
``country_code == "FRA"`` and non-blank ``admin1_code``. ``actor_slot`` is
``(zlib.crc32(key) & 0xFFFFFFFF) % 8`` where ``key`` is UTF-8 bytes of
``(actor1_name or '').strip().lower()`` (stable across processes; **not** Python
``hash``).
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from ingest.event_tape import EventTapeRecord
from schemas.graph_builder_warehouse import (
    NODE_WAREHOUSE_EMBEDDING_DIM_V1,
    NodeWarehouseRowMeta,
)

NODE_WAREHOUSE_RECIPE_ID_V0 = "gdelt_cameo_hist_actor1_admin1_v0"
NODE_VECTOR_DIM: int = NODE_WAREHOUSE_EMBEDDING_DIM_V1


def _stable_actor_slot(actor1_name: str | None) -> int:
    key = (actor1_name or "").strip().lower().encode("utf-8")
    return (zlib.crc32(key) & 0xFFFFFFFF) % 8


def _root_two_chars(event_root_code: str) -> tuple[str, str]:
    text = (event_root_code or "").strip()
    if len(text) >= 2:
        return text[0], text[1]
    if len(text) == 1:
        return text[0], " "
    return " ", " "


def _root_bin(event_root_code: str) -> int:
    c0, c1 = _root_two_chars(event_root_code)
    return (ord(c0) % 8) * 8 + (ord(c1) % 8)


def _admin1_hash_bin(admin1_code: str) -> int:
    return (zlib.crc32(admin1_code.encode("utf-8")) & 0xFFFFFFFF) % 32


def _map_quad_to_neg1_1(quad: int) -> float:
    q = max(1, min(4, int(quad)))
    return (q - 1) / 3.0 * 2.0 - 1.0


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    mask = norms.ravel() > 0.0
    out[mask] = out[mask] / norms[mask]
    return out


def build_france_node_matrix_v0(
    records: Sequence[EventTapeRecord],
    *,
    as_of: date,
    window_days: int,
) -> tuple[np.ndarray, list[NodeWarehouseRowMeta]]:
    """Aggregate France FRA events into a float32 ``(N, 128)`` matrix and row metadata."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    start = as_of - timedelta(days=window_days - 1)

    grouped: dict[tuple[str, int], list[EventTapeRecord]] = defaultdict(list)
    for rec in records:
        if rec.country_code != "FRA":
            continue
        admin1 = (rec.admin1_code or "").strip()
        if not admin1:
            continue
        if not (start <= rec.event_date <= as_of):
            continue
        slot = _stable_actor_slot(rec.actor1_name)
        grouped[(admin1, slot)].append(rec)

    keys_sorted = sorted(grouped.keys(), key=lambda k: (k[0], k[1]))
    rows_meta: list[NodeWarehouseRowMeta] = []
    feats = np.zeros((len(keys_sorted), NODE_VECTOR_DIM), dtype=np.float64)

    for row_idx, (admin1, slot) in enumerate(keys_sorted):
        evs = grouped[(admin1, slot)]
        hist = np.zeros(64, dtype=np.float64)
        gold_vals: list[float] = []
        quad_vals: list[float] = []
        tone_vals: list[float] = []

        for ev in evs:
            hist[_root_bin(ev.event_root_code)] += 1.0
            if ev.goldstein_scale is not None:
                gold_vals.append(float(ev.goldstein_scale))
            if ev.quad_class is not None:
                quad_vals.append(_map_quad_to_neg1_1(ev.quad_class))
            if ev.avg_tone is not None:
                tone_vals.append(float(ev.avg_tone))

        s = float(hist.sum())
        if s > 0.0:
            hist = hist / s

        feats[row_idx, 0:64] = hist
        feats[row_idx, 64 + slot] = 1.0
        aidx = _admin1_hash_bin(admin1)
        feats[row_idx, 72 + aidx] = 1.0

        n_ev = len(evs)
        feats[row_idx, 104] = np.log1p(n_ev) / 10.0
        feats[row_idx, 105] = float(np.mean(gold_vals)) if gold_vals else 0.0
        feats[row_idx, 106] = float(np.mean(quad_vals)) if quad_vals else 0.0
        feats[row_idx, 107] = float(np.mean(tone_vals)) if tone_vals else 0.0

        node_id = f"fr_v0|{admin1}|slot{slot}"
        slice_id = f"as_of_{as_of.isoformat()}"
        rows_meta.append(
            NodeWarehouseRowMeta(
                node_id=node_id,
                slice_id=slice_id,
                admin1_code=admin1,
            )
        )

    matrix = _l2_normalize_rows(feats.astype(np.float32))
    return matrix, rows_meta


__all__ = [
    "NODE_VECTOR_DIM",
    "NODE_WAREHOUSE_RECIPE_ID_V0",
    "build_france_node_matrix_v0",
]
