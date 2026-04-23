"""France (FRA) node warehouse v0: locked 128-d features from ``EventTapeRecord``.

Point-in-time window (inclusive on both ends): for ``as_of`` and ``window_days``,
events satisfy ``event_date <= as_of`` and
``event_date >= as_of - timedelta(days=window_days - 1)``. That is exactly
``window_days`` calendar days ending on ``as_of``.

**128-dimensional layout** (built deterministically, then each full row is L2-normalized).
Rows whose **pre-normalisation** L2 norm is below ``1e-6`` are **rejected** at build time
(rather than written as zeros that could become NaNs in other pipelines):

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

import sys
import time
import re
import zlib
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
from tqdm import tqdm

from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import query_records
from baselines.node_warehouse_mmap import write_float32_matrix, write_manifest
from schemas.graph_builder_warehouse import (
    NODE_WAREHOUSE_EMBEDDING_DIM_V1,
    NodeWarehouseManifest,
    NodeWarehouseRowMeta,
)

NODE_WAREHOUSE_RECIPE_ID_V0 = "gdelt_cameo_hist_actor1_admin1_v0"
NODE_WAREHOUSE_RECIPE_ID_V1 = "gdelt_cameo_hist_actor1_admin1_monthly_v1"
# Locked: shared mmap / query-encoder ANN version for France and Arab Spring v0 builds.
NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0: Final[str] = "gdelt_cameo_hist_actor1_admin1_v0"
NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V1: Final[str] = "ar_v1"
DEFAULT_ARAB_SPRING_NODE_MMAP: Final[Path] = Path("data/arab_spring/node_warehouse_v0.mmap")
DEFAULT_ARAB_SPRING_NODE_MANIFEST: Final[Path] = Path("data/arab_spring/node_warehouse_v0.mmap.json")
ARAB_SPRING_COUNTRY_RANGE_START: Final[date] = date(2010, 1, 1)
ARAB_SPRING_COUNTRY_RANGE_END: Final[date] = date(2013, 12, 31)
# Node matrix over EG, TU, LY, and SY when present in the warehouse (FIPS 2-letter ``country_code``).
ARAB_SPRING_NODE_COUNTRY_CODES: Final[frozenset[str]] = frozenset({"EG", "TU", "LY", "SY"})

NODE_VECTOR_DIM: int = NODE_WAREHOUSE_EMBEDDING_DIM_V1
# Reject near-zero feature rows before L2 normalisation so we never write NaNs from 0/0
# or propagate silent zeros into ANN (downstream cosine/dot assumes finite directions).
_PRE_L2_NORM_EPS = 1e-6
# Matches graph_builder_query_encoder.ENTITY_HINT_KEYS; avoid importing that module (torch).
_ENTITY_HINT_KEYS = "entity_hint_keys"


def _stable_actor_slot(actor1_name: str | None) -> int:
    key = (actor1_name or "").strip().lower().encode("utf-8")
    return (zlib.crc32(key) & 0xFFFFFFFF) % 8


def _normalize_actor_name(actor1_name: str | None) -> str:
    text = (actor1_name or "").lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "unknown"
    return text[:64].strip() or "unknown"


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

        first_seen = min(ev.event_date for ev in evs)
        hint_keys: set[str] = set()
        for ev in evs:
            label = (ev.actor1_name or "").strip().lower()
            if label:
                hint_keys.add(label)

        node_id = f"fr_v0|{admin1}|slot{slot}"
        slice_id = f"as_of_{as_of.isoformat()}"
        rows_meta.append(
            NodeWarehouseRowMeta(
                node_id=node_id,
                first_seen=first_seen,
                slice_id=slice_id,
                admin1_code=admin1,
                extensions={_ENTITY_HINT_KEYS: list(hint_keys)},
            )
        )

    if feats.shape[0] > 0:
        pre_norms = np.linalg.norm(feats, axis=1)
        if bool(np.any(pre_norms < _PRE_L2_NORM_EPS)):
            bad_idx = int(np.flatnonzero(pre_norms < _PRE_L2_NORM_EPS)[0])
            bad_id = rows_meta[bad_idx].node_id
            raise ValueError(
                "node warehouse v0: row L2 norm below "
                f"{_PRE_L2_NORM_EPS:g} before normalisation (would corrupt ANN); "
                f"first offending node_id={bad_id!r}",
            )

    matrix = _l2_normalize_rows(feats.astype(np.float32))
    return matrix, rows_meta


def _hint_keys_for_record(rec: EventTapeRecord) -> set[str]:
    out: set[str] = set()
    for n in (rec.actor1_name, rec.actor2_name):
        label = (n or "").strip().lower()
        if label:
            out.add(label)
    return out


def _actor1_hint_keys_raw(evs: Sequence[EventTapeRecord]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ev in evs:
        label = (ev.actor1_name or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return sorted(out)


def _month_bucket(d: date) -> str:
    return d.strftime("%Y-%m")


def build_arab_spring_node_matrix_v0(
    records: Sequence[EventTapeRecord],
    *,
    as_of: date,
    window_days: int = 1,
    country_codes: frozenset[str] = ARAB_SPRING_NODE_COUNTRY_CODES,
    data_start: date = ARAB_SPRING_COUNTRY_RANGE_START,
    data_end: date = ARAB_SPRING_COUNTRY_RANGE_END,
) -> tuple[np.ndarray, list[NodeWarehouseRowMeta]]:
    """Aggregate Arab Spring (EG/TU/LY/SY) events into a float32 ``(N, 128)`` matrix and row metadata.

    Same 128-d recipe as :func:`build_france_node_matrix_v0`, but:
    * **Daily** windows: default ``window_days=1`` (single calendar day ``as_of``);
    * Only ``country_code`` in ``country_codes`` (2-letter, matching GDELT/ACLED tape);
    * Only events with ``data_start <= event_date <= data_end``;
    * ``entity_hint_keys`` use normalized (lower/stripped) ``actor1_name`` and ``actor2_name`` when set;
    * ``node_id`` prefix ``ar_v0``.

    :class:`NodeWarehouseRowMeta` **first_seen** is the minimum ``event_date`` in the PIT window
    (over in-range events) per ``(admin1_code, actor_slot)``.
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    start = as_of - timedelta(days=window_days - 1)
    if data_start > data_end:
        raise ValueError("data_start must be on or before data_end")

    grouped: dict[tuple[str, int], list[EventTapeRecord]] = defaultdict(list)
    for rec in records:
        if rec.country_code not in country_codes:
            continue
        if not (data_start <= rec.event_date <= data_end):
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

        first_seen = min(ev.event_date for ev in evs)
        hint_keys: set[str] = set()
        for ev in evs:
            hint_keys |= _hint_keys_for_record(ev)

        node_id = f"ar_v0|{admin1}|slot{slot}"
        slice_id = f"as_of_{as_of.isoformat()}"
        rows_meta.append(
            NodeWarehouseRowMeta(
                node_id=node_id,
                first_seen=first_seen,
                slice_id=slice_id,
                admin1_code=admin1,
                extensions={_ENTITY_HINT_KEYS: sorted(hint_keys)},
            )
        )

    if feats.shape[0] > 0:
        pre_norms = np.linalg.norm(feats, axis=1)
        if bool(np.any(pre_norms < _PRE_L2_NORM_EPS)):
            bad_idx = int(np.flatnonzero(pre_norms < _PRE_L2_NORM_EPS)[0])
            bad_id = rows_meta[bad_idx].node_id
            raise ValueError(
                "node warehouse v0: row L2 norm below "
                f"{_PRE_L2_NORM_EPS:g} before normalisation (would corrupt ANN); "
                f"first offending node_id={bad_id!r}",
            )

    matrix = _l2_normalize_rows(feats.astype(np.float32))
    return matrix, rows_meta


def build_arab_spring_node_warehouse_v0(
    warehouse_path: Path | str,
    out_mmap: Path | str,
    out_manifest: Path | str,
    *,
    as_of: date | None = None,
    window_days: int = 1,
    data_start: date = ARAB_SPRING_COUNTRY_RANGE_START,
    data_end: date = ARAB_SPRING_COUNTRY_RANGE_END,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Load Arab Spring events from DuckDB, build v0 embeddings, write mmap + JSON manifest.

    Queries ``event_date`` in ``[as_of - window_days + 1, as_of]`` (inclusive). Defaults:
    ``as_of`` = end of locked data range (2013-12-31), ``window_days`` = 1 (single day).
    For a multi-day PIT window, increase ``window_days``; ensure the machine can hold the
    resulting row list in memory.

    When ``show_progress`` is True, prints a 3-step tqdm bar on ``stderr`` (DuckDB load →
    matrix build → write mmap/manifest) with row counts and elapsed seconds.
    """
    wp = Path(warehouse_path)
    out_m = Path(out_mmap)
    out_j = Path(out_manifest)
    pit = as_of if as_of is not None else data_end
    event_start = pit - timedelta(days=window_days - 1)
    event_end = pit

    pbar = (
        tqdm(
            total=3,
            desc="arab_spring_node_warehouse_v0",
            unit="step",
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )
        if show_progress
        else None
    )
    t_query = time.perf_counter()
    try:
        if pbar is not None:
            pbar.set_postfix_str("DuckDB query …", refresh=False)
        recs = query_records(
            db_path=wp,
            event_start=event_start,
            event_end=event_end,
        )
        if pbar is not None:
            pbar.set_postfix_str(
                f"loaded {len(recs):,} events in {time.perf_counter() - t_query:.1f}s"
            )
            pbar.update(1)

        t_build = time.perf_counter()
        if pbar is not None:
            pbar.set_postfix_str("build matrix …", refresh=False)
        matrix, rows_meta = build_arab_spring_node_matrix_v0(
            recs,
            as_of=pit,
            window_days=window_days,
            data_start=data_start,
            data_end=data_end,
        )
        if pbar is not None:
            pbar.set_postfix_str(
                f"matrix {matrix.shape[0]:,}×{matrix.shape[1]} in {time.perf_counter() - t_build:.1f}s"
            )
            pbar.update(1)

        t_write = time.perf_counter()
        if pbar is not None:
            pbar.set_postfix_str("write mmap + manifest …", refresh=False)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        write_float32_matrix(out_m, matrix)
        manifest = NodeWarehouseManifest(
            manifest_version="v0",
            embedding_version=NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0,
            mmap_path=str(out_m),
            row_count=int(matrix.shape[0]),
            recipe_id=NODE_WAREHOUSE_RECIPE_ID_V0,
            window_days=window_days,
            as_of=pit,
            rows=rows_meta,
            embedding_dim=NODE_WAREHOUSE_EMBEDDING_DIM_V1,
        )
        write_manifest(out_j, manifest)
        if pbar is not None:
            pbar.set_postfix_str(
                f"wrote in {time.perf_counter() - t_write:.1f}s total {time.perf_counter() - t_query:.1f}s"
            )
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return {
        "warehouse_path": str(wp.resolve()),
        "out_mmap": str(out_m.resolve()),
        "out_manifest": str(out_j.resolve()),
        "row_count": int(matrix.shape[0]),
        "as_of": pit.isoformat(),
        "window_days": window_days,
        "event_query_start": event_start.isoformat(),
        "event_query_end": event_end.isoformat(),
    }


def build_arab_spring_node_matrix_v1(
    records: Sequence[EventTapeRecord],
    *,
    as_of: date,
    window_days: int = 1461,
    country_codes: frozenset[str] = ARAB_SPRING_NODE_COUNTRY_CODES,
    data_start: date = ARAB_SPRING_COUNTRY_RANGE_START,
    data_end: date = ARAB_SPRING_COUNTRY_RANGE_END,
) -> tuple[np.ndarray, list[NodeWarehouseRowMeta]]:
    """Aggregate Arab Spring events into monthly actor/admin1 v1 node rows."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    start = as_of - timedelta(days=window_days - 1)
    if data_start > data_end:
        raise ValueError("data_start must be on or before data_end")

    grouped: dict[tuple[str, str, str], list[EventTapeRecord]] = defaultdict(list)
    for rec in records:
        if rec.country_code not in country_codes:
            continue
        if not (data_start <= rec.event_date <= data_end):
            continue
        admin1 = (rec.admin1_code or "").strip()
        if not admin1:
            continue
        if not (start <= rec.event_date <= as_of):
            continue
        norm_actor = _normalize_actor_name(rec.actor1_name)
        time_bucket = _month_bucket(rec.event_date)
        grouped[(norm_actor, admin1, time_bucket)].append(rec)

    keys_sorted = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    rows_meta: list[NodeWarehouseRowMeta] = []
    feats = np.zeros((len(keys_sorted), NODE_VECTOR_DIM), dtype=np.float64)

    for row_idx, (norm_actor, admin1, time_bucket) in enumerate(keys_sorted):
        evs = grouped[(norm_actor, admin1, time_bucket)]
        hist = np.zeros(64, dtype=np.float64)
        gold_vals: list[float] = []
        quad_vals: list[float] = []
        tone_vals: list[float] = []
        slot = _stable_actor_slot(norm_actor)

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

        first_seen = min(ev.event_date for ev in evs)
        node_id = f"ar_v1|{norm_actor}|{admin1}|{time_bucket}"
        slice_id = f"monthly_{time_bucket}"
        rows_meta.append(
            NodeWarehouseRowMeta(
                node_id=node_id,
                first_seen=first_seen,
                slice_id=slice_id,
                admin1_code=admin1,
                extensions={
                    _ENTITY_HINT_KEYS: _actor1_hint_keys_raw(evs),
                    "actor_name_normalized": norm_actor,
                    "time_bucket": time_bucket,
                },
            )
        )

    if feats.shape[0] > 0:
        pre_norms = np.linalg.norm(feats, axis=1)
        if bool(np.any(pre_norms < _PRE_L2_NORM_EPS)):
            bad_idx = int(np.flatnonzero(pre_norms < _PRE_L2_NORM_EPS)[0])
            bad_id = rows_meta[bad_idx].node_id
            raise ValueError(
                "node warehouse v1: row L2 norm below "
                f"{_PRE_L2_NORM_EPS:g} before normalisation (would corrupt ANN); "
                f"first offending node_id={bad_id!r}",
            )

    matrix = _l2_normalize_rows(feats.astype(np.float32))
    return matrix, rows_meta


def build_arab_spring_node_warehouse_v1(
    warehouse_path: Path | str,
    out_mmap: Path | str,
    out_manifest: Path | str,
    *,
    as_of: date | None = None,
    window_days: int = 1461,
    data_start: date = ARAB_SPRING_COUNTRY_RANGE_START,
    data_end: date = ARAB_SPRING_COUNTRY_RANGE_END,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Load Arab Spring events from DuckDB, build v1 embeddings, write mmap + JSON manifest."""
    wp = Path(warehouse_path)
    out_m = Path(out_mmap)
    out_j = Path(out_manifest)
    pit = as_of if as_of is not None else date(2013, 12, 31)
    event_start = pit - timedelta(days=window_days - 1)
    event_end = pit

    pbar = (
        tqdm(
            total=3,
            desc="arab_spring_node_warehouse_v1",
            unit="step",
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )
        if show_progress
        else None
    )
    t_query = time.perf_counter()
    try:
        if pbar is not None:
            pbar.set_postfix_str("DuckDB query …", refresh=False)
        recs = query_records(
            db_path=wp,
            event_start=event_start,
            event_end=event_end,
        )
        if pbar is not None:
            pbar.set_postfix_str(
                f"loaded {len(recs):,} events in {time.perf_counter() - t_query:.1f}s"
            )
            pbar.update(1)

        t_build = time.perf_counter()
        if pbar is not None:
            pbar.set_postfix_str("build matrix …", refresh=False)
        matrix, rows_meta = build_arab_spring_node_matrix_v1(
            recs,
            as_of=pit,
            window_days=window_days,
            data_start=data_start,
            data_end=data_end,
        )
        if pbar is not None:
            pbar.set_postfix_str(
                f"matrix {matrix.shape[0]:,}×{matrix.shape[1]} in {time.perf_counter() - t_build:.1f}s"
            )
            pbar.update(1)

        t_write = time.perf_counter()
        if pbar is not None:
            pbar.set_postfix_str("write mmap + manifest …", refresh=False)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        write_float32_matrix(out_m, matrix)
        manifest = NodeWarehouseManifest(
            manifest_version="v1",
            embedding_version=NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V1,
            mmap_path=str(out_m),
            row_count=int(matrix.shape[0]),
            recipe_id=NODE_WAREHOUSE_RECIPE_ID_V1,
            window_days=window_days,
            as_of=pit,
            rows=rows_meta,
            embedding_dim=NODE_WAREHOUSE_EMBEDDING_DIM_V1,
        )
        write_manifest(out_j, manifest)
        if pbar is not None:
            pbar.set_postfix_str(
                f"wrote in {time.perf_counter() - t_write:.1f}s total {time.perf_counter() - t_query:.1f}s"
            )
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return {
        "warehouse_path": str(wp.resolve()),
        "out_mmap": str(out_m.resolve()),
        "out_manifest": str(out_j.resolve()),
        "row_count": int(matrix.shape[0]),
        "as_of": pit.isoformat(),
        "window_days": window_days,
        "event_query_start": event_start.isoformat(),
        "event_query_end": event_end.isoformat(),
    }


__all__ = [
    "ARAB_SPRING_COUNTRY_RANGE_END",
    "ARAB_SPRING_COUNTRY_RANGE_START",
    "ARAB_SPRING_NODE_COUNTRY_CODES",
    "DEFAULT_ARAB_SPRING_NODE_MANIFEST",
    "DEFAULT_ARAB_SPRING_NODE_MMAP",
    "NODE_VECTOR_DIM",
    "NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0",
    "NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V1",
    "NODE_WAREHOUSE_RECIPE_ID_V0",
    "NODE_WAREHOUSE_RECIPE_ID_V1",
    "_normalize_actor_name",
    "build_arab_spring_node_matrix_v0",
    "build_arab_spring_node_matrix_v1",
    "build_arab_spring_node_warehouse_v0",
    "build_arab_spring_node_warehouse_v1",
    "build_france_node_matrix_v0",
]
