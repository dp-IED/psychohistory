"""Optional Wikidata QID grounding for Actor and Location nodes in graph snapshots."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_EVERY_N_API = 25

from evals.wikidata_dump_grounding import resolve_qid_from_dump, select_dump_slice
from evals.wikidata_linking import (
    _negative_cache_entry,
    _negative_cache_is_fresh,
    normalize_entity_label,
    qid_from_value,
    search_wikidata_entity,
    wikidata_qid_for_node,
)

_COUNTRY_SEARCH_HINT: dict[str, str] = {
    "FR": "France",
    "DE": "Germany",
    "ES": "Spain",
    "IT": "Italy",
    "GB": "United Kingdom",
    "US": "United States",
}


def _primary_label(node: dict[str, Any]) -> str | None:
    value = node.get("label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _country_hint(node: dict[str, Any]) -> str | None:
    attrs = node.get("attributes")
    if not isinstance(attrs, dict):
        return None
    code = attrs.get("country_code")
    if not isinstance(code, str) or not code.strip():
        return None
    code = code.strip().upper()
    return _COUNTRY_SEARCH_HINT.get(code, code)


def ground_snapshot_nodes(
    nodes: list[dict[str, Any]],
    *,
    cache_path: Path | None = None,
    entity_types: frozenset[str] = frozenset({"Actor", "Location"}),
    request_delay_s: float = 0.25,
    log: bool = True,
    origin: str | None = None,
    dump_manifest_path: Path | None = None,
    api_fallback: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve Wikidata QIDs for Actor and Location nodes from PIT dump slice, then API fallback."""

    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cache = raw
        except json.JSONDecodeError:
            cache = {}

    origin_tag = origin or "?"
    candidate_nodes = sum(
        1
        for n in nodes
        if isinstance(n, dict) and n.get("type") in entity_types
    )
    if log:
        print(
            f"[wikidata-grounding] origin={origin_tag} start "
            f"candidate_actor_location_nodes={candidate_nodes} "
            f"cache_keys_loaded={len(cache)} cache_path={cache_path}",
            file=sys.stderr,
            flush=True,
        )

    out_nodes: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "attempted": 0,
        "resolved": 0,
        "dump_resolved": 0,
        "api_resolved": 0,
        "skipped_existing_qid": 0,
        "skipped_no_label": 0,
        "api_calls": 0,
    }
    as_of_date = (origin or "").strip()[:10]
    dump_id: str | None = None
    dump_date: str | None = None
    dump_cache_scope = ""
    if dump_manifest_path is not None and as_of_date:
        try:
            dump_selection = select_dump_slice(dump_manifest_path, as_of_date=as_of_date)
        except Exception as exc:
            dump_selection = None
            if log:
                print(
                    f"[wikidata-grounding] origin={origin_tag} dump_select_error err={exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        if dump_selection is not None:
            dump_id = dump_selection.dump_id
            dump_date = dump_selection.dump_date
            dump_cache_scope = f"@{dump_selection.dump_id}"
        else:
            dump_cache_scope = f"@asof:{as_of_date}"

    for node in nodes:
        if not isinstance(node, dict):
            out_nodes.append(node)
            continue
        node_type = node.get("type")
        if node_type not in entity_types:
            out_nodes.append(node)
            continue

        if wikidata_qid_for_node(node):
            stats["skipped_existing_qid"] += 1
            out_nodes.append(node)
            continue

        label = _primary_label(node)
        if not label:
            stats["skipped_no_label"] += 1
            out_nodes.append(node)
            continue

        key = f"{node_type}:{normalize_entity_label(label)}{dump_cache_scope}"
        stats["attempted"] += 1

        cached = cache.get(key)
        if isinstance(cached, dict) and cached.get("_status") == "error":
            cached = cached if _negative_cache_is_fresh(cached) else None

        if cached is None:
            country = _country_hint(node)
            if dump_manifest_path is not None and as_of_date:
                try:
                    dump_hit = resolve_qid_from_dump(
                        label=label,
                        as_of_date=as_of_date,
                        manifest_path=dump_manifest_path,
                        country_hint=country,
                    )
                except Exception as exc:
                    if log:
                        print(
                            f"[wikidata-grounding] origin={origin_tag} "
                            f"dump_error label={label[:120]!r} err={exc!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    dump_hit = None
                if dump_hit is not None and dump_hit.selection is not None:
                    dump_id = dump_hit.selection.dump_id
                    dump_date = dump_hit.selection.dump_date
                if dump_hit is not None and dump_hit.entity is not None:
                    cached = {
                        "id": dump_hit.entity.get("qid"),
                        "label": dump_hit.entity.get("label_en") or dump_hit.entity.get("label") or label,
                        "description": dump_hit.entity.get("description"),
                        "method": "dump_slice",
                        "dump_id": dump_hit.selection.dump_id if dump_hit.selection is not None else None,
                        "dump_date": dump_hit.selection.dump_date if dump_hit.selection is not None else None,
                    }
                    stats["dump_resolved"] += 1

            if cached is None and api_fallback:
                stats["api_calls"] += 1
                if log and (
                    stats["api_calls"] <= 3
                    or stats["api_calls"] % _LOG_EVERY_N_API == 1
                ):
                    print(
                        f"[wikidata-grounding] origin={origin_tag} "
                        f"api_call={stats['api_calls']} "
                        f"type={node_type} label={label[:120]!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    cached = search_wikidata_entity(label, country_code=country)
                    if cached is None:
                        cached = _negative_cache_entry("no search result")
                    elif isinstance(cached, dict):
                        cached = dict(cached)
                        cached["method"] = "search_api"
                        cached["grounded_at_utc"] = datetime.now(timezone.utc).isoformat()
                except Exception as exc:
                    if log:
                        print(
                            f"[wikidata-grounding] origin={origin_tag} "
                            f"api_error label={label[:120]!r} err={exc!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    cached = _negative_cache_entry(str(exc))
                if request_delay_s > 0:
                    time.sleep(request_delay_s)
            elif cached is None:
                cached = _negative_cache_entry("no dump match")

            if cached is not None:
                cache[key] = cached

        node_copy = dict(node)
        if isinstance(cached, dict) and cached.get("_status") != "error":
            qid = qid_from_value(cached.get("id"))
            if qid:
                ext = node_copy.setdefault("external_ids", {})
                if not isinstance(ext, dict):
                    ext = {}
                    node_copy["external_ids"] = ext
                ext["wikidata"] = qid
                if cached.get("method") == "search_api":
                    stats["api_resolved"] += 1
                stats["resolved"] += 1
        out_nodes.append(node_copy)

    if dump_id:
        stats["dump_id"] = dump_id
    if dump_date:
        stats["dump_date"] = dump_date
    if as_of_date:
        stats["as_of_date"] = as_of_date

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if log:
            try:
                nbytes = cache_path.stat().st_size
            except OSError:
                nbytes = -1
            print(
                f"[wikidata-grounding] origin={origin_tag} done "
                f"stats={stats} wrote_cache={cache_path} bytes={nbytes}",
                file=sys.stderr,
                flush=True,
            )

    return out_nodes, stats


def apply_wikidata_grounding(
    payload: dict[str, Any],
    *,
    cache_path: Path,
    request_delay_s: float = 0.25,
    log: bool = True,
    origin: str | None = None,
    dump_manifest_path: Path | None = None,
    api_fallback: bool = True,
) -> dict[str, Any]:
    """Ground Actor/Location nodes; mutates ``payload`` and returns the same object."""

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        return payload
    nodes, stats = ground_snapshot_nodes(
        raw_nodes,
        cache_path=cache_path,
        request_delay_s=request_delay_s,
        log=log,
        origin=origin,
        dump_manifest_path=dump_manifest_path,
        api_fallback=api_fallback,
    )
    payload["nodes"] = nodes
    meta = payload.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["wikidata_grounding"] = stats
    return payload
