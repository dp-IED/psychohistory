"""Wikidata identifier extraction and lightweight graph-artifact enrichment."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


_QID = re.compile(r"^Q[1-9][0-9]*$")
_WIKIDATA_ENTITY = re.compile(r"(?:^|[/#:])(Q[1-9][0-9]*)(?:$|[/?#])")
_SLUG = re.compile(r"[^a-z0-9]+")
DEFAULT_WIKIDATA_CACHE_PATH = Path("cache/wikidata_search_cache.json")
_NEGATIVE_CACHE_TTL = timedelta(days=7)


def normalize_entity_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def qid_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        raw = value.strip()
        if _QID.match(raw):
            return raw
        match = _WIKIDATA_ENTITY.search(raw)
        if match:
            return match.group(1)
    if isinstance(value, list):
        for item in value:
            got = qid_from_value(item)
            if got:
                return got
    if isinstance(value, dict):
        for key in (
            "wikidata",
            "wikidata_id",
            "wikidata_qid",
            "wikidata_qids",
            "wd",
            "qid",
            "id",
            "url",
            "same_as",
            "sameAs",
        ):
            if key in value:
                got = qid_from_value(value[key])
                if got:
                    return got
    return None


def wikidata_qid_for_node(node: dict[str, Any]) -> str | None:
    """Return a Wikidata QID from common graph artifact grounding shapes."""

    for key in (
        "wikidata",
        "wikidata_id",
        "wikidata_qid",
        "wikidata_qids",
        "external_id",
        "external_ids",
        "identifiers",
        "source_ids",
        "same_as",
        "sameAs",
        "provenance",
        "references",
    ):
        if key in node:
            got = qid_from_value(node[key])
            if got:
                return got
    return qid_from_value(node.get("id"))


def node_labels(node: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for key in ("label", "name", "title", "canonical_label"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            labels.add(normalize_entity_label(value))
    aliases = node.get("aliases") or node.get("alt_labels")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                labels.add(normalize_entity_label(alias))
    return labels


def extract_seed_entity_labels(probe_data: dict[str, Any]) -> list[str]:
    """Flatten probe seed_entities blocks into unique candidate entity labels."""

    seeds = probe_data.get("seed_entities") or probe_data.get("seedentities") or {}
    out: list[str] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            label = value.strip()
            key = normalize_entity_label(label)
            if label and key not in seen:
                seen.add(key)
                out.append(label)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(seeds)
    return out


def _seed_entity_items(probe_data: dict[str, Any]) -> list[tuple[str, str]]:
    seeds = probe_data.get("seed_entities") or probe_data.get("seedentities") or {}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def walk(value: Any, group: str) -> None:
        if isinstance(value, str):
            label = value.strip()
            key = normalize_entity_label(label)
            if label and key not in seen:
                seen.add(key)
                out.append((group, label))
            return
        if isinstance(value, list):
            for item in value:
                walk(item, group)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, str(key))

    walk(seeds, "seed")
    return out


def _node_type_for_seed_group(group: str) -> str:
    group_norm = group.strip().casefold()
    if "actor" in group_norm:
        return "Actor"
    if "event" in group_norm or "war" in group_norm or "battle" in group_norm:
        return "Event"
    if "channel" in group_norm:
        return "TransmissionChannel"
    if "institution" in group_norm or "office" in group_norm:
        return "Institution"
    if "policy" in group_norm:
        return "Policy"
    if "claim" in group_norm:
        return "Claim"
    if "legacy" in group_norm:
        return "LegacyAggregate"
    if "lineage" in group_norm:
        return "LineageClaim"
    if "narrative" in group_norm:
        return "Narrative"
    if "place" in group_norm or "region" in group_norm:
        return "Place"
    return "Actor"


def seed_entity_node_type(group: str) -> str:
    return _node_type_for_seed_group(group)


def iter_seed_entity_items(probe_data: dict[str, Any]) -> list[tuple[str, str]]:
    return _seed_entity_items(probe_data)


def _slug_for_label(label: str) -> str:
    slug = _SLUG.sub("_", label.strip().casefold()).strip("_")
    return slug or "entity"


def search_wikidata_entity(
    label: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 2,
    country_code: str | None = None,
) -> dict[str, Any] | None:
    """Search Wikidata for an entity label. Optional ``country_code`` narrows ambiguous names."""

    query = label
    if country_code:
        query = f"{label} {country_code}"

    params = urlencode(
        {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": "1",
            "search": query,
        }
    )
    request = Request(
        f"https://www.wikidata.org/w/api.php?{params}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "psychohistory-autoresearch/0.1 (Wikidata grounding enrichment)",
        },
    )
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 1.0 + attempt
            except ValueError:
                delay = 1.0 + attempt
            time.sleep(delay)
    else:
        return None
    search = payload.get("search") if isinstance(payload, dict) else None
    if not isinstance(search, list) or not search:
        return None
    first = search[0]
    return first if isinstance(first, dict) else None


def _negative_cache_entry(error: str) -> dict[str, str]:
    return {
        "_status": "error",
        "error": error,
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _negative_cache_is_fresh(value: dict[str, Any]) -> bool:
    if value.get("_status") != "error":
        return False
    cached_at = value.get("cached_at_utc")
    if not isinstance(cached_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(cached_at)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - parsed <= _NEGATIVE_CACHE_TTL


def enrich_graph_with_wikidata(
    probe_data: dict[str, Any],
    graph: dict[str, Any],
    *,
    cache_path: Path | None = None,
    request_delay_s: float = 0.25,
) -> dict[str, Any]:
    """Add seed entity nodes with Wikidata QIDs when the search API resolves them."""

    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [dict(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)]
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw_cache, dict):
                cache = raw_cache
        except json.JSONDecodeError:
            cache = {}

    by_label: dict[str, dict[str, Any]] = {}
    existing_ids = {str(node.get("id") or "") for node in nodes}
    for node in nodes:
        for label in node_labels(node):
            by_label.setdefault(label, node)

    resolved = 0
    attempted = 0
    for group, label in _seed_entity_items(probe_data):
        key = normalize_entity_label(label)
        cached = cache.get(key)
        if isinstance(cached, dict) and "error" in cached:
            cached = cached if _negative_cache_is_fresh(cached) else None
        if cached is None:
            attempted += 1
            try:
                cached = search_wikidata_entity(label)
                if cached is None:
                    cached = _negative_cache_entry("no search result")
            except Exception as exc:
                cached = _negative_cache_entry(str(exc))
            cache[key] = cached
            if request_delay_s > 0:
                time.sleep(request_delay_s)
        if not isinstance(cached, dict) or cached.get("_status") == "error":
            continue
        qid = qid_from_value(cached.get("id"))
        if not qid:
            continue
        resolved += 1
        node = by_label.get(key)
        if node is None:
            base_id = f"seed_entity__{_slug_for_label(label)}"
            node_id = base_id
            suffix = 2
            while node_id in existing_ids:
                node_id = f"{base_id}_{suffix}"
                suffix += 1
            node = {
                "id": node_id,
                "type": _node_type_for_seed_group(group),
                "label": cached.get("label") or label,
                "seed_entity": True,
            }
            nodes.append(node)
            existing_ids.add(node_id)
            by_label[key] = node
        if not isinstance(node.get("external_ids"), dict):
            node["external_ids"] = {}
        node["external_ids"]["wikidata"] = qid
        if isinstance(cached.get("label"), str):
            node.setdefault("label", cached["label"])
        if isinstance(cached.get("description"), str):
            node.setdefault("description", cached["description"])
        node.setdefault("source", "Wikidata")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out = dict(graph)
    out["nodes"] = nodes
    out["edges"] = edges
    out["wikidata_enrichment"] = {
        "seed_entities": len(_seed_entity_items(probe_data)),
        "resolved": resolved,
        "api_requests": attempted,
    }
    return out


def merge_wikidata_cache_files(cache_paths: list[Path], target_path: Path) -> None:
    merged: dict[str, Any] = {}
    for cache_path in cache_paths:
        if not cache_path.exists():
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            merged[key] = value
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
