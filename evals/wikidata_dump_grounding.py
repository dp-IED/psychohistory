"""Point-in-time Wikidata dump-slice resolver for label -> QID grounding."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.wikidata_linking import normalize_entity_label, qid_from_value


@dataclass(frozen=True)
class DumpSlice:
    dump_id: str
    dump_date: dt.date
    index_path: Path


@dataclass(frozen=True)
class DumpSelection:
    dump_id: str
    dump_date: str
    index_path: Path


@dataclass(frozen=True)
class DumpLookupResult:
    entity: dict[str, Any] | None
    selection: DumpSelection | None


_MANIFEST_CACHE: dict[Path, list[DumpSlice]] = {}
_INDEX_CACHE: dict[Path, dict[str, list[dict[str, Any]]]] = {}


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value[:10])


def load_dump_manifest(manifest_path: Path) -> list[DumpSlice]:
    path = manifest_path.expanduser().resolve()
    cached = _MANIFEST_CACHE.get(path)
    if cached is not None:
        return cached
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Wikidata dump manifest must be a list: {path}")
    slices: list[DumpSlice] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        dump_id = item.get("dump_id")
        dump_date = item.get("dump_date")
        index_path = item.get("index_path")
        if not isinstance(dump_id, str) or not dump_id.strip():
            continue
        if not isinstance(dump_date, str) or not dump_date.strip():
            continue
        if not isinstance(index_path, str) or not index_path.strip():
            continue
        index = (path.parent / index_path).resolve()
        slices.append(
            DumpSlice(
                dump_id=dump_id.strip(),
                dump_date=_parse_date(dump_date.strip()),
                index_path=index,
            )
        )
    slices.sort(key=lambda item: item.dump_date)
    _MANIFEST_CACHE[path] = slices
    return slices


def select_dump_slice(manifest_path: Path, *, as_of_date: str) -> DumpSelection | None:
    as_of = _parse_date(as_of_date)
    selected: DumpSlice | None = None
    for item in load_dump_manifest(manifest_path):
        if item.dump_date <= as_of:
            selected = item
        else:
            break
    if selected is None:
        return None
    return DumpSelection(
        dump_id=selected.dump_id,
        dump_date=selected.dump_date.isoformat(),
        index_path=selected.index_path,
    )


def _entity_labels(entity: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("label_en", "label"):
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
    for key in ("aliases_en", "aliases"):
        aliases = entity.get(key)
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                labels.append(alias.strip())
    return labels


def _load_index(index_path: Path) -> dict[str, list[dict[str, Any]]]:
    path = index_path.expanduser().resolve()
    cached = _INDEX_CACHE.get(path)
    if cached is not None:
        return cached
    by_label: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entity = json.loads(line)
            if not isinstance(entity, dict):
                continue
            qid = qid_from_value(entity.get("qid") or entity.get("id"))
            if not qid:
                continue
            entity["qid"] = qid
            for raw_label in _entity_labels(entity):
                key = normalize_entity_label(raw_label)
                if key:
                    by_label.setdefault(key, []).append(entity)
    _INDEX_CACHE[path] = by_label
    return by_label


def resolve_qid_from_dump(
    *,
    label: str,
    as_of_date: str,
    manifest_path: Path,
    country_hint: str | None = None,
) -> DumpLookupResult:
    selection = select_dump_slice(manifest_path, as_of_date=as_of_date)
    if selection is None:
        return DumpLookupResult(entity=None, selection=None)

    key = normalize_entity_label(label)
    candidates = list(_load_index(selection.index_path).get(key, []))
    if country_hint:
        hint = normalize_entity_label(country_hint)
        filtered = [
            entity
            for entity in candidates
            if normalize_entity_label(str(entity.get("country") or "")) == hint
        ]
        if filtered:
            candidates = filtered

    entity = candidates[0] if candidates else None
    return DumpLookupResult(entity=entity, selection=selection)
