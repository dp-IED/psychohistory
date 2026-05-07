"""Doubleword batch pipeline for Wikipedia UniversalEventRecord pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from baselines.node_warehouse_build_v0 import ARAB_SPRING_COUNTRY_RANGE_START
from ingest.io_utils import open_text_auto, write_jsonl_records
from ingest.universal_event_staging import (
    ACTOR_TYPES,
    DATE_PRECISIONS,
    LAG_PRECISION_INDEX,
    LAG_PRECISIONS,
    LOCATION_TYPES,
    RELATION_TYPES,
    REQUIRED_FIELDS,
    UniversalEventRecord,
    BRAUDEL_LAYERS,
    build_staging_row,
    EMBEDDING_TEXT_FIELD,
    DESCRIPTION_MAX_CHARS,
    DESCRIPTION_MIN_CHARS,
    STRUCTURAL_MECHANISM_MAX_CHARS,
    STRUCTURAL_MECHANISM_MIN_CHARS,
    distinct_model_ids,
    load_predictions,
    upsert_universal_event_rows,
)


UTC = dt.timezone.utc
DOUBLEWORD_BASE_URL = "https://api.doubleword.ai/v1"
PIT_ANCHOR_ISO = ARAB_SPRING_COUNTRY_RANGE_START.isoformat()

CATEGORY_NAMES = (
    "cascade",
    "suppression",
    "emergence",
    "precursor",
    "cross_domain_wildcard",
)

WINDOW_MAP = {
    "standard": "24h",
    "24h": "24h",
    "priority": "1h",
    "high": "1h",
    "1h": "1h",
}

MODEL_BY_LANE = {
    "397": "Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a historical event extraction engine. Given a Wikipedia article, "
    "extract structured event records. Each record captures one directional relationship "
    "between two actors at a point in time. Treat history as a causal physical system: "
    "extract mechanisms, transmission channels, constraints, incentives, mobilized groups, "
    "and measurable scale where the article provides them. Extract the structural force "
    "operating on actors, not the sequence of events that resulted. "
    "Every record must name both a subject and an object. If the object is an abstract outcome, "
    "make it a PHENOMENON or IDEA with a stable object_id rather than omitting it. "
    "Return between 3 and 8 records only. "
    "The records array must never be empty. "
    "Do not repeat records. "
    "Follow schema enums strictly and return valid JSON tool arguments."
)

GOLD_REQUIRED_PER_ARTICLE = 3
EXTRACTION_USER_PREFIX = (
    "Extraction constraints: return 3 to 8 records; do not return an empty records array. "
    "If certainty is low, use ESTIMATED/UNKNOWN precision fields rather than omitting records. "
    "Every description must be a mechanism-rich causal explanation, not a short event caption. "
    "Prefer layered time horizons: connect visible actions to medium-run institutional pressures "
    "and slow-moving constraints such as demography, ecology, fiscal capacity, legitimacy, technology, "
    "class coalitions, information networks, security capacity, or trade structure when the article supports them. "
    "For each description, explain why the relation happened and how it propagated or operated: "
    "name the relevant background conditions, transmission channel, relay actors or institutions, "
    "mobilized population, coercive/economic/ecological mechanism, and concrete quantities or dates "
    "when present in the article. Avoid descriptions that merely say X spread, influenced, opposed, "
    "or transformed Y."
)
ENUM_AND_SCORING_GUIDANCE = (
    "Enum guidance: choose the most specific relation_type. INFLUENCES is a last resort, not the default. "
    "Use SPREADS_TO when a repertoire, pathogen, technique, institution, or information pattern propagates; "
    "PRECEDES when a condition enables a later outcome without directly creating it; CREATES when a new "
    "organization, policy, infrastructure, movement, or capacity comes into existence; TRANSFORMS when the "
    "operating rules or composition of an existing system change; SUPPRESSES for coercive containment; "
    "OPPOSES for resistance/conflict without clear containment; LEGITIMISES for authority or acceptability; "
    "DESTROYS for elimination or termination. Prefer date_precision=YEAR when year is known. "
    "Use DAY/MONTH only when exact dates are explicitly present."
)
STRUCTURAL_EXTRACTION_GUIDANCE = (
    "Structural extraction examples: Bad: 'Protests spread from Tunisia to Egypt in 2011.' "
    "Good: 'Demonstrated regime vulnerability in Tunisia activated latent protest networks in Egypt "
    "by signalling that mass mobilisation could overcome security apparatus resistance; satellite TV, "
    "online video, and activist channels transmitted the example across urban youth publics.' "
    "Bad: 'The drought came in three waves.' Good: 'Repeated drought waves interacted with deep plowing "
    "and wheat monoculture to remove grassland soil anchors, so wind converted ordinary aridity into "
    "regional farm abandonment and forced migration.' Before emitting records, prefer the highest-value "
    "mechanisms over chronology: include triggers only when they reveal a channel, constraint, incentive, "
    "coordination pathway, institutional bottleneck, ecological dependency, or population-scale vulnerability."
)
LAG_YEARS_GUIDANCE = (
    "Lag guidance: always emit lag_years. Use 0 for SYNCHRONOUS, null only for UNKNOWN, "
    "and an integer estimate for SHORT, MEDIUM, LONG, GENERATIONAL, or EPOCHAL. "
    "Choose the lag between the cause and represented outcome, not the pace of the article narrative: "
    "SYNCHRONOUS=0, SHORT=1-5, MEDIUM=6-15, LONG=16-40, GENERATIONAL=41-80, EPOCHAL=80+. "
    "Do not choose SHORT merely because the article describes a visible event."
)
OBJECT_AND_DOMAIN_GUIDANCE = (
    "Object/domain guidance: object_id, object_name, and object_type are mandatory for every relation, "
    "including CREATES and PRECEDES. For PRECEDES, use the downstream condition as the object "
    "(for example political_instability, regime_collapse, labor_shortage). Never omit concept_domain: "
    "set it to null for GEOGRAPHIC/UNKNOWN rows, and set a concrete lowercase snake_case domain for "
    "CONCEPTUAL or NETWORK rows, such as protest_coordination, trade_network, disease_ecology, "
    "fiscal_order, legitimacy, scientific_network, surveillance_state, or print_network."
)
DESCRIPTION_QUALITY_GUIDANCE = (
    f"Description quality: write {DESCRIPTION_MIN_CHARS}-{DESCRIPTION_MAX_CHARS} characters. "
    "Use one dense sentence or two short sentences. Include causal language such as because, through, "
    "by, after, or as, and include concrete nouns from the article rather than generic labels. "
    "For cascades, specify the medium of spread (for example broadcast media, social media, trade routes, "
    "migration, military movement, institutional imitation), who relayed it, who was mobilized, and why "
    "nearby sites were susceptible."
)

PILOT_CATEGORY_HINTS: dict[str, dict[str, Any]] = {
    "cascade": {"location_type": "GEOGRAPHIC", "lag_precision": "SHORT", "lag_years": 2},
    "suppression": {"location_type": "GEOGRAPHIC", "lag_precision": "MEDIUM", "lag_years": 4},
    "emergence": {"location_type": "CONCEPTUAL", "lag_precision": "GENERATIONAL", "lag_years": 18},
    "precursor": {"location_type": "GEOGRAPHIC", "lag_precision": "LONG", "lag_years": 9},
    "cross_domain_wildcard": {"location_type": "NETWORK", "lag_precision": "MEDIUM", "lag_years": 5},
}


def _openai_client(api_key: str, base_url: str = DOUBLEWORD_BASE_URL):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai is required for submit/shakeout; install with `pip install openai`") from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_completion_window(tier: str) -> str:
    key = tier.strip().lower()
    if key not in WINDOW_MAP:
        raise ValueError(
            f"unknown batch tier={tier!r}; expected one of: {', '.join(sorted(WINDOW_MAP))}"
        )
    return WINDOW_MAP[key]


def _ensure_article_shape(row: dict[str, Any]) -> dict[str, Any]:
    required = {"article_id", "title", "url", "category", "pit_mode"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"article is missing fields {missing}: {row}")
    article = {
        "article_id": str(row["article_id"]),
        "title": str(row["title"]),
        "url": str(row["url"]),
        "category": str(row["category"]),
        "pit_mode": str(row["pit_mode"]),
        "as_of": str(row["as_of"]) if row.get("as_of") is not None else None,
    }
    return article


def _ensure_gold_shape(row: dict[str, Any]) -> dict[str, Any]:
    if "article_id" not in row:
        raise ValueError(f"gold record missing article_id: {row}")
    payload = dict(row)
    payload.pop("article_id", None)
    UniversalEventRecord.model_validate(payload)
    return row


def validate_pilot_artifacts(
    *,
    article_list_path: Path,
    gold_labels_path: Path,
    pit_map_path: Path,
) -> dict[str, Any]:
    article_rows = _read_json(article_list_path)
    gold_rows = _read_json(gold_labels_path)
    pit_map = _read_json(pit_map_path)
    if not isinstance(article_rows, list):
        raise ValueError("article list must be a JSON array")
    if not isinstance(gold_rows, list):
        raise ValueError("gold labels must be a JSON array")
    if not isinstance(pit_map, dict):
        raise ValueError("pit map must be a JSON object")

    articles = [_ensure_article_shape(dict(row)) for row in article_rows]
    for row in gold_rows:
        _ensure_gold_shape(dict(row))

    article_ids = [row["article_id"] for row in articles]
    if len(set(article_ids)) != len(article_ids):
        raise ValueError("article ids must be unique")
    if len(articles) != 50:
        raise ValueError(f"expected 50 pilot articles; got {len(articles)}")

    category_counts = Counter(row["category"] for row in articles)
    for category in CATEGORY_NAMES:
        if category_counts.get(category, 0) != 10:
            raise ValueError(
                f"category={category!r} must contain 10 articles; got {category_counts.get(category, 0)}"
            )

    overlap_ids = {
        row["article_id"]
        for row in articles
        if row["pit_mode"] == "arab_spring_overlap"
    }
    for row in articles:
        if row["pit_mode"] == "arab_spring_overlap":
            if row["as_of"] != PIT_ANCHOR_ISO:
                raise ValueError(
                    f"overlap article {row['article_id']} must set as_of={PIT_ANCHOR_ISO}; got {row['as_of']}"
                )
        elif row["pit_mode"] == "static":
            if row["as_of"] is not None:
                raise ValueError(f"static article {row['article_id']} must not set as_of")
        else:
            raise ValueError(f"invalid pit_mode for article {row['article_id']}: {row['pit_mode']}")

    if set(pit_map) != overlap_ids:
        missing = sorted(overlap_ids - set(pit_map))
        extra = sorted(set(pit_map) - overlap_ids)
        raise ValueError(
            f"pit map keys must exactly match overlap article ids; missing={missing}, extra={extra}"
        )
    for article_id, as_of in pit_map.items():
        if str(as_of) != PIT_ANCHOR_ISO:
            raise ValueError(
                f"pit map as_of must be {PIT_ANCHOR_ISO} for article_id={article_id}; got {as_of}"
            )

    gold_counts = Counter(str(row["article_id"]) for row in gold_rows)
    missing_gold = [article_id for article_id in article_ids if gold_counts.get(article_id, 0) == 0]
    if missing_gold:
        raise ValueError(f"missing gold records for article_ids={missing_gold}")
    bad_gold_counts = {
        article_id: count
        for article_id, count in sorted(gold_counts.items())
        if count != GOLD_REQUIRED_PER_ARTICLE
    }
    if bad_gold_counts:
        raise ValueError(
            f"expected exactly {GOLD_REQUIRED_PER_ARTICLE} gold records per article; bad={bad_gold_counts}"
        )
    if len(gold_rows) != len(articles) * GOLD_REQUIRED_PER_ARTICLE:
        raise ValueError(
            f"expected {len(articles) * GOLD_REQUIRED_PER_ARTICLE} gold rows; got {len(gold_rows)}"
        )

    return {
        "article_count": len(articles),
        "gold_count": len(gold_rows),
        "overlap_count": len(overlap_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "pit_anchor_date": PIT_ANCHOR_ISO,
    }


def emit_records_tool_schema() -> dict[str, Any]:
    return {
        "name": "emit_records",
        "description": "Emit all UniversalEventRecords extracted from this article",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "records": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "required": list(REQUIRED_FIELDS)
                        + [
                            "date_start",
                            "date_end",
                            "geo_country",
                            "geo_admin1",
                            "concept_domain",
                            "outcome",
                            "outcome_date",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "subject_id": {"type": "string"},
                            "subject_name": {"type": "string"},
                            "subject_type": {"type": "string", "enum": list(ACTOR_TYPES)},
                            "object_id": {"type": "string"},
                            "object_name": {"type": "string"},
                            "object_type": {"type": "string", "enum": list(ACTOR_TYPES)},
                            "relation_type": {"type": "string", "enum": list(RELATION_TYPES)},
                            "relation_description": {"type": "string"},
                            "braudel_layer": {
                                "type": "string",
                                "enum": list(BRAUDEL_LAYERS),
                                "description": "Temporal scale of the mechanism: event surface, medium-term conjuncture, or long-run structure.",
                            },
                            "structural_mechanism": {
                                "type": "string",
                                "minLength": STRUCTURAL_MECHANISM_MIN_CHARS,
                                "maxLength": STRUCTURAL_MECHANISM_MAX_CHARS,
                                "description": (
                                    "Why the relation holds: the causal pathway, constraint, incentive, "
                                    f"transmission channel, or system rule. This is the locked {EMBEDDING_TEXT_FIELD} "
                                    "field for downstream 128-d UniversalEventRecord text embeddings."
                                ),
                            },
                            "date_start": {
                                "type": ["string", "null"],
                                "format": "date",
                                "description": "Earliest supported date for the relation; use null if unavailable.",
                            },
                            "date_end": {
                                "type": ["string", "null"],
                                "format": "date",
                                "description": "Latest supported date for the relation; use null if unavailable.",
                            },
                            "date_precision": {"type": "string", "enum": list(DATE_PRECISIONS)},
                            "location_type": {
                                "type": "string",
                                "enum": list(LOCATION_TYPES),
                                "description": (
                                    "GEOGRAPHIC for place-bound relations; NETWORK for transmission across "
                                    "people/institutions/media/trade/science channels; CONCEPTUAL for abstract "
                                    "domains such as legitimacy, ideology, fiscal order, or protest coordination."
                                ),
                            },
                            "geo_country": {
                                "type": ["string", "null"],
                                "description": "Modern country name for GEOGRAPHIC rows when supported; otherwise null.",
                            },
                            "geo_admin1": {
                                "type": ["string", "null"],
                                "description": "First-level region/state/province for GEOGRAPHIC rows when supported; otherwise null.",
                            },
                            "concept_domain": {
                                "type": ["string", "null"],
                                "description": (
                                    "Never omit. Use null for GEOGRAPHIC/UNKNOWN. For NETWORK or CONCEPTUAL, "
                                    "use a concrete lowercase snake_case domain, e.g. protest_coordination, "
                                    "trade_network, disease_ecology, fiscal_order, legitimacy, scientific_network, "
                                    "surveillance_state, print_network."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "minLength": DESCRIPTION_MIN_CHARS,
                                "maxLength": DESCRIPTION_MAX_CHARS,
                            },
                            "outcome": {"type": ["string", "null"]},
                            "outcome_date": {"type": ["string", "null"], "format": "date"},
                            "lag_years": {"type": ["integer", "null"]},
                            "lag_precision": {"type": "string", "enum": list(LAG_PRECISIONS)},
                            "source_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                    },
                }
            },
            "required": ["records"],
        },
    }


def _lane_to_model_id(model_lane: str) -> str:
    if model_lane not in MODEL_BY_LANE:
        raise ValueError(f"unknown model lane={model_lane!r}; expected one of {', '.join(MODEL_BY_LANE)}")
    return MODEL_BY_LANE[model_lane]


def _build_batch_line(
    *,
    custom_id: str,
    model_id: str,
    article_text: str,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{EXTRACTION_USER_PREFIX}\n\nArticle text:\n{article_text}",
                },
            ],
            "tools": [{"type": "function", "function": emit_records_tool_schema()}],
            "tool_choice": {"type": "function", "function": {"name": "emit_records"}},
        },
    }


def _prepare_article_text(article_text: str, *, max_chars: int = 24000) -> str:
    """Bound prompt size to reduce truncated tool-call JSON on long wiki pages."""
    text = (article_text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _build_user_prompt(*, article_text: str, category: str | None) -> str:
    base = (
        f"{EXTRACTION_USER_PREFIX}\n{ENUM_AND_SCORING_GUIDANCE}\n"
        f"{STRUCTURAL_EXTRACTION_GUIDANCE}\n{LAG_YEARS_GUIDANCE}\n"
        f"{OBJECT_AND_DOMAIN_GUIDANCE}\n{DESCRIPTION_QUALITY_GUIDANCE}"
    )
    hints = PILOT_CATEGORY_HINTS.get(str(category or ""))
    if hints is not None:
        base += (
            "\nPilot category guidance: "
            f"category={category}; prefer location_type={hints['location_type']}; "
            f"prefer lag_precision={hints['lag_precision']} and lag_years around {hints['lag_years']}."
        )
    return f"{base}\n\nArticle text:\n{article_text}"


def build_batch_jsonl_requests(
    *,
    fetched_articles_path: Path,
    model_lane: str,
    out_jsonl: Path,
    out_manifest: Path,
    max_tokens: int = 3072,
    limit: int | None = None,
    max_article_chars: int = 24000,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    fetched_rows = _read_json(fetched_articles_path)
    if not isinstance(fetched_rows, list):
        raise ValueError("fetched articles must be a JSON array")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if max_article_chars < 1:
        raise ValueError("max-article-chars must be >= 1")

    model_id = _lane_to_model_id(model_lane)
    lines: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    selected_rows = fetched_rows[:limit] if limit is not None else fetched_rows
    for index, row in enumerate(selected_rows, start=1):
        article_id = str(row["article_id"])
        custom_id = f"wiki-{article_id}-{model_lane}-{index:03d}"
        article_text = _prepare_article_text(
            str(row.get("text") or ""),
            max_chars=max_article_chars,
        )
        if not article_text:
            raise ValueError(f"fetched article text is empty for article_id={article_id}")
        lines.append(
            _build_batch_line(
                custom_id=custom_id,
                model_id=model_id,
                article_text=article_text,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        )
        # Inject category-aware prompt hints into the request line.
        lines[-1]["body"]["messages"][1]["content"] = _build_user_prompt(
            article_text=article_text,
            category=str(row.get("category") or ""),
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "article_id": article_id,
                "model_id": model_id,
                "source_url": row["url"],
                "revision_id": row.get("revision_id"),
                "fetched_at": row.get("fetched_at"),
                "title": row.get("title"),
                "category": row.get("category"),
                "pit_mode": row.get("pit_mode"),
                "as_of": row.get("as_of"),
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(out_jsonl, lines)
    _write_json(out_manifest, manifest)

    return {
        "request_count": len(lines),
        "model_lane": model_lane,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "max_article_chars": max_article_chars,
        "jsonl_path": str(out_jsonl.resolve()),
        "manifest_path": str(out_manifest.resolve()),
    }


def run_realtime_shakeout(
    *,
    fetched_articles_path: Path,
    model_lanes: list[str],
    sample_size: int,
    api_key: str,
    base_url: str = DOUBLEWORD_BASE_URL,
    max_tokens: int = 1600,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    rows = _read_json(fetched_articles_path)
    if not isinstance(rows, list):
        raise ValueError("fetched articles must be a JSON array")
    if sample_size < 1:
        raise ValueError("sample-size must be >= 1")

    subset = rows[:sample_size]
    client = _openai_client(api_key=api_key, base_url=base_url)
    summary: dict[str, Any] = {"lanes": {}, "sample_size": len(subset)}

    for lane in model_lanes:
        model_id = _lane_to_model_id(lane)
        lane_results: list[dict[str, Any]] = []
        for row in subset:
            article_text = _prepare_article_text(str(row["text"]), max_chars=6000)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        article_text=article_text,
                        category=str(row.get("category") or ""),
                    ),
                },
            ]
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=[{"type": "function", "function": emit_records_tool_schema()}],
                tool_choice={"type": "function", "function": {"name": "emit_records"}},
                max_tokens=max_tokens,
                temperature=0,
            )
            # Keep this parser path aligned with batch ingestion parser.
            payload = {
                "custom_id": f"shakeout-{row['article_id']}",
                "response": {"body": response.model_dump(mode="json")},
            }
            body = payload["response"]["body"]
            finish_reason = None
            choices = body.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
            if finish_reason == "length":
                debug_path = (
                    Path(".context")
                    / "doubleword_shakeout_debug"
                    / f"{lane}_{row['article_id']}_truncated.json"
                )
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(
                    json.dumps(body, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise ValueError(
                    "shakeout output hit max_tokens before completing tool JSON "
                    f"(article_id={row['article_id']}, lane={lane}, model_id={model_id}); "
                    f"raw response saved to {debug_path}. "
                    "Retry with lower article text length and/or lower requested record count."
                )
            records = _extract_records_from_result_line(payload)
            if not records:
                debug_path = (
                    Path(".context")
                    / "doubleword_shakeout_debug"
                    / f"{lane}_{row['article_id']}.json"
                )
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(
                    json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise ValueError(
                    "shakeout returned no parseable records "
                    f"(article_id={row['article_id']}, lane={lane}, model_id={model_id}); "
                    f"raw response saved to {debug_path}"
                )
            valid_count = 0
            for record in records:
                try:
                    UniversalEventRecord.model_validate(record)
                    valid_count += 1
                except Exception:
                    debug_path = (
                        Path(".context")
                        / "doubleword_shakeout_debug"
                        / f"{lane}_{row['article_id']}_invalid_record.json"
                    )
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            if valid_count == 0:
                raise ValueError(
                    "shakeout returned records but none passed schema validation "
                    f"(article_id={row['article_id']}, lane={lane}, model_id={model_id})"
                )
            lane_results.append(
                {
                    "article_id": row["article_id"],
                    "record_count": len(records),
                    "valid_record_count": valid_count,
                }
            )
        summary["lanes"][lane] = {
            "model_id": model_id,
            "checked_articles": lane_results,
        }
    return summary


def submit_batch(
    *,
    input_jsonl_path: Path,
    batch_tier: str,
    run_id: str,
    api_key: str,
    base_url: str = DOUBLEWORD_BASE_URL,
) -> dict[str, Any]:
    completion_window = resolve_completion_window(batch_tier)
    client = _openai_client(api_key=api_key, base_url=base_url)

    with input_jsonl_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"run": run_id, "tier": batch_tier},
    )
    return batch.model_dump(mode="json")


def _download_file_with_resume(
    *,
    api_key: str,
    file_id: str,
    out_path: Path,
    state_path: Path,
    base_url: str = DOUBLEWORD_BASE_URL,
) -> dict[str, Any]:
    offset = 0
    if state_path.exists():
        content = state_path.read_text(encoding="utf-8").strip()
        if content:
            offset = int(content)

    url = f"{base_url}/files/{file_id}/content"
    if offset > 0:
        url = f"{url}?{urlencode({'offset': offset})}"
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=120) as response:
        blob = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if offset > 0 else "wb"
    with out_path.open(mode) as handle:
        handle.write(blob)

    is_incomplete = str(headers.get("x-incomplete", "")).lower() == "true"
    next_offset = int(headers.get("x-last-line", offset))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(str(next_offset), encoding="utf-8")

    return {
        "file_id": file_id,
        "bytes_written": len(blob),
        "incomplete": is_incomplete,
        "offset_start": offset,
        "offset_end": next_offset,
        "out_path": str(out_path.resolve()),
    }


def poll_and_download(
    *,
    api_key: str,
    batch_id: str | None,
    output_file_id: str | None,
    error_file_id: str | None,
    output_path: Path,
    error_path: Path | None,
    output_state_path: Path,
    error_state_path: Path | None,
    base_url: str = DOUBLEWORD_BASE_URL,
) -> dict[str, Any]:
    if not output_file_id and not batch_id:
        raise ValueError("provide either --output-file-id or --batch-id")

    batch_payload: dict[str, Any] | None = None
    if batch_id is not None:
        client = _openai_client(api_key=api_key, base_url=base_url)
        batch = client.batches.retrieve(batch_id)
        batch_payload = batch.model_dump(mode="json")
        output_file_id = output_file_id or batch_payload.get("output_file_id")
        error_file_id = error_file_id or batch_payload.get("error_file_id")

    if not output_file_id:
        raise ValueError("could not resolve output_file_id")

    output_result = _download_file_with_resume(
        api_key=api_key,
        file_id=str(output_file_id),
        out_path=output_path,
        state_path=output_state_path,
        base_url=base_url,
    )

    error_result: dict[str, Any] | None = None
    if error_file_id and error_path and error_state_path:
        error_result = _download_file_with_resume(
            api_key=api_key,
            file_id=str(error_file_id),
            out_path=error_path,
            state_path=error_state_path,
            base_url=base_url,
        )

    return {
        "batch": batch_payload,
        "output": output_result,
        "error": error_result,
    }


def _extract_records_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        fn = tool_call.get("function") or {}
        if fn.get("name") != "emit_records":
            continue
        raw_args = fn.get("arguments")
        if raw_args is None:
            continue
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                # Some models occasionally emit malformed tool-call arguments.
                # Keep scanning other tool calls and fall back to message content parsing below.
                continue
        else:
            parsed = raw_args
        chunk = parsed.get("records") if isinstance(parsed, dict) else None
        if isinstance(chunk, list):
            records.extend(dict(item) for item in chunk)

    # Back-compat: some providers/models return a single function_call object.
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name") == "emit_records":
        raw_args = function_call.get("arguments")
        if raw_args is not None:
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = raw_args
            if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
                records.extend(dict(item) for item in parsed["records"])

    if records:
        return records

    content = message.get("content")
    # Some SDKs emit content as a list of typed blocks; recover text blocks.
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        content = "\n".join(text_parts) if text_parts else None

    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return records
            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return records
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            return [dict(item) for item in parsed["records"]]
        if isinstance(parsed, list):
            return [dict(item) for item in parsed if isinstance(item, dict)]

    # Doubleword batch responses may place JSON directly in reasoning_content.
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        try:
            parsed = json.loads(reasoning_content)
        except json.JSONDecodeError:
            start = reasoning_content.find("{")
            end = reasoning_content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return records
            try:
                parsed = json.loads(reasoning_content[start : end + 1])
            except json.JSONDecodeError:
                return records
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            return [dict(item) for item in parsed["records"] if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [dict(item) for item in parsed if isinstance(item, dict)]

    return records


def _extract_records_from_result_line(result_line: dict[str, Any]) -> list[dict[str, Any]]:
    response = result_line.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    return _extract_records_from_message(message)


def ingest_batch_results(
    *,
    results_jsonl_path: Path,
    manifest_path: Path,
    db_path: Path,
    batch_id: str | None = None,
    dropped_records_path: Path | None = None,
) -> dict[str, Any]:
    manifest_rows = _read_json(manifest_path)
    if not isinstance(manifest_rows, list):
        raise ValueError("manifest must be a JSON array")
    by_custom_id = {str(row["custom_id"]): row for row in manifest_rows}

    staged_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    dropped = 0
    processed = 0
    with open_text_auto(results_jsonl_path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            processed += 1
            result = json.loads(line)
            custom_id = str(result.get("custom_id") or "")
            meta = by_custom_id.get(custom_id)
            if meta is None:
                dropped += 1
                dropped_rows.append(
                    {
                        "custom_id": custom_id,
                        "article_id": None,
                        "title": None,
                        "category": None,
                        "error": "custom_id not found in manifest",
                        "raw_record": result,
                    }
                )
                continue

            records = _extract_records_from_result_line(result)
            for record_payload in records:
                try:
                    record = UniversalEventRecord.model_validate(record_payload)
                    staged_rows.append(
                        build_staging_row(
                            record=record,
                            model_id=str(meta["model_id"]),
                            source_url=str(meta["source_url"]),
                            revision_id=(
                                str(meta["revision_id"]) if meta.get("revision_id") is not None else None
                            ),
                            fetched_at=meta.get("fetched_at"),
                            batch_id=batch_id,
                            custom_id=custom_id,
                            article_id=str(meta["article_id"]),
                        )
                    )
                except Exception as exc:
                    dropped += 1
                    dropped_rows.append(
                        {
                            "custom_id": custom_id,
                            "article_id": str(meta.get("article_id") or ""),
                            "title": meta.get("title"),
                            "category": meta.get("category"),
                            "model_id": meta.get("model_id"),
                            "source_url": meta.get("source_url"),
                            "revision_id": meta.get("revision_id"),
                            "error": str(exc),
                            "raw_record": record_payload,
                        }
                    )

    if dropped_records_path is not None and dropped_rows:
        write_jsonl_records(dropped_records_path, dropped_rows)

    upsert_summary = upsert_universal_event_rows(db_path=db_path, rows=staged_rows)
    return {
        "processed_lines": processed,
        "parsed_records": len(staged_rows),
        "dropped_records": dropped,
        **upsert_summary,
    }


def _tokenize(text: str) -> set[str]:
    return {tok for tok in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split() if tok}


def _description_similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _safe_mean(values: Iterable[float]) -> float:
    seq = list(values)
    return float(sum(seq) / len(seq)) if seq else 0.0


def _pair_records(
    gold_by_article: dict[str, list[dict[str, Any]]],
    pred_by_article: dict[str, list[dict[str, Any]]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], float]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    coverage_parts: list[float] = []
    for article_id, gold_records in gold_by_article.items():
        preds = pred_by_article.get(article_id, [])
        gold_sorted = sorted(gold_records, key=lambda row: row["description"])
        pred_sorted = sorted(preds, key=lambda row: row["description"])
        if gold_sorted:
            coverage_parts.append(min(len(pred_sorted), len(gold_sorted)) / len(gold_sorted))
        for gold_row, pred_row in zip(gold_sorted, pred_sorted):
            pairs.append((gold_row, pred_row))
    return pairs, _safe_mean(coverage_parts)


def _score_model(
    *,
    model_id: str,
    gold_by_article: dict[str, list[dict[str, Any]]],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        pred_by_article[str(row["article_id"])] .append(row)

    pairs, coverage = _pair_records(gold_by_article, pred_by_article)
    if not pairs:
        return {
            "model_id": model_id,
            "coverage": coverage,
            "field_completeness": 0.0,
            "enum_correctness": 0.0,
            "description_quality": 0.0,
            "lag_estimation": 0.0,
            "location_classification": 0.0,
            "overall": 0.0,
            "recommendation": "No predictions found; rerun ingestion before scoring.",
        }

    completeness_scores: list[float] = []
    enum_scores: list[float] = []
    desc_scores: list[float] = []
    lag_scores: list[float] = []
    location_scores: list[float] = []

    enum_fields = ("subject_type", "relation_type", "date_precision", "lag_precision")

    for gold_row, pred_row in pairs:
        present = 0
        for field in REQUIRED_FIELDS:
            value = pred_row.get(field)
            if value is not None and str(value).strip() != "":
                present += 1
        completeness_scores.append(present / len(REQUIRED_FIELDS))

        enum_matches = [1.0 if pred_row.get(field) == gold_row.get(field) else 0.0 for field in enum_fields]
        enum_matches.append(1.0 if pred_row.get("location_type") == gold_row.get("location_type") else 0.0)
        enum_scores.append(_safe_mean(enum_matches))

        desc_jaccard = _description_similarity(str(pred_row.get("description") or ""), str(gold_row.get("description") or ""))
        len_ok = DESCRIPTION_MIN_CHARS <= len(str(pred_row.get("description") or "")) <= DESCRIPTION_MAX_CHARS
        desc_scores.append(0.5 * desc_jaccard + 0.5 * (1.0 if len_ok else 0.0))

        lp_pred = str(pred_row.get("lag_precision") or "UNKNOWN")
        lp_gold = str(gold_row.get("lag_precision") or "UNKNOWN")
        lag_precision_score = 0.0
        if lp_pred in LAG_PRECISION_INDEX and lp_gold in LAG_PRECISION_INDEX:
            delta = abs(LAG_PRECISION_INDEX[lp_pred] - LAG_PRECISION_INDEX[lp_gold])
            lag_precision_score = 1.0 if delta <= 1 else 0.0

        years_pred = pred_row.get("lag_years")
        years_gold = gold_row.get("lag_years")
        lag_years_score = 0.0
        if years_pred is not None and years_gold is not None:
            lag_years_score = 1.0 if abs(int(years_pred) - int(years_gold)) <= 5 else 0.0
        lag_scores.append(0.7 * lag_precision_score + 0.3 * lag_years_score)

        location_scores.append(1.0 if pred_row.get("location_type") == gold_row.get("location_type") else 0.0)

    field_completeness = _safe_mean(completeness_scores) * coverage
    enum_correctness = _safe_mean(enum_scores) * coverage
    description_quality = _safe_mean(desc_scores) * coverage
    lag_estimation = _safe_mean(lag_scores) * coverage
    location_classification = _safe_mean(location_scores) * coverage

    overall = (
        0.25 * field_completeness
        + 0.25 * enum_correctness
        + 0.25 * description_quality
        + 0.15 * lag_estimation
        + 0.10 * location_classification
    )

    if "397B" in model_id:
        if overall >= 0.80:
            recommendation = "Proceed to full run on 397B."
        elif overall >= 0.65:
            recommendation = "Proceed after prompt revision; rerun pilot check on 397B."
        else:
            recommendation = "Do not proceed; revise prompt/schema and rerun pilot."
    else:
        recommendation = "Model threshold policy not defined for this model id."

    return {
        "model_id": model_id,
        "coverage": round(coverage, 4),
        "field_completeness": round(field_completeness, 4),
        "enum_correctness": round(enum_correctness, 4),
        "description_quality": round(description_quality, 4),
        "lag_estimation": round(lag_estimation, 4),
        "location_classification": round(location_classification, 4),
        "overall": round(overall, 4),
        "recommendation": recommendation,
    }


def score_pilot(
    *,
    gold_labels_path: Path,
    db_path: Path,
    out_json: Path,
    out_markdown: Path,
) -> dict[str, Any]:
    gold_rows = _read_json(gold_labels_path)
    if not isinstance(gold_rows, list):
        raise ValueError("gold labels must be a JSON array")

    gold_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        validated = _ensure_gold_shape(dict(row))
        article_id = str(validated["article_id"])
        payload = dict(validated)
        payload.pop("article_id", None)
        gold_by_article[article_id].append(payload)

    model_ids = distinct_model_ids(db_path=db_path)
    if not model_ids:
        raise ValueError("no model outputs found in universal_event_staging")

    model_scores: list[dict[str, Any]] = []
    article_id_set = set(gold_by_article)
    for model_id in model_ids:
        pred_rows = load_predictions(db_path=db_path, model_id=model_id, article_ids=article_id_set)
        model_scores.append(_score_model(model_id=model_id, gold_by_article=gold_by_article, pred_rows=pred_rows))

    payload = {
        "generated_at": dt.datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "gold_record_count": len(gold_rows),
        "model_scores": model_scores,
    }
    _write_json(out_json, payload)

    lines = [
        "# Wikipedia Pilot Scorecard",
        "",
        "| Model | Overall | Field | Enum | Description | Lag | Location | Coverage | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for score in model_scores:
        lines.append(
            "| {model} | {overall:.4f} | {field:.4f} | {enum:.4f} | {desc:.4f} | {lag:.4f} | {loc:.4f} | {cov:.4f} | {decision} |".format(
                model=score["model_id"],
                overall=score["overall"],
                field=score["field_completeness"],
                enum=score["enum_correctness"],
                desc=score["description_quality"],
                lag=score["lag_estimation"],
                loc=score["location_classification"],
                cov=score["coverage"],
                decision=score["recommendation"],
            )
        )
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _api_key_from_args(value: str | None) -> str:
    token = value or os.environ.get("DOUBLEWORD_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not token:
        raise ValueError("missing API key; pass --api-key or set DOUBLEWORD_API_KEY")
    return token


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-artifacts")
    validate.add_argument("--article-list", default="data/pilot/wikipedia_pilot_50.json")
    validate.add_argument("--gold-labels", default="data/pilot/wikipedia_gold_150.json")
    validate.add_argument("--pit-map", default="data/pilot/wikipedia_pit_as_of_map.json")
    validate.add_argument("--out", default=None)

    shakeout = sub.add_parser("shakeout")
    shakeout.add_argument("--fetched-articles", required=True)
    shakeout.add_argument("--model-lanes", default="397", help="Comma-separated model lanes")
    shakeout.add_argument("--sample-size", type=int, default=5)
    shakeout.add_argument("--api-key", default=None)
    shakeout.add_argument("--base-url", default=DOUBLEWORD_BASE_URL)
    shakeout.add_argument("--max-tokens", type=int, default=4096)
    shakeout.add_argument("--out", required=True)

    build = sub.add_parser("build-jsonl")
    build.add_argument("--fetched-articles", required=True)
    build.add_argument("--model-lane", required=True, choices=sorted(MODEL_BY_LANE))
    build.add_argument("--out-jsonl", required=True)
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--max-tokens", type=int, default=3072)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--max-article-chars", type=int, default=24000)

    submit = sub.add_parser("submit-batch")
    submit.add_argument("--input-jsonl", required=True)
    submit.add_argument("--batch-tier", default="standard")
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--api-key", default=None)
    submit.add_argument("--base-url", default=DOUBLEWORD_BASE_URL)
    submit.add_argument("--out", required=True)

    poll = sub.add_parser("poll-download")
    poll.add_argument("--batch-id", default=None)
    poll.add_argument("--output-file-id", default=None)
    poll.add_argument("--error-file-id", default=None)
    poll.add_argument("--api-key", default=None)
    poll.add_argument("--base-url", default=DOUBLEWORD_BASE_URL)
    poll.add_argument("--output-path", required=True)
    poll.add_argument("--error-path", default=None)
    poll.add_argument("--output-state", required=True)
    poll.add_argument("--error-state", default=None)
    poll.add_argument("--out", required=True)

    ingest = sub.add_parser("ingest-results")
    ingest.add_argument("--results-jsonl", required=True)
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--db-path", required=True)
    ingest.add_argument("--batch-id", default=None)
    ingest.add_argument("--dropped-records-out", default=None)
    ingest.add_argument("--out", required=True)

    score = sub.add_parser("score-pilot")
    score.add_argument("--gold-labels", default="data/pilot/wikipedia_gold_150.json")
    score.add_argument("--db-path", required=True)
    score.add_argument("--out-json", required=True)
    score.add_argument("--out-md", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-artifacts":
            summary = validate_pilot_artifacts(
                article_list_path=Path(args.article_list),
                gold_labels_path=Path(args.gold_labels),
                pit_map_path=Path(args.pit_map),
            )
            if args.out:
                _write_json(Path(args.out), summary)
            print(json.dumps(summary, sort_keys=True))
            return 0

        if args.command == "shakeout":
            api_key = _api_key_from_args(args.api_key)
            lanes = [token.strip() for token in str(args.model_lanes).split(",") if token.strip()]
            summary = run_realtime_shakeout(
                fetched_articles_path=Path(args.fetched_articles),
                model_lanes=lanes,
                sample_size=args.sample_size,
                api_key=api_key,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
            )
            _write_json(Path(args.out), summary)
            print(json.dumps({"status": "ok", "out": str(Path(args.out).resolve())}))
            return 0

        if args.command == "build-jsonl":
            summary = build_batch_jsonl_requests(
                fetched_articles_path=Path(args.fetched_articles),
                model_lane=args.model_lane,
                out_jsonl=Path(args.out_jsonl),
                out_manifest=Path(args.manifest_out),
                max_tokens=args.max_tokens,
                limit=args.limit,
                max_article_chars=args.max_article_chars,
            )
            print(json.dumps(summary, sort_keys=True))
            return 0

        if args.command == "submit-batch":
            api_key = _api_key_from_args(args.api_key)
            payload = submit_batch(
                input_jsonl_path=Path(args.input_jsonl),
                batch_tier=args.batch_tier,
                run_id=args.run_id,
                api_key=api_key,
                base_url=args.base_url,
            )
            _write_json(Path(args.out), payload)
            print(json.dumps({"batch_id": payload.get("id"), "out": str(Path(args.out).resolve())}))
            return 0

        if args.command == "poll-download":
            api_key = _api_key_from_args(args.api_key)
            summary = poll_and_download(
                api_key=api_key,
                batch_id=args.batch_id,
                output_file_id=args.output_file_id,
                error_file_id=args.error_file_id,
                output_path=Path(args.output_path),
                error_path=Path(args.error_path) if args.error_path else None,
                output_state_path=Path(args.output_state),
                error_state_path=Path(args.error_state) if args.error_state else None,
                base_url=args.base_url,
            )
            _write_json(Path(args.out), summary)
            print(json.dumps({"status": "ok", "out": str(Path(args.out).resolve())}))
            return 0

        if args.command == "ingest-results":
            summary = ingest_batch_results(
                results_jsonl_path=Path(args.results_jsonl),
                manifest_path=Path(args.manifest),
                db_path=Path(args.db_path),
                batch_id=args.batch_id,
                dropped_records_path=Path(args.dropped_records_out) if args.dropped_records_out else None,
            )
            _write_json(Path(args.out), summary)
            print(json.dumps(summary, sort_keys=True))
            return 0

        if args.command == "score-pilot":
            summary = score_pilot(
                gold_labels_path=Path(args.gold_labels),
                db_path=Path(args.db_path),
                out_json=Path(args.out_json),
                out_markdown=Path(args.out_md),
            )
            print(json.dumps({"status": "ok", "models": [m["model_id"] for m in summary["model_scores"]]}))
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
