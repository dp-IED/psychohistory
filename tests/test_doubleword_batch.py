from __future__ import annotations

import json
from pathlib import Path

from ingest.doubleword_batch import (
    MODEL_BY_LANE,
    main,
    build_batch_jsonl_requests,
    emit_records_tool_schema,
    ingest_batch_results,
    poll_and_download,
    resolve_completion_window,
    run_realtime_shakeout,
    score_pilot,
    submit_batch,
    validate_pilot_artifacts,
)
from ingest.io_utils import write_jsonl_records
from ingest.universal_event_staging import (
    DESCRIPTION_MIN_CHARS,
    EMBEDDING_TEXT_FIELD,
    STRUCTURAL_MECHANISM_MAX_CHARS,
    STRUCTURAL_MECHANISM_MIN_CHARS,
    UniversalEventRecord,
    build_staging_row,
    embedding_text_for_record,
    load_predictions,
    upsert_universal_event_rows,
)


def test_resolve_completion_window_aliases() -> None:
    assert resolve_completion_window("standard") == "24h"
    assert resolve_completion_window("24h") == "24h"
    assert resolve_completion_window("priority") == "1h"
    assert resolve_completion_window("high") == "1h"
    assert resolve_completion_window("1h") == "1h"


def test_model_lane_mapping_uses_dottxt_for_35b() -> None:
    assert MODEL_BY_LANE["35"] == "Qwen/Qwen3.5-35B-A3B-FP8-dottxt"
    assert MODEL_BY_LANE["397"] == "Qwen/Qwen3.5-397B-A17B-FP8-dottxt"


def test_emit_records_schema_enforces_non_empty_bounded_records() -> None:
    schema = emit_records_tool_schema()
    records = schema["parameters"]["properties"]["records"]
    item = records["items"]
    description = records["items"]["properties"]["description"]
    mechanism = records["items"]["properties"]["structural_mechanism"]
    assert records["minItems"] == 3
    assert records["maxItems"] == 8
    assert "braudel_layer" in item["required"]
    assert "structural_mechanism" in item["required"]
    assert "object_id" in item["required"]
    assert "object_name" in item["required"]
    assert "object_type" in item["required"]
    assert item["properties"]["object_id"]["type"] == "string"
    assert item["properties"]["object_name"]["type"] == "string"
    assert item["properties"]["object_type"]["type"] == "string"
    assert description["minLength"] == DESCRIPTION_MIN_CHARS
    assert mechanism["minLength"] == STRUCTURAL_MECHANISM_MIN_CHARS
    assert mechanism["maxLength"] == STRUCTURAL_MECHANISM_MAX_CHARS
    assert EMBEDDING_TEXT_FIELD == "structural_mechanism"
    assert schema["parameters"]["additionalProperties"] is False
    assert records["items"]["additionalProperties"] is False


def test_validate_pilot_artifacts_committed_files() -> None:
    summary = validate_pilot_artifacts(
        article_list_path=Path("data/pilot/wikipedia_pilot_50.json"),
        gold_labels_path=Path("data/pilot/wikipedia_gold_150.json"),
        pit_map_path=Path("data/pilot/wikipedia_pit_as_of_map.json"),
    )
    assert summary["article_count"] == 50
    assert summary["gold_count"] == 150
    assert summary["pit_anchor_date"] == "2010-01-01"
    assert all(summary["category_counts"][category] == 10 for category in summary["category_counts"])


def test_validate_pilot_artifacts_rejects_wrong_overlap_as_of(tmp_path: Path) -> None:
    articles = json.loads(Path("data/pilot/wikipedia_pilot_50.json").read_text(encoding="utf-8"))
    gold = json.loads(Path("data/pilot/wikipedia_gold_150.json").read_text(encoding="utf-8"))
    pit_map = json.loads(Path("data/pilot/wikipedia_pit_as_of_map.json").read_text(encoding="utf-8"))

    target_idx = next(i for i, row in enumerate(articles) if row["pit_mode"] == "arab_spring_overlap")
    target_id = articles[target_idx]["article_id"]
    articles[target_idx]["as_of"] = "2011-01-01"
    pit_map[target_id] = "2011-01-01"

    article_path = tmp_path / "articles.json"
    gold_path = tmp_path / "gold.json"
    pit_path = tmp_path / "pit.json"
    article_path.write_text(json.dumps(articles), encoding="utf-8")
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    pit_path.write_text(json.dumps(pit_map), encoding="utf-8")

    try:
        validate_pilot_artifacts(
            article_list_path=article_path,
            gold_labels_path=gold_path,
            pit_map_path=pit_path,
        )
    except ValueError as exc:
        assert "as_of=2010-01-01" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for incorrect overlap as_of")


def test_build_batch_jsonl_requests_includes_tool_schema(tmp_path: Path) -> None:
    fetched = [
        {
            "article_id": "black-death",
            "title": "Black Death",
            "url": "https://en.wikipedia.org/wiki/Black_Death",
            "category": "cascade",
            "pit_mode": "static",
            "as_of": None,
            "revision_id": "123",
            "fetched_at": "2026-04-27T00:00:00Z",
            "text": "Sample article text that is long enough to look realistic.",
        }
    ]
    fetched_path = tmp_path / "fetched.json"
    fetched_path.write_text(json.dumps(fetched), encoding="utf-8")

    out_jsonl = tmp_path / "batch.jsonl"
    out_manifest = tmp_path / "manifest.json"
    summary = build_batch_jsonl_requests(
        fetched_articles_path=fetched_path,
        model_lane="397",
        out_jsonl=out_jsonl,
        out_manifest=out_manifest,
    )
    assert summary["request_count"] == 1

    line = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"].endswith("dottxt")
    assert line["body"]["tools"][0]["type"] == "function"
    assert line["body"]["tool_choice"]["function"]["name"] == "emit_records"

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest[0]["article_id"] == "black-death"


def test_build_batch_jsonl_requests_limits_count_and_article_chars(tmp_path: Path) -> None:
    fetched = [
        {
            "article_id": f"article-{idx}",
            "title": f"Article {idx}",
            "url": f"https://example.test/{idx}",
            "category": "cascade",
            "pit_mode": "static",
            "as_of": None,
            "revision_id": str(idx),
            "fetched_at": "2026-04-27T00:00:00Z",
            "text": "x" * 200,
        }
        for idx in range(3)
    ]
    fetched_path = tmp_path / "fetched.json"
    fetched_path.write_text(json.dumps(fetched), encoding="utf-8")

    out_jsonl = tmp_path / "batch.jsonl"
    out_manifest = tmp_path / "manifest.json"
    summary = build_batch_jsonl_requests(
        fetched_articles_path=fetched_path,
        model_lane="35",
        out_jsonl=out_jsonl,
        out_manifest=out_manifest,
        max_tokens=8192,
        limit=2,
        max_article_chars=50,
    )

    assert summary["request_count"] == 2
    assert summary["max_tokens"] == 8192
    assert summary["max_article_chars"] == 50
    lines = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all(line["body"]["max_tokens"] == 8192 for line in lines)
    assert all("x" * 51 not in line["body"]["messages"][1]["content"] for line in lines)

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert [row["article_id"] for row in manifest] == ["article-0", "article-1"]


def test_ingest_and_score_roundtrip(tmp_path: Path) -> None:
    gold = []
    rels = ["INFLUENCES", "OPPOSES", "TRANSFORMS"]
    objs = [
        ("civil_society", "Civil society", "GROUP"),
        ("state_apparatus", "State apparatus", "INSTITUTION"),
        ("public_discourse", "Public discourse", "IDEA"),
    ]
    for idx in range(3):
        relation = rels[idx]
        object_id, object_name, object_type = objs[idx]
        gold.append(
            {
                "article_id": "black-death",
                "subject_id": "black_death",
                "subject_name": "Black Death",
                "subject_type": "PHENOMENON",
                "object_id": object_id,
                "object_name": object_name,
                "object_type": object_type,
                "relation_type": relation,
                "relation_description": f"Black Death {relation.lower()} {object_name.lower()}.",
                "braudel_layer": "conjonctures",
                "structural_mechanism": (
                    "Mass mortality reduced labor supply and disrupted administration, shifting bargaining power "
                    "between households, institutions, and local authorities while recurring plague shocks "
                    "changed tax capacity, labor obligations, and elite bargaining strategies across affected regions."
                ),
                "date_start": f"134{idx}-01-01",
                "date_end": f"134{idx}-12-31",
                "date_precision": "YEAR",
                "location_type": "GEOGRAPHIC",
                "geo_country": "MULTI",
                "geo_admin1": None,
                "concept_domain": None,
                "description": (
                    f"Black Death {relation.lower()} {object_name.lower()} "
                    "through mass mortality, labor scarcity, administrative disruption, "
                    "and institutional responses that shifted local incentives."
                ),
                "outcome": "Institutional adaptation.",
                "outcome_date": f"134{idx+1}-12-31",
                "lag_years": 1,
                "lag_precision": "SHORT",
                "source_confidence": 0.7,
            }
        )

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    db_path = tmp_path / "staging.duckdb"
    rows = []
    for record in gold:
        payload = dict(record)
        payload.pop("article_id")
        uer = UniversalEventRecord.model_validate(payload)
        rows.append(
            build_staging_row(
                record=uer,
                model_id="Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
                source_url="https://en.wikipedia.org/wiki/Black_Death",
                revision_id="123",
                fetched_at="2026-04-27T00:00:00Z",
                batch_id="batch-1",
                custom_id="wiki-black-death-397-001",
                article_id="black-death",
            )
        )
    upsert_universal_event_rows(db_path=db_path, rows=rows)

    out_json = tmp_path / "score.json"
    out_md = tmp_path / "score.md"
    payload = score_pilot(
        gold_labels_path=gold_path,
        db_path=db_path,
        out_json=out_json,
        out_markdown=out_md,
    )

    assert payload["model_scores"]
    score = payload["model_scores"][0]
    assert score["overall"] >= 0.8
    assert "Proceed" in score["recommendation"]

    preds = load_predictions(db_path=db_path, model_id="Qwen/Qwen3.5-397B-A17B-FP8-dottxt")
    assert len(preds) == 3


def test_universal_event_record_embedding_text_uses_structural_mechanism() -> None:
    record = UniversalEventRecord.model_validate(
        {
            "subject_id": "printing_press",
            "subject_name": "Printing press",
            "subject_type": "PHENOMENON",
            "object_id": "information_democratisation",
            "object_name": "Information democratisation",
            "object_type": "PHENOMENON",
            "relation_type": "CREATES",
            "relation_description": "Printing press creates information democratisation.",
            "braudel_layer": "longue_duree",
            "structural_mechanism": (
                "Lower copying costs and faster reproducibility weakened manuscript bottlenecks, "
                "letting urban printers scale access to texts beyond clerical gatekeepers while commercial "
                "book markets rewarded standardization, vernacular publication, and rapid replication."
            ),
            "date_precision": "CENTURY",
            "location_type": "NETWORK",
            "concept_domain": "print_network",
            "description": (
                "The printing press created information democratisation through lower copying costs, "
                "faster reproducibility, and urban print networks that weakened manuscript bottlenecks."
            ),
            "lag_years": 60,
            "lag_precision": "GENERATIONAL",
            "source_confidence": 0.8,
        }
    )
    assert embedding_text_for_record(record) == record.structural_mechanism
    assert embedding_text_for_record(record) != record.description


def test_all_relations_require_named_object_including_precedes() -> None:
    base = {
        "subject_id": "youth_bulge",
        "subject_name": "Youth bulge",
        "subject_type": "PHENOMENON",
        "object_id": "political_instability",
        "object_name": "Political instability",
        "object_type": "PHENOMENON",
        "relation_description": "Youth bulge precedes political instability.",
        "braudel_layer": "conjonctures",
        "structural_mechanism": (
            "Urban age concentration increased job-market pressure and protest recruitment capacity "
            "before identifiable regime crisis targets emerged, creating a latent mobilization reservoir "
            "that could be activated by visible repression, price shocks, or elite fragmentation."
        ),
        "date_precision": "YEAR",
        "location_type": "GEOGRAPHIC",
        "geo_country": "MULTI",
        "description": (
            "Youth bulge preceded political instability because urban age concentration increased "
            "employment pressure and protest recruitment before regime crisis targets emerged."
        ),
        "lag_years": 5,
        "lag_precision": "SHORT",
        "source_confidence": 0.75,
    }
    UniversalEventRecord.model_validate({**base, "relation_type": "PRECEDES"})
    try:
        UniversalEventRecord.model_validate(
            {**base, "relation_type": "CREATES", "object_id": "", "object_name": ""}
        )
    except ValueError as exc:
        assert "object_id and object_name are required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected empty object fields to fail")


def test_run_realtime_shakeout_parses_function_tool_calls(monkeypatch, tmp_path: Path) -> None:
    fetched = [
        {
            "article_id": "black-death",
            "title": "Black Death",
            "text": "Historical text",
        }
    ]
    fetched_path = tmp_path / "fetched.json"
    fetched_path.write_text(json.dumps(fetched), encoding="utf-8")

    response_payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "emit_records",
                                "arguments": json.dumps(
                                    {
                                        "records": [
                                            {
                                                "subject_id": "black_death",
                                                "subject_name": "Black Death",
                                                "subject_type": "PHENOMENON",
                                                "object_id": "civil_society",
                                                "object_name": "Civil society",
                                                "object_type": "GROUP",
                                                "relation_type": "INFLUENCES",
                                                "relation_description": "Black Death influences civil society.",
                                                "braudel_layer": "conjonctures",
                                                "structural_mechanism": (
                                                    "Mass mortality reduced labor supply and disrupted administration, "
                                                    "shifting bargaining power between households and institutions while "
                                                    "recurring plague shocks changed tax capacity, labor obligations, "
                                                    "and elite bargaining strategies across affected regions."
                                                ),
                                                "date_start": "1347-01-01",
                                                "date_end": "1347-12-31",
                                                "date_precision": "YEAR",
                                                "location_type": "GEOGRAPHIC",
                                                "geo_country": "MULTI",
                                                "geo_admin1": None,
                                                "concept_domain": None,
                                                "description": (
                                                    "Black Death influenced civil society through mass mortality, labor scarcity, "
                                                    "and institutional responses that changed bargaining power and local obligations."
                                                ),
                                                "outcome": "Institutional adaptation.",
                                                "outcome_date": "1348-12-31",
                                                "lag_years": 1,
                                                "lag_precision": "SHORT",
                                                "source_confidence": 0.7,
                                            }
                                        ]
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }

    class FakeResponse:
        def model_dump(self, mode="json"):
            return response_payload

    class FakeChatCompletions:
        def create(self, **kwargs):
            assert kwargs["tool_choice"]["function"]["name"] == "emit_records"
            assert kwargs["model"] in MODEL_BY_LANE.values()
            return FakeResponse()

    class FakeChat:
        completions = FakeChatCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("ingest.doubleword_batch._openai_client", lambda **_: FakeClient())

    out = run_realtime_shakeout(
        fetched_articles_path=fetched_path,
        model_lanes=["397", "35"],
        sample_size=1,
        api_key="dummy",
    )
    assert out["sample_size"] == 1
    assert out["lanes"]["397"]["checked_articles"][0]["valid_record_count"] >= 1
    assert out["lanes"]["35"]["checked_articles"][0]["valid_record_count"] >= 1


def test_shakeout_cli_passes_max_tokens(monkeypatch, tmp_path: Path) -> None:
    fetched_path = tmp_path / "fetched.json"
    fetched_path.write_text("[]", encoding="utf-8")
    out_path = tmp_path / "shakeout.json"
    captured: dict[str, object] = {}

    def fake_shakeout(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dummy")
    monkeypatch.setattr("ingest.doubleword_batch.run_realtime_shakeout", fake_shakeout)

    assert (
        main(
            [
                "shakeout",
                "--fetched-articles",
                str(fetched_path),
                "--model-lanes",
                "397",
                "--sample-size",
                "1",
                "--max-tokens",
                "8192",
                "--out",
                str(out_path),
            ]
        )
        == 0
    )
    assert captured["max_tokens"] == 8192


def test_submit_batch_uses_resolved_completion_window(monkeypatch, tmp_path: Path) -> None:
    input_jsonl = tmp_path / "batch.jsonl"
    write_jsonl_records(
        input_jsonl,
        [{"custom_id": "x", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m"}}],
    )

    captured: dict[str, object] = {}

    class FakeObj:
        def __init__(self, payload):
            self._payload = payload

        @property
        def id(self):
            return self._payload["id"]

        def model_dump(self, mode="json"):
            return self._payload

    class FakeFiles:
        def create(self, file, purpose):
            assert purpose == "batch"
            assert file.readable()
            return FakeObj({"id": "file_123"})

    class FakeBatches:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeObj({"id": "batch_123", **kwargs})

    class FakeClient:
        files = FakeFiles()
        batches = FakeBatches()

    monkeypatch.setattr("ingest.doubleword_batch._openai_client", lambda **_: FakeClient())
    payload = submit_batch(
        input_jsonl_path=input_jsonl,
        batch_tier="standard",
        run_id="run-1",
        api_key="dummy",
    )
    assert payload["id"] == "batch_123"
    assert captured["completion_window"] == "24h"
    assert captured["endpoint"] == "/v1/chat/completions"


def test_poll_and_download_uses_resume_headers(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "out.jsonl"
    out_state = tmp_path / "out.offset"
    out_state.write_text("2", encoding="utf-8")

    class FakeResp:
        def __init__(self):
            self.headers = {"X-Incomplete": "false", "X-Last-Line": "5"}

        def read(self):
            return b'{"custom_id":"a"}\\n'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=120):
        assert "offset=2" in request.full_url
        return FakeResp()

    monkeypatch.setattr("ingest.doubleword_batch.urlopen", fake_urlopen)

    summary = poll_and_download(
        api_key="dummy",
        batch_id=None,
        output_file_id="file_abc",
        error_file_id=None,
        output_path=out_path,
        error_path=None,
        output_state_path=out_state,
        error_state_path=None,
    )
    assert summary["output"]["offset_start"] == 2
    assert summary["output"]["offset_end"] == 5
    assert out_state.read_text(encoding="utf-8").strip() == "5"
    assert out_path.read_text(encoding="utf-8").strip() == '{"custom_id":"a"}\\n'


def test_ingest_results_parses_reasoning_content_shape(tmp_path: Path) -> None:
    manifest = [
        {
            "custom_id": "wiki-black-death-397-001",
            "article_id": "black-death",
            "model_id": "Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
            "source_url": "https://en.wikipedia.org/wiki/Black_Death",
            "revision_id": "123",
            "fetched_at": "2026-04-27T00:00:00Z",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = {
        "id": "batch_req_1",
        "custom_id": "wiki-black-death-397-001",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": json.dumps(
                                {
                                    "records": [
                                        {
                                            "subject_id": "black_death",
                                            "subject_name": "Black Death",
                                            "subject_type": "PHENOMENON",
                                            "object_id": "civil_society",
                                            "object_name": "Civil society",
                                            "object_type": "GROUP",
                                            "relation_type": "INFLUENCES",
                                            "relation_description": "Black Death influences civil society.",
                                            "braudel_layer": "conjonctures",
                                            "structural_mechanism": (
                                                "Mass mortality reduced labor supply and disrupted administration, "
                                                "shifting bargaining power between households and institutions while "
                                                "recurring plague shocks changed tax capacity, labor obligations, "
                                                "and elite bargaining strategies across affected regions."
                                            ),
                                            "date_start": "1347-01-01",
                                            "date_end": "1347-12-31",
                                            "date_precision": "YEAR",
                                            "location_type": "GEOGRAPHIC",
                                            "geo_country": "MULTI",
                                            "geo_admin1": None,
                                            "concept_domain": None,
                                            "description": (
                                                "Black Death influenced civil society through mass mortality, labor scarcity, "
                                                "and institutional responses that changed bargaining power and local obligations."
                                            ),
                                            "outcome": "Institutional adaptation.",
                                            "outcome_date": "1348-12-31",
                                            "lag_years": 1,
                                            "lag_precision": "SHORT",
                                            "source_confidence": 0.7,
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
                "model": "Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
            },
        },
        "error": None,
    }
    results_path = tmp_path / "output.jsonl"
    write_jsonl_records(results_path, [payload])

    db_path = tmp_path / "staging.duckdb"
    summary = ingest_batch_results(
        results_jsonl_path=results_path,
        manifest_path=manifest_path,
        db_path=db_path,
        batch_id="batch-1",
    )
    assert summary["processed_lines"] == 1
    assert summary["parsed_records"] == 1
    assert summary["upserted_count"] == 1


def test_ingest_results_counts_unknown_custom_id_and_invalid_record_as_dropped(tmp_path: Path) -> None:
    manifest = [
        {
            "custom_id": "wiki-black-death-397-001",
            "article_id": "black-death",
            "model_id": "Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
            "source_url": "https://en.wikipedia.org/wiki/Black_Death",
            "revision_id": "123",
            "fetched_at": "2026-04-27T00:00:00Z",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    invalid_known = {
        "custom_id": "wiki-black-death-397-001",
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "emit_records",
                                        "arguments": json.dumps(
                                            {"records": [{"subject_id": "too_incomplete"}]}
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        },
    }
    unknown_custom_id = {
        "custom_id": "wiki-not-in-manifest-397-999",
        "response": {"body": {"choices": []}},
    }
    results_path = tmp_path / "output.jsonl"
    write_jsonl_records(results_path, [invalid_known, unknown_custom_id])

    summary = ingest_batch_results(
        results_jsonl_path=results_path,
        manifest_path=manifest_path,
        db_path=tmp_path / "staging.duckdb",
        batch_id="batch-1",
    )

    assert summary["processed_lines"] == 2
    assert summary["parsed_records"] == 0
    assert summary["dropped_records"] == 2
    assert summary["upserted_count"] == 0
