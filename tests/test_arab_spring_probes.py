"""Tests for Arab Spring probe corpus: seeds, expansion, gate coverage, hint validation."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from datetime import date

import pytest

from baselines.arab_spring_probes import (
    ARAB_SPRING_BASE_PROBE_DEFS,
    ARAB_SPRING_GEOGRAPHIES,
    ARAB_SPRING_GRANULARITY_LABELS,
    ARAB_SPRING_TEMPORAL_BUCKET_DATES,
    ARAB_SPRING_TEMPORAL_BUCKET_LABELS,
    ARAB_SPRING_TRAINING_CONTEXT_ID,
    build_arab_spring_probe_corpus,
    validate_arab_spring_probe_gate_annotations,
    validate_gate_coverage,
    validate_probe_hints_against_manifest,
)
from baselines.graph_builder_query_encoder import build_hint_index, normalize_hint
from baselines.node_warehouse_build_v0 import build_arab_spring_node_matrix_v0
from baselines.stage1_probe_corpus import Stage1ProbeCorpus
from ingest.event_tape import EventTapeRecord
from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta

_DATE_LOWER = date(2010, 1, 1)
_DATE_UPPER = date(2013, 12, 31)


def _make_probe(
    probe_id: str,
    hint: str,
    gate: AssumptionEmphasis,
    as_of: date = date(2011, 6, 15),
    gate_coverage: AssumptionEmphasis | None = None,
) -> ProbeRecord:
    cov = gate_coverage if gate_coverage is not None else gate
    return ProbeRecord(
        probe_id=probe_id,
        origin=as_of,
        nl_text=f"nl {probe_id}",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=["Tunisia"],
                actor_type=["protest_group"],
                entity_hints=[hint] if hint else [],
                state_flags=["escalating"],
                as_of=as_of,
            ),
        ),
        lens_params=LensParamsV0(horizon_days=7, context_snippet="ctx"),
        assumption_emphasis=gate,
        generation_meta=GenerationMeta(
            template_id="test_t",
            generator_version="test_g",
            seed=1,
            assumption_gate_coverage=cov,
        ),
    )


class TestBaseSeeds:
    def test_base_count_is_20(self) -> None:
        assert len(ARAB_SPRING_BASE_PROBE_DEFS) == 20

    def test_all_gates_represented_in_seeds(self) -> None:
        gates = {p.assumption_emphasis for p in ARAB_SPRING_BASE_PROBE_DEFS}
        assert gates == set(AssumptionEmphasis)

    def test_four_seeds_per_gate(self) -> None:
        counts = Counter(p.assumption_emphasis for p in ARAB_SPRING_BASE_PROBE_DEFS)
        assert all(v == 4 for v in counts.values())

    def test_no_syria_geography_in_seeds(self) -> None:
        for probe in ARAB_SPRING_BASE_PROBE_DEFS:
            geos = [g.lower() for g in probe.q_struct.actor_state.geography]
            assert not any("syria" in g or "sy" == g for g in geos), (
                f"Syria found in probe {probe.probe_id} geographies"
            )

    def test_no_syria_entity_hints_in_seeds(self) -> None:
        """No Syrian actors should appear as primary entity hints."""
        syria_markers = {"syrian", "syria", "damascus", "aleppo", "homs", "bashar", "al-assad"}
        for probe in ARAB_SPRING_BASE_PROBE_DEFS:
            for hint in probe.q_struct.actor_state.entity_hints:
                lower = hint.lower()
                for marker in syria_markers:
                    assert marker not in lower, (
                        f"Syrian marker {marker!r} found in probe {probe.probe_id} hint {hint!r}"
                    )

    def test_seed_dates_in_range(self) -> None:
        for probe in ARAB_SPRING_BASE_PROBE_DEFS:
            as_of = probe.q_struct.actor_state.as_of
            assert _DATE_LOWER <= as_of <= _DATE_UPPER, (
                f"probe {probe.probe_id} as_of={as_of} out of range"
            )

    def test_origin_gte_as_of(self) -> None:
        for probe in ARAB_SPRING_BASE_PROBE_DEFS:
            assert probe.origin >= probe.q_struct.actor_state.as_of, (
                f"probe {probe.probe_id} origin < as_of"
            )

    def test_gate_coverage_matches_emphasis_in_seeds(self) -> None:
        for probe in ARAB_SPRING_BASE_PROBE_DEFS:
            assert probe.generation_meta.assumption_gate_coverage == probe.assumption_emphasis


class TestExpansion:
    def test_corpus_size_gte_150(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        assert len(corpus) >= 150

    def test_corpus_size_equals_full_product(self) -> None:
        """3 geographies × 3 temporal buckets × 3 granularity modes × 3 actor types × 3 horizon days = 243."""
        corpus = build_arab_spring_probe_corpus()
        assert len(corpus) == 243

    def test_probe_ids_unique(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        ids = [p.probe_id for p in corpus]
        assert len(ids) == len(set(ids))

    def test_at_least_3_distinct_geography_tokens(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        geos: set[str] = set()
        for probe in corpus:
            for g in probe.q_struct.actor_state.geography:
                geos.add(g.split("-")[0])  # strip admin1 suffixes
        assert len(geos) >= 3

    def test_all_three_geographies_present(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        found: set[str] = set()
        for probe in corpus:
            for g in probe.q_struct.actor_state.geography:
                base = g.split("-")[0]
                found.add(base)
        assert "Tunisia" in found
        assert "Egypt" in found
        assert "Libya" in found

    def test_no_syria_in_expansion(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        for probe in corpus:
            for g in probe.q_struct.actor_state.geography:
                assert "Syria" not in g and "SY" not in g, (
                    f"Syria found in expanded probe {probe.probe_id}"
                )

    def test_at_least_3_temporal_bucket_labels_in_ids(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        found_labels: set[str] = set()
        for probe in corpus:
            for label in ARAB_SPRING_TEMPORAL_BUCKET_LABELS:
                if label in probe.probe_id:
                    found_labels.add(label)
        assert len(found_labels) >= 3

    def test_all_temporal_dates_in_range(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        for probe in corpus:
            as_of = probe.q_struct.actor_state.as_of
            assert _DATE_LOWER <= as_of <= _DATE_UPPER, (
                f"probe {probe.probe_id} as_of={as_of} out of range"
            )

    def test_temporal_bucket_dates_correct(self) -> None:
        assert ARAB_SPRING_TEMPORAL_BUCKET_DATES[0] == date(2010, 12, 15)
        assert ARAB_SPRING_TEMPORAL_BUCKET_DATES[1] == date(2011, 6, 15)
        assert ARAB_SPRING_TEMPORAL_BUCKET_DATES[2] == date(2013, 6, 15)

    def test_at_least_3_distinct_granularity_labels_in_ids(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        found_labels: set[str] = set()
        for probe in corpus:
            for label in ARAB_SPRING_GRANULARITY_LABELS:
                if label in probe.probe_id:
                    found_labels.add(label)
        assert len(found_labels) >= 3

    def test_all_granularity_labels_present(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        found: set[str] = set()
        for probe in corpus:
            for label in ARAB_SPRING_GRANULARITY_LABELS:
                if label in probe.probe_id:
                    found.add(label)
        for label in ARAB_SPRING_GRANULARITY_LABELS:
            assert label in found, f"granularity label {label!r} not found in any probe_id"

    def test_round_robin_gate_coverage(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        counts = Counter(p.assumption_emphasis for p in corpus)
        assert set(counts.keys()) == set(AssumptionEmphasis)

    def test_expansion_gate_counts_balanced_within_one(self) -> None:
        """Round-robin % 5 over 243 rows: counts differ by at most 1; not axis-skewed."""
        corpus = build_arab_spring_probe_corpus()
        counts = Counter(p.assumption_emphasis for p in corpus)
        cvals = sorted(counts.values())
        assert cvals[-1] - cvals[0] <= 1, (
            f"per-gate counts should differ by at most 1; got {dict(counts)}"
        )
        assert counts == {
            AssumptionEmphasis.PERSISTENCE: 49,
            AssumptionEmphasis.PROPAGATION: 49,
            AssumptionEmphasis.PRECURSOR: 49,
            AssumptionEmphasis.SUPPRESSION: 48,
            AssumptionEmphasis.COORDINATION: 48,
        }

    def test_all_gate_coverage_populated(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        missing = [p.probe_id for p in corpus if p.generation_meta.assumption_gate_coverage is None]
        assert not missing, f"probes missing gate coverage: {missing}"

    def test_origin_gte_as_of_in_expansion(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        for probe in corpus:
            assert probe.origin >= probe.q_struct.actor_state.as_of, (
                f"probe {probe.probe_id} origin < as_of"
            )

    def test_generation_meta_populated(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        for probe in corpus:
            assert probe.generation_meta.template_id
            assert probe.generation_meta.generator_version
            assert probe.generation_meta.seed is not None


class TestGateValidation:
    def test_validate_arab_spring_gate_annotations_passes_on_valid(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        validate_arab_spring_probe_gate_annotations(corpus)

    def test_validate_gate_annotations_raises_on_missing(self) -> None:
        gates = list(AssumptionEmphasis)
        probes = [_make_probe(f"p{i}", "", gates[i % 5], gate_coverage=None) for i in range(5)]
        # patch away coverage
        for p in probes:
            object.__setattr__(
                p.generation_meta,
                "assumption_gate_coverage",
                None,
            )
        with pytest.raises(ValueError, match="assumption_gate_coverage"):
            validate_arab_spring_probe_gate_annotations(probes)

    def test_validate_gate_coverage_passes_on_full_corpus(self) -> None:
        corpus = build_arab_spring_probe_corpus()
        validate_gate_coverage(corpus, training_context_id=ARAB_SPRING_TRAINING_CONTEXT_ID)

    def test_validate_gate_coverage_raises_on_missing_gate(self) -> None:
        probes = [_make_probe("p0", "", AssumptionEmphasis.PERSISTENCE)]
        with pytest.raises(ValueError, match="gate"):
            validate_gate_coverage(probes, training_context_id=ARAB_SPRING_TRAINING_CONTEXT_ID)


class TestHintValidation:
    def _make_manifest(self, hint_keys: list[str]) -> NodeWarehouseManifest:
        row = NodeWarehouseRowMeta(
            node_id="ar_v0|EGC1|slot0",
            first_seen=date(2011, 1, 15),
            admin1_code="EGC1",
            extensions={"entity_hint_keys": hint_keys},
        )
        return NodeWarehouseManifest(
            manifest_version="v0",
            embedding_version="ar_v0",
            mmap_path="nodes.f32",
            row_count=1,
            rows=[row],
            window_days=1,
            as_of=date(2011, 1, 15),
        )

    def test_validates_successfully_when_hints_match(self) -> None:
        manifest = self._make_manifest(["Muslim Brotherhood", "ElBaradei"])
        probe = _make_probe("p0", "Muslim Brotherhood", AssumptionEmphasis.PROPAGATION)
        validate_probe_hints_against_manifest([probe], manifest)

    def test_raises_for_unresolved_hint_with_probe_id_in_message(self) -> None:
        manifest = self._make_manifest(["known actor"])
        probe = _make_probe("p_unresolved", "Unknown Entity XYZ", AssumptionEmphasis.PROPAGATION)
        with pytest.raises(ValueError) as exc_info:
            validate_probe_hints_against_manifest([probe], manifest)
        msg = str(exc_info.value)
        assert "p_unresolved" in msg
        assert "Unknown Entity XYZ" in msg

    def test_raises_listing_multiple_failures(self) -> None:
        manifest = self._make_manifest([])
        probes = [
            _make_probe("pa", "hint_a", AssumptionEmphasis.PERSISTENCE),
            _make_probe("pb", "hint_b", AssumptionEmphasis.PROPAGATION),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_probe_hints_against_manifest(probes, manifest)
        msg = str(exc_info.value)
        assert "pa" in msg
        assert "pb" in msg

    def test_case_insensitive_normalization(self) -> None:
        """Hint stored as 'UGTT' in manifest should match probe hint 'ugtt'."""
        manifest = self._make_manifest(["UGTT"])
        probe = _make_probe("p_case", "ugtt", AssumptionEmphasis.COORDINATION)
        validate_probe_hints_against_manifest([probe], manifest)

    def test_empty_hints_always_pass(self) -> None:
        manifest = self._make_manifest([])
        probe = _make_probe("p_no_hints", "", AssumptionEmphasis.SUPPRESSION)
        # No hints → no failures regardless of manifest
        validate_probe_hints_against_manifest([probe], manifest)


class TestStage1ProbeCorpusArabSpring:
    def test_arab_spring_default_builds_and_validates(self) -> None:
        bundle = Stage1ProbeCorpus.arab_spring_default()
        assert bundle.kind == "arab_spring"
        assert len(bundle.probes) >= 150
        bundle.validate()

    def test_arab_spring_validate_with_none_manifest(self) -> None:
        bundle = Stage1ProbeCorpus.arab_spring_default()
        bundle.validate(manifest=None)

    def test_arab_spring_validate_with_real_manifest_resolves_hint(self) -> None:
        """Probes with empty entity_hints should pass manifest validation."""
        probes_no_hints = [p for p in ARAB_SPRING_BASE_PROBE_DEFS if not p.q_struct.actor_state.entity_hints]
        assert probes_no_hints, "expected at least one seed with no hints"
        gates = list(AssumptionEmphasis)
        padded = list(probes_no_hints) + [
            _make_probe(f"fill_{i}", "", gates[i % 5]) for i in range(5)
        ]
        bundle = Stage1ProbeCorpus(kind="arab_spring", probes=padded)
        empty_manifest = NodeWarehouseManifest(
            manifest_version="v0",
            embedding_version="ar_v0",
            mmap_path="nodes.f32",
            row_count=1,
            rows=[
                NodeWarehouseRowMeta(
                    node_id="ar_v0|EGC1|slot0",
                    first_seen=date(2011, 1, 15),
                    admin1_code="EGC1",
                    extensions={"entity_hint_keys": []},
                )
            ],
            window_days=1,
            as_of=date(2011, 1, 15),
        )
        bundle.validate(manifest=empty_manifest)


class TestRealMatrixHintResolution:
    """Build a minimal EventTapeRecord set, run build_arab_spring_node_matrix_v0,
    and verify that actor names from records appear as resolvable hint_keys."""

    def test_actor_name_flows_into_hint_keys_and_resolves(self) -> None:
        stamp = dt.datetime(2011, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        as_of = date(2011, 1, 15)
        actor_name = "UGTT"

        record = EventTapeRecord.model_validate(
            {
                "source_name": "gdelt_v1_events",
                "source_event_id": "test_hint_001",
                "event_date": as_of,
                "source_available_at": stamp,
                "retrieved_at": stamp,
                "country_code": "TU",
                "admin1_code": "TU01",
                "location_name": "Tunis",
                "latitude": 36.8,
                "longitude": 10.2,
                "event_class": "protest",
                "event_code": "141",
                "event_base_code": "14",
                "event_root_code": "14",
                "quad_class": 2,
                "goldstein_scale": -1.0,
                "num_mentions": 1,
                "num_sources": 1,
                "num_articles": 1,
                "avg_tone": -0.5,
                "actor1_name": actor_name,
                "actor1_country_code": "TU",
                "actor2_name": None,
                "actor2_country_code": None,
                "source_url": None,
                "raw": {},
            }
        )

        _matrix, rows_meta = build_arab_spring_node_matrix_v0(
            [record],
            as_of=as_of,
            window_days=1,
            country_codes=frozenset({"TU"}),
        )

        assert rows_meta, "expected at least one row from the record"
        hint_index = build_hint_index(rows_meta)
        normalized = normalize_hint(actor_name)
        assert normalized in hint_index, (
            f"expected {normalized!r} in hint_index; got keys: {list(hint_index.keys())[:10]}"
        )

        probe = _make_probe("p_real_hint", actor_name, AssumptionEmphasis.PROPAGATION)
        minimal_manifest = NodeWarehouseManifest(
            manifest_version="v0",
            embedding_version="ar_v0",
            mmap_path="nodes.f32",
            row_count=len(rows_meta),
            rows=rows_meta,
            window_days=1,
            as_of=as_of,
        )
        validate_probe_hints_against_manifest([probe], minimal_manifest)
