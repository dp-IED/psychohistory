from __future__ import annotations

from pathlib import Path
from datetime import date

from baselines.graph_builder_stage1_train import (
    STAGE1_OBJECTIVE_PREDICTIVE_RANKCHANGE_V1,
    _build_predictive_probe_horizon_mapping,
    _parse_args,
    _resolve_horizon_weights,
)
from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)


def test_parse_objective_and_predictive_targets_meta():
    args = _parse_args(
        [
            "--manifest",
            "m.json",
            "--mmap",
            "x.mmap",
            "--output-dir",
            "out",
            "--pairs-metadata",
            "pairs.meta.json",
            "--objective",
            STAGE1_OBJECTIVE_PREDICTIVE_RANKCHANGE_V1,
            "--predictive-targets-meta",
            "targets.meta.json",
            "--predictive-tuple-weight",
            "1.5",
            "--predictive-fallback-weight",
            "0.2",
            "--predictive-stratum-weight-same-time-wrong-domain",
            "3.0",
            "--predictive-stratum-weight-same-domain-wrong-horizon",
            "1.1",
            "--predictive-stratum-weight-same-region-non-precursor",
            "0.9",
            "--predictive-horizon-weight-mode",
            "inverse_horizon",
        ]
    )
    assert args.objective == STAGE1_OBJECTIVE_PREDICTIVE_RANKCHANGE_V1
    assert args.predictive_targets_meta == Path("targets.meta.json")
    assert args.predictive_tuple_weight == 1.5
    assert args.predictive_fallback_weight == 0.2
    assert args.predictive_stratum_weight_same_time_wrong_domain == 3.0
    assert args.predictive_stratum_weight_same_domain_wrong_horizon == 1.1
    assert args.predictive_stratum_weight_same_region_non_precursor == 0.9
    assert args.predictive_horizon_weight_mode == "inverse_horizon"


def test_resolve_horizon_weight_modes_normalize():
    uniform = _resolve_horizon_weights(
        horizons_days=[30, 60],
        meta_horizon_weights=[1.0, 0.5],
        mode="uniform",
    )
    assert uniform == {30: 0.5, 60: 0.5}

    inv = _resolve_horizon_weights(
        horizons_days=[30, 60],
        meta_horizon_weights=[1.0, 0.5],
        mode="inverse_horizon",
    )
    assert abs(inv[30] - (2.0 / 3.0)) < 1e-8
    assert abs(inv[60] - (1.0 / 3.0)) < 1e-8
    assert abs(sum(inv.values()) - 1.0) < 1e-8


def test_build_predictive_probe_horizon_mapping_maps_7_14_21_to_30():
    def _probe(pid: str, horizon: int) -> ProbeRecord:
        return ProbeRecord(
            probe_id=pid,
            origin=date(2011, 1, 1),
            nl_text=pid,
            q_struct=QStructV0(actor_state=ActorStateQuery(geography=["egypt"], actor_type=["group"], as_of=date(2011, 1, 1))),
            lens_params=LensParamsV0(horizon_days=horizon),
            assumption_emphasis=AssumptionEmphasis.PRECURSOR,
            generation_meta=GenerationMeta(template_id="t", generator_version="v", assumption_gate_coverage=AssumptionEmphasis.PRECURSOR),
        )

    mapping = _build_predictive_probe_horizon_mapping(
        [_probe("p7", 7), _probe("p14", 14), _probe("p21", 21)],
        [30, 180, 730, 3650, 10950],
    )
    assert mapping == {"7": 30, "14": 30, "21": 30}
