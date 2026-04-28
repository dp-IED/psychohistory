"""Build Stage1 predictive training tuples with hard-negative strata."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from baselines.stage1_predictive_targets import load_predictive_targets
from schemas.graph_builder_probe import ProbeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest

_GEO_TO_COUNTRY = {
    "tunisia": "TU",
    "egypt": "EG",
    "libya": "LY",
    "syria": "SY",
    "bahrain": "BA",
    "yemen": "YM",
    "jordan": "JO",
}


def _expected_country(probe: ProbeRecord) -> str | None:
    for g in probe.q_struct.actor_state.geography:
        gs = g.strip().lower()
        if gs in _GEO_TO_COUNTRY:
            return _GEO_TO_COUNTRY[gs]
        if "-" in gs:
            tok = gs.split("-", 1)[0]
            if tok in _GEO_TO_COUNTRY:
                return _GEO_TO_COUNTRY[tok]
    return None


def _sample(indices: list[int], cap: int, rng: np.random.Generator) -> list[int]:
    if len(indices) <= cap:
        return sorted(indices)
    pick = rng.choice(np.asarray(indices, dtype=np.int64), size=cap, replace=False)
    return sorted(int(x) for x in pick.tolist())


def build_predictive_training_tuples(
    probes: list[ProbeRecord],
    manifest: NodeWarehouseManifest,
    predictive_targets_meta_path: Path,
    *,
    seed: int = 42,
    max_pos_per_tuple: int = 32,
    max_neg_per_stratum: int = 32,
    near_zero_threshold: float = 1e-6,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if manifest.rows is None:
        raise ValueError("manifest.rows must be present")
    if max_pos_per_tuple <= 0:
        raise ValueError("max_pos_per_tuple must be > 0")
    if max_neg_per_stratum <= 0:
        raise ValueError("max_neg_per_stratum must be > 0")

    meta, horizon_targets = load_predictive_targets(predictive_targets_meta_path, manifest)
    rng = np.random.default_rng(seed)

    rows = manifest.rows
    by_country: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        admin1 = (row.admin1_code or "").strip().upper()
        if admin1:
            by_country[admin1[:2]].append(i)

    all_indices = list(range(manifest.row_count))
    tuples: list[dict[str, object]] = []

    per_horizon_tuple_counts: dict[str, int] = {str(h): 0 for h in meta.horizons_days}
    per_stratum_negative_counts: dict[str, int] = {
        "same_time_wrong_domain": 0,
        "same_domain_wrong_horizon": 0,
        "same_region_non_precursor": 0,
    }
    empty_tuple_count = 0

    positive_threshold = near_zero_threshold
    non_precursor_threshold = near_zero_threshold

    for probe in probes:
        expected_country = _expected_country(probe)
        as_of = probe.q_struct.actor_state.as_of
        same_country = by_country.get(expected_country, []) if expected_country else []
        same_country_set = set(same_country)

        same_time_wrong_domain_candidates = [
            i
            for i in all_indices
            if i not in same_country_set
            and rows[i].first_seen is not None
            and rows[i].first_seen <= as_of
        ]
        same_time_wrong_domain_candidates.sort(key=lambda i: abs((rows[i].first_seen or as_of) - as_of).days)

        for h in meta.horizons_days:
            arr = horizon_targets[h]

            pos_candidates = [
                i
                for i in same_country
                if rows[i].first_seen is not None
                and rows[i].first_seen <= as_of
                and float(arr[i]) > positive_threshold
            ]
            pos = _sample(pos_candidates, max_pos_per_tuple, rng)

            same_domain_wrong_horizon_candidates_set: set[int] = set()
            same_region_non_precursor_candidates_set: set[int] = set()
            for i in same_country:
                if rows[i].first_seen is None or rows[i].first_seen > as_of:
                    continue
                y_h = float(arr[i])
                if y_h > non_precursor_threshold:
                    continue

                same_region_non_precursor_candidates_set.add(i)
                if any(
                    (oh != h) and (float(horizon_targets[oh][i]) > positive_threshold)
                    for oh in meta.horizons_days
                ):
                    same_domain_wrong_horizon_candidates_set.add(i)

            wrong_horizon = set(
                _sample(sorted(same_domain_wrong_horizon_candidates_set), max_neg_per_stratum, rng),
            )
            non_precursor_pool = same_region_non_precursor_candidates_set - wrong_horizon
            non_precursor = set(
                _sample(sorted(non_precursor_pool), max_neg_per_stratum, rng),
            )
            wrong_domain_pool = set(same_time_wrong_domain_candidates[: max_neg_per_stratum * 8]) - wrong_horizon - non_precursor
            wrong_domain = set(
                _sample(sorted(wrong_domain_pool), max_neg_per_stratum, rng),
            )

            neg = sorted(wrong_horizon | non_precursor | wrong_domain)
            if (not pos) or (not neg):
                empty_tuple_count += 1

            tuple_rec = {
                "probe_id": probe.probe_id,
                "horizon_days": int(h),
                "pos_global_indices": pos,
                "neg_global_indices": neg,
                "strata_counts": {
                    "same_time_wrong_domain": len(wrong_domain),
                    "same_domain_wrong_horizon": len(wrong_horizon),
                    "same_region_non_precursor": len(non_precursor),
                },
                "strata_indices": {
                    "same_time_wrong_domain": sorted(wrong_domain),
                    "same_domain_wrong_horizon": sorted(wrong_horizon),
                    "same_region_non_precursor": sorted(non_precursor),
                },
            }
            tuples.append(tuple_rec)
            per_horizon_tuple_counts[str(h)] += 1
            per_stratum_negative_counts["same_time_wrong_domain"] += len(wrong_domain)
            per_stratum_negative_counts["same_domain_wrong_horizon"] += len(wrong_horizon)
            per_stratum_negative_counts["same_region_non_precursor"] += len(non_precursor)

    total = len(tuples)
    metadata = {
        "seed": seed,
        "tuple_count": total,
        "per_horizon_tuple_counts": per_horizon_tuple_counts,
        "per_stratum_negative_counts": per_stratum_negative_counts,
        "empty_tuple_count": empty_tuple_count,
        "empty_tuple_rate": (float(empty_tuple_count) / float(total)) if total > 0 else 0.0,
        "horizons_days": list(meta.horizons_days),
    }
    return tuples, metadata


__all__ = ["build_predictive_training_tuples"]
