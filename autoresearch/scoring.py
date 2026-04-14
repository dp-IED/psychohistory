"""Composite scalar scoring from eval signals and stub metric placeholders."""

from __future__ import annotations

from typing import Any

DEFAULT_WEIGHTS: dict[str, float] = {
    "structural": 0.35,
    "functional": 0.35,
    "constraints": 0.15,
    "external_stub": 0.15,
    "traversal": 0.0,
    "discipline": 0.0,
    "ablation_gain": 0.0,
    "source_V": 0.0,
    "source_G": 0.0,
    "source_I": 0.0,
    "source_F": 0.0,
    "source_C": 0.0,
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _gate_score(value: float | bool) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _clip01(float(value))


def compose_score(
    structural_ok: float | bool,
    functional_ok: float | bool,
    constraints_ok: float | bool,
    stub_scores: dict[str, float] | None,
    weights: dict[str, float] | None = None,
    objective_scores: dict[str, float] | None = None,
    source_scores: dict[str, float] | None = None,
    traversal_promotion_threshold: float | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Weighted composite in [0,1].

    Gates may be boolean (0/1) or continuous values in [0,1].
    external_stub is the mean of provided stub metrics (each assumed [0,1]).
    """

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values()) or 1.0

    parts: dict[str, float] = {
        "structural": _gate_score(structural_ok),
        "functional": _gate_score(functional_ok),
        "constraints": _gate_score(constraints_ok),
    }
    for key in ("traversal", "discipline", "ablation_gain"):
        val = 0.0
        if objective_scores and key in objective_scores:
            val = objective_scores[key]
        parts[key] = _gate_score(val)
    for key in ("V", "G", "I", "F", "C"):
        part_key = f"source_{key}"
        val = 0.0
        if source_scores and key in source_scores:
            val = source_scores[key]
        parts[part_key] = _gate_score(val)

    stub = stub_scores or {}
    numeric_stub_values: list[float] = []
    for v in stub.values():
        try:
            numeric_stub_values.append(_clip01(float(v)))
        except (TypeError, ValueError):
            continue
    if numeric_stub_values:
        ext = sum(numeric_stub_values) / len(numeric_stub_values)
    else:
        ext = 0.0
    if traversal_promotion_threshold is not None and parts["traversal"] < _clip01(float(traversal_promotion_threshold)):
        ext = 0.0
        parts["external_stub_blocked"] = 1.0
    else:
        parts["external_stub_blocked"] = 0.0
    parts["external_stub"] = ext

    raw = sum(parts[k] * w.get(k, 0.0) for k in parts) / total_w
    return _clip01(raw), parts


def compose_from_eval_report(report: dict[str, Any], weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    stub = (report.get("stub_scores") or {}) if isinstance(report.get("stub_scores"), dict) else {}
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    gate_rates = meta.get("gate_rates") if isinstance(meta.get("gate_rates"), dict) else {}
    objective_meta = meta.get("objective_v1") if isinstance(meta.get("objective_v1"), dict) else {}
    sources_meta = meta.get("objective_sources_v1") if isinstance(meta.get("objective_sources_v1"), dict) else {}
    applied = bool(objective_meta.get("applied"))
    objective_scores = {
        "traversal": gate_rates.get("traversal", report.get("traversal_score", 0.0)),
        "discipline": gate_rates.get("discipline", report.get("discipline_score", 0.0)),
        "ablation_gain": gate_rates.get("ablation_gain", report.get("ablation_gain_score", 0.0)),
    }
    if not applied:
        objective_scores = {"traversal": 0.0, "discipline": 0.0, "ablation_gain": 0.0}
    source_scores: dict[str, float] = {}
    buckets = sources_meta.get("buckets") if isinstance(sources_meta.get("buckets"), dict) else {}
    for key in ("V", "G", "I", "F", "C"):
        metric = buckets.get(key)
        if isinstance(metric, dict):
            source_scores[key] = metric.get("score", 0.0)
    return compose_score(
        gate_rates.get("structural", bool(report.get("structural_ok"))),
        gate_rates.get("functional", bool(report.get("functional_ok"))),
        gate_rates.get("constraints", bool(report.get("constraints_ok"))),
        stub,
        weights,
        objective_scores=objective_scores,
        source_scores=source_scores,
    )
