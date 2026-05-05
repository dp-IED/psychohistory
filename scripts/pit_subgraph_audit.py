from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
import re

import torch

from baselines.graph_builder_ann import brute_topk
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    warehouse_context_from_manifest,
)
from baselines.node_warehouse_mmap import read_float32_matrix
from baselines.stage1_probe_corpus import Stage1ProbeCorpus
from schemas.graph_builder_probe import ActorStateQuery, ProbeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest


COUNTRY_TOKEN_TO_CODE = {
    "tunisia": "TU",
    "egypt": "EG",
    "libya": "LY",
    "syria": "SY",
    "bahrain": "BA",
    "yemen": "YM",
    "jordan": "JO",
}

# Post-hoc normalization map for manifests where admin1_code mixes canonical country/admin codes
# with admin1 labels. Prefer explicit country/admin fields in warehouse long-term.
ADMIN1_LABEL_TO_COUNTRY_CODE = {
    # Egypt governorates / regions observed in warehouse
    "cairo": "EG",
    "giza": "EG",
    "alexandria": "EG",
    "suez": "EG",
    "gharbia": "EG",
    "sharqia": "EG",
    "dakahlia": "EG",
    "assiut": "EG",
    "menia": "EG",
    "port said": "EG",
    "qalyubia": "EG",
    "beheira": "EG",
    "fayoum": "EG",
    "ismailia": "EG",
    "damietta": "EG",
    "menoufia": "EG",
    "kafr el-sheikh": "EG",
    "north sinai": "EG",
    "south sinai": "EG",
    "aswan": "EG",
    "qena": "EG",
    "beni suef": "EG",
    "luxor": "EG",
    "sohag": "EG",
    "red sea": "EG",
    "new valley": "EG",
    # Libya coarse regions observed in warehouse
    "west": "LY",
    "east": "LY",
    "south": "LY",
}


@dataclass
class ProbeAudit:
    probe_id: str
    assumption_emphasis: str
    geo_bucket: str
    as_of: str
    horizon_days: int | None
    geography: list[str]
    entity_hints: list[str]
    top1_node_id: str
    top1_admin1: str | None
    top1_first_seen: str | None
    top1_time_ok: bool
    top1_country_ok: bool | None
    topk_time_ok_ratio: float
    topk_country_ok_ratio: float | None
    topk_hint_hit_ratio: float
    precursor_hit: bool
    future_rank_lift: float
    horizon_consistency: bool | None



def _expected_country_code(actor_state: ActorStateQuery) -> str | None:
    for g in actor_state.geography:
        gs = g.strip().lower()
        if gs in COUNTRY_TOKEN_TO_CODE:
            return COUNTRY_TOKEN_TO_CODE[gs]
        for token, code in COUNTRY_TOKEN_TO_CODE.items():
            if gs.startswith(token + "-"):
                return code
    return None



def _infer_country_from_admin1(admin1_code: str | None) -> str | None:
    if not admin1_code:
        return None
    s = admin1_code.strip()
    if not s:
        return None
    # Canonical country code (e.g., EG) or prefixed admin code (e.g., EG-XX)
    if re.fullmatch(r"[A-Z]{2}", s):
        return s
    if re.match(r"^[A-Z]{2}-", s):
        return s.split("-", 1)[0]
    # Post-hoc admin1-label normalization fallback
    return ADMIN1_LABEL_TO_COUNTRY_CODE.get(s.lower())


def _country_ok(admin1_code: str | None, expected_country: str | None) -> bool | None:
    if expected_country is None:
        return None
    inferred = _infer_country_from_admin1(admin1_code)
    if inferred is None:
        return False
    return inferred == expected_country



def _hint_hit(node_id: str, hints: list[str]) -> bool:
    s = node_id.lower()
    for h in hints:
        if h.strip().lower() in s:
            return True
    return False



def _geo_bucket(geography: list[str]) -> str:
    tokens: list[str] = []
    for g in geography:
        gs = g.strip().lower()
        if gs in COUNTRY_TOKEN_TO_CODE:
            tokens.append(gs)
            continue
        for token in COUNTRY_TOKEN_TO_CODE:
            if gs.startswith(token + "-"):
                tokens.append(token)
                break
    uniq = sorted(set(tokens))
    if not uniq:
        return "unknown"
    if len(uniq) == 1:
        return uniq[0]
    return "multi_country"



def _summary_rows(rows: list[ProbeAudit]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "probe_count": 0,
            "top1_time_ok_rate": None,
            "topk_time_ok_ratio_mean": None,
            "top1_country_ok_rate": None,
            "topk_country_ok_ratio_mean": None,
            "topk_hint_hit_ratio_mean": None,
            "precursor_hit_rate": None,
            "future_rank_lift_mean": None,
            "horizon_consistency_rate": None,
        }

    top1_time_ok_rate = mean(1.0 if p.top1_time_ok else 0.0 for p in rows)
    topk_time_ok_ratio_mean = mean(p.topk_time_ok_ratio for p in rows)
    topk_hint_hit_ratio_mean = mean(p.topk_hint_hit_ratio for p in rows)
    precursor_hit_rate = mean(1.0 if p.precursor_hit else 0.0 for p in rows)
    future_rank_lift_mean = mean(p.future_rank_lift for p in rows)

    horizon_vals = [p.horizon_consistency for p in rows if p.horizon_consistency is not None]
    horizon_consistency_rate = mean(1.0 if v else 0.0 for v in horizon_vals) if horizon_vals else None

    country_ok_vals = [p.top1_country_ok for p in rows if p.top1_country_ok is not None]
    top1_country_ok_rate = mean(1.0 if v else 0.0 for v in country_ok_vals) if country_ok_vals else None

    country_ratio_vals = [p.topk_country_ok_ratio for p in rows if p.topk_country_ok_ratio is not None]
    topk_country_ok_ratio_mean = mean(country_ratio_vals) if country_ratio_vals else None

    return {
        "probe_count": len(rows),
        "top1_time_ok_rate": top1_time_ok_rate,
        "topk_time_ok_ratio_mean": topk_time_ok_ratio_mean,
        "top1_country_ok_rate": top1_country_ok_rate,
        "topk_country_ok_ratio_mean": topk_country_ok_ratio_mean,
        "topk_hint_hit_ratio_mean": topk_hint_hit_ratio_mean,
        "precursor_hit_rate": precursor_hit_rate,
        "future_rank_lift_mean": future_rank_lift_mean,
        "horizon_consistency_rate": horizon_consistency_rate,
    }


def _summary_delta(candidate: dict[str, float | int | None], baseline: dict[str, float | int | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, cand_val in candidate.items():
        if key == "probe_count":
            continue
        base_val = baseline.get(key)
        if isinstance(cand_val, (int, float)) and isinstance(base_val, (int, float)):
            out[key] = float(cand_val) - float(base_val)
        else:
            out[key] = None
    return out


def _grouped_delta(
    candidate: dict[str, dict[str, float | int | None]],
    baseline: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | None]]:
    keys = sorted(set(candidate) | set(baseline))
    return {
        key: _summary_delta(candidate.get(key, {}), baseline.get(key, {}))
        for key in keys
    }



def _audit_probe(
    probe: ProbeRecord,
    manifest: NodeWarehouseManifest,
    matrix,
    ctx,
    encoder: QueryEncoder,
    k: int,
):
    actor_state = probe.q_struct.actor_state
    q_vec = encode_actor_state_query(
        actor_state=actor_state,
        probe_id=probe.probe_id,
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    ).detach().numpy()

    indices, _scores = brute_topk(q_vec, matrix, k=k)

    as_of = actor_state.as_of
    expected_country = _expected_country_code(actor_state)

    topk_time_ok = []
    topk_country_ok = []
    topk_hint_hit = []
    precursor_hit = False
    min_precursor_rank = k + 1
    min_future_rank = k + 1

    for rank_1idx, idx in enumerate(indices, start=1):
        row = manifest.rows[int(idx)]
        fs = row.first_seen
        time_ok = bool(fs is not None and fs <= as_of)
        hint_ok = _hint_hit(row.node_id, actor_state.entity_hints)
        topk_time_ok.append(time_ok)

        c_ok = _country_ok(row.admin1_code, expected_country)
        if c_ok is not None:
            topk_country_ok.append(c_ok)

        topk_hint_hit.append(hint_ok)

        if fs is not None and fs <= as_of and hint_ok:
            precursor_hit = True
            if rank_1idx < min_precursor_rank:
                min_precursor_rank = rank_1idx

        if fs is not None and fs > as_of and hint_ok and rank_1idx < min_future_rank:
            min_future_rank = rank_1idx

    future_rank_lift = float(min_future_rank - min_precursor_rank)

    top1 = manifest.rows[int(indices[0])]
    top1_time_ok = bool(top1.first_seen is not None and top1.first_seen <= as_of)
    top1_country_ok = _country_ok(top1.admin1_code, expected_country)

    horizon_days = probe.lens_params.horizon_days
    horizon_consistency = None
    if horizon_days is not None and top1.first_seen is not None:
        horizon_consistency = bool(top1.first_seen <= as_of + timedelta(days=horizon_days))

    probe_audit = ProbeAudit(
        probe_id=probe.probe_id,
        assumption_emphasis=probe.assumption_emphasis.value,
        geo_bucket=_geo_bucket(actor_state.geography),
        as_of=as_of.isoformat(),
        horizon_days=horizon_days,
        geography=actor_state.geography,
        entity_hints=actor_state.entity_hints,
        top1_node_id=top1.node_id,
        top1_admin1=top1.admin1_code,
        top1_first_seen=top1.first_seen.isoformat() if top1.first_seen else None,
        top1_time_ok=top1_time_ok,
        top1_country_ok=top1_country_ok,
        topk_time_ok_ratio=sum(topk_time_ok) / len(topk_time_ok),
        topk_country_ok_ratio=(sum(topk_country_ok) / len(topk_country_ok)) if topk_country_ok else None,
        topk_hint_hit_ratio=sum(topk_hint_hit) / len(topk_hint_hit),
        precursor_hit=precursor_hit,
        future_rank_lift=future_rank_lift,
        horizon_consistency=horizon_consistency,
    )
    return probe_audit



def run_audit(manifest_path: Path, mmap_path: Path, checkpoints: list[Path], output_path: Path, k: int = 20) -> None:
    manifest = NodeWarehouseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    matrix = read_float32_matrix(mmap_path, row_count=manifest.row_count, embedding_dim=manifest.embedding_dim)
    ctx = warehouse_context_from_manifest(manifest, matrix)

    corpus = Stage1ProbeCorpus.arab_spring_default()
    corpus.validate(manifest)

    report: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "mmap_path": str(mmap_path),
        "k": k,
        "checkpoints": {},
    }

    for ckpt in checkpoints:
        encoder = QueryEncoder()
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
        encoder.eval()

        probes = [_audit_probe(p, manifest, matrix, ctx, encoder, k=k) for p in corpus.probes]

        by_assumption: dict[str, list[ProbeAudit]] = defaultdict(list)
        by_geo_bucket: dict[str, list[ProbeAudit]] = defaultdict(list)
        for p in probes:
            by_assumption[p.assumption_emphasis].append(p)
            by_geo_bucket[p.geo_bucket].append(p)

        summary_by_assumption = {
            key: _summary_rows(rows)
            for key, rows in sorted(by_assumption.items())
        }
        summary_by_geo = {
            key: _summary_rows(rows)
            for key, rows in sorted(by_geo_bucket.items())
        }

        report["checkpoints"][str(ckpt)] = {
            "summary": _summary_rows(probes),
            "summary_by_assumption_emphasis": summary_by_assumption,
            "summary_by_geo_bucket": summary_by_geo,
            "probe_results": [asdict(p) for p in probes],
        }

    baseline_key = str(checkpoints[0])
    baseline_block = report["checkpoints"][baseline_key]
    for ckpt in checkpoints[1:]:
        key = str(ckpt)
        candidate_block = report["checkpoints"][key]
        candidate_block["delta_vs_baseline"] = _summary_delta(
            candidate_block["summary"],
            baseline_block["summary"],
        )
        candidate_block["delta_by_assumption_emphasis"] = _grouped_delta(
            candidate_block["summary_by_assumption_emphasis"],
            baseline_block["summary_by_assumption_emphasis"],
        )
        candidate_block["delta_by_geo_bucket"] = _grouped_delta(
            candidate_block["summary_by_geo_bucket"],
            baseline_block["summary_by_geo_bucket"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path))



def main() -> int:
    p = argparse.ArgumentParser(description="Audit PIT-faithfulness diagnostics for Stage1 retrieval checkpoints.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mmap", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()

    run_audit(
        manifest_path=args.manifest,
        mmap_path=args.mmap,
        checkpoints=args.checkpoint,
        output_path=args.output,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
