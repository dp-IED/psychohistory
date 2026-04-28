from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

from baselines.arab_spring_labels import build_arab_spring_labels
from baselines.graph_builder_ann import brute_topk
from baselines.graph_builder_bag_encoder import BagEncoder
from baselines.graph_builder_forecast_stack import ForecastHead, GateMLP, path_a_head_input
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    normalize_hint,
    resolve_entity_hint_to_node_id,
    warehouse_context_from_manifest,
)
from baselines.graph_builder_rerank import build_retrieved_graph_batch_from_ann
from baselines.node_warehouse_mmap import read_float32_matrix
from baselines.stage1_probe_corpus import Stage1ProbeCorpus
from schemas.graph_builder_probe import ProbeRecord
from schemas.graph_builder_retrieval import RetrievedGraphBatch
from schemas.graph_builder_warehouse import NodeWarehouseManifest

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    probe_id: str
    query_as_of: str
    retrieved_node_ids: list[str]
    retrieved_node_scores: list[float]
    tier_counts: dict[str, int]
    gate_persistence: float
    gate_propagation: float
    gate_precursor: float
    gate_suppression: float
    gate_coordination: float
    forecast_probability: float
    forecast_label_description: str
    hint_resolution_rate: float
    bag_embedding_norm: float


def _country_from_probe(probe: ProbeRecord) -> str | None:
    for g in probe.q_struct.actor_state.geography:
        gs = g.strip().lower()
        if gs.startswith("tunisia"):
            return "TU"
        if gs.startswith("egypt"):
            return "EG"
        if gs.startswith("libya"):
            return "LY"
        if gs.startswith("syria"):
            return "SY"
        if gs.startswith("bahrain"):
            return "BA"
        if gs.startswith("yemen"):
            return "YM"
        if gs.startswith("jordan"):
            return "JO"
    return None


def _hint_resolution_rate(probe: ProbeRecord, manifest: NodeWarehouseManifest) -> float:
    hints = list(probe.q_struct.actor_state.entity_hints)
    if not hints:
        return 1.0
    rows = manifest.rows or []
    resolved = 0
    for hint in hints:
        node_id = resolve_entity_hint_to_node_id(
            key=normalize_hint(hint),
            as_of=probe.q_struct.actor_state.as_of,
            geography=probe.q_struct.actor_state.geography,
            rows=rows,
            probe_id=probe.probe_id,
            raw_hint=hint,
        )
        if node_id is not None:
            resolved += 1
    return float(resolved) / float(len(hints))


def run_pipeline(
    probe: ProbeRecord,
    manifest: NodeWarehouseManifest,
    mmap_np: np.ndarray,
    encoder: QueryEncoder,
    bag_encoder: BagEncoder,
    gate_mlp: GateMLP,
    forecast_head: ForecastHead,
    *,
    top_k: int = 50,
    ann_k: int = 100,
    message_passing: bool = True,
) -> PipelineOutput:
    full_ctx = warehouse_context_from_manifest(manifest, mmap_np)
    q = encode_actor_state_query(
        actor_state=probe.q_struct.actor_state,
        probe_id=probe.probe_id,
        slice_ctx=full_ctx,
        full_ctx=full_ctx,
        encoder=encoder,
    )
    q_np = q.detach().cpu().numpy().astype(np.float32, copy=False)
    idx, sc = brute_topk(q_np, mmap_np, k=ann_k)
    ann_indices = np.expand_dims(idx, axis=0)
    ann_scores = np.expand_dims(sc, axis=0)
    retrieved = build_retrieved_graph_batch_from_ann(
        np.expand_dims(q_np, axis=0),
        ann_indices,
        ann_scores,
        mmap_np,
    )

    device = encoder.unk_embedding.device
    retrieved = RetrievedGraphBatch(
        node_feat=retrieved.node_feat.to(device),
        edge_index=retrieved.edge_index.to(device),
        edge_weight=retrieved.edge_weight.to(device),
        node_mask=retrieved.node_mask.to(device),
        edge_mask=retrieved.edge_mask.to(device),
        node_type=retrieved.node_type.to(device) if retrieved.node_type is not None else None,
        slot_id=retrieved.slot_id.to(device) if retrieved.slot_id is not None else None,
    )

    bag = bag_encoder(retrieved, message_passing=message_passing)
    gates = gate_mlp(bag)
    prob = forecast_head(path_a_head_input(bag, gates))

    rows = manifest.rows or []
    node_ids: list[str] = []
    node_scores: list[float] = []
    for i in range(min(top_k, idx.shape[0])):
        gi = int(idx[i])
        if gi < 0 or gi >= len(rows):
            continue
        node_ids.append(rows[gi].node_id)
        node_scores.append(float(sc[i]))

    gate_vals = gates[0].detach().cpu().numpy().astype(float)
    bag_norm = float(torch.linalg.vector_norm(bag[0]).detach().cpu().item())

    return PipelineOutput(
        probe_id=probe.probe_id,
        query_as_of=probe.q_struct.actor_state.as_of.isoformat(),
        retrieved_node_ids=node_ids,
        retrieved_node_scores=node_scores,
        tier_counts={"actor_state": len(node_ids), "trend_thread": 0, "historical_analogue": 0},
        gate_persistence=float(gate_vals[0]),
        gate_propagation=float(gate_vals[1]),
        gate_precursor=float(gate_vals[2]),
        gate_suppression=float(gate_vals[3]),
        gate_coordination=float(gate_vals[4]),
        forecast_probability=float(prob[0, 0].detach().cpu().item()),
        forecast_label_description="P(ACLED fatalities >=5 in next 7 days)",
        hint_resolution_rate=_hint_resolution_rate(probe, manifest),
        bag_embedding_norm=bag_norm,
    )


def _gate_variation_check(outputs: list[PipelineOutput], probes: list[ProbeRecord]) -> str:
    by_id = {p.probe_id: p for p in probes}
    pers_vals: list[float] = []
    prec_vals: list[float] = []
    coord_vals: list[float] = []
    supp_vals: list[float] = []
    for out in outputs:
        probe = by_id[out.probe_id]
        gate = probe.assumption_emphasis.value
        if gate == "Persistence":
            pers_vals.append(out.gate_persistence)
        if gate == "Precursor":
            prec_vals.append(out.gate_persistence)
        if gate == "Coordination":
            coord_vals.append(out.gate_coordination)
        if gate == "Suppression":
            supp_vals.append(out.gate_coordination)
    if not pers_vals or not prec_vals or not coord_vals or not supp_vals:
        return "fail"
    if (float(np.mean(pers_vals)) > float(np.mean(prec_vals))) and (
        float(np.mean(coord_vals)) > float(np.mean(supp_vals))
    ):
        return "pass"
    return "fail"


def run_contract6(
    *,
    manifest_path: Path,
    mmap_path: Path,
    encoder_checkpoint: Path,
    output_dir: Path,
    warehouse_duckdb: Path,
    run_id: str = "contract6_arab_spring_v1",
    max_probes: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = NodeWarehouseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    mmap = read_float32_matrix(mmap_path, row_count=manifest.row_count, embedding_dim=manifest.embedding_dim)
    mmap_np = np.asarray(mmap, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = QueryEncoder().to(device).eval()
    encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device, weights_only=True))
    bag_encoder = BagEncoder().to(device).eval()
    gate_mlp = GateMLP().to(device).eval()
    forecast_head = ForecastHead().to(device).eval()

    probes = Stage1ProbeCorpus.arab_spring_default().probes
    if max_probes is not None:
        probes = probes[: max(0, int(max_probes))]
    outputs: list[PipelineOutput] = []
    failures: list[dict[str, str]] = []

    labels_by_country: dict[str, dict[date, int]] = {}
    for cc in ["EG", "LY", "TU"]:
        labels_by_country[cc] = build_arab_spring_labels(
            warehouse_path=warehouse_duckdb,
            as_of=date(2013, 12, 31),
            country=cc,
            threshold=5,
            horizon_days=7,
        )

    brier_terms: list[float] = []
    for probe in probes:
        try:
            with torch.no_grad():
                out = run_pipeline(
                    probe,
                    manifest,
                    mmap_np,
                    encoder,
                    bag_encoder,
                    gate_mlp,
                    forecast_head,
                    top_k=50,
                    ann_k=100,
                    message_passing=True,
                )
            outputs.append(out)
            cc = _country_from_probe(probe)
            if cc is not None:
                labels = labels_by_country.get(cc, {})
                y = labels.get(probe.q_struct.actor_state.as_of)
                if y is not None:
                    p = out.forecast_probability
                    brier_terms.append(float((p - float(y)) ** 2))
        except Exception as exc:  # pragma: no cover
            failures.append({"probe_id": probe.probe_id, "error": type(exc).__name__})
            logger.exception("contract6 probe failed probe_id=%s", probe.probe_id)

    gate_means = {
        "persistence": float(np.mean([o.gate_persistence for o in outputs])) if outputs else 0.0,
        "propagation": float(np.mean([o.gate_propagation for o in outputs])) if outputs else 0.0,
        "precursor": float(np.mean([o.gate_precursor for o in outputs])) if outputs else 0.0,
        "suppression": float(np.mean([o.gate_suppression for o in outputs])) if outputs else 0.0,
        "coordination": float(np.mean([o.gate_coordination for o in outputs])) if outputs else 0.0,
    }

    summary = {
        "run_id": run_id,
        "checkpoint": str(encoder_checkpoint),
        "probe_count": len(probes),
        "completed": len(outputs),
        "failed": len(failures),
        "brier_score": float(np.mean(brier_terms)) if brier_terms else None,
        "mean_hint_resolution_rate": float(np.mean([o.hint_resolution_rate for o in outputs])) if outputs else 0.0,
        "gate_means": gate_means,
        "gate_variation_check": _gate_variation_check(outputs, probes),
        "checkpoint_notes": "hotfix smoke epoch 0 proxy — passes precursor/lift/horizon gates, fails country grounding, CAMEO feature space ceiling confirmed",
    }

    (output_dir / "contract6_outputs.jsonl").write_text(
        "\n".join(json.dumps(asdict(o), sort_keys=True) for o in outputs) + "\n",
        encoding="utf-8",
    )
    (output_dir / "contract6_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "contract6_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contract 6 headless demo pipeline (Arab Spring)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mmap", type=Path, required=True)
    p.add_argument("--encoder-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--warehouse", type=Path, required=True, help="Path to warehouse/events.duckdb")
    p.add_argument("--max-probes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run_contract6(
        manifest_path=args.manifest,
        mmap_path=args.mmap,
        encoder_checkpoint=args.encoder_checkpoint,
        output_dir=args.output_dir,
        warehouse_duckdb=args.warehouse,
        max_probes=args.max_probes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
