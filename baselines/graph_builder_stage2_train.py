"""Stage 2 weak supervision: bag encoder + gated forecast head with probe labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from baselines.france_plumbing_probes import (
    build_france_plumbing_probe_corpus,
    validate_france_plumbing_gate_annotations,
)
from baselines.graph_builder_ann import brute_topk
from baselines.graph_builder_bag_encoder import BagEncoder
from baselines.graph_builder_forecast_stack import (
    ForecastHead,
    GateMLP,
    FORECAST_GATE_ORDER,
    forecast_brier_log_loss,
    path_a_head_input,
)
from baselines.graph_builder_probe_labels import LABEL_VERSION_V0, ProbeLabelRow
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    warehouse_context_from_manifest,
)
from baselines.graph_builder_rerank import build_retrieved_graph_batch_from_ann
from baselines.node_warehouse_mmap import read_float32_matrix
from schemas.graph_builder_probe import AssumptionEmphasis, ProbeRecord
from schemas.graph_builder_retrieval import RetrievedGraphBatch
from schemas.graph_builder_warehouse import NodeWarehouseManifest

logger = logging.getLogger(__name__)

_DEFAULT_LR = 1e-3
INFOCE_TEMPERATURE = 0.07

# Training expects labels emitted with this version string (see ``graph_builder_probe_labels``).
STAGE2_LABEL_VERSION = LABEL_VERSION_V0


def _load_probe_label_table(
    labels_jsonl_path: Path,
    *,
    expected_version: str,
) -> dict[str, tuple[float, AssumptionEmphasis]]:
    """Map ``probe_id`` to ``(y as 0/1 float, label gate)`` for rows matching ``expected_version``."""

    out: dict[str, tuple[float, AssumptionEmphasis]] = {}
    text = labels_jsonl_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = ProbeLabelRow.model_validate_json(line)
        if row.label_version != expected_version:
            continue
        yf = 1.0 if row.y else 0.0
        out[row.probe_id] = (yf, row.gate)
    return out


def _brier_half_mse(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((p - y) ** 2).mean()


def _mask_topk_slots(batch: RetrievedGraphBatch, row: int, k: int) -> RetrievedGraphBatch:
    """Clear the first ``k`` node slots for ``row`` (rerank order: top-k cosine neighbors)."""

    nm = batch.node_mask.clone()
    nm[row, :k] = False
    return RetrievedGraphBatch(
        node_feat=batch.node_feat,
        edge_index=batch.edge_index,
        edge_weight=batch.edge_weight,
        node_mask=nm,
        edge_mask=batch.edge_mask,
        node_type=batch.node_type,
        slot_id=batch.slot_id,
    )


def _per_probe_infonce_topk(
    q: torch.Tensor,
    *,
    node_feat_row: torch.Tensor,
    active_count: int,
    k: int,
    temperature: float,
) -> torch.Tensor | None:
    """InfoNCE: positive = mean of top-``k`` node embeddings; negatives = remaining active slots.

    Node features are detached (``detach()``) for both positives and negatives; ``q`` keeps grad.
    Returns ``None`` if ``active_count <= k`` (no negatives) or ``k < 1``.
    """

    if k < 1 or active_count <= k:
        return None
    pos_stack = torch.stack([node_feat_row[j].detach() for j in range(k)], dim=0)
    pos_mean = pos_stack.mean(dim=0)
    neg_embs = [node_feat_row[j].detach() for j in range(k, active_count)]
    pos_logit = (q * pos_mean).sum() / temperature
    if neg_embs:
        neg_logits = torch.stack([(q * neg).sum() for neg in neg_embs], dim=0) / temperature
        logits = torch.cat([pos_logit.unsqueeze(0), neg_logits], dim=0)
    else:
        logits = pos_logit.unsqueeze(0)
    loss_b = -(logits[0] - torch.logsumexp(logits, dim=0))
    return loss_b


def run_stage2_training(
    manifest_path: Path,
    mmap_path: Path,
    labels_jsonl_path: Path,
    stage1_ckpt_path: Path,
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    tau: float = 0.0,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = NodeWarehouseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    label_table = _load_probe_label_table(labels_jsonl_path, expected_version=STAGE2_LABEL_VERSION)

    mmap = read_float32_matrix(
        Path(mmap_path),
        row_count=manifest.row_count,
        embedding_dim=manifest.embedding_dim,
    )
    mmap_np = np.asarray(mmap, dtype=np.float32)
    slice_ctx = full_ctx = warehouse_context_from_manifest(manifest, mmap_np)

    corpus = build_france_plumbing_probe_corpus()
    validate_france_plumbing_gate_annotations(corpus)

    encoder = QueryEncoder().to(device)
    encoder.load_state_dict(torch.load(stage1_ckpt_path, map_location=device, weights_only=True))
    encoder.train()

    bag_encoder = BagEncoder().to(device).train()
    gate_mlp = GateMLP().to(device).train()
    forecast_head = ForecastHead().to(device).train()

    params = list(encoder.parameters()) + list(bag_encoder.parameters()) + list(gate_mlp.parameters()) + list(
        forecast_head.parameters()
    )
    opt = Adam(params, lr=_DEFAULT_LR)
    # TODO(optimizer): Stage 1 only saves query_encoder weights, not Adam state. When Stage 1 checkpointing is extended to persist optimizer state, load it here so "carry over" is literal.

    global_step = 0

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        order = np.random.permutation(len(corpus))
        for start in range(0, len(corpus), batch_size):
            batch_idx = order[start : start + batch_size]
            batch: list[ProbeRecord] = [corpus[i] for i in batch_idx]

            q_list: list[torch.Tensor] = []
            for probe in batch:
                q_list.append(
                    encode_actor_state_query(
                        actor_state=probe.q_struct.actor_state,
                        probe_id=probe.probe_id,
                        slice_ctx=slice_ctx,
                        full_ctx=full_ctx,
                        encoder=encoder,
                    )
                )
            queries = torch.stack(q_list, dim=0)
            queries_np = queries.detach().cpu().numpy().astype(np.float32, copy=False)

            ann_indices = np.zeros((queries_np.shape[0], 100), dtype=np.int64)
            ann_scores = np.zeros((queries_np.shape[0], 100), dtype=np.float32)
            for b in range(queries_np.shape[0]):
                idx, sc = brute_topk(queries_np[b], mmap_np, k=100)
                ann_indices[b] = idx
                ann_scores[b] = sc

            retrieved = build_retrieved_graph_batch_from_ann(
                queries_np,
                ann_indices,
                ann_scores,
                mmap_np,
            )
            retrieved = RetrievedGraphBatch(
                node_feat=retrieved.node_feat.to(device),
                edge_index=retrieved.edge_index.to(device),
                edge_weight=retrieved.edge_weight.to(device),
                node_mask=retrieved.node_mask.to(device),
                edge_mask=retrieved.edge_mask.to(device),
                node_type=retrieved.node_type.to(device) if retrieved.node_type is not None else None,
                slot_id=retrieved.slot_id.to(device) if retrieved.slot_id is not None else None,
            )

            bag = bag_encoder(retrieved)
            gates = gate_mlp(bag)
            p_all = forecast_head(path_a_head_input(bag, gates))

            gate_batch_count: dict[AssumptionEmphasis, int] = {g: 0 for g in AssumptionEmphasis}
            for probe in batch:
                cov = probe.generation_meta.assumption_gate_coverage
                if cov is None:
                    raise ValueError(f"probe {probe.probe_id!r} missing assumption_gate_coverage after validation")
                gate_batch_count[cov] += 1

            forecast_terms: list[torch.Tensor] = []
            infonce_terms: list[torch.Tensor] = []

            for b, probe in enumerate(batch):
                if probe.probe_id not in label_table:
                    logger.debug("skip probe %r: no label row for version %s", probe.probe_id, STAGE2_LABEL_VERSION)
                    continue
                y_f, label_gate = label_table[probe.probe_id]
                cov = probe.generation_meta.assumption_gate_coverage
                if cov is None or cov != label_gate:
                    logger.debug(
                        "skip probe %r: label gate %s does not match probe coverage %s",
                        probe.probe_id,
                        label_gate,
                        cov,
                    )
                    continue

                y = torch.tensor([[y_f]], device=device, dtype=p_all.dtype)
                forecast_terms.append(forecast_brier_log_loss(p_all[b : b + 1], y))

                active_count = int(retrieved.node_mask[b].sum().item())
                k = min(10, active_count)

                delta_brier = torch.zeros((), device=device, dtype=p_all.dtype)
                if active_count <= k:
                    # No strict subset to ablate: ref equals full or empty bag; ``delta_brier`` stays 0 (test-friendly).
                    pass
                else:
                    with torch.no_grad():
                        ref_batch = _mask_topk_slots(retrieved, b, k)
                        bag_r = bag_encoder(ref_batch)
                        gates_r = gate_mlp(bag_r)
                        p_r = forecast_head(path_a_head_input(bag_r, gates_r))[b : b + 1]
                        b_full = _brier_half_mse(p_all[b : b + 1].detach(), y)
                        b_ref = _brier_half_mse(p_r, y)
                        delta_brier = b_ref - b_full

                if float(delta_brier.item()) > tau:
                    loss_n = _per_probe_infonce_topk(
                        queries[b],
                        node_feat_row=retrieved.node_feat[b],
                        active_count=active_count,
                        k=k,
                        temperature=INFOCE_TEMPERATURE,
                    )
                    if loss_n is not None:
                        infonce_terms.append(loss_n)

            if not forecast_terms:
                logger.warning("step=%s: no probes with matching labels in this batch", global_step)
                global_step += 1
                continue

            l_forecast = torch.stack(forecast_terms, dim=0).mean()
            if infonce_terms:
                l_infonce = torch.stack(infonce_terms, dim=0).mean()
            else:
                l_infonce = torch.zeros((), device=device, dtype=l_forecast.dtype)
            step_loss = l_forecast + l_infonce

            opt.zero_grad(set_to_none=True)
            step_loss.backward()
            opt.step()
            epoch_losses.append(float(step_loss.detach().cpu().item()))
            logger.info(
                "step=%s loss=%.6f forecast=%.6f infonce=%.6f",
                global_step,
                epoch_losses[-1],
                float(l_forecast.detach().cpu().item()),
                float(l_infonce.detach().cpu().item()),
            )

            for i, g in enumerate(FORECAST_GATE_ORDER):
                mean_act = float(gates[:, i].mean().detach().cpu().item())
                n_cov = gate_batch_count[g]
                logger.info(
                    "gate=%s coverage_count=%s mean_sigmoid_activation=%.4f",
                    g.value,
                    n_cov,
                    mean_act,
                )

            global_step += 1

        mean_epoch = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        ckpt = output_dir / f"query_encoder_stage2_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "query_encoder": encoder.state_dict(),
                "bag_encoder": bag_encoder.state_dict(),
                "gate_mlp": gate_mlp.state_dict(),
                "forecast_head": forecast_head.state_dict(),
            },
            ckpt,
        )
        state_path = output_dir / "train_state_stage2.json"
        state_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "embedding_version": manifest.embedding_version,
                    "label_version": STAGE2_LABEL_VERSION,
                    "tau": tau,
                    "stage1_init_checkpoint": str(stage1_ckpt_path.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("epoch=%s saved checkpoint %s mean_loss=%.6f", epoch, ckpt.name, mean_epoch)


__all__ = [
    "INFOCE_TEMPERATURE",
    "STAGE2_LABEL_VERSION",
    "run_stage2_training",
]
