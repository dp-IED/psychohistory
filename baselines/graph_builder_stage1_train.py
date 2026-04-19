"""Minimal Stage 1 SSL training: probes → query encoder → ANN → rerank → InfoNCE.

Trains ``QueryEncoder`` only (mmap and reranker frozen). No forecast head or
assumption MLPs. See module docstrings on loss for the ``pos_mean`` norm choice.
"""

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
from baselines.graph_builder_positive_pairs import load_positive_pairs
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    warehouse_context_from_manifest,
)
from baselines.graph_builder_rerank import ann_rerank_global_indices, build_retrieved_graph_batch_from_ann
from baselines.node_warehouse_mmap import read_float32_matrix
from schemas.graph_builder_probe import AssumptionEmphasis, ProbeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest

logger = logging.getLogger(__name__)

_DEFAULT_LR = 1e-3


def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i in range(int(pairs.shape[0])):
        a, b = int(pairs[i, 0]), int(pairs[i, 1])
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def _positive_nodes_in_retrieved(R: set[int], pair_set: set[tuple[int, int]]) -> set[int]:
    """Nodes in ``R`` that participate in at least one precomputed pair with another node in ``R``."""
    pos: set[int] = set()
    for g in R:
        for h in R:
            if h == g:
                continue
            a, b = (g, h) if g < h else (h, g)
            if (a, b) in pair_set:
                pos.add(g)
                break
    return pos


def _per_probe_infonce(
    q: torch.Tensor,
    *,
    global_indices_row: np.ndarray,
    node_mask_row: torch.Tensor,
    node_feat_row: torch.Tensor,
    pair_set: set[tuple[int, int]],
    temperature: float,
    probe_id: str,
) -> tuple[torch.Tensor | None, bool]:
    """Return (scalar loss, contributed) or (None, False) if skipped.

    ``q`` is L2-normalized from ``QueryEncoder``. ``pos_mean`` is the element-wise
    mean of positive node embeddings in the retrieved set and is **not** re-L2
    normalized; dots ``q·pos_mean`` and ``q·neg`` scale with key magnitude (v0 choice).
    """
    device = q.device
    dtype = q.dtype
    slots = int(global_indices_row.shape[0])
    R: set[int] = set()
    slot_by_global: dict[int, int] = {}
    for s in range(slots):
        if not bool(node_mask_row[s].item()):
            continue
        gix = int(global_indices_row[s])
        if gix < 0:
            continue
        R.add(gix)
        slot_by_global[gix] = s

    if not R:
        return None, False

    pos_nodes = _positive_nodes_in_retrieved(R, pair_set)
    if not pos_nodes:
        logger.warning(
            "skipping probe %r: no intra-retrieved positive pairs (pair_set vs retrieved set)",
            probe_id,
        )
        return None, False

    pos_embs = torch.stack(
        [node_feat_row[slot_by_global[g]].detach() for g in sorted(pos_nodes)],
        dim=0,
    )
    pos_mean = pos_embs.mean(dim=0)

    neg_nodes = sorted(R - pos_nodes)
    neg_embs = [node_feat_row[slot_by_global[g]].detach() for g in neg_nodes]

    pos_logit = (q * pos_mean).sum() / temperature
    if neg_embs:
        neg_logits = torch.stack([(q * neg).sum() for neg in neg_embs], dim=0) / temperature
        logits = torch.cat([pos_logit.unsqueeze(0), neg_logits], dim=0)
    else:
        logits = pos_logit.unsqueeze(0)

    loss_b = -(logits[0] - torch.logsumexp(logits, dim=0))
    return loss_b, True


def run_stage1_training(
    manifest_path: Path,
    mmap_path: Path,
    pairs_metadata_path: Path,
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    temperature: float = 0.07,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = NodeWarehouseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    pairs, pair_meta = load_positive_pairs(pairs_metadata_path, manifest)
    pair_set = _pair_set(pairs)

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
    encoder.train()
    opt = Adam(encoder.parameters(), lr=_DEFAULT_LR)

    positive_pair_version = str(pair_meta.get("positive_pair_version", ""))
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
            global_idx = ann_rerank_global_indices(queries_np, ann_indices, mmap_np)

            gate_batch_count: dict[AssumptionEmphasis, int] = {g: 0 for g in AssumptionEmphasis}
            gate_contrib_count: dict[AssumptionEmphasis, int] = {g: 0 for g in AssumptionEmphasis}
            gate_loss_sum: dict[AssumptionEmphasis, float] = {g: 0.0 for g in AssumptionEmphasis}

            loss_terms: list[torch.Tensor] = []
            for b, probe in enumerate(batch):
                cov = probe.generation_meta.assumption_gate_coverage
                if cov is None:
                    raise ValueError(f"probe {probe.probe_id!r} missing assumption_gate_coverage after validation")
                gate_batch_count[cov] += 1

                loss_b, ok = _per_probe_infonce(
                    queries[b],
                    global_indices_row=global_idx[b],
                    node_mask_row=retrieved.node_mask[b],
                    node_feat_row=retrieved.node_feat[b],
                    pair_set=pair_set,
                    temperature=temperature,
                    probe_id=probe.probe_id,
                )
                if ok and loss_b is not None:
                    loss_terms.append(loss_b)
                    gate_contrib_count[cov] += 1
                    gate_loss_sum[cov] += float(loss_b.detach().cpu().item())

            if loss_terms:
                step_loss = torch.stack(loss_terms, dim=0).mean()
                opt.zero_grad(set_to_none=True)
                step_loss.backward()
                opt.step()
                epoch_losses.append(float(step_loss.detach().cpu().item()))
                logger.info("step=%s mean_loss=%.6f", global_step, epoch_losses[-1])
            else:
                logger.warning("step=%s: no probes contributed InfoNCE in this batch", global_step)

            for gate in AssumptionEmphasis:
                n_batch = gate_batch_count[gate]
                c = gate_contrib_count[gate]
                mean_l = gate_loss_sum[gate] / c if c > 0 else 0.0
                logger.info(
                    "gate=%s count=%s contributors=%s loss=%.4f",
                    gate.value,
                    n_batch,
                    c,
                    mean_l,
                )

            global_step += 1

        mean_epoch = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        ckpt = output_dir / f"query_encoder_epoch_{epoch:03d}.pt"
        torch.save(encoder.state_dict(), ckpt)
        state_path = output_dir / "train_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "embedding_version": manifest.embedding_version,
                    "positive_pair_version": positive_pair_version,
                    "mean_loss": mean_epoch,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("epoch=%s saved checkpoint %s mean_loss=%.6f", epoch, ckpt.name, mean_epoch)


__all__ = ["run_stage1_training"]
