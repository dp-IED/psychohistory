"""Minimal Stage 1 SSL training: probes → query encoder → ANN → rerank → InfoNCE.

Trains ``QueryEncoder`` only (mmap and reranker frozen). No forecast head or
assumption MLPs. See module docstrings on loss for the ``pos_mean`` norm choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from baselines.graph_builder_ann import brute_topk
from baselines.graph_builder_positive_pairs import (
    META_JSON_BASENAME,
    PositivePairLookup,
    build_positive_pairs,
    load_positive_pairs,
)
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    warehouse_context_from_manifest,
)
from baselines.graph_builder_rerank import ann_rerank_global_indices, build_retrieved_graph_batch_from_ann
from baselines.node_warehouse_mmap import read_float32_matrix
from baselines.stage1_probe_corpus import Stage1ProbeCorpus
from schemas.graph_builder_probe import AssumptionEmphasis, ProbeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest

logger = logging.getLogger(__name__)

_DEFAULT_LR = 1e-3


def _positive_nodes_in_retrieved(R: set[int], pair_lookup: PositivePairLookup) -> set[int]:
    """Nodes in ``R`` that participate in at least one precomputed pair with another node in ``R``."""
    nodes = sorted(R)
    if len(nodes) < 2:
        return set()
    candidate_pairs = np.asarray(
        [(nodes[a], nodes[b]) for a in range(len(nodes)) for b in range(a + 1, len(nodes))],
        dtype=np.int32,
    )
    hits = pair_lookup.contains_many(candidate_pairs)
    pos: set[int] = set()
    for (a, b), hit in zip(candidate_pairs, hits):
        if bool(hit):
            pos.add(int(a))
            pos.add(int(b))
    return pos


def _per_probe_infonce(
    q: torch.Tensor,
    *,
    global_indices_row: np.ndarray,
    node_mask_row: torch.Tensor,
    node_feat_row: torch.Tensor,
    pair_lookup: PositivePairLookup,
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

    pos_nodes = _positive_nodes_in_retrieved(R, pair_lookup)
    if not pos_nodes:
        logger.warning(
            "skipping probe %r: no intra-retrieved positive pairs (pair lookup vs retrieved set)",
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
    corpus: Stage1ProbeCorpus | None = None,
    show_progress: bool = True,
    progress_style: Literal["tqdm", "plain"] = "plain",
) -> None:
    """Run Stage 1 SSL. Loads ``manifest_path`` then calls ``corpus.validate(manifest)``.

    When ``corpus`` is ``None``, uses ``Stage1ProbeCorpus.france_default()``. For Arab
    Spring training, pass e.g. ``Stage1ProbeCorpus.arab_spring_default()`` so
    ``validate`` receives the loaded manifest and runs ``entity_hint`` resolution checks
    (hint checks are skipped if you only call ``validate()`` without a manifest elsewhere).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if show_progress:
        print(f"[Stage1] Reading manifest {manifest_path}…", flush=True)
    manifest = NodeWarehouseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    if show_progress:
        print(
            "[Stage1] Loading positive pairs (.npy can be hundreds of MB; this can take 1–3 minutes)…",
            flush=True,
        )
    pairs, pair_meta = load_positive_pairs(pairs_metadata_path, manifest)
    pair_lookup = PositivePairLookup(pairs)
    if show_progress:
        print(
            f"[Stage1] Loaded {int(pairs.shape[0]):,} pair rows; pair lookup ready; reading node mmap…",
            flush=True,
        )

    mmap = read_float32_matrix(
        Path(mmap_path),
        row_count=manifest.row_count,
        embedding_dim=manifest.embedding_dim,
    )
    mmap_np = np.asarray(mmap, dtype=np.float32)
    if show_progress:
        print(
            f"[Stage1] Embeddings in RAM {mmap_np.shape[0]:,}×{mmap_np.shape[1]}; building warehouse index…",
            flush=True,
        )
    slice_ctx = full_ctx = warehouse_context_from_manifest(manifest, mmap_np)

    bundle = corpus if corpus is not None else Stage1ProbeCorpus.france_default()
    bundle.validate(manifest)

    probes = bundle.probes
    encoder = QueryEncoder().to(device)
    encoder.train()
    opt = Adam(encoder.parameters(), lr=_DEFAULT_LR)

    positive_pair_version = str(pair_meta.get("positive_pair_version", ""))
    global_step = 0

    use_tqdm_bars = bool(show_progress and progress_style == "tqdm")
    use_plain = bool(show_progress and progress_style == "plain")
    _tqdm_file = sys.stderr
    epoch_pbar_obj: tqdm | None = None
    start_time = time.perf_counter() if show_progress else None
    
    if use_tqdm_bars:
        epoch_pbar_obj = tqdm(
            range(epochs),
            desc="Stage1",
            unit="epoch",
            total=epochs,
            file=_tqdm_file,
            position=0,
            disable=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            dynamic_ncols=True,
            mininterval=0.25,
        )

    if show_progress:
        print(
            f"[Stage1] Training: {len(probes)} probes, {epochs} epoch(s), batch_size={batch_size}, "
            f"device={device}, progress={progress_style}",
            flush=True,
        )

    for epoch in epoch_pbar_obj or range(epochs):
        epoch_losses: list[float] = []
        epoch_contributors = 0
        epoch_probes = 0
        epoch_gate_contrib: dict[AssumptionEmphasis, int] = {g: 0 for g in AssumptionEmphasis}
        logger.debug(
            "epoch=%s/%s start probes=%s batch_size=%s",
            epoch + 1,
            epochs,
            len(probes),
            batch_size,
        )
        order = np.random.permutation(len(probes))
        batch_starts = list(range(0, len(probes), batch_size))
        n_batches = len(batch_starts)
        pbar: tqdm | None = None
        if use_tqdm_bars and batch_starts:
            pbar = tqdm(
                batch_starts,
                desc=f" batches e{epoch + 1}/{epochs}",
                unit="batch",
                file=_tqdm_file,
                leave=False,
                disable=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                dynamic_ncols=True,
                mininterval=0.25,
                position=1,
            )
            batch_iter = pbar
        else:
            batch_iter = batch_starts

        if use_plain:
            print(
                f"[Stage1] epoch {epoch + 1}/{epochs}: {n_batches} batches",
                flush=True,
            )

        for batch_i, start in enumerate(batch_iter):
            batch_idx = order[start : start + batch_size]
            batch: list[ProbeRecord] = [probes[i] for i in batch_idx]
            epoch_probes += len(batch)
            n_batch = len(batch)

            q_list: list[torch.Tensor] = []
            for qi, probe in enumerate(batch):
                if pbar is not None:
                    pbar.set_postfix_str(f"encode {qi + 1}/{n_batch}", refresh=True)
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
            for bi in range(queries_np.shape[0]):
                if pbar is not None:
                    pbar.set_postfix_str(f"ANN {bi + 1}/{queries_np.shape[0]}", refresh=True)
                idx, sc = brute_topk(queries_np[bi], mmap_np, k=100)
                ann_indices[bi] = idx
                ann_scores[bi] = sc

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
                    pair_lookup=pair_lookup,
                    temperature=temperature,
                    probe_id=probe.probe_id,
                )
                if ok and loss_b is not None:
                    loss_terms.append(loss_b)
                    gate_contrib_count[cov] += 1
                    epoch_contributors += 1
                    gate_loss_sum[cov] += float(loss_b.detach().cpu().item())

            if loss_terms:
                step_loss = torch.stack(loss_terms, dim=0).mean()
                opt.zero_grad(set_to_none=True)
                step_loss.backward()
                opt.step()
                epoch_losses.append(float(step_loss.detach().cpu().item()))
                last_loss = epoch_losses[-1]
                logger.debug("step=%s mean_loss=%.6f", global_step, last_loss)
                if pbar is not None:
                    run_mean = float(np.mean(epoch_losses))
                    pbar.set_postfix(
                        batch_loss=f"{last_loss:.4f}",
                        epoch_avg=f"{run_mean:.4f}",
                        step=global_step,
                    )
                elif use_plain:
                    run_mean = float(np.mean(epoch_losses))
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[Stage1] e{epoch + 1}/{epochs} batch {batch_i + 1:2d}/{n_batches} "
                        f"loss={last_loss:.4f} avg={run_mean:.4f} ({elapsed:.0f}s elapsed)",
                        flush=True,
                    )
            else:
                logger.warning("step=%s: no probes contributed InfoNCE in this batch", global_step)
                if pbar is not None:
                    pbar.set_postfix(skipped=1, step=global_step)

            for gate in AssumptionEmphasis:
                n_batch = gate_batch_count[gate]
                c = gate_contrib_count[gate]
                mean_l = gate_loss_sum[gate] / c if c > 0 else 0.0
                epoch_gate_contrib[gate] += c
                logger.debug(
                    "gate=%s count=%s contributors=%s loss=%.4f",
                    gate.value,
                    n_batch,
                    c,
                    mean_l,
                )

            global_step += 1
        if pbar is not None:
            pbar.close()

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
        if use_tqdm_bars and epoch_pbar_obj is not None:
            epoch_pbar_obj.set_postfix(
                mean_loss=f"{mean_epoch:.4f}",
                contrib=f"{epoch_contributors}/{epoch_probes}",
                step=global_step,
                ckpt=ckpt.name,
            )
        if use_plain:
            elapsed = time.perf_counter() - start_time
            print(
                f"[Stage1] epoch {epoch + 1}/{epochs} done mean_loss={mean_epoch:.4f} "
                f"({elapsed:.0f}s elapsed) → {ckpt.name}",
                flush=True,
            )
        gate_epoch_summary = " ".join(
            f"{g.value}={epoch_gate_contrib[g]}" for g in AssumptionEmphasis
        )
        logger.info(
            "epoch=%s/%s complete mean_loss=%.6f contributors=%s/%s steps=%s checkpoint=%s | %s",
            epoch + 1,
            epochs,
            mean_epoch,
            epoch_contributors,
            epoch_probes,
            global_step,
            ckpt.name,
            gate_epoch_summary,
        )

    if epoch_pbar_obj is not None:
        epoch_pbar_obj.close()
    if use_plain:
        elapsed = time.perf_counter() - start_time
        print(f"[Stage1] All {epochs} epochs finished in {elapsed:.0f}s.", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 query-encoder training (InfoNCE over ANN+rerank).")
    p.add_argument("--manifest", type=Path, required=True, help="Path to node warehouse manifest JSON.")
    p.add_argument("--mmap", type=Path, required=True, help="Path to float32 node embedding mmap.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and train_state.json.")
    p.add_argument(
        "--corpus",
        choices=["france", "arab_spring"],
        default="france",
        help="Probe corpus: france (default) or arab_spring.",
    )
    pairs_group = p.add_mutually_exclusive_group(required=True)
    pairs_group.add_argument(
        "--pairs-metadata",
        type=Path,
        help=f"Path to existing positive-pairs metadata JSON (basename {META_JSON_BASENAME!r} from build_positive_pairs).",
    )
    pairs_group.add_argument(
        "--build-pairs-to",
        type=Path,
        help="Directory to write positive pairs via build_positive_pairs; uses META_JSON_BASENAME inside it.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level (default: INFO, showing epoch/step progress).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all progress output (no tqdm, no plain lines).",
    )
    p.add_argument(
        "--progress",
        choices=["plain", "tqdm"],
        default="tqdm",
        help="plain=newline logs on stdout (IDE-friendly); tqdm=animated bars on stderr (default: tqdm).",
    )
    return p.parse_args(argv)


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main() -> int:
    """Prerequisites: build DuckDB/warehouse and mmap+manifest offline before running.

    Arab Spring training needs a manifest whose entity_hint_keys align with the probe
    entity_hints (validate resolves hints against the loaded manifest). Requires PyTorch.
    """
    args = _parse_args()
    _configure_logging(args.log_level)
    manifest_path: Path = args.manifest
    mmap_path: Path = args.mmap
    if args.build_pairs_to is not None:
        build_dir: Path = args.build_pairs_to
        manifest = NodeWarehouseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8"),
        )
        n_rows = manifest.row_count
        logger.info(
            "Building positive pairs for %s rows (CPU; can take many minutes on large manifests)…",
            f"{n_rows:,}",
        )
        build_positive_pairs(
            manifest,
            mmap_path,
            build_dir,
            show_progress=not args.no_progress,
        )
        logger.info("Positive pairs written under %s", build_dir)
        pairs_metadata = build_dir / META_JSON_BASENAME
    else:
        pairs_metadata = args.pairs_metadata
    corpus = (
        Stage1ProbeCorpus.france_default() if args.corpus == "france" else Stage1ProbeCorpus.arab_spring_default()
    )
    run_stage1_training(
        manifest_path,
        mmap_path,
        pairs_metadata,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        corpus=corpus,
        show_progress=not args.no_progress,
        progress_style=args.progress,
    )
    return 0


__all__ = ["main", "run_stage1_training"]


if __name__ == "__main__":
    sys.exit(main())
