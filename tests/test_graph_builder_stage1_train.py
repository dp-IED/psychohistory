"""Stage 1 SSL training smoke and edge-case tests (synthetic fixtures + patches)."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from baselines.graph_builder_positive_pairs import (
    META_JSON_BASENAME,
    POSITIVE_PAIR_VERSION,
    build_positive_pairs,
)
from baselines.graph_builder_query_encoder import ENTITY_HINT_KEYS
from baselines.graph_builder_stage1_train import run_stage1_training
from baselines.node_warehouse_mmap import write_float32_matrix, write_manifest
from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta

_ORIGIN = date(2019, 6, 1)
_AS_OF = date(2019, 6, 1)
# Within manifest PIT window [as_of - (window_days-1), as_of] so positive-pair builder keeps rows.
_SEEN = date(2019, 5, 15)


def _l2_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.standard_normal((n, 128)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _probe(probe_id: str, hint: str, gate: AssumptionEmphasis) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        origin=_ORIGIN,
        nl_text=f"nl {probe_id}",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=["France"],
                actor_type=["government"],
                entity_hints=[hint],
                state_flags=["escalating"],
                as_of=_AS_OF,
            ),
        ),
        lens_params=LensParamsV0(horizon_days=7, context_snippet="ctx"),
        assumption_emphasis=gate,
        generation_meta=GenerationMeta(
            template_id="test_t",
            generator_version="test_g",
            seed=1,
            assumption_gate_coverage=gate,
        ),
    )


def _write_smoke_warehouse(tmp_path: Path, *, embedding_version: str = "smoke_emb_v1") -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(2027)
    mmap = _l2_rows(rng, 4)
    hints = ["hint_a", "hint_b", "hint_c", "hint_d"]
    rows = [
        NodeWarehouseRowMeta(
            node_id=f"n{i}",
            first_seen=_SEEN,
            admin1_code="FR-IDF",
            extensions={ENTITY_HINT_KEYS: [hints[i]]},
        )
        for i in range(4)
    ]
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=embedding_version,
        mmap_path="nodes.f32",
        row_count=4,
        rows=rows,
        window_days=30,
        as_of=_AS_OF,
    )
    mmap_path = tmp_path / "nodes.f32"
    manifest_path = tmp_path / "manifest.json"
    write_float32_matrix(mmap_path, mmap)
    write_manifest(manifest_path, manifest)
    pairs_dir = tmp_path / "pairs_out"
    meta_path = build_positive_pairs(manifest, mmap_path, pairs_dir)
    return manifest_path, mmap_path, meta_path


def test_version_mismatch_before_training_raises(tmp_path: Path) -> None:
    manifest_path, mmap_path, meta_path = _write_smoke_warehouse(tmp_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["embedding_version"] = "wrong_version"
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = tmp_path / "train_out"
    with pytest.raises(ValueError, match=r"embedding_version mismatch"):
        run_stage1_training(
            manifest_path,
            mmap_path,
            meta_path,
            out,
            epochs=1,
            batch_size=1,
            seed=0,
        )
    assert list(out.glob("query_encoder_epoch_*.pt")) == []


def test_smoke_one_step_saves_checkpoint_and_state(tmp_path: Path) -> None:
    manifest_path, mmap_path, meta_path = _write_smoke_warehouse(tmp_path)
    probes = [_probe("p0", "hint_a", AssumptionEmphasis.PERSISTENCE), _probe("p1", "hint_b", AssumptionEmphasis.PROPAGATION)]
    out = tmp_path / "train_out"

    with patch("baselines.graph_builder_stage1_train.build_france_plumbing_probe_corpus", return_value=probes):
        run_stage1_training(
            manifest_path,
            mmap_path,
            meta_path,
            out,
            epochs=1,
            batch_size=2,
            seed=0,
        )

    ckpt = out / "query_encoder_epoch_000.pt"
    assert ckpt.is_file()
    state = json.loads((out / "train_state.json").read_text(encoding="utf-8"))
    assert state["embedding_version"] == "smoke_emb_v1"
    assert state["epoch"] == 0
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(sd, dict)
    assert any(k.startswith("fuse") or k.startswith("unk") for k in sd)


def test_gate_coverage_logs_all_five_gates(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    manifest_path, mmap_path, meta_path = _write_smoke_warehouse(tmp_path)
    gates = list(AssumptionEmphasis)
    assert len(gates) == 5
    probes = [_probe(f"p{i}", f"h{i}", gates[i]) for i in range(5)]
    out = tmp_path / "train_out"

    with patch("baselines.graph_builder_stage1_train.build_france_plumbing_probe_corpus", return_value=probes):
        with caplog.at_level(logging.INFO, logger="baselines.graph_builder_stage1_train"):
            run_stage1_training(
                manifest_path,
                mmap_path,
                meta_path,
                out,
                epochs=1,
                batch_size=5,
                seed=0,
            )

    text = caplog.text
    for g in gates:
        assert g.value in text


def _fake_brute_topk_first_four(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    del query, corpus
    idx = np.full(k, -1, dtype=np.int64)
    sc = np.full(k, -np.inf, dtype=np.float32)
    idx[:4] = np.arange(4, dtype=np.int64)
    sc[:4] = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    return idx, sc


def _write_pairs_manual_only_45(tmp_path: Path, manifest: NodeWarehouseManifest, mmap_path: Path) -> Path:
    """Single pair (4, 5) — not the full admin1_14day closure (tests ANN-restricted retrieved set)."""
    arr = np.array([[4, 5]], dtype=np.int32)
    pairs_name = "only45.npy"
    np.save(tmp_path / pairs_name, arr)
    meta = {
        "as_of": manifest.as_of.isoformat() if manifest.as_of else None,
        "embedding_version": manifest.embedding_version,
        "mmap_path": str(mmap_path),
        "pair_count": 1,
        "pairs_path": pairs_name,
        "positive_pair_version": POSITIVE_PAIR_VERSION,
        "recipe_id": manifest.recipe_id,
        "window_days": manifest.window_days,
    }
    meta_path = tmp_path / "only45.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def test_no_positive_in_retrieved_skips_without_crash(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    mmap = _l2_rows(rng, 6)
    hints = [f"h{i}" for i in range(6)]
    rows = [
        NodeWarehouseRowMeta(
            node_id=f"n{i}",
            first_seen=_SEEN,
            admin1_code="FR-IDF",
            extensions={ENTITY_HINT_KEYS: [hints[i]]},
        )
        for i in range(6)
    ]
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version="np_emb",
        mmap_path="nodes.f32",
        row_count=6,
        rows=rows,
        window_days=30,
        as_of=_AS_OF,
    )
    mmap_path = tmp_path / "nodes.f32"
    manifest_path = tmp_path / "manifest.json"
    write_float32_matrix(mmap_path, mmap)
    write_manifest(manifest_path, manifest)
    meta_path = _write_pairs_manual_only_45(tmp_path, manifest, mmap_path)

    probes = [_probe("only", "h0", AssumptionEmphasis.PERSISTENCE)]
    out = tmp_path / "train_out"

    with patch("baselines.graph_builder_stage1_train.build_france_plumbing_probe_corpus", return_value=probes):
        with patch("baselines.graph_builder_stage1_train.brute_topk", side_effect=_fake_brute_topk_first_four):
            with caplog.at_level(logging.WARNING, logger="baselines.graph_builder_stage1_train"):
                run_stage1_training(
                    manifest_path,
                    mmap_path,
                    meta_path,
                    out,
                    epochs=1,
                    batch_size=1,
                    seed=0,
                )

    assert "skipping probe" in caplog.text.lower() or "no intra-retrieved" in caplog.text.lower()
    assert (out / "query_encoder_epoch_000.pt").is_file()
