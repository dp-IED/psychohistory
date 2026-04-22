"""Stage 2 weak supervision training smoke (synthetic fixtures + patches)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from baselines.graph_builder_positive_pairs import build_positive_pairs
from baselines.graph_builder_probe_labels import LABEL_VERSION_V0, ProbeLabelRow
from baselines.graph_builder_query_encoder import ENTITY_HINT_KEYS, QueryEncoder
from baselines.graph_builder_stage2_train import run_stage2_training
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
_SEEN = date(2019, 5, 15)


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


def _write_smoke_warehouse(tmp_path: Path, *, embedding_version: str = "smoke_emb_v1") -> tuple[Path, Path]:
    rng = np.random.default_rng(2027)
    mmap = rng.standard_normal((4, 128)).astype(np.float32)
    norms = np.linalg.norm(mmap, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    mmap = mmap / norms
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
    build_positive_pairs(manifest, mmap_path, pairs_dir)
    return manifest_path, mmap_path


def _write_labels(tmp_path: Path) -> Path:
    rows = [
        ProbeLabelRow(
            probe_id="p0",
            label_version=LABEL_VERSION_V0,
            gate=AssumptionEmphasis.PERSISTENCE,
            y=True,
            t0="2019-06-01",
        ),
        ProbeLabelRow(
            probe_id="p1",
            label_version=LABEL_VERSION_V0,
            gate=AssumptionEmphasis.PROPAGATION,
            y=False,
            t0="2019-06-01",
        ),
    ]
    p = tmp_path / "labels.jsonl"
    lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_stage2_smoke_saves_checkpoint_and_state(tmp_path: Path) -> None:
    manifest_path, mmap_path = _write_smoke_warehouse(tmp_path)
    labels_path = _write_labels(tmp_path)
    probes = [
        _probe("p0", "hint_a", AssumptionEmphasis.PERSISTENCE),
        _probe("p1", "hint_b", AssumptionEmphasis.PROPAGATION),
    ]
    enc = QueryEncoder()
    stage1_ckpt = tmp_path / "query_encoder_epoch_000.pt"
    torch.save(enc.state_dict(), stage1_ckpt)
    out = tmp_path / "stage2_out"

    with patch("baselines.graph_builder_stage2_train.build_france_plumbing_probe_corpus", return_value=probes):
        run_stage2_training(
            manifest_path,
            mmap_path,
            labels_path,
            stage1_ckpt,
            out,
            epochs=1,
            batch_size=2,
            tau=1e9,
            seed=0,
        )

    ckpt = out / "query_encoder_stage2_epoch_000.pt"
    assert ckpt.is_file()
    bundle = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert set(bundle.keys()) == {"bag_encoder", "forecast_head", "gate_mlp", "query_encoder"}

    state = json.loads((out / "train_state_stage2.json").read_text(encoding="utf-8"))
    assert state["epoch"] == 0
    assert state["embedding_version"] == "smoke_emb_v1"
    assert state["label_version"] == LABEL_VERSION_V0
    assert state["tau"] == 1e9
    assert Path(state["stage1_init_checkpoint"]) == stage1_ckpt.resolve()
