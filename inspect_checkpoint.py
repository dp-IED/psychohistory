from pathlib import Path
import torch
import numpy as np

from schemas.graph_builder_warehouse import NodeWarehouseManifest
from baselines.node_warehouse_mmap import read_float32_matrix
from baselines.graph_builder_query_encoder import (
    QueryEncoder,
    encode_actor_state_query,
    warehouse_context_from_manifest,
)
from baselines.graph_builder_ann import brute_topk
from schemas.graph_builder_probe import ActorStateQuery

# Paths — adjust to your actual output locations
manifest_path = Path("shared_data/arab_spring/node_warehouse_v1_manifest.json")
mmap_path     = Path("shared_data/arab_spring/node_warehouse_v1.mmap")
ckpt_path     = Path("shared_data/arab_spring/stage1_v1_lead_lag_out/query_encoder_epoch_009.pt")

manifest = NodeWarehouseManifest.model_validate_json(
    manifest_path.read_text(encoding="utf-8")
)
matrix = read_float32_matrix(mmap_path, row_count=manifest.row_count, embedding_dim=manifest.embedding_dim)
ctx    = warehouse_context_from_manifest(manifest, matrix)

encoder = QueryEncoder()
encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
encoder.eval()

# Three test queries — one per structural type you care about
queries = [
    # Precursor: Bouazizi incident, Sidi Bouzid, December 2010
    ActorStateQuery(
        geography=["Tunisia"],
        actor_type=["individual"],
        state_flags=["escalating"],
        entity_hints=["Mohamed Bouazizi"],
        as_of="2010-12-17",
    ),
    # Propagation: Tahrir Square mobilisation, Egypt, January 2011
    ActorStateQuery(
        geography=["Egypt"],
        actor_type=["civil_resistance"],
        state_flags=["escalating"],
        entity_hints=["April 6 Youth Movement"],
        as_of="2011-01-25",
    ),
    # Suppression: Gaddafi regime response, Libya, February 2011
    ActorStateQuery(
        geography=["Libya"],
        actor_type=["security_force"],
        state_flags=["repressive"],
        entity_hints=["Muammar Gaddafi"],
        as_of="2011-02-20",
    ),
]

for i, q in enumerate(queries):
    q_vec = encode_actor_state_query(
        actor_state=q,
        probe_id=f"test_query_{i}",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    ).detach().numpy()
    scores, indices = brute_topk(q_vec, matrix, k=10)
    
    print(f"\n--- Query {i+1}: {q.geography} {q.actor_type} ({q.as_of}) hints={q.entity_hints} ---")
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        row = manifest.rows[int(idx)]
        print(f"  {rank+1:2d}. score={score:.4f}  node={row.node_id}  "
              f"admin1={row.admin1_code}  first_seen={row.first_seen}")
