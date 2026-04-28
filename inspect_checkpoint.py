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
manifest_path = Path("/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/node_warehouse_v1_fixed_manifest.json")
mmap_path     = Path("/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/node_warehouse_v1_fixed.mmap")
ckpt_path     = Path("/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_v1_fixed/query_encoder_epoch_049.pt")

manifest = NodeWarehouseManifest.model_validate_json(
    manifest_path.read_text(encoding="utf-8")
)
matrix = read_float32_matrix(mmap_path, row_count=manifest.row_count, embedding_dim=manifest.embedding_dim)
ctx    = warehouse_context_from_manifest(manifest, matrix)

encoder = QueryEncoder()
encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
encoder.eval()

# Three test queries — one per structural type you care about
# NOTE: entity_hints must use GDELT actor type strings present in the warehouse
queries = [
    # Precursor: Tunisia December 2010 — protester escalation
    ActorStateQuery(
        geography=["Tunisia"],
        actor_type=["individual"],
        state_flags=["escalating"],
        entity_hints=["protester"],
        as_of="2010-12-17",
    ),
    # Propagation: Egypt January 2011 — protester escalation at Tahrir
    ActorStateQuery(
        geography=["Egypt"],
        actor_type=["civil_resistance"],
        state_flags=["escalating"],
        entity_hints=["protester"],
        as_of="2011-01-25",
    ),
    # Suppression: Libya February 2011 — military/government response
    ActorStateQuery(
        geography=["Libya"],
        actor_type=["security_force"],
        state_flags=["repressive"],
        entity_hints=["military"],
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
    indices, scores = brute_topk(q_vec, matrix, k=10)
    
    print(f"\n--- Query {i+1}: {q.geography} {q.actor_type} ({q.as_of}) hints={q.entity_hints} ---")
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        row = manifest.rows[int(idx)]
        print(f"  {rank+1:2d}. score={score:.4f}  node={row.node_id}  "
              f"admin1={row.admin1_code}  first_seen={row.first_seen}")
