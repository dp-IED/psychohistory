"""Create isolated candidate directories and apply patch plans."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from schemas.schema_registry import load_graph_schema

from autoresearch.agent_interface import AgentProposal, FilePatch


def prepare_candidate_tree(
    root: Path,
    candidate_id: str,
    seed_schema_spec: str,
    schema_source_root: Path | None = None,
    extra_copy_globs: list[str] | None = None,
) -> Path:
    """
    Materialize a candidate folder with a copy of `schemas/` so agents mutate safely.

    `seed_schema_spec` is recorded for eval; initial files mirror `schema_source_root/schemas`
    when provided, else repo `schemas/`.
    """

    cand = root / "autoresearch" / "experiments" / "candidates" / candidate_id
    if cand.exists():
        shutil.rmtree(cand)
    cand.mkdir(parents=True)

    source_schemas = (schema_source_root or root) / "schemas"
    if source_schemas.is_dir():
        shutil.copytree(source_schemas, cand / "schemas")

    meta = cand / "candidate_meta.txt"
    meta.write_text(
        f"candidate_id={candidate_id}\nseed_schema_spec={seed_schema_spec}\n",
        encoding="utf-8",
    )
    # Touch optional paths for future assets
    for g in extra_copy_globs or []:
        _ = g  # reserved for future: copy evals/templates etc.
    return cand


def apply_proposal(candidate_root: Path, proposal: AgentProposal, repo_root: Path) -> None:
    """Apply write/copy patches under candidate_root."""

    for p in proposal.file_patches:
        rel = p.path.lstrip("/")
        dest = (candidate_root / rel).resolve()
        if not str(dest).startswith(str(candidate_root.resolve())):
            raise ValueError(f"Refusing to write outside candidate tree: {p.path}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        act = (p.action or "write").lower()
        if act == "copy":
            if not p.source_path:
                raise ValueError("copy patch requires source_path")
            src = (repo_root / p.source_path.lstrip("/")).resolve()
            if not str(src).startswith(str(repo_root.resolve())):
                raise ValueError(f"Refusing to read outside repo: {p.source_path}")
            shutil.copyfile(src, dest)
        else:
            if p.content is None:
                raise ValueError("write patch requires content")
            dest.write_text(p.content, encoding="utf-8")


def schema_spec_for_candidate(candidate_root: Path, seed_schema_spec: str) -> str:
    """
    If candidate contains `schemas/base_schema.py`, eval uses that file path.
    Otherwise fall back to the original module spec.
    """

    cand_schema = candidate_root / "schemas" / "base_schema.py"
    if cand_schema.exists():
        return str(cand_schema)
    return seed_schema_spec


def load_schema_from_candidate(candidate_root: Path, seed_schema_spec: str):
    spec = schema_spec_for_candidate(candidate_root, seed_schema_spec)
    return load_graph_schema(spec), spec


def new_candidate_id(prefix: str = "cand") -> str:
    return f"{prefix}_{uuid4().hex[:10]}"
