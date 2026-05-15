"""Parse Obsidian-style markdown notes (YAML frontmatter + body)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def split_vault_markdown(raw: str) -> tuple[dict[str, Any], str]:
    text = raw.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing = idx
            break
    if closing is None:
        return {}, text

    fm_block = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :]).lstrip("\n")
    loaded = yaml.safe_load(fm_block)
    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise ValueError("YAML frontmatter must map to a dict")
    return loaded, body


def read_vault_note(path: Path) -> tuple[dict[str, Any], str]:
    return split_vault_markdown(path.read_text(encoding="utf-8"))


def iter_vault_markdown_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file())


__all__ = ["iter_vault_markdown_files", "read_vault_note", "split_vault_markdown"]
