"""
Cursor Agent CLI contract: proposals are JSON with rationale, patch plan, impacts, risks.

The harness does not call remote APIs; the human/agent writes `proposal.json` per iteration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


PatchAction = Literal["write", "copy"]


@dataclass
class FilePatch:
    """Sandbox mutation: write or copy a file into the candidate tree."""

    path: str
    action: PatchAction = "write"
    content: str | None = None
    source_path: str | None = None


@dataclass
class AgentProposal:
    rationale: str
    file_patches: list[FilePatch] = field(default_factory=list)
    expected_metric_impact: dict[str, float] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "file_patch_plan": [
                {
                    "path": p.path,
                    "action": p.action,
                    "content": p.content,
                    "source_path": p.source_path,
                }
                for p in self.file_patches
            ],
            "expected_metric_impact": self.expected_metric_impact,
            "risks": self.risks,
            "meta": self.meta,
        }


def proposal_from_json(data: dict[str, Any]) -> AgentProposal:
    patches_raw = data.get("file_patch_plan") or data.get("file_patches") or []
    patches: list[FilePatch] = []
    for p in patches_raw:
        if not isinstance(p, dict):
            continue
        patches.append(
            FilePatch(
                path=str(p["path"]),
                action=str(p.get("action", "write")),
                content=p.get("content"),
                source_path=p.get("source_path"),
            )
        )
    return AgentProposal(
        rationale=str(data.get("rationale", "")),
        file_patches=patches,
        expected_metric_impact=dict(data.get("expected_metric_impact") or {}),
        risks=list(data.get("risks") or []),
        meta=dict(data.get("meta") or {}),
    )


def load_proposal(path: Path) -> AgentProposal:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Proposal JSON root must be an object")
    return proposal_from_json(data)


def noop_proposal() -> AgentProposal:
    return AgentProposal(
        rationale="No-op baseline: no file mutations; evaluates seed schema in isolation.",
        file_patches=[],
        expected_metric_impact={},
        risks=[],
        meta={"kind": "noop"},
    )
