from __future__ import annotations

from pathlib import Path

import pytest

from scripts.reflect import _extract_json_object, apply_reflection_payload


def test_extract_json_object_handles_fence() -> None:
    raw = """Here you go:
```json
{"policy_markdown": "---\\nfoo: bar\\n---\\n", "strategy_markdown": "# S", "vault_files": []}
```
"""
    obj = _extract_json_object(raw)
    assert obj["strategy_markdown"] == "# S"


def test_apply_reflection_payload_writes_policy_and_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    policy_path = tmp_path / "policy.md"
    policy_path.write_text("---\nblind_spot_checks: []\nmax_steps: 4\nconvergence_epsilon: 0.01\nshrinkage: null\n---\n", encoding="utf-8")

    payload = {
        "policy_markdown": (
            "---\nblind_spot_checks:\n  - geopolitical_stability_check\n"
            "max_steps: 3\nconvergence_epsilon: 0.01\nshrinkage: null\n---\n"
        ),
        "strategy_markdown": "# Strategy\n\n```dataview\nTABLE question\nFROM \"runs\"\n```\n",
        "vault_files": [
            {"path": "approaches/test.md", "content": "# Approach\n"},
        ],
    }
    log = apply_reflection_payload(payload, policy_path=policy_path, vault_dir=vault, dry_run=False)

    loaded = policy_path.read_text(encoding="utf-8")
    assert "geopolitical_stability_check" in loaded
    assert (vault / "_strategy.md").is_file()
    assert "dataview" in (vault / "_strategy.md").read_text(encoding="utf-8")
    assert (vault / "approaches/test.md").is_file()
    assert len(log) >= 2


def test_apply_reflection_payload_delete_note(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.md"
    policy_path.write_text("---\nblind_spot_checks: []\nmax_steps: 4\nconvergence_epsilon: 0.01\nshrinkage: null\n---\n", encoding="utf-8")
    vault = tmp_path / "vault"
    vault.mkdir()
    junk = vault / "obsolete.md"
    junk.write_text("# old\n", encoding="utf-8")
    payload = {
        "policy_markdown": "---\nblind_spot_checks: []\nmax_steps: 4\nconvergence_epsilon: 0.01\nshrinkage: null\n---\n",
        "vault_files": [{"path": "obsolete.md", "delete": True}],
    }
    apply_reflection_payload(payload, policy_path=policy_path, vault_dir=vault, dry_run=False)
    assert not junk.is_file()


def test_apply_reflection_payload_rejects_escape(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.md"
    policy_path.write_text("---\nblind_spot_checks: []\n---\n", encoding="utf-8")
    payload = {
        "policy_markdown": "---\nblind_spot_checks: []\n---\n",
        "vault_files": [{"path": "../outside.md", "content": "no"}],
    }
    with pytest.raises(ValueError):
        apply_reflection_payload(payload, policy_path=policy_path, vault_dir=tmp_path / "v", dry_run=False)
