from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOTS = ("skills", "agents", "references")


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    _, rest = text.split("---", 1)
    raw, body = rest.split("---", 1)
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def test_plugin_manifest_declares_in_place_plugin() -> None:
    # Omit version so in-place training edits are not pinned to a semver (plugins-reference).
    # Omit custom skills/agents paths so defaults (skills/, agents/ at plugin root) apply.
    manifest_path = REPO_ROOT / ".claude-plugin" / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "psychohistory"
    assert "version" not in manifest
    assert "skills" not in manifest
    assert "agents" not in manifest


def test_overlay_lives_at_plugin_root() -> None:
    skill = REPO_ROOT / "skills" / "predict" / "SKILL.md"
    parent = REPO_ROOT / "agents" / "parent.md"
    worker = REPO_ROOT / "agents" / "claim-worker.md"
    reflector = REPO_ROOT / "agents" / "reflector.md"
    discovery = REPO_ROOT / "references" / "discovery.md"
    structure = REPO_ROOT / "references" / "structure.md"
    analog_prior = REPO_ROOT / "references" / "analog-prior.md"
    cases = REPO_ROOT / "references" / "cases"
    assert skill.is_file()
    assert parent.is_file()
    assert worker.is_file()
    assert reflector.is_file()
    assert discovery.is_file()
    assert structure.is_file()
    assert analog_prior.is_file()
    analog_prior_body = analog_prior.read_text(encoding="utf-8").lower()
    assert "base rate" in analog_prior_body
    assert "falsifier" in analog_prior_body
    assert "source-split" in analog_prior_body
    assert cases.is_dir()
    assert any(cases.glob("*.md"))
    tariff_card = cases / "tariff-proclamation-deadline-delay.md"
    assert tariff_card.is_file()
    assert "typical openings" in tariff_card.read_text(encoding="utf-8").lower()
    assert not (REPO_ROOT / "references" / "vault.md").exists()
    skill_meta, skill_body = _frontmatter(skill)
    assert skill_meta["name"] == "predict"
    assert skill_meta.get("disable-model-invocation") == "true"
    assert "ledger.md" in skill_body
    assert "references/discovery.md" in skill_body
    assert "references/structure.md" in skill_body
    assert "references/analog-prior.md" in skill_body
    discover = REPO_ROOT / "skills" / "discover" / "SKILL.md"
    assert discover.is_file()
    discover_meta, discover_body = _frontmatter(discover)
    assert discover_meta["name"] == "discover"
    assert discover_meta.get("disable-model-invocation") == "true"
    assert "ledger.md" in discover_body
    assert "references/discovery.md" in discover_body
    assert "references/structure.md" in discover_body
    assert "predict" in discover_body.lower()
    reflect = REPO_ROOT / "skills" / "reflect" / "SKILL.md"
    assert reflect.is_file()
    reflect_meta, reflect_body = _frontmatter(reflect)
    assert reflect_meta["name"] == "reflect"
    assert reflect_meta.get("disable-model-invocation") == "true"
    assert "after_resolution" in reflect_body
    assert "justification" in reflect_body.lower()
    assert "graph-vault" in reflect_body
    assert "new or rewritten" in reflect_body.lower()
    assert "analog case cards" in reflect_body.lower()
    assert "references/structure.md" in reflect_body
    parent_meta, parent_body = _frontmatter(parent)
    assert parent_meta["name"] == "parent"
    assert "skills/predict/SKILL.md" in parent_body
    assert "skills/reflect/SKILL.md" in parent_body
    assert "agents/reflector.md" in parent_body
    assert "skills/discover/SKILL.md" in parent_body
    worker_meta, worker_body = _frontmatter(worker)
    assert worker_meta["name"] == "claim-worker"
    assert "ledger.md" in worker_body
    assert "skills/" in worker_body
    assert "references/structure.md" in worker_body
    assert "references/analog-prior.md" in worker_body
    assert "references/vault.md" not in worker_body
    pointer = REPO_ROOT / ".cursor" / "agents" / "parent.md"
    pointer_meta, pointer_body = _frontmatter(pointer)
    assert pointer_meta["name"] == "parent"
    assert "agents/parent.md" in pointer_body
    assert "predict" in pointer_body


def test_overlay_markdown_stays_portable() -> None:
    forbidden = ("import hermes", "harness.orchestrator")
    roots = [REPO_ROOT / name for name in OVERLAY_ROOTS]
    for root in roots:
        assert root.is_dir(), f"missing overlay root {root}"
    roots.append(REPO_ROOT / ".cursor" / "agents")
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{path} mentions {needle}"


def test_ledger_owners_have_worker_agents() -> None:
    from harness.ledger import parse_ledger

    book = parse_ledger((REPO_ROOT / "ledger.md").read_text(encoding="utf-8"))
    for claim in book.claims:
        agent = REPO_ROOT / "agents" / f"{claim.owner}.md"
        assert agent.is_file(), f"missing agent for owner {claim.owner}"
