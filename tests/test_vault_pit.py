from datetime import date
from pathlib import Path

from harness.vault_pit import (
    is_path_admissible,
    list_admissible_paths,
    materialize_pit_snapshot,
    quarter_end_date,
)


def test_quarter_end_date() -> None:
    assert quarter_end_date("2024-Q1") == date(2024, 3, 31)
    assert quarter_end_date("2024-Q4") == date(2024, 12, 31)


def test_timeline_filter(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "timeline").mkdir(parents=True)
    (vault / "timeline" / "2023-Q4.md").write_text("---\npit_cutoff: 2023-12-31\n---\n", encoding="utf-8")
    (vault / "timeline" / "2026-Q2.md").write_text("---\npit_cutoff: 2026-06-30\n---\n", encoding="utf-8")
    cutoff = date(2024, 6, 1)
    paths = list_admissible_paths(vault, cutoff)
    assert "timeline/2023-Q4.md" in paths
    assert "timeline/2026-Q2.md" not in paths


def test_entity_pit_cutoff(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    entities = vault / "domains" / "usa" / "entities"
    entities.mkdir(parents=True)
    (entities / "old.md").write_text(
        "---\ntype: entity\npit_cutoff: 2023-12-31\n---\n",
        encoding="utf-8",
    )
    (entities / "new.md").write_text(
        "---\ntype: entity\npit_cutoff: 2026-05-18\n---\n",
        encoding="utf-8",
    )
    cutoff = date(2024, 1, 1)
    old_ok, _ = is_path_admissible(vault, "domains/usa/entities/old.md", cutoff)
    new_ok, _ = is_path_admissible(vault, "domains/usa/entities/new.md", cutoff)
    assert old_ok
    assert not new_ok


def test_excludes_meta_and_runs(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "meta" / "reflections").mkdir(parents=True)
    (vault / "runs").mkdir(parents=True)
    (vault / "meta" / "reflections" / "r.md").write_text("# x", encoding="utf-8")
    (vault / "runs" / "run.md").write_text("# x", encoding="utf-8")
    cutoff = date(2025, 1, 1)
    paths = list_admissible_paths(vault, cutoff)
    assert not any(p.startswith("meta/") for p in paths)
    assert not any(p.startswith("runs/") for p in paths)


def test_materialize_snapshot(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "timeline").mkdir(parents=True)
    (src / "timeline" / "2023-Q4.md").write_text("q4", encoding="utf-8")
    (src / "timeline" / "2026-Q2.md").write_text("q6", encoding="utf-8")
    (src / "_forecast_instructions.md").write_text("rules", encoding="utf-8")
    copied = materialize_pit_snapshot(src, dst, date(2024, 1, 1))
    assert "timeline/2023-Q4.md" in copied
    assert not (dst / "timeline" / "2026-Q2.md").exists()
    assert (dst / "_forecast_instructions.md").read_text(encoding="utf-8") == "rules"
