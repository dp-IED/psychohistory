"""Unit tests for vault relevance probe scoring (no Hermes)."""

from harness.vault_probe import (
    VaultNode,
    classify_verdict,
    question_for_node,
    score_probe_response,
)


def test_question_for_concept():
    node = VaultNode("concepts/foo.md", "concept", "foo", "Foo Bar")
    q = question_for_node(node, __import__("datetime").date(2026, 5, 19))
    assert "Foo Bar" in q
    assert "2026-05-19" in q


def test_score_passes_when_target_read_and_detailed():
    node = VaultNode("concepts/incumbent-withdrawal-cascade.md", "concept", "incumbent-withdrawal-cascade", "Withdrawal")
    payload = {
        "vault_files_read": ["concepts/incumbent-withdrawal-cascade.md"],
        "explanation": "x" * 220,
        "today_connection": "Still relevant for 2026 leadership dynamics because " + "y" * 40,
        "relevance_score": 0.85,
        "disposition": "keep",
        "gaps": "",
    }
    ok, errs, rel = score_probe_response(node, payload, min_chars=200)
    assert ok, errs
    assert rel == 0.85
    assert classify_verdict(ok, rel, relevance_floor=0.4) == "keep"


def test_below_floor_reorganize_even_if_retrieval_ok():
    node = VaultNode("entities/minor-figure.md", "entity", "minor-figure", "Minor")
    payload = {
        "vault_files_read": ["entities/minor-figure.md"],
        "explanation": "a" * 220,
        "today_connection": "Only mentioned once in a quarter chronicle " + "b" * 40,
        "relevance_score": 0.2,
        "disposition": "merge",
        "merge_target": "timeline/2022-Q3.md",
    }
    ok, _, rel = score_probe_response(node, payload, min_chars=200)
    assert ok
    assert classify_verdict(ok, rel, relevance_floor=0.4, disposition="merge") == "reorganize"


def test_above_floor_weak_retrieval_expands():
    ok, _, rel = score_probe_response(
        VaultNode("threads/russia-ukraine-war.md", "thread", "russia-ukraine-war", "War"),
        {
            "vault_files_read": [],
            "explanation": "short",
            "today_connection": "",
            "relevance_score": 0.9,
        },
        min_chars=200,
    )
    assert not ok
    assert classify_verdict(ok, rel, relevance_floor=0.4) == "expand"
