from __future__ import annotations

from datetime import date

import pytest

from harness.agent_loop import AgentLoopResult
from harness.competition_runner import (
    SPRING_2026_AIB_PROJECT_ID,
    main,
    run_batch,
    run_one_question,
    try_resolve_question,
)
from harness.metaculus_client import MetaculusAPIError, MetaculusQuestion
from harness.resolution import BrierUpdateResult


class _FakeClient:
    def __init__(self, questions: list[MetaculusQuestion], resolution: bool | None = None) -> None:
        self.questions = questions
        self.last_project_id = None
        self.posted: list[tuple[int, float, str]] = []
        self._resolution = resolution

    def get_open_questions(self, project_id: int):
        self.last_project_id = project_id
        return self.questions

    def post_forecast(self, question_id: int, p_yes: float, comment: str) -> None:
        self.posted.append((question_id, p_yes, comment))

    def get_resolution(self, question_id: int) -> bool | None:
        _ = question_id
        return self._resolution


class _FakeMemory:
    pass


def _result() -> AgentLoopResult:
    return AgentLoopResult(
        job_id="job-1",
        final_p_yes=0.61,
        confidence_interval=(0.53, 0.69),
        reasoning_summary="Evidence-backed summary.",
        blind_spot_checks_fired=["coalition_stability_check"],
        blind_spot_checks_skipped=[],
        gnn_score_trajectory=[0.55, 0.6, 0.61],
        tool_call_count=3,
    )


def _questions() -> list[MetaculusQuestion]:
    return [
        MetaculusQuestion(1234, "Will X happen by date Y?", "d", "c", date(2026, 7, 1), date(2026, 6, 20)),
        MetaculusQuestion(5678, "Will Z happen by date Q?", "d2", "c2", date(2026, 8, 1), date(2026, 7, 20)),
    ]


def test_run_one_question_uses_default_aib_project_and_posts() -> None:
    client = _FakeClient(_questions())

    out = run_one_question(client=client, run_loop=lambda *_: _result())

    assert client.last_project_id == SPRING_2026_AIB_PROJECT_ID
    assert client.posted == [(1234, 0.61, "Evidence-backed summary.")]
    assert out.question_id == 1234


def test_run_one_question_targets_specific_question_id() -> None:
    client = _FakeClient(_questions())

    out = run_one_question(client=client, run_loop=lambda *_: _result(), question_id=5678)

    assert out.question_id == 5678
    assert client.posted[0][0] == 5678


def test_run_one_question_raises_when_question_id_not_found() -> None:
    client = _FakeClient(_questions())
    with pytest.raises(ValueError, match="not found"):
        run_one_question(client=client, run_loop=lambda *_: _result(), question_id=9999)


def test_run_batch_posts_n_questions() -> None:
    client = _FakeClient(_questions())

    out = run_batch(client=client, run_loop=lambda *_: _result(), batch_size=2)

    assert len(out) == 2
    assert [r.question_id for r in out] == [1234, 5678]
    assert len(client.posted) == 2


def test_try_resolve_question_returns_none_when_unresolved() -> None:
    client = _FakeClient(_questions(), resolution=None)

    got = try_resolve_question(
        client=client,
        question_id=1234,
        job_id="job-1",
        memory=_FakeMemory(),
        tools=None,
        resolver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("resolver should not run")),
    )

    assert got is None


def test_try_resolve_question_returns_brier_result_when_resolved() -> None:
    client = _FakeClient(_questions(), resolution=True)
    expected = BrierUpdateResult(
        job_id="job-1",
        market_id="m-1",
        outcome=True,
        brier_score=0.04,
        misses=["check-a"],
        p_yes_at_resolution=0.8,
    )

    got = try_resolve_question(
        client=client,
        question_id=1234,
        job_id="job-1",
        memory=_FakeMemory(),
        tools=None,
        resolver=lambda **_kwargs: expected,
    )

    assert got == expected


def test_run_one_question_with_resolve_includes_resolution() -> None:
    client = _FakeClient(_questions(), resolution=True)
    expected = BrierUpdateResult(
        job_id="job-1",
        market_id="m-1",
        outcome=True,
        brier_score=0.01,
        misses=[],
        p_yes_at_resolution=0.9,
    )

    out = run_one_question(
        client=client,
        run_loop=lambda *_: _result(),
        question_id=1234,
        resolve=True,
        memory=_FakeMemory(),
        tools=object(),
        resolver=lambda **_kwargs: expected,
    )

    assert out.resolution == expected


def test_run_one_question_raises_when_no_open_questions() -> None:
    client = _FakeClient([])
    with pytest.raises(RuntimeError, match="No open questions"):
        run_one_question(client=client, run_loop=lambda *_: _result())


def test_cli_exit_2_on_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("METACULUS_API_TOKEN", raising=False)
    rc = main(argv=["--question-id", "12345"])
    assert rc == 2


def test_cli_exit_2_on_invalid_question_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METACULUS_API_TOKEN", "token")
    rc = main(argv=["--question-id", "0"])
    assert rc == 2


def test_cli_exit_2_when_resolve_without_question_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METACULUS_API_TOKEN", "token")
    rc = main(argv=["--resolve", "--batch", "1"])
    assert rc == 2


def test_cli_exit_1_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METACULUS_API_TOKEN", "token")

    class _ErrClient:
        def get_open_questions(self, project_id: int):
            raise MetaculusAPIError("401 boom")

    rc = main(argv=["--batch", "1"], client_factory=lambda _token: _ErrClient(), run_loop=lambda *_: _result())
    assert rc == 1
