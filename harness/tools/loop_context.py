"""Per-thread research context for tool closures (question, dates, tool_calls tail).

Agent loop sets these before graph_query / gnn_score so real tool implementations
can read fuller state without changing AgentToolset callable signatures.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import date
from typing import Iterator

from harness.memory_schema import ToolCallRecord

_question_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("research_question", default=None)
_cutoff_var: contextvars.ContextVar[date | None] = contextvars.ContextVar("research_cutoff", default=None)
_resolution_var: contextvars.ContextVar[date | None] = contextvars.ContextVar("research_resolution", default=None)
_market_family_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("research_market_family", default=None)
_tool_calls_var: contextvars.ContextVar[list[ToolCallRecord] | None] = contextvars.ContextVar(
    "research_tool_calls", default=None
)
_vault_synthesis_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vault_synthesis_context", default=None
)


def get_research_question() -> str | None:
    return _question_var.get()


def get_research_cutoff() -> date | None:
    return _cutoff_var.get()


def get_research_resolution() -> date | None:
    return _resolution_var.get()


def get_research_market_family() -> str | None:
    return _market_family_var.get()


def get_research_tool_calls() -> list[ToolCallRecord] | None:
    return _tool_calls_var.get()


def get_vault_synthesis_context() -> str | None:
    return _vault_synthesis_var.get()


def set_vault_synthesis_context(text: str | None) -> contextvars.Token[str | None]:
    return _vault_synthesis_var.set(text)


def reset_vault_synthesis_context(token: contextvars.Token[str | None]) -> None:
    _vault_synthesis_var.reset(token)


@contextmanager
def agent_research_context(
    question: str,
    cutoff_date: date,
    resolution_date: date,
) -> Iterator[None]:
    t_q = _question_var.set(question)
    t_c = _cutoff_var.set(cutoff_date)
    t_r = _resolution_var.set(resolution_date)
    t_m = _market_family_var.set(None)
    t_tc = _tool_calls_var.set(None)
    try:
        yield
    finally:
        _question_var.reset(t_q)
        _cutoff_var.reset(t_c)
        _resolution_var.reset(t_r)
        _market_family_var.reset(t_m)
        _tool_calls_var.reset(t_tc)


def set_market_family_for_research(family: str) -> contextvars.Token[str | None]:
    return _market_family_var.set(family)


def reset_market_family(token: contextvars.Token[str | None]) -> None:
    _market_family_var.reset(token)


def set_step_tool_calls(calls: list[ToolCallRecord]) -> contextvars.Token[list[ToolCallRecord] | None]:
    return _tool_calls_var.set(calls)


def reset_step_tool_calls(token: contextvars.Token[list[ToolCallRecord] | None]) -> None:
    _tool_calls_var.reset(token)
