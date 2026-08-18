from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_PROBLEM_HEADING = re.compile(
    r"^###\s+(?P<id>\S+)\s+[—-]\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)
_CLAIM_HEADING = re.compile(r"^###\s+(?P<id>\S+)\s*$", re.MULTILINE)
_K_LINE = re.compile(r"(?m)^K:\s*(\d+)\s*$")
_RESOLUTION = re.compile(r"(?m)^Resolution:\s*(\d{4}-\d{2}-\d{2})\s*$")
_MOTIVATION = re.compile(
    r"(?ms)^Motivation:\s*(.+?)(?=^Resolution:|\Z)",
)
_FIELD = re.compile(
    r"(?m)^-\s+(?P<name>Problem|Forecast|Owner|Claim|Justification):\s*(?P<value>.*)$"
)


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    motivation: str
    resolution_day: date


@dataclass(frozen=True)
class Claim:
    id: str
    problem_id: str
    forecast_day: date
    owner: str
    claim: str
    justification: str


@dataclass(frozen=True)
class Ledger:
    k: int  # max new problems per discover tick
    problems: tuple[Problem, ...] = ()
    claims: tuple[Claim, ...] = ()

    def problem(self, problem_id: str) -> Problem | None:
        for item in self.problems:
            if item.id == problem_id:
                return item
        return None

    def live_problems(self, as_of: date) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if p.resolution_day >= as_of)

    def after_resolution(self, as_of: date) -> tuple[Claim, ...]:
        past = {p.id for p in self.problems if as_of > p.resolution_day}
        return tuple(c for c in self.claims if c.problem_id in past)

    def claims_for(self, problem_id: str) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.problem_id == problem_id)


def parse_ledger(text: str) -> Ledger:
    k_match = _K_LINE.search(text)
    if k_match is None:
        raise ValueError("ledger is missing K")
    return Ledger(
        k=int(k_match.group(1)),
        problems=_parse_problems(text),
        claims=_parse_claims(text),
    )


def _section(text: str, heading: str) -> str:
    pattern = rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _blocks(body: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?m)^(?=### )", body) if part.strip()]


def _parse_problems(text: str) -> tuple[Problem, ...]:
    problems: list[Problem] = []
    for part in _blocks(_section(text, "Problems")):
        heading = _PROBLEM_HEADING.search(part)
        if heading is None:
            continue
        body = part[heading.end() :]
        res_match = _RESOLUTION.search(body)
        if res_match is None:
            raise ValueError(f"problem {heading.group('id')} is missing Resolution")
        motivation_match = _MOTIVATION.search(body)
        motivation = motivation_match.group(1).strip() if motivation_match else ""
        problems.append(
            Problem(
                id=heading.group("id"),
                title=heading.group("title").strip(),
                motivation=motivation,
                resolution_day=date.fromisoformat(res_match.group(1)),
            )
        )
    return tuple(problems)


def _parse_claims(text: str) -> tuple[Claim, ...]:
    claims: list[Claim] = []
    for part in _blocks(_section(text, "Claims")):
        heading = _CLAIM_HEADING.search(part)
        if heading is None:
            continue
        fields: dict[str, str] = {}
        for field in _FIELD.finditer(part):
            fields[field.group("name")] = field.group("value").strip()
        claims.append(
            Claim(
                id=heading.group("id"),
                problem_id=fields.get("Problem", ""),
                forecast_day=date.fromisoformat(fields["Forecast"]),
                owner=fields.get("Owner", ""),
                claim=fields.get("Claim", ""),
                justification=fields.get("Justification", ""),
            )
        )
    return tuple(claims)
