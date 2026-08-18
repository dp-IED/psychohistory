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
_FIELD = re.compile(r"(?m)^-\s+(?P<name>Problem|Due|Owner|Claim|Justification):\s*(?P<value>.*)$")


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    motivation: str


@dataclass(frozen=True)
class Claim:
    id: str
    problem_id: str
    due: date
    owner: str
    claim: str
    justification: str


@dataclass(frozen=True)
class Ledger:
    k: int  # max new problems per discovery tick
    problems: tuple[Problem, ...] = ()
    claims: tuple[Claim, ...] = ()

    def due_today(self, as_of: date) -> tuple[Claim, ...]:
        return tuple(claim for claim in self.claims if claim.due == as_of)


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
        motivation_match = re.search(
            r"(?ms)^Motivation:\s*(.+?)\s*\Z",
            part[heading.end() :],
        )
        motivation = motivation_match.group(1).strip() if motivation_match else ""
        problems.append(
            Problem(
                id=heading.group("id"),
                title=heading.group("title").strip(),
                motivation=motivation,
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
                due=date.fromisoformat(fields["Due"]),
                owner=fields.get("Owner", ""),
                claim=fields.get("Claim", ""),
                justification=fields.get("Justification", ""),
            )
        )
    return tuple(claims)
