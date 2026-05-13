from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TimelineEntry:
    date: date
    label: str
    value: float | None
    source: str | None


def _sample_indices(n: int, max_entries: int) -> list[int]:
    if n <= max_entries:
        return list(range(n))

    head = [0, 1, 2]
    tail = [n - 3, n - 2, n - 1]
    middle_slots = max_entries - len(head) - len(tail)
    middle_start = 3
    middle_end = n - 4

    if middle_slots <= 0 or middle_end < middle_start:
        return sorted(set(head + tail))

    span = middle_end - middle_start + 1
    picks = []
    for i in range(middle_slots):
        # evenly distributed positions
        idx = middle_start + round(i * (span - 1) / max(1, middle_slots - 1))
        picks.append(idx)

    return sorted(set(head + picks + tail))[:max_entries]


def compress_timeline(entries: list[TimelineEntry], max_entries: int = 20) -> str:
    if max_entries < 1:
        raise ValueError("max_entries must be >= 1")
    if not entries:
        return ""

    ordered = sorted(entries, key=lambda e: e.date)
    idxs = _sample_indices(len(ordered), max_entries)

    lines: list[str] = []
    for i in idxs:
        item = ordered[i]
        if item.value is None:
            lines.append(f"{item.date.isoformat()} | {item.label}")
        else:
            lines.append(f"{item.date.isoformat()} | {item.label} | {item.value:g}")

    return "\n".join(lines)


__all__ = ["TimelineEntry", "compress_timeline"]
