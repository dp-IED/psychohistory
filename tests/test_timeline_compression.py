from __future__ import annotations

from datetime import date, timedelta

from harness.skills.timeline_compression import TimelineEntry, compress_timeline


def _entries(n: int = 20) -> list[TimelineEntry]:
    base = date(2024, 1, 1)
    return [
        TimelineEntry(
            date=base + timedelta(days=i),
            label=f"Event {i}",
            value=float(i) if i % 2 == 0 else None,
            source=None,
        )
        for i in range(n)
    ]


def test_exactly_20_entries_all_preserved() -> None:
    entries = _entries(20)
    out = compress_timeline(entries, max_entries=20)
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 20
    assert "2024-01-01" in out
    assert "2024-01-20" in out


def test_40_entries_sampled_to_max_with_first3_last3_present() -> None:
    entries = _entries(40)
    out = compress_timeline(entries, max_entries=20)
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) <= 20

    assert "2024-01-01" in out
    assert "2024-01-02" in out
    assert "2024-01-03" in out

    assert "2024-02-07" in out
    assert "2024-02-08" in out
    assert "2024-02-09" in out


def test_value_none_renders_without_value_column_text() -> None:
    entries = [TimelineEntry(date(2024, 1, 1), "No value event", None, None)]
    out = compress_timeline(entries)
    assert "No value event" in out
    assert "value=" not in out.lower()


def test_output_non_empty_string() -> None:
    out = compress_timeline(_entries(3))
    assert isinstance(out, str)
    assert out.strip()
