"""GDELT CAMEO-style root-code severity tiers (v0).

Maps ``event_root_code`` (two-digit GDELT ``EventRootCode`` strings) to integer
tiers where **higher = more severe**. Non-GDELT tapes may use non-numeric
``event_root_code`` values; those map to tier ``0`` (lowest).
"""

from __future__ import annotations


def cameo_tier(root_code: str) -> int:
    """Return escalation tier for a GDELT-style root code.

    **Tier bands (locked):** codes ``01``–``09`` → tier ``0``; ``10``–``17`` →
    tier ``1`` (includes ``14`` protest); ``18``–``20`` → tier ``2``.

    Unknown, blank, non-numeric, or out-of-range codes → tier ``0`` (lowest).

    Parsing: leading whitespace is stripped; the first two characters are used
    when both are ASCII digits (``"14"``, ``"09"``, ``"141"`` → ``"14"`` prefix).
    """
    if not root_code:
        return 0
    s = root_code.strip()
    if len(s) < 2:
        return 0
    prefix = s[:2]
    if not (prefix[0].isdigit() and prefix[1].isdigit()):
        return 0
    n = int(prefix)
    if n < 1 or n > 20:
        return 0
    if n <= 9:
        return 0
    if n <= 17:
        return 1
    return 2


__all__ = ["cameo_tier"]
