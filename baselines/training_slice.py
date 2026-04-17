"""
Pinned calendar windows for reproducible train/holdout splits (`next_steps.md` §2.1 step B).

Scaffold uses **France** regional event tape conventions; naming stays explicit so the same
pattern can be copied for other domains without implying France-only models.
"""

from __future__ import annotations

import datetime as dt

# Weekly forecast origins in [TRAIN_ORIGIN_START, TRAIN_ORIGIN_END] (inclusive) — model development.
FRANCE_SCAFFOLD_TRAIN_ORIGIN_START: dt.date = dt.date(2020, 1, 6)
FRANCE_SCAFFOLD_TRAIN_ORIGIN_END: dt.date = dt.date(2022, 12, 26)

# Holdout origins for reporting — do not tune on this range when claiming generalization.
FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START: dt.date = dt.date(2023, 1, 2)
FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END: dt.date = dt.date(2023, 12, 25)
