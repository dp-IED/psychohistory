"""Single source of truth for default weekly-origin train/holdout windows.

The France protest tape-driven snapshot builder in :mod:`ingest.snapshot_export` is the
**reference scaffold** today. These constants are named so another region or domain can
swap values without overloading ``FR``-specific identifiers in call sites.

**Development** (train) and **holdout** here mean inclusive bounds of valid Monday
``forecast_origin`` dates passed to :func:`ingest.snapshot_export.build_snapshot_payload`.
The holdout block is the final ~12 calendar months of configured origins in this scaffold;
the development block is the contiguous prior range used for model fitting.
"""

from __future__ import annotations

import datetime as dt

# Informational: ``metadata["domain"]`` for the reference implementation (not used in split math).
FORECAST_SCAFFOLD_REFERENCE_DOMAIN_KEY = "france_protest"

WEEKLY_ORIGIN_DEVELOPMENT_START: dt.date = dt.date(2021, 1, 4)
WEEKLY_ORIGIN_DEVELOPMENT_END: dt.date = dt.date(2024, 12, 30)

WEEKLY_ORIGIN_HOLDOUT_START: dt.date = dt.date(2025, 1, 6)
WEEKLY_ORIGIN_HOLDOUT_END: dt.date = dt.date(2025, 12, 29)
