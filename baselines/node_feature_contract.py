"""
Enumerated node feature expectations at snapshot time `t` (`next_steps.md` §2.1 step C).

The France harness uses **admin1_code** grid cells as prediction units; the heterogeneous GNN
uses typed graph nodes from snapshots. This table is the contract to extend when new node
types or WM v0 tensors land — not an implementation of those features.
"""

from __future__ import annotations

from typing import Any

# Logical node or unit kind -> feature groups available at forecast_origin (PIT-safe inputs).
NODE_FEATURE_CONTRACT: dict[str, dict[str, Any]] = {
    "Location_admin1_forecast_unit": {
        "tabular_from_tape": [
            "event_count_prev_1w",
            "event_count_prev_4w",
            "event_count_prev_12w",
            "rate_change_1w_vs_4w",
            "mean_goldstein_prev_4w",
            "mean_avg_tone_prev_4w",
            "mean_num_mentions_prev_4w",
            "mean_num_articles_prev_4w",
            "distinct_actor_count_prev_4w",
            "national_event_count_prev_1w",
            "national_event_count_prev_4w",
            "weeks_since_last_event",
            "admin1_code_idx",
        ],
        "optional_future": ["frozen_text_embedding"],
        "not_in_baseline": ["wikidata_property_bundle", "latent_wm_state"],
    },
    "Event_snapshot_node": {
        "typical": ["type", "time_span", "source_refs"],
        "optional_future": ["text_embedding"],
    },
}
