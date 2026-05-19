"""PIT search shim — re-exports harness.pit_research for synthesis tools."""

from harness.pit_research import (
    PitSearchResponse,
    PitSearchResult,
    pit_search,
    results_to_prompt_block,
)

__all__ = [
    "PitSearchResponse",
    "PitSearchResult",
    "pit_search",
    "results_to_prompt_block",
]
