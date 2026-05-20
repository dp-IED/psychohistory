# Gap 2: Instrumented Probe Results

## Purpose

When a probe misses the market calibration band, we currently have almost no trace of *why*: we get `p_yes`, `reasoning`, and the librarian brief — but not which vault files the agent read, which rules from `_forecast_instructions.md` were triggered, or what the agent's internal deliberation was. This spec adds full instrumentation so every probe result is debuggable.

## What to Build

### New fields in `MarketProbeResult`

Add to the `@dataclass` in `harness/pit_market_probe.py`:

```python
vault_files_read: list[str] = field(default_factory=list)
    # Relative paths of vault files the agent reported reading.
    # Only counts files that exist in the PIT snapshot (post-filtering).
    
rules_checked: list[str] = field(default_factory=list)
    # Which rules from _forecast_instructions.md the agent checked.
    # Each entry is a rule number/name, e.g. "Rule 1: Central Bank Questions"
    
rules_triggered: list[str] = field(default_factory=list)
    # Subset of rules_checked that the agent considered relevant enough
    # to influence the forecast.
```

Also add to the `to_dict()` method and the `from_dict()` constructor for persistence.

### Instrumented Agent Prompt

Modify the output format requested from the agent (in `_build_structured_prompt()` in `harness/orchestrator.py` or `build_forecast_prompt()` in `harness/pit_market_probe.py`). The current format is:

```json
{"p_yes": 0.XX, "reasoning": "one-sentence summary"}
```

Change to:

```json
{
  "p_yes": 0.XX,
  "reasoning": "one-sentence summary",
  "vault_files_read": ["_forecast_instructions.md", "concepts/foo.md", ...],
  "rules_checked": ["Rule 1: Central Bank Questions", "Rule 2: Domestic Politics Gap Check", ...],
  "rules_triggered": ["Rule 2: Domestic Politics Gap Check"],
  "deliberation": "2-3 sentences explaining key reasoning steps and which vault evidence most influenced the forecast"
}
```

All new fields are optional — the probe should not fail if the agent omits them. Default to empty lists.

### Return Instrumented Data Through the Pipeline

1. `harness/orchestrator.py` line 84 `validate_structured()`: extract the new fields from agent JSON output. Return them alongside `(p_yes, reasoning, errors)` — update the return type.

2. `harness/orchestrator.py` `run_structured()`: pass the instrumented fields back to the caller.

3. `harness/pit_market_probe.py` `run_market_probe()`: store instrumented fields in `MarketProbeResult`.

4. `harness/pit_market_probe.py` `format_market_calibration_feedback()`: include `vault_files_read` and `rules_triggered` in miss diagnostics.

### Reflection Integration

Update `scripts/pit_reflect.py` to include `vault_files_read` and `rules_triggered` in the feedback it sends to the reflection agent. Change the miss-formatting block to include:

```
  vault_files_read: _forecast_instructions.md, concepts/foo.md, entities/bar.md
  rules_checked: Rule 1, Rule 2, Rule 9
  rules_triggered: Rule 2
  deliberation: {deliberation text}
```

### Persistence

The instrumented fields should be stored in the JSONL results rows alongside existing fields. Update `MarketProbeResult.to_dict()` and `MarketProbeResult.from_dict()` accordingly.

## Files to Modify

- `harness/pit_market_probe.py` — `MarketProbeResult` (new fields), `build_forecast_prompt()` (output format), `run_market_probe()` (pass through), `format_market_calibration_feedback()` (display)
- `harness/orchestrator.py` — `validate_structured()` (extract new fields), `_build_structured_prompt()` (output format in prompt), `run_structured()` (return new fields)
- `scripts/pit_reflect.py` — miss diagnostic formatting

## Backward Compatibility

All existing result rows in `results.jsonl` lack the new fields. Code that loads results must handle missing keys gracefully — default `vault_files_read` to `[]`, `rules_checked` to `[]`, etc. Re-running probes will populate them.

## Test

1. Run `python scripts/pit_market_calibration.py run --skip-existing` on one probe that's already in the catalog.
2. Check that the new result row in `results.jsonl` contains `vault_files_read`, `rules_checked`, `rules_triggered`.
3. Run `python scripts/pit_market_calibration.py score` — verify scoring still works with both old and new result rows.
