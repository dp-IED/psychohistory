# Gap 1: Vault-Ablated Control

## Purpose

Determine whether the PIT vault actually adds forecasting value above the agent's parametric knowledge. Run each market probe **without** any vault context (no PIT librarian brief, no vault files accessible, no _forecast_instructions.md) and compare Brier/MAE to the vault-augmented run.

## What to Build

### Changes to `harness/pit_market_probe.py`

Add a `no_vault: bool = False` parameter to `run_market_probe()`:
- When `no_vault=True`:
  1. Skip the PIT librarian entirely (`run_pit_research` is not called)
  2. Do NOT set `enforce_pit=True` in `run_structured()`
  3. Do NOT pass `pit_brief_block` to the forecaster
  4. Set `graph_only=False` — the agent CAN use `web_search` BUT must respect the cutoff date
  5. Pass a modified prompt that says "You have NO vault access. Forecast using general knowledge and web search only, respecting the cutoff date."

Add `vault_ablated_p_yes: float | None` and `vault_ablated_reasoning: str` fields to `MarketProbeResult`.

Add `score_ablation()`: compare vault-augmented vs vault-ablated results per probe. Compute:
  - `vault_value_add = abs(ablated_mae) - abs(vault_mae)` (negative = vault helped)
  - `vault_worsened = vault_mae > ablated_mae` (True when vault made it worse)

### Changes to `scripts/pit_market_calibration.py`

Add a `--no-vault` flag to the `run` subcommand that enables ablation mode.

Add an `ablate` subcommand:
```
python scripts/pit_market_calibration.py ablate --catalog data/pit_market_probes/catalog.jsonl
```
This re-runs all completed probes without vault access and stores results to `data/pit_market_probes/ablated_results.jsonl`.

Add a `compare` subcommand:
```
python scripts/pit_market_calibration.py compare
```
This loads both `results.jsonl` (vault) and `ablated_results.jsonl` and prints a table:
```
Probe                          | Vault MAE | Ablated MAE | Delta  | Vault Helped?
gold-gold_01_...               |  0.005    |   0.120     | -0.115 | ✅ YES
gold-gold_18_...               |  0.265    |   0.080     | +0.185 | ❌ NO
```

### Prompt for no-vault mode

When `no_vault=True`, build a prompt like:
```
Question: {question}
Cutoff (strict PIT): {cutoff}
Polymarket YES at cutoff: {market_yes}

NO VAULT MODE: You do not have access to the graph-vault.
Forecast using ONLY general knowledge and web search.
You MUST respect the cutoff date — no post-cutoff information.
Web search IS allowed (unlike vault mode).

Output JSON: {{"p_yes": 0.XX, "reasoning": "..."}}
```

## Files to Modify

- `harness/pit_market_probe.py` — `run_market_probe()`, `MarketProbeResult`, add `score_ablation()`
- `scripts/pit_market_calibration.py` — add `cmd_ablate`, `cmd_compare`
- `harness/orchestrator.py` — `_build_structured_prompt()` or `run_structured()` may need a `use_vault=False` parameter that skips vault setup and does NOT set `enforce_pit=True`

## Test

Run `python scripts/pit_market_calibration.py ablate --catalog data/pit_market_probes/catalog.jsonl` then `python scripts/pit_market_calibration.py compare`. Verify the comparison table prints correctly.

## Key Metrics

- How many probes does vault help vs hurt?
- Mean vault MAE vs mean ablated MAE
- Is there a domain where vault consistently hurts (suggesting noise/incomplete coverage)?
