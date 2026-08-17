# Psychohistory

A **training bed** for forecasting skills and subagents that can run in any harness.

## What “done” means here

1. **Point-in-time vault.** For cutoff `t`, only evidence admissible at `t` is visible (`harness/vault_pit.py`).
2. **Portable skills.** Compression, retrieval, blind-spot checks, and search tools do not import hermes or a specific agent CLI.
3. **Eval loops.** Polymarket probes, backtests, and resolution/Brier records measure skill quality.
4. **Harness-agnostic protocol.** Memory schema, episodes, and resolution stay usable when the runner changes.

## Temporary vs durable

- **Durable:** `graph-vault/`, PIT, `harness/skills/`, `harness/tools/`, `harness/policy/`, memory/agent-loop types, Polymarket ingest/schemas, probe data.
- **Temporary:** `harness/orchestrator.py` and chain scripts that shell out to hermes. Replace next session.

## Not this repo anymore

The France GDELT GNN, warehouse, learned graph builder, and world-model track were removed from the working tree. See [`docs/history/research-track.md`](docs/history/research-track.md).
