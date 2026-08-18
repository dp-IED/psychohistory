# Psychohistory

A **training bed** for a forecasting **plugin** that any harness can load (Cursor, Claude Code, Codex, …).

## What “done” means here

1. **Plugin overlay.** `skills/`, `agents/`, `references/` plus deterministic helpers. The host owns orchestration (`/loop`, `/automate`). No hermes import.
2. **Live training loop.** Discover problems and claims → `ledger.md` → Parent wakes due agents → at `Y` reflect into plugin **and** vault graph. Fan-out is many live OSINT / political / socioeconomic bets, including self-chosen **problems** (method that should pay forward).
3. **Vault graph.** Reflection may add entities, threads, concepts. GNN analysis of that graph is later (research track stays parked).
4. **Epochs improve the plugin.** Frozen model weights. Guard skill bloat.

Historical PIT filters and gold probes remain in-tree as legacy; they are not the epoch objective. The Polymarket **testbed** (gold/schemas) is parked; live markets are a discovery surface.

## Temporary vs durable

- **Durable:** `graph-vault/`, plugin overlay (to add), `ledger.md` (to add), `CONTEXT.md`, `docs/adr/`.
- **Temporary:** `harness/orchestrator.py` and chain scripts that shell out to hermes. Do not rewrite them into the product.

## Not this repo anymore

The France GDELT GNN, warehouse, learned graph builder, and world-model track were removed from the working tree. See [`docs/history/research-track.md`](docs/history/research-track.md).
