# Next steps

This branch is a harness-agnostic **plugin training bed**. Do not extend the hermes v1 runner. Do not treat historical PIT Brier as the training objective.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next session (product)

1. **Plugin overlay** — add `skills/`, `agents/` (Parent + workers), `references/` on top of existing `harness/` + `graph-vault/`. Guard skill bloat (`/writing-for-agents`).
2. **`ledger.md`** at repo root — problems (with motivation) + dated claims (justification). `K=1` intake knob.
3. **Parent behavior** — due-today: read ledger, wake scheduled agents, reflect at `Y` into **plugin + vault graph**. Discovery: separate rarer tick, ungated problems, at most K new per tick.
4. **Host ticker** — wire due-today vs discovery with the harness `/loop` or `/automate` (not a repo daemon). Deferred until the overlay and ledger exist.
5. **Polymarket testbed** — historical gold/probes/schemas; next item after the live loop exists. Live markets are a discovery surface, not that testbed.

## Do not pick up

- Replacing `harness/orchestrator.py` as the product (plugin + host harness instead)
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN (`docs/history/research-track.md`)
- `orchestrator_v2` / GNN tag calibration (deleted)
- Merging this branch into `origin/main` until you choose to
- Conductor worktrees (`testbed`, `gnn-agent`) — left as artifacts

## Scripts inventory (current)

- Vault/PIT (legacy, not the training loop): `validate_vault.py`, `vault_relevance_probe.py`, `pit_market_calibration.py`, `pit_phrasing_scan.py`, `thread_continuity_audit.py`, `bootstrap_timeline.py`, `pit_train.py`
- Temporary hermes: `run_backtest.py`, `run_chain.py`, `batch_chain.sh`
- Polymarket data: `fetch_polymarket_resolved.py`, `build_polymarket_branch_graphs.py`, `build_polymarket_gold_dataset.py`
