# Psychohistory

Skills and agents **training bed**: a harness-agnostic plugin for live forecasting. Hosts (Cursor, Claude Code, Codex, …) load it and own `/loop` / `/automate`. Epochs improve the plugin and the vault graph, not model weights.

## Start here

| Document | Purpose |
|----------|---------|
| [`CONTEXT.md`](CONTEXT.md) | Domain language |
| [`project.md`](project.md) | What this repo is |
| [`next_steps.md`](next_steps.md) | Current work order |
| [`docs/adr/`](docs/adr/) | Locked decisions |
| [`docs/history/research-track.md`](docs/history/research-track.md) | Deleted France/GNN/warehouse program |

## Layout

| Path | Role |
|------|------|
| `graph-vault/` | Knowledge graph (timelines, entities, threads) |
| `ledger.md` | Schedule book (problems + dated claims) — to add |
| `skills/` `agents/` `references/` | Plugin overlay — to add |
| `harness/` | Existing Python helpers; `orchestrator.py` is TEMPORARY |
| `ingest/` `schemas/` `data/polymarket/` | Parked Polymarket testbed + live-market data |
| `data/pit_market_probes/` | Legacy PIT probes; not the training loop |
| `scripts/` | CLIs |

## Training loop (forward only)

Discover problems (ungated, **K** new per discovery tick, start `K=1`) → write **motivation** + dated claims with **justification** → daily Parent due-today tick → at `Y` reflection writes plugin + vault. Discovery is a separate, rarer host job. Wire ticker later via the harness.

## Temporary runner (do not build on this)

```text
python -m scripts.run_backtest --source polymarket --max-questions 5
```

Requires `hermes` on PATH. Portable skills must not import `harness.orchestrator`.
