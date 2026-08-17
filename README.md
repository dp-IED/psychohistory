# Psychohistory

Skills and subagent **training bed** for temporally clean forecasting.

The durable product is a point-in-time **graph-vault**, PIT filters, compression skills, and eval probes. Agents should learn to forecast from admissible evidence only — not from a single CLI runner.

A **temporary** hermes v1 orchestrator still posts forecasts. Portable skills must not import it. Next session replaces that runner.

## Start here

| Document | Purpose |
|----------|---------|
| [`project.md`](project.md) | What this repo is |
| [`next_steps.md`](next_steps.md) | Current work order |
| [`docs/polymarket_agentic_harness.md`](docs/polymarket_agentic_harness.md) | Market frames, branches, PIT labels |
| [`docs/history/research-track.md`](docs/history/research-track.md) | Deleted France/GNN/warehouse program (git history only) |

## Layout

| Path | Role |
|------|------|
| `graph-vault/` | Knowledge store (timelines, entities, threads, forecast runs) |
| `harness/` | PIT, skills, tools, memory protocol; `orchestrator.py` is TEMPORARY |
| `ingest/` | Polymarket resolved markets and branch graphs |
| `schemas/` | Polymarket agentic contracts |
| `data/polymarket/` | Hypotheses, tracked markets, forecasts |
| `data/pit_market_probes/` | PIT calibration probes |
| `scripts/` | Vault/PIT/backtest CLIs |

## Temporary runner (do not build on this)

```text
python -m scripts.run_backtest --source polymarket --max-questions 5
```

Requires `hermes` on PATH. Skills and subagents should take vault + cutoff + question as inputs instead.

## Non-negotiable

No post-`t` facts in inputs. Replay by cutoff.
