# Psychohistory

Harness-agnostic **plugin** for live forecasting. The repository **is** the plugin. Load in place (no marketplace cache):

```text
claude --plugin-dir .
```

Then `/predict`, `/reflect`, or `/discover`. Cursor: project-root Parent pointer at `.cursor/agents/parent.md`.

Epochs grow `skills/`, `agents/`, `references/`, and `scripts/` when a grade earns it. The host owns three separate jobs (`/loop` or `/automate`). Do not add a repo daemon. Weights stay frozen.

Suggested host cadence (host may change it): predict daily, reflect daily later the same day, discover weekly after both. Each job runs one tick only. Jobs must not overlap: pull fast-forward, then one tick, then commit and push `harness-only` (no pull request).

## Start here

| Document | Purpose |
|----------|---------|
| [`CONTEXT.md`](CONTEXT.md) | Domain language |
| [`project.md`](project.md) | What this repo is |
| [`next_steps.md`](next_steps.md) | Current work order |
| [`docs/adr/`](docs/adr/) | Locked decisions |
| [`ledger.md`](ledger.md) | Schedule book |

## Layout

| Path | Role |
|------|------|
| `.claude-plugin/plugin.json` | Plugin manifest (no version pin) |
| `skills/` `agents/` `references/` | Overlay |
| `ledger.md` | Problems + dated claims |
| `harness/ledger.py` | Deterministic schedule reader |
| `docs/adr/` | ADRs 0001–0010 |

Retired PIT, hermes, Polymarket eval, and the old `graph-vault/` corpus live in git history (`origin/main`, `e867bec3`), not this branch.
