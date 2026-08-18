# Psychohistory

Harness-agnostic **plugin** for live forecasting. The repository **is** the plugin. Load in place (no marketplace cache):

```text
claude --plugin-dir .
```

Then `/due-today`, `/reflect`, or `/discover`. Cursor: project-root Parent pointer at `.cursor/agents/parent.md`.

Epochs grow `skills/`, `agents/`, `references/`, and `scripts/` when a grade earns it. The host owns three separate jobs (`/loop` or `/automate`). Do not add a repo daemon. Weights stay frozen.

Suggested host cadence (host may change it): due-today daily, reflect daily, discover weekly. Each job runs one tick only.

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
| `docs/adr/` | ADRs 0001–0007 |

Retired PIT, hermes, Polymarket eval, and the old `graph-vault/` corpus live in git history (`origin/main`, `e867bec3`), not this branch.
