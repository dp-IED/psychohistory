# Psychohistory

Harness-agnostic **plugin** for live forecasting. The repository **is** the plugin. Load in place (no marketplace cache):

```text
claude --plugin-dir .
```

Then `/due-today` or `/reflect`. Cursor: project-root Parent pointer at `.cursor/agents/parent.md`.

Epochs improve `skills/`, `agents/`, `references/`, and (when Parent first writes it) a vault graph. The host owns `/loop` / `/automate`. Weights stay frozen.

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
| `docs/adr/` | ADRs 0001–0006 |

Retired PIT, hermes, Polymarket eval, and the old `graph-vault/` corpus live in git history (`origin/main`, `e867bec3`), not this branch.
