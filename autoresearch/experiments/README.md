# Experiments

- `log.jsonl` — append-only JSON lines from `autoresearch.runner` (proposal, scores, failure class, artifact paths).
- `candidates/` — per-run sandbox trees (copied `schemas/` + `eval_report.json`).
- `best/latest/` — most recent candidate that matched or improved the rolling best composite score (see runner logic).

No databases: all artifacts are plain files for inspection and git-ignoring as you prefer.
