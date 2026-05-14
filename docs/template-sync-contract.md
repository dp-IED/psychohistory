# Template ↔ Harness Sync Contract (v0)

Discovery result: `metac-bot-template/main.py` does **not** persist machine-readable outputs by default (`folder_to_save_reports_to=None`).

So sync relies on JSONL exports placed in a shared directory.

## Expected input format (`*.jsonl`)
One JSON object per line. Required fields:
- `question_id` (int)
- `question_text` (string)
- `run_timestamp` (ISO datetime string)
- `posted_probability` (float 0..1)
- one of `close_date|scheduled_close_time|cutoff_date` (ISO date/datetime)
- one of `resolution_date|resolve_time|scheduled_resolve_time|close_date` (ISO date/datetime)

Optional:
- `resolved_outcome|outcome|resolution` (bool) to trigger Brier resolution update

## Idempotency
- Deterministic job id: `template-{question_id}` (one episode per question)
- If `read_episode_by_id(job_id)` exists, import is skipped.
- If resolution already applied, `AlreadyResolvedError` is swallowed and counted.

## Script
- `scripts/sync_template_runs.py`

CLI:
```bash
python -m scripts.sync_template_runs \
  --template-output-dir /path/to/template/exports \
  --memory-dir .harness_memory
```
