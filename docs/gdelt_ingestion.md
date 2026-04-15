# GDELT Ingestion Operations

This path ingests raw GDELT 2.0 event exports for the France protest benchmark and turns them into a point-in-time event tape. The raw fragments are intentionally kept before normalization so each downstream graph snapshot or baseline row can be audited back to the source export file, retrieval time, and filter configuration.

The first supported domain is narrow: France protest events from GDELT event rows where `ActionGeo_CountryCode == "FR"` and `EventRootCode == "14"`. The prediction unit is the country-qualified GDELT action geocode, using `FR` plus `ActionGeo_ADM1Code`; this keeps the evaluation grid stable even when a location has no prior events before a forecast origin. X/Twitter may later provide narrative or attention signals, but it is deferred and is not part of this event-source tape or baseline benchmark.

## Time Fields

`event_date` and `source_available_at` are separate because they answer different questions.

`event_date` is when the protest event is reported to have happened. It defines feature windows and target windows.

`source_available_at` is when the GDELT export containing the record became available. It defines what a forecast made at a historical origin was allowed to know. Feature reconstruction must use only records where `source_available_at < forecast_origin` and `event_date < forecast_origin_date`.

## Smoke Run

Use this tiny live run to validate network access, raw filtering, event-tape normalization, batch construction, and graph snapshot export.

```bash
python -m ingest.gdelt_raw fetch-france-protests \
  --event-start 2023-03-20 \
  --event-end 2023-04-03 \
  --source-start 2023-03-23T00:00:00Z \
  --source-end 2023-04-03T00:00:00Z \
  --out data/gdelt/raw/france_protest_smoke \
  --workers 4

python -m ingest.event_tape normalize-france-protests \
  --raw data/gdelt/raw/france_protest_smoke \
  --out data/gdelt/tape/france_protest_smoke/events.jsonl

python -m ingest.historical_injection build-batches \
  --tape data/gdelt/tape/france_protest_smoke/events.jsonl \
  --out data/gdelt/injection/france_protest_smoke/batches.jsonl

python -m ingest.snapshot_export export-weekly \
  --tape data/gdelt/tape/france_protest_smoke/events.jsonl \
  --origin-start 2023-03-27 \
  --origin-end 2023-03-27 \
  --out data/gdelt/snapshots/france_protest_smoke
```

This window intentionally extends through the forecast week for the `2023-03-27` origin. It should produce both pre-origin feature events and target-window labels. If the fetch metadata has `kept_row_count == 0`, rerun with the wider source window:

```bash
python -m ingest.gdelt_raw fetch-france-protests \
  --event-start 2023-03-20 \
  --event-end 2023-04-03 \
  --source-start 2023-03-20T00:00:00Z \
  --source-end 2023-04-10T00:00:00Z \
  --out data/gdelt/raw/france_protest_smoke \
  --workers 4
```

If GDELT lists historical exports that now return 404, the fetcher records those failures in `fetch_manifest.jsonl`. Rerun the same command with `--allow-partial` only when the metadata still has `kept_row_count > 0` and the missing files are acceptable for a smoke check. Use `--force` when replacing an older smoke run with a different event or source window.

## Full Historical Fetch

This is expensive because it scans years of 15-minute GDELT export files. Run it only when you are ready to build the full benchmark tape.

```bash
python -m ingest.gdelt_raw fetch-france-protests \
  --event-start 2019-01-01 \
  --event-end 2026-01-04 \
  --source-start 2019-01-01T00:00:00Z \
  --source-end latest \
  --out data/gdelt/raw/france_protest \
  --workers 8
```

After the fetch, run the same normalization, injection, snapshot, and baseline commands against the non-smoke paths.

```bash
python -m ingest.event_tape normalize-france-protests \
  --raw data/gdelt/raw/france_protest \
  --out data/gdelt/tape/france_protest/events.jsonl

python -m ingest.historical_injection build-batches \
  --tape data/gdelt/tape/france_protest/events.jsonl \
  --out data/gdelt/injection/france_protest/batches.jsonl

python -m ingest.snapshot_export export-weekly \
  --tape data/gdelt/tape/france_protest/events.jsonl \
  --origin-start 2021-01-04 \
  --origin-end 2025-12-29 \
  --out data/gdelt/snapshots/france_protest

python -m baselines.backtest recurrence \
  --tape data/gdelt/tape/france_protest/events.jsonl \
  --origin-start 2021-01-04 \
  --origin-end 2025-12-29 \
  --out data/gdelt/baselines/france_protest/recurrence_predictions.jsonl
```

## Generated Paths

Raw fetch output:

```text
data/gdelt/raw/france_protest*/fetch_metadata.json
data/gdelt/raw/france_protest*/fetch_manifest.jsonl
data/gdelt/raw/france_protest*/fragments/**.jsonl
```

Normalized event tape:

```text
data/gdelt/tape/france_protest*/events.jsonl
data/gdelt/tape/france_protest*/events.audit.json
```

Historical injection batches:

```text
data/gdelt/injection/france_protest*/batches.jsonl
```

Graph snapshots:

```text
data/gdelt/snapshots/france_protest*/as_of_YYYY-MM-DD.json
```

Baseline predictions:

```text
data/gdelt/baselines/france_protest/recurrence_predictions.jsonl
data/gdelt/baselines/france_protest/recurrence_predictions.audit.json
```

`data/gdelt/**` is gitignored and should not be committed. Commit code, tests, and docs only.

## Audit Checks

Inspect fetch metadata first:

```bash
python -m json.tool data/gdelt/raw/france_protest_smoke/fetch_metadata.json
```

Check failed files and kept rows in the manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/gdelt/raw/france_protest_smoke/fetch_manifest.jsonl")
failed = []
kept = 0
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    kept += int(row.get("kept_row_count") or 0)
    if row.get("status") == "failed":
        failed.append((row["source_file_timestamp"], row["error"]))
print("kept rows:", kept)
print("failed files:", len(failed))
for timestamp, error in failed[:10]:
    print(timestamp, error)
PY
```

Inspect normalization counts:

```bash
python -m json.tool data/gdelt/tape/france_protest_smoke/events.audit.json
wc -l data/gdelt/tape/france_protest_smoke/events.jsonl
```

Confirm that the smoke snapshot has real labels:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/gdelt/snapshots/france_protest_smoke/as_of_2023-03-27.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print("feature records:", payload["metadata"]["feature_record_count"])
print("label records:", payload["metadata"]["label_record_count"])
print("unscored labels:", payload["metadata"]["label_audit"]["unscored_admin1_event_count"])
PY
```

Validate a snapshot:

```bash
python - <<'PY'
import json
from pathlib import Path
from evals.graph_artifact_contract import GraphArtifactV1

path = Path("data/gdelt/snapshots/france_protest_smoke/as_of_2023-03-27.json")
GraphArtifactV1.model_validate(json.loads(path.read_text(encoding="utf-8")))
print("valid:", path)
PY
```
