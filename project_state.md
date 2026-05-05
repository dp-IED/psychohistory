# Project state (ground truth)

Last updated: Arab Spring ingest v0.1 landed (ingest pipeline + tests). Stage 2 graph-builder probe scaffolding remains; warehouse and event tape are now `event_root_code`-canonical (no `event_class`).

## Tests

- Full suite: **199** tests (`pytest`), including mocked ACLED OAuth Arab Spring pagination (`tests/test_acled_arab_spring_fetch.py`).

## Arab Spring ingest (GDELT 1.0 + ACLED OAuth read API)

- **GDELT:** `python -m ingest.gdelt_raw fetch-arab-spring` — daily GDELT 1.0 URLs under `--events-base-url` (default `http://data.gdeltproject.org/events`), fragments `arab_spring_YYYYMMDD_000000.jsonl`, `fetch_manifest.json`, zip bytes not kept on disk.
- **ACLED:** `ACLED_EMAIL` (or `ACLED_USERNAME`) + `ACLED_PASSWORD` → OAuth token → `python -m ingest.acled_raw fetch-arab-spring` — `https://acleddata.com/api/acled/read` (Bearer), fragments `acled_arab_spring_page_NNNN.jsonl`, pagination stops on an **empty** `data` page (not on `len(data) < limit`). The myACLED account must have **API access** (ACLED grants `restful get acled_api_endpoint`); a login alone is not enough — request access via [ACLED](https://acleddata.com/) / their access team if you see that permission error.
- **Tape merge:** `python -m ingest.event_tape merge-arab-spring` — optional `--cleanup-fragments`.
- **France GDELT tape:** `normalize-france-protests` passes explicit `GdeltTapeNormalizeParams` (FR + CAMEO `14`).

### Smoke order before a full 2010–2013 fetch

Do **not** start the full multi-year pull blindly. Suggested sequence:

1. **GDELT smoke:** narrow window first, e.g.  
   `python -m ingest.gdelt_raw fetch-arab-spring --raw-dir … --date-start 2011-01-01 --date-end 2011-01-07`  
   Confirm fragments, manifest, disk warnings, and that failures are localized.
2. **ACLED smoke:** same idea with `--date-start` / `--date-end`; confirm live JSON matches `normalize_acled_arab_spring_row` (country names → EG/TU/LY/SY, `source_available_at` end-of-day UTC).
3. **Full window** only after (1) and (2) look good.

## Forecast / labels

- `baselines/train_loop_skeleton.occurs_protest_in_forward_window` filters by **`_record_is_protest_forecast_target`** (GDELT `event_root_code == "14"`; ACLED / `acled_v3` protest-type strings such as `Protests`).
- `baselines/graph_builder_probe_labels.py` notes ACLED vs CAMEO escalation.

## Warehouse CLI note

- `import-jsonl` uses **`--input`**, not `--db`; set DB via **`--warehouse-path`** (and optional `--data-root`).
