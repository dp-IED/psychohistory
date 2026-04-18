# ACLED Ingestion Operations

The preferred ACLED workflow fetches France protest rows and normalizes them
directly into the central DuckDB warehouse. Raw API pages are ephemeral by
default; manifests and metadata remain for audit.

Credentials must live outside the repo:

```bash
source ~/.config/psychohistory/acled.env
test -n "$ACLED_USERNAME"
test -n "$ACLED_EMAIL"
test -n "$ACLED_PASSWORD"
```

Do not print the password, bearer token, request bodies, or raw API failure
payloads.

## Warehouse Fetch

```bash
python -m ingest.event_warehouse init

python -m ingest.acled_raw fetch-france-protests \
  --event-start 2019-01-01 \
  --event-end 2026-01-04 \
  --limit 5000 \
  --max-pages 100 \
  --raw-retention none \
  --normalize-to-warehouse \
  --availability-policy event_date_lag \
  --availability-lag-days 7
```

For a small smoke:

```bash
python -m ingest.acled_raw fetch-france-protests \
  --event-start 2023-03-20 \
  --event-end 2023-04-03 \
  --limit 5000 \
  --max-pages 5 \
  --raw-retention none \
  --normalize-to-warehouse \
  --availability-policy event_date_lag \
  --availability-lag-days 7
```

Inspect the result:

```bash
python -m ingest.event_warehouse audit
```

## Raw Retention

`--raw-retention none` is the default. It writes:

- `fetch_metadata.json`
- `fetch_manifest.jsonl`
- normalized warehouse rows when `--normalize-to-warehouse` is set

Use `--raw-retention compressed` to retain `fragments/page_*.jsonl.gz`, or
`--raw-retention full` for plain JSONL fragments. These modes are for debugging
or reproducibility; do not keep large raw folders in repo worktrees.

## JSONL normalization (import path)

JSONL exports remain available for compatibility:

```bash
python -m ingest.acled_tape normalize-france-protests \
  --raw data/acled/raw/france_protest \
  --out data/acled/tape/france_protest/events.jsonl \
  --availability-policy event_date_lag \
  --availability-lag-days 7
```

Treat those files as import/export artifacts. Prefer importing them into the
warehouse:

```bash
python -m ingest.event_warehouse import-jsonl \
  --data-root "${PSYCHOHISTORY_DATA_ROOT:-/Users/darenpalmer/conductor/shared-data/psychohistory-v2}" \
  --input data/acled/tape/france_protest/events.jsonl
```

## Availability Policy

The ACLED `timestamp` column is an API upload/update timestamp, not guaranteed
to be the first date a historical event became knowable. For retrospective
forecast benchmarks, the default is:

```text
--availability-policy event_date_lag --availability-lag-days 7
```

Available policies:

- `event_date_lag`: benchmark approximation; recommended unless historical
  ACLED snapshots are available.
- `timestamp`: use ACLED's upload/update timestamp; conservative but can remove
  many historical rows from early forecast origins.
- `retrieved_at`: mark all rows available at fetch time; useful for static data
  exploration, but not for historical point-in-time backtests.

## HTTP 403

If the fetcher returns:

```text
error: ACLED request failed: status=403 endpoint=/api/acled/read
```

the credentials were sufficient to request OAuth but the account was not allowed
to read the ACLED endpoint. Confirm the account has API access to ACLED event
data in My ACLED or with ACLED support. The fetcher intentionally reports only
the status code and endpoint path.
