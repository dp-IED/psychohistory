"""Wikipedia article fetch helpers with pilot PIT policy support."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from baselines.node_warehouse_build_v0 import ARAB_SPRING_COUNTRY_RANGE_START


UTC = dt.timezone.utc
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
PIT_ANCHOR_ISO = ARAB_SPRING_COUNTRY_RANGE_START.isoformat()


def _api_get(params: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    query = urlencode(params)
    req = Request(f"{WIKIPEDIA_API_URL}?{query}", headers={"User-Agent": "psychohistory-v2/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_get_with_retries(
    params: dict[str, Any],
    *,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 0.25,
) -> dict[str, Any]:
    """Fetch MediaWiki JSON with a small retry budget for transient transport failures."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _api_get(params, timeout=timeout)
        except (TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(backoff_seconds * attempt)
    assert last_exc is not None  # for type checkers; loop always sets this before breaking
    raise last_exc


def _extract_page(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query") or {}
    pages = query.get("pages") or []
    if not pages:
        raise ValueError("wikipedia query returned no pages")
    page = pages[0]
    if page.get("missing"):
        title = page.get("title", "unknown")
        raise ValueError(f"wikipedia page missing: {title}")
    return page


def _extract_revision(page: dict[str, Any]) -> dict[str, Any]:
    revisions = page.get("revisions") or []
    if not revisions:
        raise ValueError(f"no revisions returned for page: {page.get('title', 'unknown')}")
    return revisions[0]


def _revision_text(revision: dict[str, Any]) -> str:
    # MediaWiki may return either slots.main.content or slots.main['*'] depending on endpoint/version.
    slots = revision.get("slots") or {}
    main = slots.get("main") or {}
    for key in ("content", "*", "text"):
        if main.get(key):
            return str(main[key])
    if revision.get("*"):
        return str(revision["*"])
    return ""


def _as_of_timestamp(as_of_iso: str) -> str:
    as_of = dt.date.fromisoformat(as_of_iso)
    stamp = dt.datetime.combine(as_of, dt.time(0, 0, 0), tzinfo=UTC)
    return stamp.isoformat().replace("+00:00", "Z")


def fetch_article(
    *,
    title: str,
    url: str,
    pit_mode: str,
    as_of: str | None,
    timeout: int = 30,
) -> dict[str, Any]:
    params_base: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "titles": title,
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "rvlimit": "1",
    }

    pit_status = "static_latest"
    pit_warning: str | None = None

    if pit_mode == "arab_spring_overlap":
        if as_of != PIT_ANCHOR_ISO:
            raise ValueError(
                f"overlap article requires as_of={PIT_ANCHOR_ISO}; got {as_of!r} for title={title!r}"
            )
        params_older = dict(params_base)
        params_older["rvdir"] = "older"
        params_older["rvstart"] = _as_of_timestamp(as_of)
        payload = _api_get_with_retries(params_older, timeout=timeout)
        page = _extract_page(payload)
        revisions = page.get("revisions") or []
        if revisions:
            revision = revisions[0]
            pit_status = "pit_at_or_before_as_of"
        else:
            # Explicit PIT miss logging: page had no revision at or before the anchor.
            params_newer = dict(params_base)
            params_newer["rvdir"] = "newer"
            params_newer["rvstart"] = _as_of_timestamp(as_of)
            payload = _api_get_with_retries(params_newer, timeout=timeout)
            page = _extract_page(payload)
            revisions = page.get("revisions") or []
            if not revisions:
                raise ValueError(
                    f"no revisions returned for page: {page.get('title', 'unknown')} "
                    f"(both PIT <= {as_of} and fallback >= {as_of})"
                )
            revision = revisions[0]
            pit_status = "pit_fallback_after_as_of"
            pit_warning = (
                f"no revision existed at or before as_of={as_of}; "
                "used first available revision after as_of"
            )
    elif pit_mode != "static":
        raise ValueError(f"unknown pit_mode: {pit_mode}")
    else:
        payload = _api_get_with_retries(params_base, timeout=timeout)
        page = _extract_page(payload)
        revision = _extract_revision(page)
        pit_status = "static_latest"

    canonical_title = str(page.get("title") or title)
    canonical_url = str(url or f"https://en.wikipedia.org/wiki/{quote(canonical_title.replace(' ', '_'))}")
    text = _revision_text(revision)
    if not text.strip():
        raise ValueError(f"empty revision content for page: {canonical_title}")

    return {
        "title": canonical_title,
        "url": canonical_url,
        "text": text,
        "revision_id": str(revision.get("revid")),
        "revision_timestamp": str(revision.get("timestamp")),
        "fetched_at": dt.datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "pit_status": pit_status,
        "pit_warning": pit_warning,
    }


def fetch_articles(
    *,
    articles: list[dict[str, Any]],
    pit_map: dict[str, str],
    timeout: int = 30,
    sleep_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, article in enumerate(articles):
        article_id = str(article["article_id"])
        pit_mode = str(article["pit_mode"])
        as_of = pit_map.get(article_id)
        if pit_mode == "arab_spring_overlap" and as_of is None:
            raise ValueError(f"missing PIT mapping for overlap article_id={article_id}")
        if pit_mode == "static":
            as_of = None

        fetched = fetch_article(
            title=str(article["title"]),
            url=str(article["url"]),
            pit_mode=pit_mode,
            as_of=as_of,
            timeout=timeout,
        )
        row = {
            "article_id": article_id,
            "category": article["category"],
            "pit_mode": pit_mode,
            "as_of": as_of,
            **fetched,
        }
        out.append(row)
        if sleep_seconds > 0 and index < len(articles) - 1:
            time.sleep(sleep_seconds)
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--article-list", required=True, help="Path to wikipedia_pilot_50.json")
    fetch.add_argument("--pit-map", required=True, help="Path to wikipedia_pit_as_of_map.json")
    fetch.add_argument("--out", required=True, help="Path to write fetched article payloads")
    fetch.add_argument("--timeout", type=int, default=30)
    fetch.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        try:
            article_list = _load_json(Path(args.article_list))
            pit_map = _load_json(Path(args.pit_map))
            fetched = fetch_articles(
                articles=list(article_list),
                pit_map=dict(pit_map),
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
            )
            _write_json(Path(args.out), fetched)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
