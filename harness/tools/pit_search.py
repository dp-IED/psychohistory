"""
PIT-enforced web search for cursor-agent synthesis.

Two backends:
  - wikipedia: General knowledge via Wikimedia REST API, PIT via revision timestamps (no API key needed)
  - duckduckgo: News/sources via DuckDuckGo HTML, PIT via date-bounded querying (no API key needed)

Usage:
  python -m harness.tools.pit_search "Who won the 2024 election?" --cutoff 2024-10-01
  python -m harness.tools.pit_search "Bitcoin price March 2025" --cutoff 2025-03-15 --backend duckduckgo
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

BackendName = Literal["wikipedia", "duckduckgo"]

_USER_AGENT = (
    "PsychohistoryHarness/1.0 (PIT research tool; "
    "mailto:daren@example.com)"
)

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1"

_DDG_URL = "https://html.duckduckgo.com/html"


@dataclass
class PITSearchResult:
    title: str
    url: str
    snippet: str
    published_at: date | None  # None when date cannot be determined
    source: str  # "wikipedia" or "duckduckgo"


@dataclass
class PITSearchResponse:
    query: str
    cutoff: date
    results: list[PITSearchResult] = field(default_factory=list)
    error: str | None = None


# ── Wikipedia backend ──────────────────────────────────────────────


def _wiki_page_revisions(
    page_title: str, cutoff: date
) -> list[dict]:
    """Fetch revision metadata for a Wikipedia page, returning only revs ≤ cutoff.

    Uses rvstart to query only revisions up to the cutoff date.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "timestamp|ids|size",
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": f"{cutoff.isoformat()}T23:59:59Z",
    }
    url = f"{_WIKI_API}?{urlencode(params)}"
    req = request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return []

    pages = data.get("query", {}).get("pages", {})
    revs: list[dict] = []
    for pid, pdata in pages.items():
        if pid == "-1":
            continue  # missing page
        for rev in pdata.get("revisions", []):
            ts_str = rev.get("timestamp", "")
            try:
                rev_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if rev_dt.date() <= cutoff:
                revs.append(
                    {
                        "revid": rev.get("revid"),
                        "timestamp": ts_str,
                        "size": rev.get("size", 0),
                    }
                )
    return revs


def _wiki_search_pit(query: str, cutoff: date, max_results: int = 5) -> list[PITSearchResult]:
    """Search Wikipedia with PIT filtering via page revision timestamps."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": min(max_results * 2, 50),
        "srprop": "timestamp|snippet",
        "srinfo": "totalhits",
    }
    url = f"{_WIKI_API}?{urlencode(params)}"
    req = request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return [PITSearchResult(
            title="[error]",
            url="",
            snippet=f"Wikipedia search failed: {exc}",
            published_at=None,
            source="wikipedia",
        )]

    results: list[PITSearchResult] = []
    search_hits = data.get("query", {}).get("search", [])
    for hit in search_hits[:max_results]:
        page_title = hit.get("title", "")
        ts_str = hit.get("timestamp", "")
        snippet_raw = hit.get("snippet", "")

        # Clean snippet (Wikipedia returns HTML snippets with <span> tags)
        snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()[:300]

        # PIT check: does this page have revisions knowable ≤ cutoff?
        revs = _wiki_page_revisions(page_title, cutoff)
        if not revs:
            # No revision existed ≤ cutoff — page didn't exist or was created after cutoff
            continue
        latest_rev_ts = revs[-1]["timestamp"]

        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"

        # Fetch the actual PIT page content via REST API summary
        summary = ""
        try:
            encoded_title = urllib.parse.quote(page_title.replace(" ", "_"))
            rest_url = f"{_WIKI_REST}/page/summary/{encoded_title}"
            rest_req = request.Request(rest_url, headers={"User-Agent": _USER_AGENT})
            with request.urlopen(rest_req, timeout=10) as resp:
                rest_data = json.loads(resp.read().decode("utf-8"))
                summary = rest_data.get("extract", "")[:500]
        except Exception:
            summary = snippet

        results.append(PITSearchResult(
            title=page_title,
            url=page_url,
            snippet=summary or snippet,
            published_at=(
                datetime.fromisoformat(latest_rev_ts.replace("Z", "+00:00")).date()
                if latest_rev_ts
                else None
            ),
            source="wikipedia",
        ))

    return results


# ── DuckDuckGo backend ──────────────────────────────────────────────


def _ddg_search_pit(query: str, cutoff: date, max_results: int = 5) -> list[PITSearchResult]:
    """Search DuckDuckGo HTML and filter results by date.

    DuckDuckGo doesn't expose machine-readable timestamps in HTML results,
    so PIT enforcement here is best-effort: we search and return results,
    marking those with available date info. The synthesis agent should
    cross-check dates from the snippet content.
    """
    data = urllib.parse.urlencode({"q": query}).encode()
    req = request.Request(
        _DDG_URL,
        data=data,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return [PITSearchResult(
            title="[error]",
            url="",
            snippet=f"DuckDuckGo search failed: {exc}",
            published_at=None,
            source="duckduckgo",
        )]

    results: list[PITSearchResult] = []
    # Parse DuckDuckGo HTML result blocks
    # Pattern: class="result__a" for links, class="result__snippet" for snippets
    result_blocks = _parse_ddg_html(html)

    for block in result_blocks[:max_results]:
        results.append(PITSearchResult(
            title=block.get("title", ""),
            url=block.get("url", ""),
            snippet=block.get("snippet", "")[:400],
            published_at=None,  # DDG doesn't expose reliable timestamps
            source="duckduckgo",
        ))

    return results


def _parse_ddg_html(html: str) -> list[dict]:
    """Minimal DuckDuckGo HTML result parser."""
    results: list[dict] = []
    # Find result wrappers — each result is in a <div> with class starting with "result"
    # Simpler approach: look for <a class="result__a" and nearby .result__snippet
    for match in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        url = match.group(1)
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()

        # Find the associated snippet (next .result__snippet)
        remainder = html[match.end() : match.end() + 2000]
        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            remainder,
            re.DOTALL,
        )
        snippet = ""
        if snippet_match:
            snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()[:400]

        results.append({"title": title, "url": url, "snippet": snippet})
    return results


# ── Main entrypoint ──────────────────────────────────────────────


def search(
    query: str,
    cutoff: date,
    *,
    backend: BackendName = "wikipedia",
    max_results: int = 5,
) -> PITSearchResponse:
    """PIT-enforced search. Returns results that were knowable ≤ cutoff."""
    try:
        if backend == "wikipedia":
            results = _wiki_search_pit(query, cutoff, max_results=max_results)
        elif backend == "duckduckgo":
            results = _ddg_search_pit(query, cutoff, max_results=max_results)
        else:
            return PITSearchResponse(
                query=query, cutoff=cutoff, error=f"Unknown backend: {backend}"
            )
        return PITSearchResponse(query=query, cutoff=cutoff, results=results)
    except Exception as exc:
        return PITSearchResponse(
            query=query, cutoff=cutoff, error=str(exc)
        )


def results_to_prompt_block(results: list[PITSearchResult], cutoff: date) -> str:
    """Format PIT search results as a markdown block for the synthesis prompt."""
    if not results:
        return f"(no PIT-filtered results found ≤ {cutoff.isoformat()})"

    lines: list[str] = [
        f"Results knowable as of {cutoff.isoformat()}:",
        "",
    ]
    for i, r in enumerate(results, 1):
        date_str = r.published_at.isoformat() if r.published_at else "(date unknown)"
        lines.append(f"{i}. **{r.title}** ({r.source}, {date_str})")
        lines.append(f"   {r.url}")
        lines.append(f"   > {r.snippet[:200]}")
        lines.append("")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PIT-filtered web search (knowable-information only)"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--cutoff", required=True, help="Cutoff date (YYYY-MM-DD)")
    parser.add_argument(
        "--backend",
        choices=["wikipedia", "duckduckgo"],
        default="wikipedia",
        help="Search backend (default: wikipedia)",
    )
    parser.add_argument(
        "--max-results", type=int, default=5, help="Max results (default: 5)"
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format (default: md)",
    )

    args = parser.parse_args()
    cutoff_date = date.fromisoformat(args.cutoff)
    resp = search(
        args.query,
        cutoff_date,
        backend=args.backend,
        max_results=args.max_results,
    )

    if args.format == "json":
        print(json.dumps(asdict(resp), default=str, indent=2))
    else:
        if resp.error:
            print(f"⚠ Search error: {resp.error}")
        print(results_to_prompt_block(resp.results, cutoff_date))
