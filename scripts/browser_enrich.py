#!/usr/bin/env python3
"""Browser enrichment agent — takes a URL + page content, cross-references the vault, enriches it.

Usage:
  # Pipe page text:
  cat page.txt | python3 browser_enrich.py --url "https://..." --title "Page Title"

  # Dry-run mode (propose only, no writes):
  python3 browser_enrich.py --url "..." --title "..." --body "..." --dry-run

What it does:
  1. Reads vault entity/concept/thread indices
  2. Matches page content against vault entities using aliases
  3. Spots new candidate entities (proper nouns not yet in vault)
  4. Writes observations to matching entities
  5. Creates new entity stubs for high-relevance candidates
  6. Links everything together
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# ── vault paths ──────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
VAULT = HERE.parent / "graph-vault"
ENTITIES_DIR = VAULT / "entities"
THREADS_DIR = VAULT / "threads"
CONCEPTS_DIR = VAULT / "concepts"
AGENT_ROLES_DIR = VAULT / "agent-roles"


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def load_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    entries = []
    for m in re.finditer(r"- \[\[([^\]]+)\]\]", text):
        link_text = m.group(1)
        if "|" in link_text:
            title, slug = link_text.split("|", 1)
        else:
            slug = link_text
            title = link_text
        entries.append({"slug": slug.strip(), "title": title.strip()})
    return entries


def build_entity_index(entities: list[dict]) -> list[dict]:
    """Build enriched entity index with computed aliases for matching."""
    enriched = []
    for e in entities:
        title = e["title"]
        slug = e["slug"]
        aliases = build_aliases(title, slug)
        enriched.append({**e, "aliases": aliases})
    enriched.sort(key=lambda x: -len(x["title"]))  # longer titles first for priority
    return enriched


def build_aliases(title: str, slug: str) -> list[str]:
    """Generate matching aliases for an entity.

    Examples:
      "federal-reserve-system" -> ["federal reserve system", "federal reserve", "the fed"]
      "donald-trump" -> ["donald trump", "president trump", "trump"]
      "jerome-powell" -> ["jerome powell", "chair powell", "powell"]
    """
    aliases = set()
    t_lower = title.lower()
    aliases.add(t_lower)

    # Base slug variant (hyphens -> spaces)
    base = slug.replace("-", " ")
    aliases.add(base)

    # Strip common prefixes for shorter aliases
    for prefix in ["the ", "u.s. ", "us ", "dr. "]:
        if t_lower.startswith(prefix):
            stripped = t_lower[len(prefix):]
            aliases.add(stripped)
            aliases.add(stripped.replace("-", " "))

    # Shorter partial from multi-word entity (skip if 2 words = already covered)
    # "federal reserve system" -> "federal reserve"
    parts = base.split()
    if len(parts) >= 3:
        for i in range(len(parts) - 1, 1, -1):
            shorter = " ".join(parts[:i])
            if len(shorter) >= 5:
                aliases.add(shorter)

    # Last-name only support for people (min 4 chars to avoid false positives)
    if len(parts) >= 2:
        last = parts[-1]
        stop_words = {"of", "the", "and", "for", "in", "at", "a", "an", "to", "on"}
        if len(last) >= 4 and last not in stop_words:
            aliases.add(last)

    # Title prefix variants for the full name
    title_prefixes = ["president ", "fed chair ", "chair ", "senator ", "governor ", "secretary "]
    for prefix in title_prefixes:
        prefixed = prefix + base
        if prefixed != base:
            aliases.add(prefixed)

    # Built-in nickname mappings for common entities
    nickname_map = {
        "federal-reserve-system": ["fed", "the fed", "the federal reserve"],
        "federal-open-market-committee": ["fomc", "the fomc"],
        "donald-trump": ["president donald trump", "president trump"],
        "united-states": ["us", "usa", "the us"],
        "central-bank-forward-guidance": ["fed forward guidance", "forward guidance"],
    }
    if slug in nickname_map:
        for nick in nickname_map[slug]:
            aliases.add(nick.lower())

    return sorted(aliases, key=len, reverse=True)


def find_matches(body: str, enriched_entities: list[dict]) -> list[dict]:
    """Find entities whose names or aliases appear in the body."""
    body_lower = body.lower()
    matched = []
    matched_slugs = set()

    for ent in enriched_entities:
        for alias in ent["aliases"]:
            if alias in body_lower:
                if ent["slug"] not in matched_slugs:
                    matched.append(ent)
                    matched_slugs.add(ent["slug"])
                break  # one alias match is enough

    return matched


def extract_candidate_entities(
    body: str, title: str, existing_slugs: set[str],
    enriched_entities: list[dict]
) -> list[dict]:
    """Extract proper noun candidates from page content that aren't in the vault yet.

    Excludes: entities already in vault (even if mentioned by a variant),
    and single-word capitalized things (not rich enough for a stub).
    """
    # Collect all alias phrases that already match existing entities
    body_lower = body.lower()
    covered_spans = set()
    for ent in enriched_entities:
        for alias in ent["aliases"]:
            idx = body_lower.find(alias)
            if idx != -1:
                covered_spans.add((idx, idx + len(alias)))

    def is_covered(start: int, end: int) -> bool:
        for c_start, c_end in covered_spans:
            if start >= c_start and end <= c_end:
                return True
        return False

    candidates = []
    seen_slugs = set(existing_slugs)

    # Find capitalized multi-word phrases (2-5 words)
    phrase_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")
    for m in phrase_pattern.finditer(body):
        phrase = m.group(1)
        start, end = m.start(), m.end()
        if is_covered(start, end):
            continue

        slug = slugify(phrase)
        if slug in seen_slugs or len(slug) < 5:
            continue
        seen_slugs.add(slug)
        candidates.append({"slug": slug, "title": phrase})

    # Also capture acronyms in parens: "Federal Open Market Committee (FOMC)"
    for m in re.finditer(r"\(([A-Z]{2,6})\)", body):
        acronym = m.group(1)
        slug = slugify(acronym)
        if slug in seen_slugs or len(slug) < 2:
            continue
        seen_slugs.add(slug)
        candidates.append({"slug": slug, "title": acronym})

    return candidates


# ── vault writes ─────────────────────────────────────────────────────────

def write_observation(entity_slug: str, entity_title: str, observation: str, url: str, page_title: str) -> str:
    path = ENTITIES_DIR / f"{entity_slug}.md"
    if not path.exists():
        return f"SKIP: entities/{entity_slug}.md does not exist"

    existing = path.read_text(encoding="utf-8")
    today = date.today().isoformat()

    obs_block = (
        f"\n\n## Observation: {page_title[:80]}\n"
        f"**Date:** {today}  \n"
        f"**Source:** [{page_title[:80]}]({url})  \n"
        f"**Note:** {observation}\n"
    )
    path.write_text(existing.strip() + "\n" + obs_block, encoding="utf-8")
    return f"OBSERVATION added to entities/{entity_slug}.md"


def create_entity_stub(title: str, slug: str, url: str, context: str) -> str:
    path = ENTITIES_DIR / f"{slug}.md"
    if path.exists():
        return f"SKIP: entities/{slug}.md already exists"

    today = date.today().isoformat()
    kind = guess_kind(title)

    content = (
        f"---\n"
        f"type: entity\n"
        f"kind: {kind}\n"
        f"title: \"{title}\"\n"
        f"slug: {slug}\n"
        f"pit_cutoff: {today}\n"
        f"---\n"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"*Auto-created stub from browser enrichment.*  \n"
        f"Source: [{title}]({url})  \n"
        f"{context[:200]}\n"
        f"\n"
        f"## Significance\n"
        f"\n"
        f"*Forecasting significance to be determined.*\n"
        f"\n"
        f"## Wikilinks\n"
        f"\n"
        f"- Source: [{url}]({url})\n"
    )
    path.write_text(content, encoding="utf-8")
    return f"STUB created: entities/{slug}.md"


def guess_kind(title: str) -> str:
    corporate = ["Co.", "Inc.", "Group", "Corp", "LLC", "Ltd", "Bank", "Fund",
                 "Board", "Commission", "Department", "University", "Institute",
                 "Association", "Organization", "Party", "Authority", "Agency",
                 "Council", "Committee", "Office", "Administration"]
    place_words = ["City", "State", "County", "Province", "River", "Mountain",
                   "Island", "Sea", "Ocean", "Valley", "Region", "Republic",
                   "Kingdom", "Union", "Nation", "Kingdom"]
    if " " in title and not any(ind in title for ind in corporate + place_words):
        return "person"
    if any(p in title for p in corporate):
        return "organization"
    if "(" in title or len(title.split()) > 5:
        return "event"
    return "organization"


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Browser enrichment agent")
    parser.add_argument("--url", required=True, help="Page URL")
    parser.add_argument("--title", default="", help="Page title")
    parser.add_argument("--body", default="", help="Page text content (pass - for stdin)")
    parser.add_argument("--dry-run", action="store_true", help="Propose only, don't write")
    parser.add_argument("--auto-create", action="store_true", help="Auto-create stubs without asking")
    args = parser.parse_args()

    body = args.body
    if body == "-" or not body:
        body = sys.stdin.read()

    title = args.title or args.url
    today = date.today().isoformat()

    # ── load vault ──────────────────────────────────────────────────
    raw_entities = load_index(ENTITIES_DIR / "_index.md")
    enriched_entities = build_entity_index(raw_entities)
    concepts = load_index(CONCEPTS_DIR / "_index.md")
    threads_data = load_index(THREADS_DIR / "_index.md")
    existing_slugs = {e["slug"] for e in raw_entities}

    print(f"── Browser Enrichment Report ──────────────────────────")
    print(f"Source: {title}")
    print(f"URL:    {args.url}")
    print(f"Date:   {today}")
    print(f"Vault:  {VAULT}")
    print(f"Body:   {len(body)} chars")
    print(f"Entities: {len(raw_entities)} in vault")
    print(f"──────────────────────────────────────────────────────")

    # ── step 1: match entities (with aliases) ─────────────────────
    matches = find_matches(body, enriched_entities)
    print(f"\n📌 Entities matched ({len(matches)}):")
    for m in matches[:25]:
        print(f"   [[{m['title']}]] ({m['slug']})")

    # ── step 2: match concepts ────────────────────────────────────
    concept_matches = find_matches(body, build_entity_index(concepts))
    print(f"\n🔗 Concepts matched ({len(concept_matches)}):")
    for m in concept_matches[:10]:
        print(f"   [[{m['title']}]]")

    # ── step 3: match threads ─────────────────────────────────────
    thread_matches = find_matches(body, build_entity_index(threads_data))
    print(f"\n🧵 Threads matched ({len(thread_matches)}):")
    for m in thread_matches[:10]:
        print(f"   [[{m['title']}]]")

    # ── step 4: candidate new entities ────────────────────────────
    candidates = extract_candidate_entities(body, title, existing_slugs, enriched_entities)
    print(f"\n🔍 Candidate new entities ({len(candidates)}):")
    for c in candidates[:25]:
        clean = clean_entity_title(c["title"])
        guess = guess_kind(clean)
        print(f"   {clean} → entities/{slugify(clean)}.md [{guess}]")

    # ── step 5: check thread gap ──────────────────────────────────
    if not thread_matches and len(body) > 1500:
        suggested_thread = slugify(title)
        print(f"\n⚠️  No thread overlap. Consider: threads/{suggested_thread}.md")

    # ── writes ────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n── DRY RUN — no files written ────────────────────────")
        return 0

    print(f"\n── Writing enrichments ───────────────────────────────")
    results = []

    for m in matches:
        # Get the aliases that matched for this entity for snippet extraction
        m_aliases = [a for a in m.get("aliases", []) if a in body.lower()]
        snippet = extract_snippet(body, m["title"], m_aliases)
        obs = f"Mentioned in page [{title[:60]}]({args.url}). {snippet[:300]}"
        r = write_observation(m["slug"], m["title"], obs, args.url, title)
        results.append(r)
        print(f"   {r}")

    if args.auto_create:
        for c in candidates[:10]:
            clean_title = clean_entity_title(c["title"])
            clean_slug = slugify(clean_title)
            context = f"Appears in [{title[:60]}]({args.url})."
            r = create_entity_stub(clean_title, clean_slug, args.url, context)
            results.append(r)
            print(f"   {r}")

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"   Observations added: {len(matches)}")
    print(f"   Entity stubs created: {sum(1 for r in results if 'STUB' in r)}")
    print(f"   Source: {args.url}")
    print(f"──────────────────────────────────────────────────────")

    return 0


def clean_entity_title(title: str) -> str:
    """Strip role prefixes from entity titles for cleaner stubs.
    "Fed Chair Jerome Powell" -> "Jerome Powell"
    "President Donald Trump" -> "Donald Trump"
    """
    prefixes = [
        "Fed Chair ", "President ", "Senator ", "Governor ", "Secretary ",
        "Chair ", "Rep. ", "Sen. ", "Gov. ", "Sec. ",
        "Vice President ", "Ambassador ", "Attorney General ",
        "Speaker ", "Majority Leader ", "Minority Leader ",
        "Chief ", "Director ", "Commissioner ", "Judge ",
    ]
    result = title
    for p in prefixes:
        if result.startswith(p):
            result = result[len(p):]
            break
    return result


def extract_snippet(body: str, keyword: str, matched_aliases: list[str] | None = None, max_len: int = 200) -> str:
    """Extract a relevant sentence or phrase around any matching alias."""
    # First try the aliases that actually matched
    search_terms = [keyword] + (matched_aliases or [])
    best_snippet = ""

    for term in search_terms:
        idx = body.lower().find(term.lower())
        if idx == -1:
            continue
        start = body.rfind(". ", max(0, idx - 150), idx)
        if start == -1:
            start = max(0, idx - 100)
        else:
            start += 2
        end = body.find(". ", idx, idx + 200)
        if end == -1:
            end = min(len(body), idx + 200)
        else:
            end += 1
        snippet = body[start:end].strip()
        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."
        if len(snippet) > len(best_snippet):
            best_snippet = snippet

    return best_snippet


if __name__ == "__main__":
    raise SystemExit(main())
