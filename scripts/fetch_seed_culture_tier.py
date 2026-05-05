#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

from io import StringIO

import pandas as pd

UTC = dt.timezone.utc

BILLBOARD_PAGES = [
    "https://en.wikipedia.org/wiki/List_of_Billboard_Hot_100_number_ones_of_2024",
    "https://en.wikipedia.org/wiki/List_of_Billboard_Hot_100_number_ones_of_2025",
]

OSCAR_PAGES = [
    "https://en.wikipedia.org/wiki/96th_Academy_Awards",
    "https://en.wikipedia.org/wiki/97th_Academy_Awards",
]

GRAMMY_PAGES = [
    "https://en.wikipedia.org/wiki/66th_Annual_Grammy_Awards",
]

BOX_OFFICE_WEEKLY_PAGE = "https://en.wikipedia.org/wiki/List_of_2024_box_office_number-one_films_in_the_United_States"
BOX_OFFICE_FILMS_PAGE = "https://en.wikipedia.org/wiki/List_of_American_films_of_2024"


def fmt_z(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date_guess(text: str, default_year: int | None = None) -> dt.datetime | None:
    text = (text or "").strip()
    if default_year and re.match(r"^[A-Za-z]+\s+\d{1,2}$", text):
        text = f"{text}, {default_year}"
    for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"]:
        try:
            d = dt.datetime.strptime(text, fmt).replace(tzinfo=UTC)
            return d
        except Exception:
            pass
    return None


def clean_title(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\[[^\]]+\]", "", t)
    t = re.sub(r"[†‡*]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fetch_url_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (psychohistory-priority4-culture)"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def ingest_billboard() -> list[dict]:
    out = []
    for url in BILLBOARD_PAGES:
        html = fetch_url_text(url)
        tables = pd.read_html(StringIO(html))
        ym = re.search(r"_(20\d\d)$", url)
        default_year = int(ym.group(1)) if ym else None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            # Expected columns typically include Date/Issue date + Song + Artist
            date_col = None
            for c in t.columns:
                cl = str(c).strip().lower()
                if "date" in cl:
                    date_col = c
                    break
            song_col = None
            for c in t.columns:
                cl = str(c).strip().lower()
                if "song" in cl or "single" in cl:
                    song_col = c
                    break
            artist_col = None
            for c in t.columns:
                cl = str(c).strip().lower()
                if "artist" in cl:
                    artist_col = c
                    break
            if not date_col or not song_col:
                continue

            for _, r in t.iterrows():
                d = parse_date_guess(str(r.get(date_col, "")), default_year=default_year)
                if not d:
                    continue
                pub = d.replace(hour=12, minute=0)
                # For weekly chart state, approximate interval as prior 7-day window ending on chart date
                state_end = d.replace(hour=0, minute=0)
                state_start = (state_end - dt.timedelta(days=6))
                out.append(
                    {
                        "source_name": "wikipedia_billboard_hot100",
                        "entity_type": "chart",
                        "chart_name": "Billboard Hot 100",
                        "song": str(r.get(song_col, "")).strip(),
                        "artist": str(r.get(artist_col, "")).strip() if artist_col else None,
                        "state_interval_start": fmt_z(state_start),
                        "state_interval_end": fmt_z(state_end),
                        "publication_time": fmt_z(pub),
                        "event_time": fmt_z(state_end),
                        "source_url": url,
                        "pit_policy": "chart_state_week_interval + publication_time_gate",
                        "provenance_notes": "Chart week state is separate from publication metadata.",
                    }
                )
    return out


def ingest_oscars() -> list[dict]:
    out = []
    nom_re = re.compile(r"Nominations? (?:were )?announced on ([A-Za-z]+ \d{1,2}, \d{4})", re.IGNORECASE)
    date_re = re.compile(r"<th[^>]*>Date</th>\s*<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)

    category_patterns = [
        "Best Picture",
        "Best Directing",
        "Best Actor in a Leading Role",
        "Best Actress in a Leading Role",
        "Best Actor in a Supporting Role",
        "Best Actress in a Supporting Role",
        r"Best Writing \(Original Screenplay\)",
        r"Best Writing \(Adapted Screenplay\)",
        "Best International Feature Film",
        "Best Animated Feature Film",
        "Best Documentary Feature Film",
        "Best Original Score",
        "Best Original Song",
    ]
    category_re = re.compile(r"(" + "|".join(category_patterns) + r")")

    def parse_nominee_block(cat: str, block: str, award_slug: str, source_url: str, pub_ts: dt.datetime | None) -> list[dict]:
        rows = []
        txt = re.sub(r"\s+", " ", block).strip()
        if not txt:
            return rows

        winner_text = ""
        others_text = ""
        if " ‡ " in txt:
            winner_text, others_text = txt.split(" ‡ ", 1)
            winner_text = winner_text.strip()
            others_text = others_text.strip()
        else:
            winner_text = txt

        def split_candidates(text: str, picture_mode: bool) -> list[str]:
            if not text:
                return []
            pat = r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\".\- ]+?)\s–\s" if not picture_mode else r"([A-Z0-9][A-Za-z0-9À-ÖØ-öø-ÿ:'’\".,\- ]+?)\s–\s"
            matches = list(re.finditer(pat, text))
            if not matches:
                return [text.strip()]
            cands = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                cands.append(text[start:end].strip())
            return [c for c in cands if c]

        nominee_texts = [winner_text] + split_candidates(others_text, picture_mode=(cat == "Best Picture"))

        for candidate in nominee_texts:
            person = None
            film = None

            if " – " in candidate:
                left, right = candidate.split(" – ", 1)
                left = left.strip()
                right = right.strip()
                if cat == "Best Picture":
                    film = left
                else:
                    person = left
                    r0 = right.split(" as ", 1)[0].split(";", 1)[0].strip()
                    film = r0 if r0 else None
            else:
                film = candidate.strip()

            rows.append(
                {
                    "source_name": "wikipedia_academy_awards",
                    "entity_type": "award_nominee",
                    "award": award_slug,
                    "category": cat,
                    "nominee_text": candidate,
                    "person_name": person,
                    "film_title": film,
                    "winner_flag": candidate == winner_text,
                    "publication_time": fmt_z(pub_ts) if pub_ts else None,
                    "event_time": fmt_z(pub_ts) if pub_ts else None,
                    "source_url": source_url,
                    "pit_policy": "nomination_announcement_time_is_observable",
                    "provenance_notes": "Parsed from Oscars nominee table; winner indicated by dagger marker.",
                }
            )
        return rows

    for url in OSCAR_PAGES:
        html = fetch_url_text(url)
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean)
        award_slug = url.rsplit("/", 1)[-1].replace("_", " ")

        nomination_ts = None
        ceremony_ts = None

        # Nomination announcement as PIT-observable milestone
        m = nom_re.search(clean)
        if m:
            d = parse_date_guess(m.group(1))
            if d:
                nomination_ts = d.replace(hour=13, minute=0)
                out.append(
                    {
                        "source_name": "wikipedia_academy_awards",
                        "entity_type": "award_nomination_milestone",
                        "award": award_slug,
                        "state_interval_start": fmt_z(d.replace(hour=0, minute=0)),
                        "state_interval_end": fmt_z(d.replace(hour=23, minute=59)),
                        "publication_time": fmt_z(nomination_ts),
                        "event_time": fmt_z(nomination_ts),
                        "source_url": url,
                        "pit_policy": "nomination_announcement_time_is_observable",
                        "provenance_notes": "Nomination announcement time used for nominee-state observability.",
                    }
                )

        # Ceremony date as resolution-side milestone
        m2 = date_re.search(html)
        if m2:
            date_text = re.sub(r"<[^>]+>", " ", m2.group(1))
            date_text = re.sub(r"\s+", " ", date_text).strip()
            d2 = parse_date_guess(date_text)
            if d2:
                ceremony_ts = d2.replace(hour=13, minute=0)
                out.append(
                    {
                        "source_name": "wikipedia_academy_awards",
                        "entity_type": "award_ceremony_milestone",
                        "award": award_slug,
                        "state_interval_start": fmt_z(d2.replace(hour=0, minute=0)),
                        "state_interval_end": fmt_z(d2.replace(hour=23, minute=59)),
                        "publication_time": fmt_z(ceremony_ts),
                        "event_time": fmt_z(ceremony_ts),
                        "source_url": url,
                        "pit_policy": "ceremony_time_is_resolution_event",
                        "provenance_notes": "Ceremony event is not observable before ceremony publication timestamp.",
                    }
                )

        # Parse nominee/winner blocks from the known packed two-column table
        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            tables = []

        if len(tables) > 1:
            nominee_table = tables[1]
            nominee_pub_ts = nomination_ts or (ceremony_ts - dt.timedelta(days=45) if ceremony_ts else None)
            for col in nominee_table.columns[:2]:
                blob = " ".join(str(x) for x in nominee_table[col].dropna().tolist())
                blob = re.sub(r"\s+", " ", blob)
                matches = list(category_re.finditer(blob))
                for i, mcat in enumerate(matches):
                    cat = mcat.group(1)
                    start = mcat.end()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
                    block = blob[start:end].strip()
                    out.extend(parse_nominee_block(cat, block, award_slug, url, nominee_pub_ts))

        # Parse film nomination counts (if present)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if len(cols) >= 2 and "nominations" in cols[0] and "film" in cols[1]:
                for _, r in t.iterrows():
                    film = str(r.get(t.columns[1], "")).strip()
                    n = r.get(t.columns[0])
                    if not film or str(film).lower() == "nan":
                        continue
                    try:
                        n_int = int(n)
                    except Exception:
                        continue
                    out.append(
                        {
                            "source_name": "wikipedia_academy_awards",
                            "entity_type": "film_nomination_count",
                            "award": award_slug,
                            "film_title": film,
                            "nomination_count": n_int,
                            "publication_time": fmt_z(nomination_ts) if nomination_ts else None,
                            "event_time": fmt_z(nomination_ts) if nomination_ts else None,
                            "source_url": url,
                            "pit_policy": "nomination_announcement_time_is_observable",
                            "provenance_notes": "Parsed from films-with-multiple-nominations table.",
                        }
                    )

    return out


def ingest_grammys() -> list[dict]:
    out = []
    category_patterns = [
        "Album of the Year",
        "Song of the Year",
        "Record of the Year",
        "Best Pop Solo Performance",
        "Best Pop Duo/Group Performance",
        "Best Rock Performance",
        "Best Metal Performance",
        "Best R&B Performance",
    ]
    category_re = re.compile(r"(" + "|".join(category_patterns) + r")")

    def parse_nominee_block(cat: str, block: str, award_slug: str, source_url: str, pub_ts: dt.datetime | None) -> list[dict]:
        rows = []
        txt = re.sub(r"\s+", " ", block).strip().rstrip(".")
        if not txt:
            return rows

        def split_candidates(text: str, cat_name: str) -> list[str]:
            if not text:
                return []
            if cat_name in {
                "Song of the Year",
                "Record of the Year",
                "Best Pop Solo Performance",
                "Best Pop Duo/Group Performance",
                "Best Rock Performance",
                "Best Metal Performance",
                "Best R&B Performance",
            }:
                q = list(re.finditer(r'"[^"]+"', text))
                if q:
                    parts = []
                    for i, m in enumerate(q):
                        start = m.start()
                        end = q[i + 1].start() if i + 1 < len(q) else len(text)
                        parts.append(text[start:end].strip())
                    return [p for p in parts if p]

            anchor = list(re.finditer(r"(?=[A-Z][A-Za-z0-9'&().,:/\- ]+\s–\s)", text))
            if not anchor:
                return [text.strip()]
            out_cands = []
            for i, m in enumerate(anchor):
                start = m.start()
                end = anchor[i + 1].start() if i + 1 < len(anchor) else len(text)
                c = text[start:end].strip()
                if c:
                    out_cands.append(c)
            return out_cands

        if " ‡ " in txt:
            winner_text, others_text = txt.split(" ‡ ", 1)
            winner_text = winner_text.strip()
            nominee_texts = [winner_text] + split_candidates(others_text.strip(), cat)
        else:
            nominee_texts = split_candidates(txt, cat)
            winner_text = nominee_texts[0].strip() if nominee_texts else txt

        seen = set()
        deduped = []
        for x in nominee_texts:
            k = re.sub(r"\s+", " ", x).strip()
            if k and k not in seen:
                seen.add(k)
                deduped.append(k)

        for nominee in deduped:
            work_title = None
            artist = None
            if " – " in nominee:
                left, right = nominee.split(" – ", 1)
                work_title = left.strip().strip('"')
                artist = right.split(";", 1)[0].strip()
            else:
                work_title = nominee.strip().strip('"')

            rows.append(
                {
                    "source_name": "wikipedia_grammy_awards",
                    "entity_type": "award_nominee",
                    "award": award_slug,
                    "category": cat,
                    "nominee_text": nominee,
                    "work_title": work_title,
                    "artist_name": artist,
                    "winner_flag": nominee == winner_text,
                    "publication_time": fmt_z(pub_ts) if pub_ts else None,
                    "event_time": fmt_z(pub_ts) if pub_ts else None,
                    "source_url": source_url,
                    "pit_policy": "grammy_nomination_state_is_observable_pre_ceremony",
                    "provenance_notes": "Parsed from Grammy nominee table; winner indicated by dagger marker or first-listed convention.",
                }
            )
        return rows

    for url in GRAMMY_PAGES:
        html = fetch_url_text(url)
        award_slug = url.rsplit("/", 1)[-1].replace("_", " ")
        ceremony_ts = dt.datetime(2024, 2, 4, 13, 0, tzinfo=UTC)

        out.append(
            {
                "source_name": "wikipedia_grammy_awards",
                "entity_type": "award_ceremony_milestone",
                "award": award_slug,
                "state_interval_start": fmt_z(dt.datetime(2024, 2, 4, 0, 0, tzinfo=UTC)),
                "state_interval_end": fmt_z(dt.datetime(2024, 2, 4, 23, 59, tzinfo=UTC)),
                "publication_time": fmt_z(ceremony_ts),
                "event_time": fmt_z(ceremony_ts),
                "source_url": url,
                "pit_policy": "ceremony_time_is_resolution_event",
                "provenance_notes": "Ceremony event for 66th Annual Grammy Awards.",
            }
        )

        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            tables = []

        if len(tables) > 4:
            nominee_table = tables[4]
            for col in nominee_table.columns[:2]:
                blob = " ".join(str(x) for x in nominee_table[col].dropna().tolist())
                blob = re.sub(r"\s+", " ", blob)
                matches = list(category_re.finditer(blob))
                for i, mcat in enumerate(matches):
                    cat = mcat.group(1)
                    start = mcat.end()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
                    block = blob[start:end].strip()
                    out.extend(parse_nominee_block(cat, block, award_slug, url, ceremony_ts - dt.timedelta(days=90)))

    return out


def ingest_box_office_2024_us() -> list[dict]:
    out = []

    # Build release-date lookup from quarterly film tables
    release_by_title: dict[str, dt.datetime] = {}
    films_html = fetch_url_text(BOX_OFFICE_FILMS_PAGE)
    try:
        film_tables = pd.read_html(StringIO(films_html))
    except Exception:
        film_tables = []

    for t in film_tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "title" not in cols or "opening" not in cols:
            continue
        title_col = t.columns[cols.index("title")]
        opening_col = t.columns[cols.index("opening")]
        for _, r in t.iterrows():
            title = clean_title(str(r.get(title_col, "")))
            opening_raw = str(r.get(opening_col, "")).strip()
            if not title or title.lower() == "nan":
                continue
            d = parse_date_guess(opening_raw, default_year=2024)
            if d:
                release_by_title[title] = d.replace(hour=12, minute=0)

    weekly_html = fetch_url_text(BOX_OFFICE_WEEKLY_PAGE)
    try:
        tables = pd.read_html(StringIO(weekly_html))
    except Exception:
        tables = []

    weekly_table = None
    yearly_table = None
    for t in tables:
        col_l = [str(c).strip().lower() for c in t.columns]
        norm = [c.replace(".", "") for c in col_l]
        if {"#", "weekend end date", "film", "gross"}.issubset(set(norm)) and weekly_table is None:
            weekly_table = t

        has_rank = any("rank" == c or c.endswith("rank") for c in norm)
        has_title = any("title" == c or c.endswith("title") for c in norm)
        has_domestic = any("domestic gross" in c for c in norm)
        if has_rank and has_title and has_domestic and len(t) <= 25:
            yearly_table = t

    if weekly_table is not None:
        cols = {str(c).strip().lower(): c for c in weekly_table.columns}
        date_col = cols.get("weekend end date")
        film_col = cols.get("film")
        gross_col = cols.get("gross")
        rank_col = cols.get("#")
        for _, r in weekly_table.iterrows():
            film = clean_title(str(r.get(film_col, "")).strip())
            if not film or film.lower() == "nan":
                continue
            end_d = parse_date_guess(str(r.get(date_col, "")), default_year=2024)
            if not end_d:
                continue
            period_end = end_d.replace(hour=0, minute=0)
            period_start = period_end - dt.timedelta(days=6)
            pub_ts = period_end.replace(hour=12, minute=0)

            out.append(
                {
                    "source_name": "wikipedia_box_office_us_2024",
                    "entity_type": "box_office_weekly_number_one",
                    "market": "US_domestic",
                    "year": 2024,
                    "week_number": int(r.get(rank_col)) if str(r.get(rank_col, "")).strip().isdigit() else None,
                    "film_title": film,
                    "weekend_gross": str(r.get(gross_col, "")).strip(),
                    "release_date": fmt_z(release_by_title[film]) if film in release_by_title else None,
                    "state_interval_start": fmt_z(period_start),
                    "state_interval_end": fmt_z(period_end),
                    "publication_time": fmt_z(pub_ts),
                    "event_time": fmt_z(period_end),
                    "source_url": BOX_OFFICE_WEEKLY_PAGE,
                    "pit_policy": "weekly_gross_state_interval + publication_time_gate",
                    "provenance_notes": "Weekend #1 reflects week-level observed gross state, observable after publication timestamp.",
                }
            )

    if yearly_table is not None:
        cols = {str(c).strip().lower(): c for c in yearly_table.columns}
        rank_col = next((orig for k, orig in cols.items() if "rank" in k), None)
        title_col = next((orig for k, orig in cols.items() if "title" in k), None)
        gross_col = next((orig for k, orig in cols.items() if "domestic gross" in k), None)
        pub_ts = dt.datetime(2025, 1, 2, 12, 0, tzinfo=UTC)
        for _, r in yearly_table.iterrows():
            title = clean_title(str(r.get(title_col, "")).strip())
            if not title or title.lower() == "nan":
                continue
            rank_val = str(r.get(rank_col, "")).strip()
            m = re.search(r"\d+", rank_val)
            rank_int = int(m.group()) if m else None
            out.append(
                {
                    "source_name": "wikipedia_box_office_us_2024",
                    "entity_type": "box_office_year_rank",
                    "market": "US_domestic",
                    "year": 2024,
                    "rank": rank_int,
                    "film_title": title,
                    "domestic_gross": str(r.get(gross_col, "")).strip(),
                    "release_date": fmt_z(release_by_title[title]) if title in release_by_title else None,
                    "state_interval_start": fmt_z(dt.datetime(2024, 1, 1, 0, 0, tzinfo=UTC)),
                    "state_interval_end": fmt_z(dt.datetime(2024, 12, 31, 23, 59, tzinfo=UTC)),
                    "publication_time": fmt_z(pub_ts),
                    "event_time": fmt_z(pub_ts),
                    "source_url": BOX_OFFICE_WEEKLY_PAGE,
                    "pit_policy": "yearly_rank_publication_is_post_year_aggregate",
                    "provenance_notes": "Year aggregate should only be visible after publication time; not usable at early-year cutoffs.",
                }
            )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/culture/raw/seed_culture_tier")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "seed_culture_events.jsonl"
    manifest = out_dir / "fetch_manifest.json"

    rows = []
    rows.extend(ingest_billboard())
    rows.extend(ingest_oscars())
    rows.extend(ingest_grammys())
    rows.extend(ingest_box_office_2024_us())
    rows.sort(key=lambda r: (r.get("publication_time") or "", r.get("source_name") or ""))

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    man = {
        "ok": True,
        "created_at": fmt_z(dt.datetime.now(tz=UTC)),
        "records_total": len(rows),
        "sources": {
            "wikipedia_billboard_hot100": sum(1 for r in rows if r["source_name"] == "wikipedia_billboard_hot100"),
            "wikipedia_academy_awards": sum(1 for r in rows if r["source_name"] == "wikipedia_academy_awards"),
            "wikipedia_grammy_awards": sum(1 for r in rows if r["source_name"] == "wikipedia_grammy_awards"),
            "wikipedia_box_office_us_2024": sum(1 for r in rows if r["source_name"] == "wikipedia_box_office_us_2024"),
        },
        "outputs": {
            "jsonl": str(out_jsonl.resolve()),
            "manifest": str(manifest.resolve()),
        },
        "pit_policy_ref": ".context/priority4_culture_pit_policy_2026-05-03.md",
        "notes": [
            "Priority 4 start checkpoint: Billboard + Oscars milestone ingestion.",
            "Box-office connector implemented: 2024 US weekly #1 timeline + year-end domestic rank context.",
        ],
    }
    manifest.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
