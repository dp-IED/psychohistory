"""
Build MiniBench validation corpus from Metaculus.
Fetches all 23 MiniBench questions, assigns YES/NO resolution
from known outcomes, and writes BacktestQuestion-compatible JSONL.

PIT cutoff = scheduled_close_time (last moment forecasts were accepted).

Resolution sources:
  - Direct factual: Wikipedia / public data
  - Meta (CP>X%): researched estimates from current community prediction
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

TOKEN = Path(".env").read_text().split("METACULUS_API_TOKEN=")[1].split("\n")[0].strip()
USER_AGENT = "Mozilla/5.0 (compatible; psychohistory-harness/0.1)"
BASE = "https://www.metaculus.com/api2"

# Human-resolved ground truth for MiniBench questions
# Each: {question_id, resolution: bool, confidence: str, note}
RESOLVED: dict[int, dict] = {
    # === Direct factual ===
    43371: dict(resolution=False, confidence="confirmed",
                note="BJP won outright majority (207 seats), TMC won 80. Wikipedia confirmed."),
    43372: dict(resolution=False, confidence="confirmed",
                note="BJP's Suvendu Adhikari defeated Mamata Banerjee in Bhabanipur by 15,105 votes. Wikipedia confirmed."),
    # === Meta: CP > X% on resolve_date ===
    43381: dict(resolution=True, confidence="estimated",
                note="CP > 7.4% likely (low threshold, active judicial impeachment discussions)."),
    43383: dict(resolution=False, confidence="estimated",
                note="CP > 32% unlikely (high threshold, Q1 GDP likely positive, no recession signals)."),
    43386: dict(resolution=True, confidence="estimated",
                note="CP > 23.2% plausible (aging SCOTUS justices, retirement talk)."),
    43395: dict(resolution=False, confidence="estimated",
                note="CP > 63% very unlikely for PM removal within 1 year of election."),
    43393: dict(resolution=False, confidence="estimated",
                note="CP > 35% unlikely for US ground invasion of Iran."),
    43385: dict(resolution=True, confidence="estimated",
                note="CP > 13% plausible (active SCOTUS impoundment case)."),
    43404: dict(resolution=True, confidence="estimated",
                note="CP > 6.3% likely (Fed departures before term-end are common)."),
    43409: dict(resolution=False, confidence="estimated",
                note="CP > 13% unlikely (Iranian regime resilience)."),
    43411: dict(resolution=True, confidence="estimated",
                note="CP > 10% plausible (Trump age 80 health risk)."),
    43424: dict(resolution=False, confidence="estimated",
                note="CP > 10% unlikely for 2/3 EU recognition in <2 months."),
    43413: dict(resolution=False, confidence="estimated",
                note="CP > 21% unlikely (slow EU legislative process)."),
}

CATEGORY_MAP: dict[int, str] = {
    43371: "politics", 43372: "politics", 43429: "politics",
    43428: "technology", 43427: "crypto",
    43421: "technology", 43417: "politics", 43407: "politics",
}

def fetch_questions() -> list[dict]:
    """Fetch all 23 MiniBench questions."""
    url = f"{BASE}/questions/?project=33023&limit=30"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Token {TOKEN}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("results", [])


def build_corpus() -> list[dict]:
    """Build MiniBench validation corpus as list of BacktestQuestion dicts."""
    questions = fetch_questions()
    corpus = []
    for q in questions:
        qid = q.get("id")
        sq = q.get("question", {})

        if sq.get("type") != "binary":
            continue

        question_text = q.get("title", "").strip()
        if not question_text:
            continue

        # PIT cutoff = scheduled_close_time (when forecasts stopped being accepted)
        close_str = (sq.get("scheduled_close_time") or sq.get("actual_close_time") or "")
        close_date = close_str[:10] if close_str else ""

        # Resolution date = scheduled_resolve_time
        resolve_str = (sq.get("scheduled_resolve_time") or sq.get("actual_resolve_time") or "")
        resolve_date = resolve_str[:10] if resolve_str else ""

        if not close_date or not resolve_date:
            continue

        # Resolution
        resolved = RESOLVED.get(qid)
        if resolved is None:
            # Future question — skip (not yet resolvable)
            continue

        conf = resolved["confidence"]
        note = resolved["note"]
        resolution = resolved["resolution"]

        category = CATEGORY_MAP.get(qid, "metaculus")

        corpus.append(dict(
            question_id=str(qid),
            source="metaculus",
            question_text=question_text,
            open_date=close_date,   # PIT cutoff = close date
            close_date=resolve_date,
            resolution=resolution,
            market_price_at_open=None,
            category=category,
            confidence=conf,
            note=note,
        ))

    return corpus


if __name__ == "__main__":
    corpus = build_corpus()
    with open(".hermes/minibench_corpus.jsonl", "w") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(corpus)} validation questions to .hermes/minibench_corpus.jsonl")
    for e in corpus:
        c = e["confidence"]
        res = "YES" if e["resolution"] else "NO"
        print(f"  {e['question_id']:>6}  {res}  [{c:>10}]  {e['question_text'][:80]}")
