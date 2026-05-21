"""Quick single-question forecast test."""
import sys, os, logging, time
from pathlib import Path

# Add project root so we can import harness modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test")

from harness.tournament_watcher import (
    list_tournament_posts, extract_open_questions, 
    get_post_details, QuestionInfo
)

# -- Import run_forecast_pipeline from watch_tournaments --
# We can't do `from scripts.watch_tournaments import ...` because scripts/ isn't a package.
# Instead, exec the file as a module.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "watch_tournaments", 
    ROOT / "scripts" / "watch_tournaments.py"
)
wt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wt)

run_forecast_pipeline = wt.run_forecast_pipeline
KNOWN_TOURNAMENTS = wt.KNOWN_TOURNAMENTS

# Load token
token = os.environ.get("METACULUS_TOKEN")
if not token:
    token_path = os.path.expanduser("~/.metaculus_token")
    if os.path.exists(token_path):
        token = open(token_path).read().strip()
if not token:
    print("METACULUS_TOKEN not set", file=sys.stderr)
    sys.exit(1)

# Fetch open questions from Cup
posts = list_tournament_posts(KNOWN_TOURNAMENTS["cup"], token)
open_pairs = extract_open_questions(posts)
print(f"Found {len(open_pairs)} open questions")

# Fetch details for first few
questions = []
for qid, pid in open_pairs:
    try:
        time.sleep(0.5)
        details = get_post_details(pid, token)
        qdata = details.get("question", {})
        if qdata.get("status") == "open":
            q = QuestionInfo(
                question_id=qid,
                post_id=pid,
                title=qdata.get("title", ""),
                description=qdata.get("description", ""),
                resolution_criteria=qdata.get("resolution_criteria", ""),
                fine_print=qdata.get("fine_print", ""),
                question_type=qdata.get("type", "binary"),
                close_time=qdata.get("scheduled_close_time", ""),
                resolve_time=qdata.get("scheduled_resolve_time", ""),
                status=qdata.get("status", ""),
            )
            questions.append(q)
            print(f"  Q{q.question_id}: [{q.question_type}] {q.title[:80]}")
    except Exception as e:
        print(f"  Error fetching Q{qid}: {e}")

print(f"\nLoaded {len(questions)} questions")

# Pick first binary and first non-binary
binary_q = next((q for q in questions if q.question_type == "binary"), None)
nonbinary_q = next((q for q in questions if q.question_type != "binary"), None)

if binary_q:
    print(f"\n{'='*60}")
    print(f"TESTING BINARY: Q{binary_q.question_id} '{binary_q.title[:60]}'")
    print(f"{'='*60}")
    try:
        result = run_forecast_pipeline(binary_q, token, dry_run=True)
        orch = result.get("orchestrator", {})
        print(f"\n✅ Binary result: p_yes={orch.get('p_yes')}, output_type={orch.get('output_type')}")
    except Exception as e:
        logger.exception("Binary forecast FAILED")
        sys.exit(1)

if nonbinary_q:
    print(f"\n{'='*60}")
    print(f"TESTING NON-BINARY: Q{nonbinary_q.question_id} [{nonbinary_q.question_type}] '{nonbinary_q.title[:60]}'")
    print(f"{'='*60}")
    try:
        result = run_forecast_pipeline(nonbinary_q, token, dry_run=True)
        orch = result.get("orchestrator", {})
        print(f"\n✅ Non-binary result: output_type={orch.get('output_type')}, value={orch.get('value')}, distribution={orch.get('distribution')}")
    except Exception as e:
        logger.exception("Non-binary forecast FAILED")
        sys.exit(1)

print(f"\n{'='*60}")
print("ALL TESTS PASSED ✅")
