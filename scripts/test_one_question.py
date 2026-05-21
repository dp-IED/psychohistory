"""Run one question through the full cognitive pipeline to completion."""
import sys, os, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reconfigure stdout for line buffering
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

# Load token
token = os.environ.get("METACULUS_TOKEN")
if not token:
    tk = os.path.expanduser("~/.metaculus_token")
    if os.path.exists(tk):
        token = Path(tk).read_text().strip()
assert token, "No METACULUS_TOKEN"

from harness.tournament_watcher import (
    list_tournament_posts, extract_open_questions, get_post_details, QuestionInfo,
    KNOWN_TOURNAMENTS,
)

# Fetch first binary question
posts = list_tournament_posts(KNOWN_TOURNAMENTS["cup"], token)
open_pairs = extract_open_questions(posts)
print(f"Found {len(open_pairs)} open questions")

for qid, pid in open_pairs:
    time.sleep(0.3)
    details = get_post_details(pid, token)
    qdata = details.get("question", {})
    if qdata.get("status") != "open":
        continue
    qtype = qdata.get("type", "binary")
    if qtype != "binary":
        continue
    
    q = QuestionInfo(
        question_id=qid, post_id=pid,
        title=qdata.get("title", ""),
        description=qdata.get("description", ""),
        resolution_criteria=qdata.get("resolution_criteria", ""),
        fine_print=qdata.get("fine_print", ""),
        question_type=qtype,
        close_time=qdata.get("scheduled_close_time", ""),
        resolve_time=qdata.get("scheduled_resolve_time", ""),
        status=qdata.get("status", ""),
    )
    
    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: Q{q.question_id} [{q.question_type}]")
    print(f"  '{q.title[:100]}'")
    print(f"{'='*60}")
    
    # Import watch_tournaments to get run_forecast_pipeline
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "watch_tournaments", ROOT / "scripts" / "watch_tournaments.py"
    )
    wt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wt)
    
    try:
        result = wt.run_forecast_pipeline(q, token, dry_run=True)
        orch = result.get("orchestrator", {})
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"   p_yes:       {orch.get('p_yes')}")
        print(f"   output_type: {orch.get('output_type')}")
        print(f"   confidence:  {orch.get('confidence')}")
        print(f"   ci_low/high: {orch.get('ci_low')} / {orch.get('ci_high')}")
        print(f"   rationale:   {str(orch.get('rationale', ''))[:300]}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    break
else:
    print("No binary questions found!")
    sys.exit(1)
