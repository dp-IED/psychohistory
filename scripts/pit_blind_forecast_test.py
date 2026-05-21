"""
Blind PIT-Context Forecasting Test.

For each of the 30 Polymarket gold cases:
  1. Build PIT quarter summaries for the 4 quarters before the question was created
     (blind — the agent never sees the question)
  2. Load all quarter summaries + show the question → agent forecasts YES/NO
  3. Score against ground truth

The quarters are built ONCE (shared across questions needing the same quarter).

Usage:
  python scripts/pit_blind_forecast_test.py [--phase quarters|forecast|score|all]
"""

import json, subprocess, sys, os, time, re, shutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
sys.path.insert(0, str(TESTBED))
WORKSPACE = TESTBED.parent
GOLD_PATH = TESTBED / "data" / "polymarket" / "gold_branch_dataset.json"
OUTPUT = TESTBED / "pit_blind_test"
QUARTERS_DIR = OUTPUT / "quarters"
FORECASTS_DIR = OUTPUT / "forecasts"
RESULTS_PATH = OUTPUT / "results.json"

HERMES_PROFILE = "forecasting"
HERMES_TIMEOUT = 1200
MAX_WORKERS = 4
from harness.config import VAULT_DIR

VAULT_PATH = VAULT_DIR


# ── Helpers ──────────────────────────────────────────────────────────

def quarters_before(dt: datetime, n: int = 4):
    """Return list of n quarter labels (e.g. ['2024-Q3','2024-Q2',…]) before dt."""
    q = (dt.month - 1) // 3 + 1
    result = []
    for _ in range(n):
        q -= 1
        if q < 1:
            q = 4
            dt = dt.replace(year=dt.year - 1)
        result.append(f"{dt.year}-Q{q}")
    return result


def quarter_start_end(label: str):
    """Return (start_date, end_date) as date objects for a quarter label like '2024-Q3'."""
    year = int(label.split("-Q")[0])
    q = int(label.split("Q")[1])
    month = (q - 1) * 3 + 1
    start = datetime(year, month, 1)
    if q == 4:
        end = datetime(year, 12, 31)
    else:
        end = datetime(year, month + 3, 1) - timedelta(days=1)
    return start, end


def call_hermes(prompt: str, label: str = "", timeout: int = HERMES_TIMEOUT):
    """Run hermes -z with the given prompt, return stdout."""
    cmd = ["hermes", "-z", prompt, "--profile", HERMES_PROFILE]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(OUTPUT)
        )
        if r.returncode != 0:
            return f"[ERROR] exit={r.returncode} stderr={r.stderr[:500]}"
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"[ERROR] timeout after {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"


def quarter_summary_prompt(label: str, start: datetime, end: datetime):
    """Blind prompt — no question references, just quarter summary."""
    return f"""=== PIT SUMMARY: {label} ({start.date()} to {end.date()} ) ===

You are at the end of {label}. Research what happened during this period
using ONLY information available up to {end.date()}. No future knowledge.

=== YOUR TOOLS ===
- Web search (PIT-constrained to {end.date()})
- Write_file to create files

=== YOUR TASK ===
Write a PIT-constrained summary for {label}. Make it rich and detailed.
Focus on events, decisions, turning points, and emerging patterns across
all domains — geopolitics, economics, technology, society, environment.
Cite your sources inline with the date of publication.

Write to: {QUARTERS_DIR / f"{label}.md"}

=== PIT CONSTRAINT ===
No information from after {end.date()}. If you can't find PIT-safe sources
for a claim, do not include it."""


def forecast_prompt(case: dict, quarter_summaries: list[tuple[str, str]]):
    """
    Build the prompt for the forecasting agent.
    Shows the quarter summaries (already built, blind) THEN the question.
    """
    context_blocks = []
    for label, summary_path in quarter_summaries:
        try:
            text = Path(summary_path).read_text()
            context_blocks.append(f"[CONTEXT: {label}]\n{text[:3000]}")
        except Exception as e:
            context_blocks.append(f"[CONTEXT: {label}]\n(UNAVAILABLE: {e})")

    context = "\n\n".join(context_blocks)
    question = case['full_text'][:4000]  # truncated to fit

    return f"""=== BLIND FORECAST TASK ===

Below is historical context that was prepared without knowledge of the question.
Use ONLY this context and your knowledge of what was known up to the context cutoffs.

{"=" * 60}

HISTORICAL CONTEXT (prepared blind):
{"=" * 60}

{context}

{"=" * 60}

FORECAST QUESTION:
{"=" * 60}

{question}

Based on the historical context above, what is your forecast for this question?
You MUST output your answer in this exact format at the end:

---
FINAL ANSWER:
Prediction: YES or NO
Confidence: 0.00 to 1.00
Rationale: (1-2 sentences)
---"""


def load_gold_cases():
    with open(GOLD_PATH) as f:
        return json.load(f)["cases"]


# ── Phase 1: Build all unique quarter summaries ────────────────────

def phase_quarters():
    """Build all 13 unique quarter summaries (blind — no questions)."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    QUARTERS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_gold_cases()
    all_qs = set()
    for case in cases:
        created = case["record"]["created_at"]
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        for q in quarters_before(dt):
            all_qs.add(q)

    all_qs = sorted(all_qs)
    print(f"Phase 1: Building {len(all_qs)} unique quarter summaries (blind)")
    print(f"  Quarters: {', '.join(all_qs)}")
    print()

    # Check which already exist
    to_build = []
    for label in all_qs:
        qfile = QUARTERS_DIR / f"{label}.md"
        if qfile.exists():
            print(f"  [SKIP] {label} already exists ({qfile.stat().st_size} bytes)")
        else:
            to_build.append(label)

    if not to_build:
        print("\n  All quarters exist — skipping builds.")
        return all_qs

    # Build in parallel batches of 4
    batches = [to_build[i:i+4] for i in range(0, len(to_build), 4)]
    for batch_idx, batch in enumerate(batches):
        print(f"\n  Batch {batch_idx + 1}/{len(batches)}: {', '.join(batch)}")
        results = {}

        def _build(label):
            start, end = quarter_start_end(label)
            prompt = quarter_summary_prompt(label, start, end)
            output = call_hermes(prompt, label=label)
            result_path = QUARTERS_DIR / f"{label}.md"
            if result_path.exists() and result_path.stat().st_size > 100:
                size = result_path.stat().st_size
                return label, f"OK ({size} bytes)"
            else:
                return label, f"ERROR: {output[:200]}"

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch))) as pool:
            futures = {pool.submit(_build, label): label for label in batch}
            for future in as_completed(futures):
                label, status = future.result()
                results[label] = status

        for label in batch:
            print(f"    {label}: {results.get(label, '?')}")

        # Brief pause between batches
        if batch_idx < len(batches) - 1:
            time.sleep(5)

    # Verify all built
    missing = [q for q in all_qs if not (QUARTERS_DIR / f"{q}.md").exists()]
    if missing:
        print(f"\n  WARNING: {len(missing)} quarters missing: {missing}")
    else:
        print(f"\n  All {len(all_qs)} quarters built successfully.")

    return all_qs
# ── Phase 2: Per-question forecast + immediate reflection ──────────

def phase_forecast(question_ids: list[str] | None = None, force: bool = False):
    """
    Per-question forecast + immediate reflection loop.

    For EACH question:
      1. Agent reads the relevant vault timeline files
      2. Agent forecasts YES/NO
      3. Ground truth is REVEALED to the agent
      4. Agent reflects: why was it right/wrong? What was vault missing?
      5. Agent improves _spec.md, _procedure.md, threads, concepts
      6. Git commit → next question starts with an improved vault
    """
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATE_SCRIPT = HERE / "validate_vault.py"
    cases = load_gold_cases()

    if question_ids:
        cases = [c for c in cases if c["case_id"] in question_ids]

    print(f"Phase 2: Running {len(cases)} questions with per-question reflection loop")
    print()

    # Track cumulative stats
    cumulative = {"correct": 0, "wrong": 0, "unparseable": 0}

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        expected = case["expected_target_value"]
        expected_label = "YES" if expected == 1.0 else "NO"

        # Resume: skip if this question already has a committed reflection
        # (check git for a commit message containing the case_id)
        if not force:
            skip_check = subprocess.run(
                ["git", "log", "--oneline", "--grep", f"reflection after {case_id}"],
                capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=10,
            )
            if skip_check.stdout.strip():
                print(f"  [{i+1}/{len(cases)}] {case_id[:45]} — already reflected, skipping")
                continue

        created = case["record"]["created_at"]
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except:
            dt = datetime.now()
        pre_qs = quarters_before(dt)

        print(f"\n{'='*70}")
        print(f"  QUESTION {i+1}/{len(cases)}: {case_id[:50]}")
        print(f"  Expected: {expected_label} | Pre-quarters: {pre_qs}")
        print(f"{'='*70}")

        # ── Step A: Read vault state before forecast ──
        vault_tree = "\n".join(_tree(VAULT_PATH))

        # ── Step A.5: Create PIT snapshot ──
        from harness.vault_pit import materialize_pit_snapshot, format_admissible_block
        import tempfile
        cutoff_date = dt.date()
        pit_tmp = tempfile.mkdtemp(prefix="pit-forecast-")
        pit_snapshot_path = Path(pit_tmp)
        copied = materialize_pit_snapshot(
            VAULT_PATH, pit_snapshot_path, cutoff_date, strict=True
        )

        # ── Step B: Forecast (using PIT snapshot, NOT live vault) ──
        q_files = "\n".join(f"  timeline/{q}.md" for q in pre_qs)
        question_text = case["full_text"][:4000]

        pit_block = format_admissible_block(copied, vault_dir=pit_snapshot_path)

        forecast_prompt = f"""=== FORECAST TASK (question {i+1}/{len(cases)}) ===

=== PIT CONSTRAINT ===
You are in a PIT-constrained vault snapshot at:
{pit_snapshot_path}

The LIVE vault at {VAULT_PATH} contains information from AFTER your cutoff.
DO NOT read from the live vault. Only use files from the snapshot.
Your cutoff date is {cutoff_date.isoformat()}.

{pit_block}

Relevant timeline files (read from the snapshot path above):
{q_files}

=== QUESTION TO FORECAST ===

{question_text}

Based on the historical context in the PIT snapshot:
- Read each timeline file listed above from the snapshot
- You may also check domains/*/threads/ and domains/*/concepts/ for context
- Then produce your forecast

You MUST output your answer in this exact format at the end:

---
FINAL ANSWER:
Prediction: YES or NO
Confidence: 0.00 to 1.00
Rationale: (1-2 sentences)
---"""

        print(f"\n  ── Step 1: Forecast ──")
        output = call_hermes(forecast_prompt, label=f"{case_id}-forecast", timeout=HERMES_TIMEOUT)
        prediction = parse_prediction(output)
        correct = (
            (prediction == "YES" and expected == 1.0) or
            (prediction == "NO" and expected == 0.0)
        ) if prediction else None

        status = "✓" if correct else ("✗" if correct is False else "?")
        print(f"  Prediction: {prediction}  Expected: {expected_label}  {status}")

        # Save forecast result
        result = {
            "case_id": case_id,
            "expected": expected,
            "prediction": prediction,
            "correct": correct,
            "quarters_used": pre_qs,
            "question": case["full_text"][:200],
            "raw_output": output[:500],
        }
        fpath = FORECASTS_DIR / f"{case_id}.json"
        fpath.write_text(json.dumps(result, indent=2))

        if correct:
            cumulative["correct"] += 1
        elif correct is False:
            cumulative["wrong"] += 1
        else:
            cumulative["unparseable"] += 1

        # ── Step C: Reveal ground truth + reflect ──
        print(f"\n  ── Step 2: Reflection (revealing ground truth) ──")

        # Load current spec/procedure
        spec_path = VAULT_PATH / "_spec.md"
        proc_path = VAULT_PATH / "_procedure.md"
        spec_content = spec_path.read_text() if spec_path.exists() else "(no _spec.md)"
        proc_content = proc_path.read_text() if proc_path.exists() else "(no _procedure.md)"

        # Build error details for cumulative stats
        results_so_far = []
        for rfile in sorted(FORECASTS_DIR.glob("gold_*.json")):
            rd = json.loads(rfile.read_text())
            results_so_far.append(rd)

        correct_list = [r for r in results_so_far if r.get("correct") is True]
        wrong_list = [r for r in results_so_far if r.get("correct") is False]

        reflect_prompt = f"""=== PER-QUESTION REFLECTION (after question {i+1}/{len(cases)}) ===

You just made a forecast for this question.

=== VAULT CONTEXT (abbreviated) ===

{vault_tree[:2000]}

=== YOUR FORECAST ===

Question: {case['full_text'][:300]}
Your prediction: {prediction or 'unparseable'}
Ground truth (actual outcome): {expected_label}

This prediction was {'CORRECT ✓' if correct else 'WRONG ✗' if correct is False else 'UNPARSEABLE ?'}

=== CUMULATIVE RESULTS SO FAR ===
Questions completed: {i + 1}
Correct: {len(correct_list)}
Wrong: {len(wrong_list)}

{'=== WRONG PREDICTIONS ===' if wrong_list else ''}
{chr(10).join(f"- {r['case_id']}: expected={'YES' if r['expected'] else 'NO'}, predicted={r['prediction']}" for r in wrong_list[-5:])}

=== CURRENT SPEC ===
{spec_content[:2000]}

=== CURRENT PROCEDURE ===
{proc_content[:2000]}

=== YOUR TASK ===

This is a per-question reflection. The vault must learn from EVERY question
because there are only {len(cases)} total.

1. DIAGNOSE: Why was your prediction right or wrong?
   - What information in the vault helped or misled you?
   - What was missing from the quarter summaries?
   - What causal chain or dynamic was under-represented?

2. IMPROVE THE SYSTEM — write files to {VAULT_PATH}:
   - Update _spec.md if the schema needs refinement (e.g. "include diplomatic signals")
   - Update _procedure.md if the forecast methodology needs changing
   - Create/update a thread file in domains/*/threads/ for a causal chain that was missed
   - Create/update a concept file in domains/*/concepts/ for a recurring dynamic
   - Create entity stubs in domains/*/entities/ for people/orgs that were relevant but missing

3. REPORT: What did you change and why?

Be specific. Each question is a learning opportunity.
The goal: the vault is strictly better after this reflection, so the next
question benefits from improved context."""

        reflect_output = call_hermes(reflect_prompt, label=f"{case_id}-reflect", timeout=HERMES_TIMEOUT)
        print(f"  Reflection ({len(reflect_output)} chars):")
        for line in reflect_output.split("\n")[:5]:
            print(f"    {line}")
        if len(reflect_output.split("\n")) > 6:
            print(f"    ... ({len(reflect_output.split('\\n')) - 5} more lines)")

        # Save reflection
        ref_path = OUTPUT / f"reflections/{case_id}-reflection.md"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(reflect_output)

        # ── Step D: Validate vault ──
        print(f"\n  ── Step 3: Validate vault ──")
        try:
            val = subprocess.run(
                [sys.executable, str(VALIDATE_SCRIPT), "--json"],
                capture_output=True, text=True, timeout=120,
                cwd=str(VAULT_PATH.parent),
            )
            vdata = json.loads(val.stdout) if val.stdout.strip() else {}
        except Exception as e:
            print(f"  Validation error: {e}")
            vdata = {}

        for check in vdata.get("checks", []):
            if not check.get("passed"):
                issues = check.get("issues", [])
                if issues:
                    print(f"  [{check['name']}] {len(issues)} issues")
                    for iss in issues[:3]:
                        print(f"    {iss}")

        # ── Step E: Git commit ──
        subprocess.run(["git", "add", "-A"], capture_output=True, cwd=str(VAULT_PATH), timeout=30)
        diff_stat = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=30,
        )
        if diff_stat.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m", f"reflection after {case_id}: {status} (pred={prediction}, actual={expected_label})"],
                capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=30,
            )
            print(f"  [git] Committed: {diff_stat.stdout.strip()[:150]}")
        else:
            print(f"  [git] No vault changes from this reflection.")

        # Brief pause between questions
        time.sleep(2)

        # Clean up PIT snapshot temp dir
        shutil.rmtree(pit_snapshot_path, ignore_errors=True)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  ALL {len(cases)} QUESTIONS COMPLETE")
    print(f"{'='*70}")
    print(f"  Correct: {cumulative['correct']}")
    print(f"  Wrong:   {cumulative['wrong']}")
    print(f"  Unparseable: {cumulative['unparseable']}")

    # Save aggregate
    all_results = []
    for rfile in sorted(FORECASTS_DIR.glob("gold_*.json")):
        all_results.append(json.loads(rfile.read_text()))
    agg_path = FORECASTS_DIR / "aggregate.json"
    agg_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Aggregate saved to {agg_path}")

    print_summary(all_results)
    return all_results


def parse_prediction(output: str):
    """Extract YES/NO and confidence from agent output."""
    if not output:
        return None
    output_upper = output.upper()
    if "PREDICTION: YES" in output_upper or "PREDICTION:YES" in output_upper:
        pred = "YES"
    elif "PREDICTION: NO" in output_upper or "PREDICTION:NO" in output_upper:
        pred = "NO"
    else:
        # Fallback: scan for YES/NO near the end
        last_200 = output[-600:]
        if "YES" in last_200.upper() and "NO" not in last_200.upper():
            pred = "YES"
        elif "NO" in last_200.upper() and "YES" not in last_200.upper():
            pred = "NO"
        else:
            return None

    # Extract confidence
    conf = None
    m = re.search(r'[Cc]onfidence:\s*([0-9]+\.[0-9]+)', output)
    if m:
        conf = float(m.group(1))

    return pred  # return string for now, keep it simple


# ── Phase 3: Score ─────────────────────────────────────────────────

def phase_score():
    """Score the forecasts against ground truth."""
    results_path = FORECASTS_DIR / "aggregate.json"
    if not results_path.exists():
        print("No aggregate results found. Run phase_forecast first.")
        return

    results = json.loads(results_path.read_text())
    print_summary(results)
    return results


def print_summary(results: list[dict]):
    """Print a formatted summary of forecast results."""
    total = len(results)
    predicted = [r for r in results if r.get("prediction")]
    correct = [r for r in predicted if r["correct"] is True]
    wrong = [r for r in predicted if r["correct"] is False]

    # By category
    categories = defaultdict(list)
    for r in results:
        cat = r["case_id"].split("_")[1] if "_" in r["case_id"] else "other"
        categories[cat].append(r)

    print()
    print("=" * 70)
    print("  BLIND PIT-CONTEXT FORECAST TEST RESULTS")
    print("=" * 70)
    print(f"\n  Total cases:    {total}")
    print(f"  Predicted:      {len(predicted)}")
    print(f"  Correct:        {len(correct)}  ({len(correct)/len(predicted)*100:.1f}% if predicted)")
    print(f"  Wrong:          {len(wrong)}")
    print(f"  Unparseable:    {total - len(predicted)}")

    if predicted:
        print(f"\n  ── By category ──")
        for cat in sorted(categories):
            items = categories[cat]
            preds = [r for r in items if r.get("prediction")]
            corr = [r for r in preds if r["correct"]]
            if preds:
                print(f"    {cat:20s} {len(corr)}/{len(preds)} ({len(corr)/len(preds)*100:.0f}%)")

    print(f"\n  ── Detail ──")
    for r in results:
        pred = r.get("prediction", "?")
        exp = "YES" if r["expected"] else "NO"
        mark = "✓" if r.get("correct") else ("✗" if r.get("correct") is False else "?")
        short_id = r["case_id"][:35]
        print(f"    {mark} {short_id:40s} expected={exp:3s} got={pred}" if isinstance(pred, str) else f"    {mark} {short_id:40s} expected={exp:3s} got={pred}")

    print()

    # Save to RESULTS_PATH
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DATA = {
        "total": total,
        "predicted": len(predicted),
        "correct": len(correct),
        "accuracy_pct": round(len(correct) / len(predicted) * 100, 1) if predicted else 0,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(RESULT_DATA, indent=2))
    print(f"  Full results saved to {RESULTS_PATH}")


# ── Phase 4: Reflect (with loop-until-ok pattern) ──────────────────

def phase_reflect():
    """
    Reflection phase with loop-until-ok pattern (mirrors pit_reflect.py).

    1. Agent reviews forecast results + vault state
    2. Improves _spec.md, _procedure.md, threads, concepts
    3. validate_vault.py checks structural integrity
    4. Loop until structural failures fixed or max_attempts reached
    5. Git commit
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)

    VALIDATE_SCRIPT = HERE / "validate_vault.py"

    # Load forecast results
    agg_path = FORECASTS_DIR / "aggregate.json"
    if not agg_path.exists():
        print("No aggregate results found. Run phase_forecast first.")
        return
    results = json.loads(agg_path.read_text())

    # Compute stats
    total = len(results)
    predicted = [r for r in results if r.get("prediction")]
    correct = [r for r in predicted if r["correct"] is True]
    wrong = [r for r in predicted if r["correct"] is False]

    # Build error analysis
    error_details = []
    for r in wrong:
        error_details.append(
            f"- {r['case_id']}: expected={'YES' if r['expected'] else 'NO'}, "
            f"predicted={r['prediction']}, q={r['quarters_used']}, "
            f"question={r['question'][:120]}"
        )
    correct_details = []
    for r in correct:
        correct_details.append(
            f"- {r['case_id']}: expected={'YES' if r['expected'] else 'NO'}, "
            f"predicted={r['prediction']}"
        )

    # Build vault tree
    vault_tree = "\n".join(_tree(VAULT_PATH))

    # Read current _spec.md / _procedure.md
    spec_path = VAULT_PATH / "_spec.md"
    proc_path = VAULT_PATH / "_procedure.md"
    spec_content = spec_path.read_text() if spec_path.exists() else "(no _spec.md)"
    proc_content = proc_path.read_text() if proc_path.exists() else "(no _procedure.md)"

    # Initial prompt
    prompt = f"""=== PIT FORECAST TEST — REFLECTION on GRAPH VAULT ===

You are in the graph vault at {VAULT_PATH}.
This vault was used to run 30 blind forecasts. The quarter summaries were
built without knowledge of the forecasting questions.

Below are the results. Review what happened, diagnose the errors,
and improve the vault system (_spec.md, _procedure.md, threads, concepts).

{"=" * 60}

FORECAST RESULTS
{"=" * 60}

Total: {total}
Correct: {len(correct)} / {len(predicted)} = {len(correct)/len(predicted)*100:.1f}%
Wrong: {len(wrong)}
Unparseable: {total - len(predicted)}

CORRECT:
{chr(10).join(correct_details) if correct_details else "(none)"}

WRONG:
{chr(10).join(error_details) if error_details else "(none)"}

{"=" * 60}

VAULT STRUCTURE
{"=" * 60}

{vault_tree}

{"=" * 60}

CURRENT SPEC
{"=" * 60}

{spec_content[:3000]}

{"=" * 60}

CURRENT PROCEDURE
{"=" * 60}

{proc_content[:3000]}

{"=" * 60}

YOUR TASK
{"=" * 60}

1. DIAGNOSE: Why were the wrong predictions wrong? Look for systematic biases:
   - Does the vault over-emphasize conflict narratives and miss diplomatic signals?
   - Are there missing thread files for key causal chains?
   - Are quarter summaries missing certain types of signals?
   - Is the spec missing guidance on what to include?

2. IMPROVE THE SYSTEM:
   - Update _spec.md: define/refine the schema for quarter summaries
   - Update _procedure.md: document how the pipeline works
   - Create/update thread files in domains/*/threads/ for causal chains that were missed
   - Create/update concept files in domains/*/concepts/ for recurring dynamics
   - Create entity stubs in domains/*/entities/ for key people/orgs referenced

3. REPORT: What did you change and why?

Write files using write_file with absolute paths under {VAULT_PATH}.
The goal: the NEXT iteration gets better accuracy because the vault learned."""

    STRUCTURAL_CHECKS = {
        "Dual directory", "Frontmatter drift", "Zero-byte files",
        "Missing annual summaries", "Entity backlinks",
        "Quarter cutoff", "Related Periods",
    }

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Reflection attempt {attempt}/{max_attempts} ---")
        print(f"  Calling hermes for reflection...", end=" ", flush=True)
        output = call_hermes(prompt, label=f"reflect-attempt-{attempt}", timeout=HERMES_TIMEOUT)
        print("done.")
        print(f"  Raw output ({len(output)} chars):")
        # Print first/last few lines
        lines = output.split("\n")
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... ({len(lines) - 20} lines omitted)")
            for line in lines[-10:]:
                print(f"    {line}")

        # Save reflection output
        reflect_path = OUTPUT / f"_reflection-{datetime.now().strftime('%Y-%m-%d')}-attempt-{attempt}.md"
        reflect_path.write_text(output)

        # Run validate_vault.py
        print(f"\n  Running validate_vault.py...", end=" ", flush=True)
        try:
            val = subprocess.run(
                [sys.executable, str(VALIDATE_SCRIPT), "--json"],
                capture_output=True, text=True, timeout=120,
                cwd=str(VAULT_PATH.parent),
            )
            try:
                vdata = json.loads(val.stdout) if val.stdout.strip() else {"passed": False, "error": "empty output"}
            except json.JSONDecodeError:
                print(f"parse error:\n{val.stdout[:500]}")
                vdata = {"passed": False, "error": "json parse failure"}
        except Exception as e:
            print(f"error: {e}")
            vdata = {"passed": False, "error": str(e)}

        # Separate structural failures (gating) from aspirational
        structural_failures = []
        aspirational_issues = 0
        for check in vdata.get("checks", []):
            if not check.get("passed"):
                if check["name"] in STRUCTURAL_CHECKS:
                    structural_failures.append(check)
                else:
                    aspirational_issues += len(check.get("issues", []))

        if aspirational_issues:
            print(f"  [info] {aspirational_issues} aspirational issues (non-gating)")
        if not structural_failures:
            print(f"\n  ✓ All {len(vdata.get('checks', []))} validation checks passed (structural clean).")
            break

        total_structural = sum(len(c.get("issues", [])) for c in structural_failures)
        print(f"\n  ✗ {total_structural} structural issues remain (attempt {attempt}/{max_attempts})")
        for check in structural_failures:
            issues = check.get("issues", [])
            print(f"    [{check['name']}] {len(issues)} issues")
            for iss in issues[:5]:
                print(f"      {iss}")
            if len(issues) > 5:
                print(f"      ... ({len(issues) - 5} more)")

        if attempt >= max_attempts:
            print(f"\n  Max attempts ({max_attempts}) reached. Exiting with structural issues.")
        else:
            # Append failures to prompt for retry
            prompt += f"\n\n=== STRUCTURAL FAILURES TO FIX (attempt {attempt}) ===\n"
            for check in structural_failures:
                prompt += f"\n## {check['name']}\n"
                for iss in check.get("issues", [])[:30]:
                    prompt += f"- {iss}\n"
                if len(check.get("issues", [])) > 30:
                    prompt += f"- ... and {len(check['issues']) - 30} more\n"
            prompt += "\nFix ALL of the above structural issues before the next validation pass."

    # Commit reflection changes (into the main vault)
    print(f"\n  Committing reflection changes...")
    git_add = subprocess.run(
        ["git", "add", "-A"],
        capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=30,
    )
    diff_stat = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=30,
    )
    if diff_stat.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", "reflection: post-blind-forecast-test review"],
            capture_output=True, text=True, cwd=str(VAULT_PATH), timeout=30,
        )
        print(f"  [git] Committed: {diff_stat.stdout.strip()[:200]}")
    else:
        print("  [git] No changes from reflection.")

    print(f"\n  Full reflection saved to {reflect_path}")


def _tree(dir_path: Path, prefix: str = "", max_files: int = 50) -> list[str]:
    """Build a compact ASCII tree view of a directory (mirrors pit_reflect.py)."""
    lines: list[str] = []
    entries = sorted(dir_path.iterdir())
    count = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if count >= max_files:
            remaining = len([e for e in entries if not e.name.startswith(".")]) - max_files
            lines.append(f"{prefix}+-- ... ({remaining} more)")
            break
        is_last = (count == len([e for e in entries if not e.name.startswith(".")]) - 1) or count >= max_files - 1
        connector = "+-- " if is_last else "|-- "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            if count < max_files:
                indent = "    " if is_last else "|   "
                sub = _tree(entry, prefix + indent, max_files - count - 1)
                lines.extend(sub)
        else:
            size = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name} ({size}b)")
        count += 1
    return lines


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    phase = args[0] if args else "all"

    if phase in ("quarters", "all"):
        phase_quarters()

    if phase in ("forecast", "all"):
        phase_forecast(force=force)

    if phase in ("score", "all"):
        phase_score()

    if phase in ("reflect", "all"):
        phase_reflect()

    if phase == "all":
        print("\n  Done. See results.json for full report.")
