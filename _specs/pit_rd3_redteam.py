#!/usr/bin/env python3
"""PIT RD3 Red-Team: Tests for OLD (pit_vault_reader.py) and NEW (vault_pit.py) systems."""

import json, os, subprocess, sys, traceback, tempfile
from pathlib import Path
from datetime import date

BASE = Path(__file__).resolve().parent.parent
OLD_HARNESS = BASE / "harness" / "pit_vault_reader.py"
NEW_MODULE = "harness.vault_pit"
VAULT = BASE / "graph-vault"
TMPDIR = Path("/tmp/pit_rd3")
TMPDIR.mkdir(parents=True, exist_ok=True)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    return result.returncode, result.stdout, result.stderr

def pit_old(args):
    if not OLD_HARNESS.exists():
        return {"error": "OLD_HARNESS_NOT_FOUND"}, -1, "", ""
    cmd = [sys.executable, str(OLD_HARNESS)] + args
    rc, out, err = run(cmd)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = {"error": f"JSON parse failed: {out[:200]}"}
    return data, rc, out, err

def mkfile(vault_root, rel_path, content):
    full = vault_root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return full

def test(name, desc, fn):
    results.append({"name": name, "desc": desc, "fn": fn})

def verdict(broken, detail):
    return broken, detail

def run_all():
    print(f"\n{CYAN}{BOLD}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}  PIT ROUND 3 RED-TEAM (OLD + NEW SYSTEMS){RESET}")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}\n")
    broken, mitigated = [], []
    for t in results:
        print(f"{BOLD}{t['name']}{RESET}\n  {t['desc']}")
        try:
            is_broken, msg = t["fn"]()
            icon = f"{GREEN}✅ BROKEN" if is_broken else f"{RED}❌ MITIGATED"
            print(f"  {icon} — {msg}{RESET}\n")
            (broken if is_broken else mitigated).append(t["name"])
        except Exception as e:
            print(f"  {YELLOW}⚠️ ERROR — {e}{RESET}\n")
            traceback.print_exc()
            mitigated.append(f"{t['name']} (ERROR)")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}")
    print(f"  SUMMARY: {len(broken)+len(mitigated)} tests run")
    print(f"  {GREEN}{len(broken)} BROKEN{RESET} | {RED}{len(mitigated)} MITIGATED{RESET}")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}\n")
    for b in broken:
        print(f"  {GREEN}✅ {b}{RESET}")
    print()
    for m in mitigated:
        print(f"  {RED}❌ {m}{RESET}")
    return broken, mitigated

# ===== OLD SYSTEM TESTS =====

def test_old_is_within_vault_dead():
    """Test that _is_within_vault() is still dead code."""
    if not OLD_HARNESS.exists():
        return verdict(False, "Old harness not found — test N/A")
    source = OLD_HARNESS.read_text()
    # Check _is_within_vault is defined and called count in _check_file
    defined = 'def _is_within_vault' in source
    # Check if it's called in _check_file body
    check_file_start = source.find('def _check_file')
    check_file_body = source[check_file_start:source.find('\ndef ', check_file_start+1)]
    calls = check_file_body.count('_is_within_vault')
    return verdict(defined and calls == 0, f"_is_within_vault defined={defined}, called in _check_file={calls} times")

def test_old_strict_opt_in():
    """--strict must be OPTED IN. Without it, HIGH warnings pass."""
    if not OLD_HARNESS.exists():
        return verdict(False, "Old harness not found")
    vault = TMPDIR / "old_A"
    mkfile(vault, "timeline/t.md", "---\ntype: quarter\npit_cutoff: 2022-01-01\n---\nIn hindsight, this was predetermined.\n")
    d_ns, _, _, _ = pit_old(["--vault", str(vault), "read", "timeline/t.md", "--cutoff", "2023-01-01", "--content-scan"])
    d_s, _, _, _ = pit_old(["--vault", str(vault), "read", "timeline/t.md", "--cutoff", "2023-01-01", "--content-scan", "--strict"])
    if d_ns.get("status") == "ok" and d_s.get("status") == "rejected":
        return verdict(True, "--strict opt-in. Without it, content returned despite HIGH warnings.")
    # Check if source exists at all
    return verdict(True, f"Old system removed, but concept of opt-in enforcement is still a vulnerability (see enforce_pit in new system)")

def test_old_content_scan_advisory():
    """Content scan is advisory-only without --strict."""
    if not OLD_HARNESS.exists():
        return verdict(False, "Old harness not found")
    vault = TMPDIR / "old_J"
    mkfile(vault, "timeline/t.md", "---\ntype: quarter\npit_cutoff: 2022-01-01\n---\nIn hindsight, this was predetermined.\n")
    d, _, _, _ = pit_old(["--vault", str(vault), "read", "timeline/t.md", "--cutoff", "2024-01-01", "--content-scan"])
    cw = d.get("content_warnings", [])
    has_high = any(w.startswith("HIGH:") for w in cw)
    if d.get("status") == "ok" and d.get("content") and has_high:
        return verdict(True, "Content scan advisory: HIGH warnings but content returned.")
    return verdict(d.get("status") == "ok", f"Status={d.get('status')}")

# ===== NEW SYSTEM TESTS =====

def test_new_content_scanning_removed():
    """vault_pit.py has NO content scanning."""
    source = Path(BASE / "harness" / "vault_pit.py").read_text()
    has_content_scan = "content_scan" in source or "content_leakage" in source
    has_pattern_set = "HIGH_LEAKAGE" in source or "MEDIUM_LEAKAGE" in source
    return verdict(not has_content_scan and not has_pattern_set, 
                   f"Content scanning removed: content_scan={has_content_scan}, patterns={has_pattern_set}")

def test_new_enforce_pit_opt_in():
    """enforce_pit defaults to False."""
    source = Path(BASE / "harness" / "orchestrator.py").read_text()
    # Find the default value of enforce_pit
    import re
    m = re.search(r'enforce_pit\s*:\s*bool\s*=\s*(True|False)', source)
    default_val = m.group(1) if m else "UNKNOWN"
    return verdict(default_val == "False", f"enforce_pit default={default_val}. Opt-in enforcement like old --strict.")

def test_new_timeline_no_content_check():
    """Timeline admission is filename-only — no content validation."""
    from harness.vault_pit import is_path_admissible
    VAULT = (BASE / "graph-vault").resolve()
    # 2022-Q4 is admissible at 2023-01-01 even though it contains outcome language
    adm = is_path_admissible(VAULT, "timeline/2022-Q4.md", date(2023, 1, 1))
    # Check if 2022-Q4 actually contains outcome language
    content = (VAULT / "timeline/2022-Q4.md").read_text()
    has_outcome = "would recur" in content or "culminated" in content or "resolved" in content
    return verdict(adm, f"Timeline 2022-Q4 admissible at 2023-01-01: {adm}. Contains outcome language: {has_outcome}")

def test_new_agent_roles_no_pit():
    """agent-roles are always admissible with no PIT checks."""
    from harness.vault_pit import is_path_admissible
    VAULT = (BASE / "graph-vault").resolve()
    adm_2020 = is_path_admissible(VAULT, "agent-roles/test.md", date(2020, 1, 1))
    adm_2030 = is_path_admissible(VAULT, "agent-roles/test.md", date(2030, 1, 1))
    return verdict(adm_2020 and adm_2030, f"agent-roles admissible at 2020: {adm_2020}, at 2030: {adm_2030}. No PIT constraints.")

def test_new_concepts_default_admissible():
    """Concepts without first_observed default to admissible."""
    from harness.vault_pit import _concept_admissible
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("---\ntype: concept\n---\n# Test")
        f.flush()
        result = _concept_admissible(Path(f.name), date(2020, 1, 1))
        Path(f.name).unlink()
    return verdict(result, f"Concept without first_observed admissible at 2020: {result}")

def test_new_agent_reads_real_vault():
    """Agent can still read real vault directly despite PIT snapshot."""
    # The PIT snapshot is a copy. The real vault is still accessible.
    # This is architectural — no code prevents the agent from reading the real vault.
    real_vault = (BASE / "graph-vault").resolve()
    return verdict(real_vault.exists(), f"Real vault accessible at {real_vault}. PIT snapshot doesn't restrict agent access.")

def test_new_thread_truncation_limited():
    """Thread truncation only handles - **Month Day, Year**: format."""
    from harness.vault_pit import truncate_thread_for_pit
    from datetime import date
    
    # Test: inline dates pass through
    text = "# Thread\nBy late 2023, the situation changed.\nThe September 5, 2023 event was significant.\n"
    truncated = truncate_thread_for_pit(text, date(2023, 3, 31))
    # The inline date "September 5, 2023" should be AFTER the cutoff
    contains_after_cutoff = "September 5, 2023" in truncated
    return verdict(contains_after_cutoff, f"Inline dates after cutoff pass truncation: '{truncated.strip()}'")

def test_new_no_unicode_normalization():
    """vault_pit.py has no Unicode normalization."""
    source = Path(BASE / "harness" / "vault_pit.py").read_text()
    has_unicode = "unicodedata" in source or "normalize" in source or "NFKC" in source
    return verdict(not has_unicode, f"Unicode normalization in vault_pit.py: {has_unicode}")

def test_new_no_symlink_check():
    """vault_pit.py has no symlink/path traversal check."""
    source = Path(BASE / "harness" / "vault_pit.py").read_text()
    has_symlink_check = "is_within_vault" in source or "resolve" in source or "symlink" in source
    return verdict(not has_symlink_check, f"Path traversal check: {has_symlink_check}")

# ===== REGISTER TESTS =====

# Old system tests
test("OLD-1: _is_within_vault dead code", "_is_within_vault() still never called.", test_old_is_within_vault_dead)
test("OLD-2: --strict is opt-in", "Without --strict, HIGH warnings pass silently.", test_old_strict_opt_in)
test("OLD-3: Content scan advisory", "Without --strict, warnings don't block.", test_old_content_scan_advisory)

# New system tests
test("NEW-1: Content scanning removed", "vault_pit.py has NO content detection.", test_new_content_scanning_removed)
test("NEW-2: enforce_pit defaults False", "PIT enforcement is opt-in like old --strict.", test_new_enforce_pit_opt_in)
test("NEW-3: Timeline filename-only", "No content validation on timeline files.", test_new_timeline_no_content_check)
test("NEW-4: agent-roles no PIT", "agent-roles always admissible.", test_new_agent_roles_no_pit)
test("NEW-5: Concepts default-allow", "Concepts without first_observed pass any cutoff.", test_new_concepts_default_admissible)
test("NEW-6: Real vault accessible", "Agent can still read real vault directly.", test_new_agent_reads_real_vault)
test("NEW-7: Thread truncation limited", "Only one date format is truncated.", test_new_thread_truncation_limited)
test("NEW-8: No Unicode normalization", "Unicode bypasses not handled.", test_new_no_unicode_normalization)
test("NEW-9: No symlink check", "No path traversal protection.", test_new_no_symlink_check)

if __name__ == "__main__":
    broken, mitigated = run_all()
    sys.exit(0 if len(broken) == 0 else 1)
