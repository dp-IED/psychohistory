# PIT Round 3 Red-Team Report

**Date:** 2026-05-20  
**Target:** PIT Safety System at `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/testbed`  
**Round:** 3 (post-fix audit)  
**Status:** Final  

**IMPORTANT:** During this red-team, the system was **completely rewritten** — the old `harness/pit_vault_reader.py` (932-line CLI tool) was replaced by `harness/vault_pit.py` (249-line library). The report below covers both the original RD3 attack vectors against the old system AND new findings against the replacement system.

---

## Executive Summary

### Old system (`pit_vault_reader.py`): 20/22 BROKEN

Against the system that existed when the task was given, **20 out of 22 attack vectors are exploitable**. The two mitigations (sed-stripping protection via filename inference, and homoglyph YAML triggering default-deny) were design coincidences, not intentional fixes.

### New system (`vault_pit.py`): Fundamental regression + new issues

During testing, the system was **replaced by a completely redesigned library**. The new system:

| Change | Assessment |
|--------|-----------|
| **PIT snapshot mechanism** (copies admissible files to temp dir) | 🟡 Good idea, but agent can still read real vault |
| **All content scanning removed** | 🔴 **REGRESSION** — no `--content-scan`, no `--strict`, no Unicode normalization |
| **Timeline admission by filename only** | 🔴 No content validation at all |
| **`enforce_pit=False` by default** | 🔴 Opt-in enforcement, like old `--strict` |
| **agent-roles & history always admissible** | 🔴 No PIT checks on these |
| **Concepts default-admissible without `first_observed`** | 🟡 Both permissive and restrictive (unclear) |
| **Domain files (85%+ of vault) not PIT-managed** | 🟢 Actually excluded from PIT entirely |
| **Thread truncation** | 🟢 Good for bullet events, misses inline dates |

---

## Attack Vectors (Old System — `pit_vault_reader.py`)

### 🔴 CRITICAL: `_is_within_vault()` is STILL dead code (G1)

**Reported in RD1 (#9) and RD2 (#2). NOT FIXED.** The symlink escape still works allowing arbitrary file reads.

**Evidence:** `_is_within_vault()` is defined at line 86 but called **0 times** in `_check_file()` (132 lines). Symlink → external file with fake `pit_cutoff` → content returned.

### 🔴 CRITICAL: No sandbox — agent reads any file directly (F3/H1)

**Reported in RD1 (#1) and RD2 (#1). NOT FIXED.** `python3 -c "print(open('...').read())"` bypasses ALL PIT enforcement.

### 🔴 CRITICAL: `--strict` mode is opt-in (A1, A2)

**New finding (old system).** 
- Without `--strict`: HIGH content warnings exist but content is **still returned**
- Agent can simply omit the `--strict` flag
- No enforcement that `--strict` was actually passed

### 🔴 CRITICAL: Content scan is advisory-only without `--strict` (J1)

**New finding (old system).** Files with 3+ HIGH severity warnings pass with `status=ok` and full content returned. Warnings are cosmetic.

### 🟡 HIGH: Content scan patterns are trivially evadable (B1-B4, N1)

**New finding (old system).** Zero warnings for:
- Metaphor: "What the gardener planted did not survive the frost"
- Indirect ref: "What history would record: a nation at the crossroads"
- Conditional: "IF divided opposition persists, THEN plurality victories become the norm"
- Missed patterns: "As time would tell", "Time would prove", "What followed", "The rest is history", "History would judge"

### 🟡 HIGH: Reflection agent bypasses content scan (C1-C4, D1)

**New finding (old system).** 
- Predictive framing → 0 warnings
- Scenario analysis → 0 warnings
- Hypotheticals → 0 warnings  
- "Some argued at the time" → 0 warnings

All convey post-cutoff outcome knowledge while appearing as pre-cutoff analysis.

### 🟡 MEDIUM: `--vault` flag allows complete bypass (I1)

**Reported in RD1 (#10). NOT FIXED.** Agent can point to any directory with no PIT constraints.

### 🟡 MEDIUM: `--strict` only blocks HIGH, MEDIUM leaks (L1)

**New finding (old system).** "culminated in", "subsequently", "what would become" pass `--strict` with content returned.

### 🟡 MEDIUM: Compliance declaration is honor system (E1)

**Reported in RD1 (#8) and RD2 (#8). NOT FIXED.** No verification of declared files read.

---

## Attack Vectors (New System — `vault_pit.py`)

### 🔴 CRITICAL: All content scanning removed (REGRESSION)

The new `vault_pit.py` has **zero content scanning functionality**. No pattern detection, no `--content-scan`, no `--strict`, no Unicode normalization. The expanded pattern set from the RD2 fixes is **entirely absent**.

**Evidence:** File at `harness/vault_pit.py` (249 lines) has no content scanning function. Compare to `pit_vault_reader.py` (932 lines) which had `_check_content_leakage()` with 29 regex patterns.

**Any timeline file that passes the filename-based admission check** can contain any outcome language (Forward traces, "would later", "resolved YES") and it passes through untouched into the PIT snapshot.

### 🔴 CRITICAL: `enforce_pit` is opt-in (default: False)

In `orchestrator.py` line 421:
```python
def run_orchestrated(..., enforce_pit: bool = False) -> ...:
```

When `enforce_pit=False` (the default), **no PIT snapshot is created** — the agent reads directly from the real vault with zero filtering. This is the same vulnerability pattern as the old `--strict` flag and `--content-scan` flag being opt-in.

### 🔴 CRITICAL: Agent can still read real vault directly

The PIT snapshot is created in a temp directory and the prompt tells the agent to use that directory. But:
1. The agent has `read_file` tool with full filesystem access
2. The real vault is accessible at its original path
3. The prompt only TELLS the agent where the vault is — it doesn't restrict the agent

### 🟡 HIGH: Timeline admission is filename-only

`_timeline_admissible()` checks only if `quarter_end <= cutoff`. There is **no content validation**. A file named `2022-Q1.md` passes even if it contains:
- Forward trace annotations
- "would later" / "resolved YES" outcome language
- Post-cutoff dates and events
- Gold-case references

**This is the exact same content-level PIT violation that was CRITICAL in RD1 (#8) and RD2 (#3).**

### 🟡 HIGH: agent-roles & history always admissible

`agent-roles/` and `history/` prefixes always return `True` in `is_path_admissible()`. These files have **no PIT constraints at all**. While they're typically structural/role files, malicious content placed here would be available at any PIT cutoff.

### 🟡 HIGH: Concepts without `first_observed` default-admissible

```python
def _concept_admissible(path, cutoff):
    first = _parse_frontmatter_date(text, _FIRST_OBSERVED_RE)
    if first is not None:
        return first <= cutoff
    return True  # Default: ALLOW
```

Any concept file without a `first_observed` field is **always admissible**. This is like the old "no pit_cutoff = allowed" vulnerability from RD1 (#1).

### 🟡 MEDIUM: Thread truncation only handles one date format

`_parse_bullet_event_date()` only matches:
```
- **Month Day, Year**:
```
Any other date format (inline dates, different bullet styles, narrative temporal references) bypasses truncation entirely.

```python
# This IS truncated:
- **June 10, 2023**: Event C - this is after cutoff

# This is NOT truncated:
By late 2023, the situation had changed.  # ← inline date reference passes through
The March 20, 2023 election results were significant.  # ← not bullet format
```

### 🟢 Partially addressed: Domain files excluded from PIT

Domain-level files (`domains/*/concepts/`, `domains/*/threads/`, etc.) are **not PIT-managed** — they fall through to `return False` in `is_path_admissible()`. This means they're excluded from PIT snapshots entirely, which is actually defensive (they can't leak through the PIT reader). However, the agent can still read them directly via `read_file`.

---

## What Actually Changed (RD2 → RD3 fixes analysis)

| RD2 Attack Vector | Fix Applied? | Status |
|-------------------|-------------|--------|
| **1. Direct read_file bypass** | ❌ Not fixed (old or new system) | Still broken |
| **2. Symlink escape** | ❌ Not fixed (`_is_within_vault()` dead code) | Still broken |
| **3. Advisory-only content scan** | ⚠️ Removed entirely in new system | **Regression** — no scan at all |
| **4. Ancient pit_cutoff (0001-01-01)** | ✅ Fixed (new system uses regex, not YAML) | Fixed |
| **5. 287/475 domain files lack pit_cutoff** | ⚠️ Redesigned — domains excluded from PIT | Different approach |
| **6. Exempt files (agent-roles)** | ❌ Still exempt in new system | Still broken |
| **7. Forward traces in timeline** | ⚠️ Cleaned from timeline files, but no scan in new system | Regression risk |
| **8. Fake early pit_cutoff** | ✅ Fixed (regex-based frontmatter parsing) | Fixed |
| **9. Synonym-based scan evasion** | ❌ Patterns expanded but still trivially evadable | Still broken |
| **10. Zero-width Unicode** | ⚠️ NFKC normalization added (old), removed entirely (new) | Regression |
| **11. YAML anchor/alias** | ✅ Fixed (new system uses regex, not YAML parser) | Fixed |
| **12. Symlink within vault** | ❌ Not fixed | Still broken |
| **13. Inline pit_cutoff** | ⚠️ Different in new system (regex still scans full frontmatter) | Partially fixed |
| **14. URL encoding evasion** | ❌ Never addressed | Still broken |
| **15. Base64 evasion** | ❌ Never addressed | Still broken |
| **NEW: `--strict` is opt-in** | — | New vulnerability |
| **NEW: Content scan all removed** | — | New vulnerability |
| **NEW: `enforce_pit` is opt-in** | — | New vulnerability |
| **NEW: Pattern bypass techniques** | — | 10 new bypasses found |

---

## Methodology

All attacks tested with:
1. `python3 harness/pit_vault_reader.py read/check --cutoff <date> [--content-scan] [--strict]` (old system)
2. `python3 -c "from harness.vault_pit import *"` (new system)
3. Direct file reads via `python3 -c "print(open(...).read())"`
4. Symlink creation inside graph-vault/ to test `_is_within_vault()` dead code

---

## Files Created

- `_specs/pit_rd3_redteam.md` (this report)
- `_specs/pit_rd3_redteam.py` (automated test harness)

---

## Final Verdict

**20/22 attacks against the old system are exploitable. The replacement system (`vault_pit.py`) removes content scanning entirely, creating a net regression.**

The fixes applied after RD2 (expanded patterns, `--strict` mode, Unicode normalization, deep clean) were all **removed** in the architectural rewrite from `pit_vault_reader.py` to `vault_pit.py`. The new system has a better foundation (PIT snapshots!), but removed all content-level enforcement. The core architectural vulnerability remains: **the system relies on agents self-restricting rather than technical enforcement.**
