# Gap 5: Thread Continuity Auditor

## Purpose

Spec Rule 21 (thread continuity is mandatory) says: "All threads with `status: active` MUST be updated in each subsequent quarter file where relevant events occur. Creating a thread and then failing to maintain it is a structural vault failure." But there's no automated check. This spec builds a script that audits thread continuity and flags stale threads.

## What to Build

A standalone script `scripts/thread_continuity_audit.py` that:
1. Scans all files in `graph-vault/threads/` and reads their frontmatter
2. For threads with `status: active` (or no status — treat as active), checks which timeline quarter files reference them
3. Reports threads that are stale (not mentioned in the most recent 2+ quarters)

### Frontmatter Fields

Every thread file has frontmatter like:
```yaml
---
type: thread
title: "..."
slug: iran-israel-escalation
inception: 2024-04-01
pit_body_cutoff: 2025-06-22
conclusion: 2025-06-24
status: resolved
tags: [middle-east, state-conflict, escalation, ceasefire-mediation]
---
```

Thread status values:
- `active` — thread is ongoing, should be updated in each quarter
- `fading` — thread is winding down, may skip some quarters
- `resolved` — thread has concluded, no updates needed
- Absent/no status — treat as active (conservative)

### Thread-Quarter Linkage

A thread is "mentioned in" a timeline quarter file if:
- The quarter file contains a wikilink to the thread: `[[threads/slug]]` or `[[slug]]`
- The thread file contains a wikilink to the quarter: `[[2024-Q2]]`
- Either direction is sufficient

### Audit Logic

For each thread:

1. **Read frontmatter**: Extract `status`, `inception`, `pit_body_cutoff`, `conclusion`, and all `[[wikilinks]]` from the body.

2. **Collect linked quarters**: Find all timeline quarter wikilinks (matches pattern like `[[2024-Q1]]`, `[[2024-Q2]]`, etc.) in the thread body.

3. **Cross-reference**: Search all timeline quarter files for wikilinks back to this thread. Use regex `\[\[slug\]\]` or `\[\[threads/slug\]\]` or `\[\[threads/slug|.*\]\]`.

4. **Check continuity**: 
   - Find the last quarter the thread was mentioned in (most recent linked quarter date)
   - Find the most recent timeline quarter file that exists
   - Compute gap = (most recent quarter end date) - (last linked quarter end date), in quarters
   - If gap >= 2 and status == "active": flag as **STALE** (severe)
   - If gap >= 3 and status == "fading": flag as **STALE** (moderate)
   - If status == "resolved" and `conclusion` date exists, check that the concluding quarter is the last linked quarter: flag as **INCONSISTENT** if it's not
   - If status is absent: flag as **NO_STATUS** (minor — suggest setting it)

### Output Format

```
Thread Continuity Audit — {date}
Scanned {N} threads

=== STALE ACTIVE THREADS (must fix) ===
  iran-israel-escalation (status=active)
    Last updated: 2024-Q4 (2024-12-31)
    Most recent quarter: 2025-Q3 (2025-09-30)
    Gap: 3 quarters — UPDATE REQUIRED
    Linked quarters: 2024-Q2, 2024-Q3, 2024-Q4

=== STALE FADING THREADS (check if still alive) ===
  (none)

=== INCONSISTENT RESOLVED THREADS ===
  gaza-ceasefire-negotiations-2025 (status=resolved, conclusion=2025-10-10)
    Last linked quarter: 2025-Q2
    Concluding quarter should be 2025-Q3 or 2025-Q4 — thread not linked from any quarter file after 2025-Q2

=== THREADS MISSING STATUS ===
  (none)

=== THREADS WITH OK COVERAGE ===
  israel-iran-shadow-war-gaza-2023-2024 (status=resolved)
    Linked in: 2024-Q1, 2024-Q2 ✓
```

### CLI

```
python scripts/thread_continuity_audit.py

# Only show issues (no "ok" section)
python scripts/thread_continuity_audit.py --issues-only

# Output JSON
python scripts/thread_continuity_audit.py --json

# Auto-fix: set status=fading for threads stale >= 2 quarters with no status
python scripts/thread_continuity_audit.py --auto-fix-stale
```

### Integration

The auditor should be callable from:
1. `pit_reflect.py` — add an `--audit-threads` flag that runs the continuity auditor as part of the reflection cycle
2. The vault relevance probe harness (`scripts/vault_relevance_probe.py`) — thread continuity is a natural extension of the relevance check

## Files to Create

- `scripts/thread_continuity_audit.py`

## Files to Modify

- `scripts/pit_reflect.py` — optional `--audit-threads` flag
- `scripts/vault_relevance_probe.py` — optional `--thread-audit` flag (lower priority)

## Test

1. Run `python scripts/thread_continuity_audit.py` against the current vault
2. Manually verify results for a known-stale thread
3. Run `python scripts/thread_continuity_audit.py --issues-only` — verify only issues shown
4. Run `python scripts/thread_continuity_audit.py --auto-fix-stale` — verify frontmatter changes are correct (dry-run first or use git to diff)
