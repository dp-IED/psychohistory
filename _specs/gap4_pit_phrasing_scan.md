# Gap 4: Automated PIT-Phrasing Scan

## Purpose

The PIT vault library (`vault_pit.py`) filters files by frontmatter dates (pit_cutoff, inception, pit_body_cutoff, timeline quarter dates). But even files that pass the date filter can contain phrases that smuggle in post-cutoff knowledge — e.g., "this ultimately led to [future event]" or "would later prove decisive." This is known to be a problem: pit_reflect's anti-leakage rules (Rule 9) explicitly tell the forecaster not to treat vault paragraphs describing outcomes as proof. An automated scan catches these before they reach the forecaster.

## What to Build

A standalone script `scripts/pit_phrasing_scan.py` that scans vault markdown files for retrocausal/leakage phrasing patterns.

### Detection Patterns

Scan each non-root vault .md file flagged as admissible for a given cutoff (threads/, concepts/, entities/, timeline/ — NOT _forecast_instructions.md, _procedure.md, _spec.md, _index.md, runs/, forecasts/, meta/).

Flag lines containing these patterns (configurable list):

**Category 1: Explicit retrocausal framing**
- "ultimately led to" / "ultimately lead to" / "ultimately proved"
- "would later" (as in "would later become" / "would later prove")
- "would go on to"
- "in hindsight" / "in retrospect" / "looking back"
- "set the stage for" / "paved the way for" / "laid the groundwork for"
- "foreshadowed" / "presaged" / "portended"

**Category 2: Conclusion/outcome framing (in threads with status: resolved)**
- For files with `conclusion` in frontmatter, watch for "The thread concluded..." / "The conflict ended with..." language that treats the outcome as inevitable

**Category 3: Temporal inconsistency with cutoff**
- Reference to a date that's later than the file's `pit_cutoff` or `pit_body_cutoff` (if present) or `conclusion` date
- E.g., "As of 2025, the situation remains..." when the file has `pit_cutoff: 2024-03-31`

**Category 4: Predictive certainty about the future**
- "was certain to" / "was inevitable that"
- "clearly heading toward"
- "the only possible outcome was"
- "increasingly clear that [future event] would"

### Output Format

Print a report to stdout:

```
PIT Phrasing Scan — cutoff={cutoff}
Scanned {N} files, {M} flagged

== Category 1: Retro causal framing (6 matches) ==
  threads/iran-israel-escalation.md:23 - "this ultimately led to full-scale war in June 2025"
  concepts/leadership-persistence-under-threat.md:45 - "would later become a defining pattern"

== Category 2: Conclusion/outcome framing (2 matches) ==
  threads/gaza-ceasefire-negotiations-2025.md:12 - "The conflict ended with..."

== Category 3: Temporal inconsistency (1 match) ==
  timeline/2024-Q1.md:89 - "As of 2025" (pit_cutoff=2024-03-31)

== Category 4: Predictive certainty (0 matches) ==
  (none)
```

### CLI Usage

```
# Scan entire vault (no cutoff filter — just scan all files for patterns)
python scripts/pit_phrasing_scan.py

# Scan with a cutoff — only scan files admissible at cutoff, check temporal consistency
python scripts/pit_phrasing_scan.py --cutoff 2024-06-30

# Output JSONL for machine consumption
python scripts/pit_phrasing_scan.py --jsonl > pit_phrasing_report.jsonl
```

### Integration

The scan should be callable from `pit_reflect.py` — after running a batch of probes, scan the vault and include any new leakage findings in the reflection feedback. Add an optional `--pit-scan` flag to `pit_reflect.py`.

### Configuration File

Store pattern lists in a YAML config file at `data/pit_scan_config.yaml` so patterns can be added/removed without code changes. Default location: `data/pit_scan/phrasing_patterns.yaml`.

```yaml
patterns:
  retrocausal:
    - phrase: "ultimately led to"
      severity: high
    - phrase: "would later"
      severity: high
    - phrase: "set the stage for"
      severity: medium
    ...
  conclusion_framing:
    - phrase: "The conflict ended with"
      severity: medium
    ...
  temporal_inconsistency:
    enabled: true
  predictive_certainty:
    - phrase: "was certain to"
      severity: high
    ...
```

## Files to Create

- `scripts/pit_phrasing_scan.py` — main CLI script
- `data/pit_scan/phrasing_patterns.yaml` — default pattern config

## Files to Modify

- `scripts/pit_reflect.py` — optional `--pit-scan` flag to include scan results in reflection

## Test

1. Run `python scripts/pit_phrasing_scan.py` — verify it produces a report
2. Manually verify a few flagged lines are genuinely retrocausal
3. Run with `--cutoff 2024-03-31` — verify temporal inconsistency detection works
4. Check false positive rate on _forecast_instructions.md (should be excluded from scanning)
