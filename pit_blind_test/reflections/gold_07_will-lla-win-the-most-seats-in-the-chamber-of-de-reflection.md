## Report: What Changed and Why

### Diagnosis

This prediction was correct because the Argentina legislative domain is fully saturated (5+ questions across all test runs). The vault already had the complete seat counts, the causal thread, entity stubs for all actors, and multiple cross-national concepts. The prediction was a lookup, not a reasoning problem.

### What I changed (abstraction shift per Spec Rule 48)

Since the domain is saturated, I created zero new Argentina-specific content and shifted entirely to cross-national abstraction:

**1. CREATED `domains/global/concepts/presidential-coattail-variability/_concept.md`**

This was a broken wikilink — the `legislative-plurality-forecast` procedure referenced `[[presidential-coattail-variability]]` but it didn't exist. The concept fills this gap with a cross-national framework explaining why coattail magnitude varies across:
- Electoral systems (PR national → regional PR → MMP → FPTP)
- Timing (same-cycle positive vs. midterm weak/negative)
- Political context (approval, governance wins, opposition fragmentation, baseline, voter volatility)

Includes a coattail magnitude estimation heuristic and validation against the LLA 2025 case.

**2. UPDATED `domains/latin-america/concepts/radical-reformer-political-survival/_concept.md`**

This concept was referenced in reasoning chains but had no "Validated By" table — a spec violation. Added two validation entries:
- Milei survival forecast (NO → correct): direct test of all 5 survival mechanisms
- LLA legislative forecast (YES → correct): indirect validation — downstream consequence of survival

**3. UPDATED `domains/global/procedures/legislative-plurality-forecast.md`**

Added **Appendix A: D'Hondt Seat Projection Method** with:
- Python calculator functions (`d_hondt_district`, `d_hondt_national`)
- Common pitfalls: threshold waste, district magnitude, coalition effects, 1-seat distortion
- Argentina 2025 worked example
- Added the repeat LLA instance to the validation record table

**4. CREATED `meta/reflections/_reflection-2026-05-20-per-q7-lla-seats.md`**

Per-question reflection documenting the saturated domain state and full compliance with Rule 48 (abstraction-over-coverage).

**5. UPDATED `_index.md`**

Added Q7/30 entry documenting all changes.

### Key lesson

When a domain reaches saturation (thread + entities + concepts all exist), the vault's reflex to create domain-specific content must be overridden. The spec provides this override (Rule 48), but it requires active enforcement. Every abstraction created in this pass is a generic tool for future questions about different countries (Brazil 2026, Spain 2028, Netherlands 2027, etc.).