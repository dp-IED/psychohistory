## REPORT: What Was Changed and Why

### Diagnosis: Why I Predicted NO (Wrong)

This was the 6th wrong prediction out of 19 questions -- and the 6th consecutive NO prediction that should have been YES. The error had three layers:

**1. Resolution-criteria ambiguity (primary cause)**: I interpreted "wins the election" as "assumes office" (the political-power outcome). Polymarket resolves it as "receives the most votes" (the electoral outcome). In authoritarian contexts, these diverge dramatically -- González got ~67% of the vote per parallel tabulation, but Maduro prevented him from taking office. The market still resolved YES because González won the vote.

**2. Unknown opposition vote-monitoring infrastructure (enabling cause)**: The PUD coalition's ConVzla operation collected tally sheets from 81% of polling centers. This parallel vote tabulation made the true outcome knowable beyond dispute, regardless of the CNE's fabricated results. I was unaware of this capability.

**3. Zero vault coverage (structural cause)**: The vault had no entity stubs for any Venezuelan actor, no thread on Venezuela politics, and no concept for authoritarian electoral dynamics. There was no infrastructure to support careful reasoning about the specific mechanisms at play.

### Files Created (6 new files)

| File | Type | Purpose |
|------|------|---------|
| `entities/edmundo-gonzalez.md` | Entity | Opposition candidate who won the vote |
| `entities/nicolas-maduro.md` | Entity | Incumbent who maintained power despite losing |
| `entities/maria-corina-machado.md` | Entity | Barred opposition leader whose endorsement was decisive |
| `entities/plataforma-unitaria.md` | Entity | PUD coalition + ConVzla parallel vote tabulation operation |
| `threads/venezuela-authoritarian-resilience.md` | Thread | Tracks Venezuela's political dynamics, regime control mechanisms, opposition strategy (status: active) |
| `concepts/authoritarian-electoral-facade.md` | Concept | Framework for forecasting opposition victory in rigged-authoritarian elections: parallel vote tabulation, institutional control, distinguishing vote outcome from office assumption |
| `forecasts/2026-05-18-venezuela-election-gonzalez.md` | Forecast entry | Full documentation of prediction, error diagnosis, vault gaps, and changes made |

### Files Updated (4 modified)

| File | Key Changes |
|------|-------------|
| `_spec.md` | Added Principle 15: Mandatory resolution-criteria clarity for authoritarian election questions -- check resolution text, create entity stubs for all actors, assess PVT infrastructure, apply authoritarian-electoral-facade concept |
| `_procedure.md` | Added Step 16 to Pre-Forecast Audit: winning-the-vote vs assuming-office distinction with 5-part checklist. Added authoritarian-election sub-pattern to status-quo bias pitfalls. Added 2 new lessons to "Lessons from Cycle 4" |
| `timeline/2024-Q3.md` | Revised Venezuela entry: explicitly states González won the vote (~67% to ~30%), details ConVzla operation from 81% of polling centers, notes Carter Center/UN/OAS rejection, adds wiki-links to new thread + concept |
| `_index.md` | Added Cycle 19 entry documenting the Venezuela domain transition from 0% to foundational coverage |

### Key Systemic Insight

The Venezuela error exposed a new failure mode not previously captured: **resolution-criteria ambiguity in election questions**. The vault already had:
- Status-quo bias pitfall (6/6 wrong predictions are NO)
- "Dual-frame analysis" spec principle
- Named-entity completeness spec principle

But none of these addressed the specific trap where "wins" can mean two different things in authoritarian contexts. The new spec principle (15) and procedure step (16) close this gap. The vault now has a systematic process for checking resolution criteria before forecasting any election question.