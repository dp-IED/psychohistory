## Reflection: pm_pol_04 — Correct but Vault Had Critical Gap

### Diagnosis

The prediction was correct, but the vault provided **zero signal** for this question. The blind-built quarter summaries completely omitted US domestic politics:

- No mention of Biden, Trump, Harris, or the 2024 presidential election
- No mention of Trump's May 30 felony conviction (first former US president convicted)
- No mention of the June 27 debate and its aftermath
- No coverage of the VP selection process

**Root cause:** The subagent that built the summaries prioritized international events (India election, Mexico, European Parliament, South Africa) over US domestic politics. This is a systemic bias in the blind-build process — without knowing what questions are coming, the builder defaults to "internationally significant" framing, which de-prioritizes US domestic stories that are globally consequential.

**Impact:** This is the most important gap found so far because it's a structural limitation of the blind-building method, not just an omission. Three more questions in the pilot (pm_cul_01, pm_cul_02) also rely on US/North American cultural context that the summaries might have covered differently.

### Files Created

- `threads/2024-us-presidential-election.md` — Thread tracking the full 2024 election cycle from the campaign through the election.
- `entities/joe-biden.md` — Entity stub for the 46th president.
- `entities/donald-trump.md` — Entity stub for the 45th/47th president.
- `entities/kamala-harris.md` — Entity stub for the 49th VP and 2024 Democratic nominee.
- `entities/tim-walz.md` — Entity stub for the 2024 Democratic VP nominee.

### Files Modified

- `timeline/2024-Q2.md` — Added US presidential election section covering the Trump conviction (May 30), the June 27 debate, and the VP selection context.
- `timeline/2024-Q1.md` — Added US election primary coverage.

### _forecast_instructions.md Update

**Rule 2 added: Domestic politics gap check** — Before forecasting any question about US politics, elections, or leadership changes, verify that the vault's quarter summaries cover US domestic politics for the relevant quarters. The blind-build process systematically under-covers US domestic stories in favor of international events. If US domestic coverage is missing, note the gap explicitly and use general knowledge to supplement.

### _spec.md Update

**Principle added:** US domestic politics is mandatory coverage in every contemporary quarter summary. The blind-build process's international-event bias must be explicitly corrected.
