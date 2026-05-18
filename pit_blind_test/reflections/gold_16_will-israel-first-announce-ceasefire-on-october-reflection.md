All vault changes are verified. Here is the diagnostic report.

---

## DIAGNOSIS: Gold #16 — "Will Israel first announce ceasefire on October 9?"

### WHAT WENT WRONG

I predicted YES but the answer was NO. The Gaza ceasefire was first announced by Israel on **October 8** (PMO confirmation after Trump's announcement), not October 9. The cabinet **ratified** it on October 9, but ratification is an internal approval process, not an announcement. The Polymarket resolution criteria tracked "the next date (ET) that Israel officially announces it has agreed to a ceasefire" — that was October 8.

**Root cause: Announcement-vs-Ratification Conflation.** The vault correctly documented both dates:
- "Oct 8: Trump announces + Israel PMO confirms"
- "Oct 9: Israeli cabinet approves"

But when forecasting the October 9 question, I used the cabinet ratification date (Oct 9) as if it were the announcement date. These are distinct events in ceasefire diplomacy.

### WHAT THE VAULT HAD RIGHT (after prior fixes)
- The October 2025 ceasefire arc was well-documented (thread, concept, Q4 stub)
- The diplomatic-pressure-tipping-point concept predicted the October timeframe correctly
- The thread noted both October 8 and October 9 events

### WHAT THE VAULT MISSED
1. **No explicit distinction** between "announcement of agreement" and "cabinet ratification"
2. **No annotation** of which date resolves Polymarket-style "first announce" questions
3. **No cross-check step** for date-specific questions — "does the event matching the resolution criteria actually fall on this date?"

### FILES CHANGED

| File | Change |
|------|--------|
| `threads/gaza-ceasefire-negotiations-2025.md` | Added explicit "first official announcement" label to Oct 8, "ratification" label to Oct 9, plus a CRITICAL FOR FORECASTING callout box distinguishing the two |
| `timeline/2025-Q4.md` | Rewrote ceasefire section to clarify PMO confirmation (Oct 8) vs cabinet ratification (Oct 9); added FORECASTING NOTE callout |
| `concepts/diplomatic-pressure-tipping-point.md` | Added "CRITICAL SEQUENCE DETAIL" section on announcement vs ratification to the Gaza ceasefire canonical example; added Oct 9 entry to Validated By table |
| `_procedure.md` | Added sub-step to step 9 (Track diplomatic signals) requiring all three dates (announcement, ratification, effective) to be documented separately for ceasefire forecasts |
| `forecasts/2026-05-18-gaza-ceasefire-october-9.md` | Created new forecast-entry documenting the error with diagnosis and vault gaps |

### PATTERN TO REMEMBER
When a question asks about a **specific date** for a ceasefire announcement:
1. Identify the **first official announcement** by the party in question (executive/PMO statement)
2. Separately identify the **ratification/approval** date (cabinet/parliament vote)
3. Separately identify the **effective date** (when fighting actually stops)
4. Verify which of these three the question's resolution criteria refers to — they are NOT interchangeable