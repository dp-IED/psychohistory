---
name: predict
description: Parent predict tick — write claims on live problems; append a revision when the outcome changes.
disable-model-invocation: true
---

# Predict

Run this from the repository root. Cursor: `.cursor/agents/parent.md`. Claude Code: `claude --plugin-dir .`. Live-tick contract: `references/host-jobs.md`. Today is the session calendar date. This tick is predict, not discover and not reflect. Formerly due-today.

## Steps

1. Pull the current `harness-only` branch with fast-forward only. If git is dirty from another tick or pull fails, stop. Completion: you are on a clean `harness-only` matching origin.
2. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Completion: you have `K`, every problem motivation, every **resolution day**, and every dated claim.
3. Select live problems: `ledger.live_problems(as_of)` where `as_of` is today. Completion: every selected problem has **resolution day** on or after today. Do not touch problems past resolution. Do not open new problems.
4. For each live problem, spawn the subagent `claim-worker` in this repo root. Pass problem id, resolution day, motivation, and the latest dated claim if any (id, forecast day, claim, justification). Completion: every live problem has been handed to a worker.
5. After workers return, re-read `ledger.md`. Completion: each live problem either has a new dated claim dated today whose **claim** (outcome line) differs from the previous row, or it has no new row because the outcome did not change. No new problem headings. No overwrite of old rows. An existing problem’s **Resolution** may move later when a published instrument has already named a later answering date; that is the same problem, not a discover open.
6. If the ledger changed, commit and push to `harness-only`. Do not open a pull request. Completion: origin has the commit, or you stopped because nothing changed.

This tick writes forecasts. Workers follow `references/structure.md` for the Structure block and `references/analog-prior.md` when a card matches. On an administrative tariff or trade clock they read `references/cases/tariff-proclamation-deadline-delay.md`, score delay separately from a signed deal, and treat an already-extended / post-pause date as take-effect unless a new delay instrument appears. If that delay names a new public date, slide this problem’s **resolution day** later so the post-pause phase stays live and can be scored. The day’s claim is then that new clock’s modal (take-effect unless a further delay instrument exists), not a pause recap. Open problems on a discover tick; that procedure is `references/discovery.md`. Do not invent a second heading for a short pause of a clock that is still the same question.
