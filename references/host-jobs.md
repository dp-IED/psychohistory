# Host jobs

Cursor dashboard automations already run the ticker: predict daily, reflect daily later, discover weekly. This file is the **in-repo contract** those jobs follow. Do not add a repo daemon. Do not treat ticker setup as unfinished work.

Procedure lives in `skills/` and `agents/` on `harness-only`. Dashboard prompts should point at those files rather than embed an old skill copy.

## Contract

- Repo: `dp-IED/psychohistory`. Live branch: `harness-only`.
- Fast-forward pull first. If git is dirty or pull fails, stop.
- Exactly one tick per run. Do not combine predict, reflect, and discover.
- Commit and push `harness-only` only. **Do not open a pull request.** Cloud-agent defaults that create PRs do not apply. Do not call `ManagePullRequest`.
- Cursor loads Parent via `.cursor/agents/parent.md`. Claude Code: `claude --plugin-dir .`.
- Predict: `skills/predict/SKILL.md` (spawn `claim-worker`).
- Reflect: Parent starts `agents/reflector.md`, which follows `skills/reflect/SKILL.md`. May start one automation on `exp/<slug>` (max three `exp/` branches; ADR 0016). Experiment runs do not spawn children.
- Discover: `skills/discover/SKILL.md`, `references/discovery.md`, `references/party-sources.md`. `K` is a cap. No claim or justification.

## In-repo names that changed after the automations were first written

Skills on `harness-only` are current. A dashboard prompt that still uses the old names should follow the current skill, not the old word:

| Old | Current |
| --- | --- |
| due-today | predict |
| Due and Y as two clocks | one **resolution day** on the problem; **forecast day** on each claim |
| wake only rows due today | live problems: resolution day on or after today |
| Parent grades / edits on reflect | Parent starts **reflector** |
| fill `K` / a sports or sociology quota | `K` is a cap; anti-cluster is a brake; party-sourced orientation |
| open a pull request | push `harness-only` only |
| write claims on discover | Motivation + resolution day only |
