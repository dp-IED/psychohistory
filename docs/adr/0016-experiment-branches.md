# Tiny live edits; risky experiments on branches; live reflect may start a host run

Open-ended evolution: after a score, reflection may propose new skills, agents, tools, and organisation (ADR 0007). That does not make the reflector the daily job router. The **harness** still starts predict / discover / reflect (ADR 0001).

**Live plugin:** the branch daily ticks use (`harness-only`). Tiny overlay edits (wording, a card line) may commit there.

**Experiment branch:** a git branch for a new agent, skill, script, or organisation. Live predict stays on the live plugin until the experiment **transfers** on later scored problems.

**What the experiment run does:** the reflector chooses per experiment (same live questions, load-only check, extra questions, etc.). Record that choice on the branch (commit message and/or branch slug). There is no single global experiment tick.

**How experiments are tracked:** git branches with prefix `exp/` (locked; example `exp/claim-critic`). Live reflect lists `exp/*`, starts/merges/deletes them. No sidecar experiment database. The prefix is the index.

**Who starts the experiment run:** the **reflection host job** (a Cursor automation / `/automate` / `/loop` run), autonomously: create/push the `exp/…` branch, then start a host run **on that branch**. This repo does not ship a daemon. The reflect skill *instructs* the already-running host agent. If the host has no way to start another automation, stop and say so; do not fake an orchestrator in-tree.

**Spawn bound (until superseded):** only a reflect job on the **live** plugin may start experiment automations. A run already on an experiment branch does not start further experiments. At most **three** `exp/` branches at once; if at cap, cull or merge before opening another.

Merge or delete the branch from a later **live** reflect tick. Do not require a PR for daily live ticks; experiment merge is still the live reflect writer (or a host rule it follows).

Does not restore an in-repo orchestrator. Does not change predict/discover to invent branches.
