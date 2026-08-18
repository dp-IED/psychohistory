# Reflection writes plugin and vault; the host owns the ticker

Superseded in part by ADR 0007: reflection writes only the plugin. Do not create `graph-vault/`.

Scheduling is the consuming harness (`/loop`, `/automate`, or equivalent). Do not add a repo-owned daemon.

Claims live in a **single ledger** markdown file. Parent reads it and wakes agents due today. Each row has a **justification** (reasoning trace), not a second clock.
