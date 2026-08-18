# Reflection writes plugin and vault; the host owns the ticker

After `Y`, reflection updates skills/agents/references and the vault graph (entities, threads, concepts). GNN analysis of that graph is deferred. Parent creates the vault directory on first write; this branch does not ship a pre-filled corpus.

Scheduling is the consuming harness (`/loop`, `/automate`, or equivalent). Do not add a repo-owned daemon.

Claims live in a **single ledger** markdown file. Parent reads it and wakes agents due today. Each row has a **justification** (reasoning trace), not a second clock.
