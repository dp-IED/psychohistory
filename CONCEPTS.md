# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Plugin loop

### Plugin overlay
The loadable instruction surface of this repository: skills, agents, references, and any scripts or tools Parent adds. Training epochs change this overlay. It is not a marketplace cache copy and not a second wiki beside the plugin.
*Avoid:* graph-vault, sidecar vault, inherited corpus

### Reflection
The host tick after a problem’s resolution day. Parent starts the reflector, which grades the whole dated-claim series **and** the overlay system that produced it, then grows or culls the plugin overlay. It is not a predict wakeup and not a weight update.
*Avoid:* vault append, tighten-only patch, grading only the last row, grading claims while skipping scripts

Reflection may add new or rewritten skills, agents, references (including analog case cards), **deterministic scripts/models**, strategies, and instructions when the grade earned a new capability. The overlay is not markdown-only. Later reflection deletes or merges overlay that failed to transfer, including scripts that missed.

### Training epoch
One scored pass whose intended outcome is plugin overlay change, including new files, not a change to foundation model weights.
*Avoid:* fine-tuning run, historical Brier label

### Parent
One host job, one tick. On predict it wakes claim workers for live problems. On reflection it starts the reflector, which edits the plugin. On discovery it opens problems and sets resolution day.

### Claim worker
A consumer of the overlay. It updates Claim and Justification on one live problem, including a Structure block from analog cards, and does not edit the overlay.

### Analog card
A `references/` case card for a discovered class of past cases: mechanism, instantiations, base rate, disanalogy, falsifiers. Written on reflection, consulted on predict. Not a ledger problem and not a historical forecast to score.
*Avoid:* graph-vault, global taxonomy, PIT labels

### Ledger
The Parent’s one schedule book: problems (id, title, motivation, resolution day), dated claims (id, problem, forecast day, owner, claim, justification), and K (max new problems per discover tick).

### Resolution day
The date the world answers the problem. Discover sets it. Predict runs while today is on or before it. Reflection runs after it. Due and Y were this one date. Discover may mix near and far resolution days in one tick; it does not set a date already in the past.

### Forecast day
The date predict wrote a dated claim. How early the outcome showed up.

## Flagged ambiguities

- "Vault" had been used for both the loadable plugin’s durable facts and a sidecar `graph-vault/` corpus — durable facts live in the overlay; do not recreate the sidecar.
