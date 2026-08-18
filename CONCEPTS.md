# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Plugin loop

### Plugin overlay
The loadable instruction surface of this repository: skills, agents, references, and any scripts or tools Parent adds. Training epochs change this overlay. It is not a marketplace cache copy and not a second wiki beside the plugin.
*Avoid:* graph-vault, sidecar vault, inherited corpus

### Reflection
The host tick after a claim’s resolution date. Parent grades the claim and the justification, then grows or culls the plugin overlay. It is not a due-today wakeup and not a weight update.
*Avoid:* vault append, tighten-only patch

Reflection may add new skills, agents, tools, strategies, and instructions when the grade earned a new capability. Later reflection deletes or merges overlay that failed to transfer.

### Training epoch
One scored pass whose intended outcome is plugin overlay change, including new files, not a change to foundation model weights.
*Avoid:* fine-tuning run, historical Brier label

### Parent
The overlay writer. On due-today it wakes claim owners. On reflection it edits the plugin. On discovery it opens problems. One host job runs exactly one of those ticks.

### Claim worker
A consumer of the overlay. It updates Claim and Justification on one due ledger row and does not edit the overlay.

### Ledger
The Parent’s one schedule book: problems with motivation, and dated claims with due date, optional resolution date, owner, claim, and justification.

### Y
The resolution date on a dated claim. Reflection selects claims whose Y is strictly before today and that are not due today. If Y is omitted on the ledger, it equals Due.

## Flagged ambiguities

- "Vault" had been used for both the loadable plugin’s durable facts and a sidecar `graph-vault/` corpus — durable facts live in the overlay; do not recreate the sidecar.
