# Ledger at repo root; K is a knob; host jobs are deferred

The schedule book is `ledger.md` at the repository root, not inside the plugin overlay. Promoted facts go into the overlay on reflection (ADR 0007); the ledger stays the Parent’s one read.

`K` (new problems per discover tick) lives on the ledger; start at 1. Predict vs discover are two host jobs (`/loop` or `/automate`).
