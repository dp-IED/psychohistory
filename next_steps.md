# Next steps

This branch is a clean skills/subagent training bed. The hermes v1 runner is labeled TEMPORARY. Do not extend it.

## Next session (product)

1. **Skill/subagent interface** — inputs: question, cutoff `t`, admissible vault paths; outputs: `p_yes` (or typed forecast), reasoning, evidence pointers. No hermes import.
2. **Replace `harness/orchestrator.py`** — keep `run_structured` / `run_orchestrated` signatures only if a thin adapter helps; the implementation should call skills/subagents.
3. **Eval on existing probes** — `data/pit_market_probes/` (vault vs ablated), seed30 blind artifacts, hypotheses in `data/polymarket/hypotheses.json`.
4. **Tournament watcher** — rewrite later from scratch; v2 cognitive pipeline was deleted on purpose.

## Do not pick up

- Graph builder / WM / France GNN (`docs/history/research-track.md`)
- `orchestrator_v2` / outside-view / GNN tag calibration (deleted)
- Merging this branch into `origin/main` until you choose to
- Conductor worktrees (`testbed`, `gnn-agent`) — left as artifacts

## Scripts inventory (current)

- Vault/PIT: `validate_vault.py`, `vault_relevance_probe.py`, `pit_market_calibration.py`, `pit_phrasing_scan.py`, `thread_continuity_audit.py`, `bootstrap_timeline.py`, `pit_train.py`
- Temporary hermes: `run_backtest.py`, `run_chain.py`, `batch_chain.sh`
- Polymarket data: `fetch_polymarket_resolved.py`, `build_polymarket_branch_graphs.py`, `build_polymarket_gold_dataset.py`
