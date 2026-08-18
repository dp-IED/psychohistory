# Ship a harness-agnostic plugin; do not replace `orchestrator.py` as the product

This repo’s durable shape is a Claude-style plugin (`skills/`, `agents/`, `references/`, `scripts/`). Cursor, Claude Code, Codex, and other harnesses load it and own orchestration. `harness/orchestrator.py` stays a temporary leftover, not the thing to rewrite into a portable Python runner with an injected LLM port.

**Considered options**: (1) typed Python skill/subagent interface plus thin `run_structured` adapter; (2) this plugin layout. (1) was rejected because orchestration and LLM execution belong to the consuming harness.
