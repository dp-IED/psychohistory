import json
import pathlib

base = pathlib.Path(__file__).resolve().parent.parent
policy = (base / ".harness/policy.md").read_text()
strategy = (base / "vault/_strategy.md").read_text()
vault_files = [
    {
        "path": "vault/approaches/sports-fixtures.md",
        "content": (base / "vault/approaches/sports-fixtures.md").read_text(),
    },
    {
        "path": "vault/runs/batch-postmortem-2026-05.md",
        "content": (base / "vault/runs/batch-postmortem-2026-05.md").read_text(),
    },
]
out = {
    "policy_markdown": policy,
    "strategy_markdown": strategy,
    "vault_files": vault_files,
}
print(json.dumps(out, ensure_ascii=False))
