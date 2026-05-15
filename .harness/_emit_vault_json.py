import json
import pathlib

base = pathlib.Path(__file__).resolve().parent.parent
policy = (base / ".harness/policy.md").read_text(encoding="utf-8")
strategy = (base / "vault/_strategy.md").read_text(encoding="utf-8")
vault_files = [
    {
        "path": "vault/approaches/sports-fixtures.md",
        "content": (base / "vault/approaches/sports-fixtures.md").read_text(
            encoding="utf-8"
        ),
    },
    {
        "path": "vault/runs/batch-postmortem-2026-05.md",
        "content": (base / "vault/runs/batch-postmortem-2026-05.md").read_text(
            encoding="utf-8"
        ),
    },
]
out = {
    "policy_markdown": policy,
    "strategy_markdown": strategy,
    "vault_files": vault_files,
}
text = json.dumps(out, ensure_ascii=False) + "\n"
here = pathlib.Path(__file__).resolve().parent
(here / "reflect_bundle.json").write_text(text, encoding="utf-8")
(here / "vault_export.json").write_text(text, encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
