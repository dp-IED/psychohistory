"""Delegates to `_emit_vault_json.py`, which mirrors this bundle layout."""

from __future__ import annotations

import pathlib
import runpy

if __name__ == "__main__":
    emit = pathlib.Path(__file__).resolve().with_name("_emit_vault_json.py")
    runpy.run_path(str(emit), run_name="__main__")
