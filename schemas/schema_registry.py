"""
Load a GraphSchema from a Python module path or file path.

Examples:
  schemas.base_schema
  /abs/path/to/custom_schema.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from schemas.schema_types import GraphSchema


def _load_module_from_file(path: Path) -> ModuleType:
    path = path.resolve()
    name = f"_schema_dyn_{hash(str(path)) % (10**9)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load schema module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_graph_schema(spec: str) -> GraphSchema:
    """
    Resolve a GraphSchema instance.

    Resolution order:
    1. If `spec` is an existing file path -> import and look for get_seed_graph_schema or GRAPH_SCHEMA.
    2. Else treat as dotted module name -> import and look for get_seed_graph_schema or GRAPH_SCHEMA.
    """
    p = Path(spec)
    mod: ModuleType
    if p.suffix == ".py" and p.exists():
        mod = _load_module_from_file(p)
    else:
        mod = importlib.import_module(spec)

    if hasattr(mod, "get_seed_graph_schema") and callable(mod.get_seed_graph_schema):
        return mod.get_seed_graph_schema()
    if hasattr(mod, "GRAPH_SCHEMA"):
        gs = mod.GRAPH_SCHEMA
        if isinstance(gs, GraphSchema):
            return gs
    raise AttributeError(
        f"Schema module {spec!r} must export get_seed_graph_schema() or GRAPH_SCHEMA: GraphSchema"
    )
