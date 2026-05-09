#!/usr/bin/env python
"""CLI wrapper for ingest.polymarket_resolved."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.polymarket_resolved import main

if __name__ == "__main__":
    raise SystemExit(main())
