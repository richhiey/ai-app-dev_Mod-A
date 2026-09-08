#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_server import mcp


if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run()
    else:
        raise SystemExit("Run with: mcp run src/mcp_server.py")
