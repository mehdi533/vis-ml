#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.results.optimization.scripts.export_replay_trace_panel import *  # noqa: F401,F403

if __name__ == "__main__":
    if "main" in globals():
        main()
