#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _repo_root_from(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path.parents[3]


ROOT = _repo_root_from(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "scripts/results/optimization/pipelines/cluster/run_single_benchmark_case.py"), run_name="__main__")
