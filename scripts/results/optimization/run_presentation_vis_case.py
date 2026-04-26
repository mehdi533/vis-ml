#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _repo_root_from(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path.parents[2]


ROOT = _repo_root_from(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.results.optimization.presentation_vis_pipeline import main


if __name__ == "__main__":
    main()
