from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.thesis_optimization_results.src.build_outputs import build_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis optimization tables/figures from run artifacts.")
    parser.add_argument(
        "--config",
        default="results/thesis_optimization_results/configs/analysis/results_pack.yaml",
        help="Analysis config YAML.",
    )
    parser.add_argument(
        "--require-replay",
        action="store_true",
        help="Fail if replay outputs are not available.",
    )
    args = parser.parse_args()

    payload = build_outputs(config_path=args.config, require_replay=bool(args.require_replay))
    print("[scheduling.build_outputs] done")
    print(f"[scheduling.build_outputs] tables={len(payload.get('tables', {}))} figures={len(payload.get('figures', []))}")


if __name__ == "__main__":
    main()
