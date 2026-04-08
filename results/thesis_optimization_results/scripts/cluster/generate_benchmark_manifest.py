#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.thesis_optimization_results.src.benchmark_utils import (  # noqa: E402
    generate_scenario_manifest,
    load_benchmark_config,
    load_formulation_suite,
    save_manifest_bundle,
)


def _print_dry_run(summary: dict[str, object], manifest_df) -> None:
    counts = (
        manifest_df.groupby(["scenario_family"], dropna=False)
        .size()
        .rename("n_scenarios")
        .reset_index()
        .sort_values("scenario_family")
    )
    print("[benchmark_manifest] scenario counts by family")
    for _, row in counts.iterrows():
        print(f"  - {row['scenario_family']}: {int(row['n_scenarios'])}")
    print("[benchmark_manifest] selected outages")
    for row in list(summary.get("selected_outages") or []):
        print(
            "  - "
            f"line_uid={int(row['line_uid'])} "
            f"severity_bin={row['severity_bin']} "
            f"severity_score={float(row['severity_score']):.4f}"
        )
    print(f"[benchmark_manifest] main benchmark jobs={int(summary.get('n_main_jobs', 0))}")
    print(f"[benchmark_manifest] cross-method jobs={int(summary.get('n_cross_method_jobs', 0))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic benchmark manifests for thesis optimization/replay.")
    parser.add_argument(
        "--benchmark-config",
        default="results/thesis_optimization_results/configs/thesis_optimization_benchmark.yaml",
        help="Benchmark config YAML.",
    )
    parser.add_argument(
        "--formulation-suite",
        default="results/thesis_optimization_results/configs/suites/01_formulation_comparison.yaml",
        help="Formulation suite YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenario counts, selected outages, and job counts after writing manifests.",
    )
    args = parser.parse_args()

    benchmark_cfg = load_benchmark_config(Path(args.benchmark_config))
    suite_cfg = load_formulation_suite(Path(args.formulation_suite))
    manifest_df, subset_df, severity_df = generate_scenario_manifest(benchmark_cfg)
    summary = save_manifest_bundle(
        manifest_df=manifest_df,
        subset_df=subset_df,
        severity_df=severity_df,
        benchmark_cfg=benchmark_cfg,
        suite_cfg=suite_cfg,
    )

    print("[benchmark_manifest] wrote scenario manifest tables under results/thesis_optimization_results/outputs/tables")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        _print_dry_run(summary, manifest_df)


if __name__ == "__main__":
    main()
