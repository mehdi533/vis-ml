from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "results" / "thesis_optimization_results"
CONFIGS_DIR = ANALYSIS_ROOT / "configs"
NOTEBOOKS_DIR = ANALYSIS_ROOT / "notebooks"
SRC_DIR = ANALYSIS_ROOT / "src"
RESULTS_DIR = ANALYSIS_ROOT / "results"
OUTPUTS_DIR = ANALYSIS_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MERGED_RESULTS_DIR = OUTPUTS_DIR / "merged_results"

BASE_OPTIMIZATION_CONFIG = CONFIGS_DIR / "base_optimization.yaml"
FORMULATION_SUITE_CONFIG = CONFIGS_DIR / "suites" / "formulation_comparison.yaml"
REPLAY_CONFIG = CONFIGS_DIR / "replay" / "replay_validation.yaml"
ANALYSIS_CONFIG = CONFIGS_DIR / "analysis" / "results_pack.yaml"

FORMULATION_SUMMARY_CSV = RESULTS_DIR / "formulation_comparison_summary.csv"
FORMULATION_SUMMARY_JSON = RESULTS_DIR / "formulation_comparison_summary.json"
REPLAY_SUMMARY_CSV = RESULTS_DIR / "replay_validation_summary.csv"
REPLAY_DETAIL_CSV = RESULTS_DIR / "replay_validation_detail.csv"


class OptimizationResultsNotReadyError(RuntimeError):
    """Raised when the expected optimization or replay artifacts are not ready yet."""


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
