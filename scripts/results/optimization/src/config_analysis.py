from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "results" / "thesis_optimization_results"
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs" / "optimization"
DEFAULT_GENERATED_CONFIGS_DIR = ANALYSIS_ROOT / "configs" / "generated"
DEFAULT_RESULTS_DIR = ANALYSIS_ROOT / "results"
DEFAULT_OUTPUTS_DIR = ANALYSIS_ROOT / "outputs"


def _resolve_env_path(name: str, default: Path) -> Path:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


CONFIGS_DIR = _resolve_env_path("THESIS_OPT_CONFIGS_DIR", DEFAULT_CONFIGS_DIR)
GENERATED_CONFIGS_DIR = _resolve_env_path(
    "THESIS_OPT_GENERATED_CONFIGS_DIR",
    DEFAULT_GENERATED_CONFIGS_DIR,
)
NOTEBOOKS_DIR = ANALYSIS_ROOT / "notebooks"
SRC_DIR = ANALYSIS_ROOT / "src"
RESULTS_DIR = _resolve_env_path("THESIS_OPT_RESULTS_DIR", DEFAULT_RESULTS_DIR)
OUTPUTS_DIR = _resolve_env_path("THESIS_OPT_OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR)
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MERGED_RESULTS_DIR = OUTPUTS_DIR / "merged_results"
PLOT_DATA_DIR = OUTPUTS_DIR / "plot_data"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmark"
BENCHMARK_MANIFESTS_DIR = BENCHMARK_RESULTS_DIR / "manifests"
BENCHMARK_GENERATED_CONFIGS_DIR = BENCHMARK_RESULTS_DIR / "generated_configs"
BENCHMARK_MAIN_RESULTS_DIR = BENCHMARK_RESULTS_DIR / "main"
BENCHMARK_CROSS_RESULTS_DIR = BENCHMARK_RESULTS_DIR / "cross_method_subset"
BENCHMARK_REPLAY_RESULTS_DIR = BENCHMARK_RESULTS_DIR / "replay"

BASE_OPTIMIZATION_CONFIG = _resolve_env_path(
    "THESIS_OPT_BASE_CONFIG",
    CONFIGS_DIR / "base_optimization.yaml",
)
FORMULATION_SUITE_CONFIG = _resolve_env_path(
    "THESIS_OPT_FORMULATION_SUITE_CONFIG",
    CONFIGS_DIR / "suites" / "01_formulation_comparison.yaml",
)
REPLAY_CONFIG = _resolve_env_path(
    "THESIS_OPT_REPLAY_CONFIG",
    CONFIGS_DIR / "replay" / "replay_validation.yaml",
)
ANALYSIS_CONFIG = _resolve_env_path(
    "THESIS_OPT_ANALYSIS_CONFIG",
    CONFIGS_DIR / "analysis" / "results_pack.yaml",
)
BENCHMARK_CONFIG = _resolve_env_path(
    "THESIS_OPT_BENCHMARK_CONFIG",
    CONFIGS_DIR / "thesis_optimization_benchmark.yaml",
)

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
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_GENERATED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_MAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CROSS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_REPLAY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
