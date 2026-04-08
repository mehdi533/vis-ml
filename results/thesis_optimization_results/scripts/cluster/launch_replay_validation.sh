#!/usr/bin/env bash
#SBATCH --job-name=opt_bench_replay
#SBATCH --output=results/thesis_optimization_results/results/benchmark/logs/replay_%A_%a.out
#SBATCH --error=results/thesis_optimization_results/results/benchmark/logs/replay_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
unset SCRIPT_DIR
source "${COMMON_ENV_DIR}/common_env.sh"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-results/thesis_optimization_results/configs/thesis_optimization_benchmark.yaml}"
FORMULATION_SUITE="${FORMULATION_SUITE:-results/thesis_optimization_results/configs/suites/01_formulation_comparison.yaml}"
BENCHMARK_GROUP="${BENCHMARK_GROUP:-main}"
WORKER="${WORKER:-results/thesis_optimization_results/scripts/cluster/run_single_benchmark_case.py}"
MANIFEST_SCRIPT="${MANIFEST_SCRIPT:-results/thesis_optimization_results/scripts/cluster/generate_benchmark_manifest.py}"
RUN_PYTHON_BIN="${RUN_PYTHON_BIN:-../venv/bin/python}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_LIMIT="${LOCAL_LIMIT:-2}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-24}"

mkdir -p results/thesis_optimization_results/results/benchmark/logs
"${RUN_PYTHON_BIN}" "${MANIFEST_SCRIPT}" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" >/dev/null

TASK_COUNT="$("${RUN_PYTHON_BIN}" "${WORKER}" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" --mode replay --group "${BENCHMARK_GROUP}" --count-tasks)"
echo "[launch_replay_validation] group=${BENCHMARK_GROUP} task_count=${TASK_COUNT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${RUN_PYTHON_BIN}" "${MANIFEST_SCRIPT}" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" --dry-run
  exit 0
fi

if [[ "${1:-}" == "--local" ]]; then
  for ((idx=0; idx<LOCAL_LIMIT && idx<TASK_COUNT; idx++)); do
    prepare_cmd=("${RUN_PYTHON_BIN}" "${WORKER}" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" --mode replay --group "${BENCHMARK_GROUP}" --task-index "${idx}" --prepare-only)
    if [[ "${FORCE}" == "1" ]]; then
      prepare_cmd+=(--force)
    fi
    echo "[launch_replay_validation] local idx=${idx}"
    replay_config="$("${prepare_cmd[@]}")"
    env KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE MPLCONFIGDIR=/tmp/matplotlib "${RUN_PYTHON_BIN}" scheduling/replay_validation.py --config "${replay_config}"
  done
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if command -v sbatch >/dev/null 2>&1; then
    sbatch --array="0-$((TASK_COUNT - 1))%${ARRAY_CONCURRENCY}" "$0"
    exit 0
  fi
  echo "[launch_replay_validation] sbatch not found; use '--local' for a smoke run." >&2
  exit 1
fi

prepare_cmd=("${RUN_PYTHON_BIN}" "${WORKER}" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" --mode replay --group "${BENCHMARK_GROUP}" --task-index "${SLURM_ARRAY_TASK_ID}" --prepare-only)
if [[ "${FORCE}" == "1" ]]; then
  prepare_cmd+=(--force)
fi
replay_config="$("${prepare_cmd[@]}")"
env KMP_DUPLICATE_LIB_OK=TRUE KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE MPLCONFIGDIR=/tmp/matplotlib "${RUN_PYTHON_BIN}" scheduling/replay_validation.py --config "${replay_config}"
