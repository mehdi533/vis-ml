#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p results/thesis_data_generation_results/logs

sbatch results/thesis_data_generation_results/scripts/run_load_mismatch_only.sh
sbatch results/thesis_data_generation_results/scripts/run_line_outages_only.sh
sbatch results/thesis_data_generation_results/scripts/run_line_outages_plus_global_load_mismatch.sh
sbatch results/thesis_data_generation_results/scripts/run_zone_based_load_mismatch.sh
sbatch results/thesis_data_generation_results/scripts/run_line_outages_plus_zone_based_load_mismatch.sh
