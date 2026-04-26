#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -d "${REPO_ROOT}/.git" && "${REPO_ROOT}" != "/" ]]; do
  REPO_ROOT="$(dirname "${REPO_ROOT}")"
done

"${REPO_ROOT}/scripts/results/optimization/pipelines/cluster/launch_replay_validation.sh" "$@"
