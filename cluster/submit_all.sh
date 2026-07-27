#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit the full campaign with a safety gate:
#   canary (must pass)  ->  6 data-gen scenarios  ->  train  ->  optimize
# SLURM dependencies ensure nothing heavy runs unless the canary's physics
# checks pass. Run from the repo root AFTER cluster/setup_env.sh:
#   bash cluster/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p results/logs
jid() { awk '{print $NF}'; }   # extract SLURM job id from "Submitted batch job N"

# 1) Canary: small run + physics assertions. Big jobs depend on afterok.
CANARY=$(sbatch --job-name=vis_canary --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem-per-cpu=4G --time=00:40:00 \
  --output=results/logs/canary_%j.out --error=results/logs/canary_%j.err \
  --wrap="bash cluster/canary.sh" | jid)
echo "canary        -> job $CANARY"

# 2) Data generation: all six scenarios, each gated on the canary passing.
DG_IDS=()
for cfg in load_mismatch_only line_outages_only no_mismatch \
           zone_based_load_mismatch line_outages_plus_global_load_mismatch \
           line_outages_plus_zone_based_load_mismatch; do
  id=$(sbatch --dependency=afterok:${CANARY} --job-name=dg_${cfg} \
       cluster/datagen.slurm configs/data_generation/${cfg}.yaml | jid)
  DG_IDS+=("$id"); echo "datagen ${cfg} -> job $id"
done
DG_DEP=$(IFS=:; echo "${DG_IDS[*]}")

# 3) Train after ALL data-gen jobs succeed.
TRAIN=$(sbatch --dependency=afterok:${DG_DEP} cluster/train.slurm | jid)
echo "train         -> job $TRAIN"

# 4) Optimize after training.
OPT=$(sbatch --dependency=afterok:${TRAIN} cluster/optimize.slurm | jid)
echo "optimize      -> job $OPT"

echo "Submitted. Monitor with:  squeue -u \$USER"
