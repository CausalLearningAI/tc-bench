#!/bin/bash
#SBATCH --job-name=tc_diag
#SBATCH --output=logs/tc_diag_%A_%a.out
#SBATCH --error=logs/tc_diag_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2     # one task per diagnostic (q_stat, q_dyn, q_con)

# Build the diagnostic CSVs for every model in one go.
#
# The array index selects the diagnostic; everything else is a Hydra
# multirun sweep over feature_type / target / seed. q_con uses both
# pressure and wind probes internally, so it ignores the target sweep.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

DIAGNOSTICS=(q_stat q_dyn q_con)
DIAGNOSTIC="${DIAGNOSTICS[${SLURM_ARRAY_TASK_ID:-0}]}"

PROBE="${PROBE:-ridge}"
DATA="${DATA:-balanced_980}"
FEATURE_TYPES="${FEATURE_TYPES:-cls,spatial_mean}"
TARGETS="${TARGETS:-pressure,wind}"
SEEDS="${SEEDS:-42}"

banner "diagnose_all: ${DIAGNOSTIC}"

if [[ "${DIAGNOSTIC}" == "q_con" ]]; then
    python -m probing.diagnose \
        --multirun \
        diagnostic="${DIAGNOSTIC}" \
        probe="${PROBE}" \
        data=us_only \
        feature_type="${FEATURE_TYPES}" \
        seed="${SEEDS}" \
        "$@"
else
    python -m probing.diagnose \
        --multirun \
        diagnostic="${DIAGNOSTIC}" \
        probe="${PROBE}" \
        data="${DATA}" \
        feature_type="${FEATURE_TYPES}" \
        target="${TARGETS}" \
        seed="${SEEDS}" \
        "$@"
fi

echo "[done] ${DIAGNOSTIC} → ${DIAG_DIR}/${DIAGNOSTIC}"
