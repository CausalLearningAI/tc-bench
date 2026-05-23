#!/bin/bash
#SBATCH --job-name=tc_fit
#SBATCH --output=logs/tc_fit_%A_%a.out
#SBATCH --error=logs/tc_fit_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Sweep `probing.fit` over (probe × model × feature_type × target × seed)
# as a single Hydra multirun. SLURM's array index selects the *model* so
# that jobs run in parallel without head-of-line blocking.
#
# Override the sweep axes from the shell, e.g.
#
#     PROBE=ridge,lasso  FEATURE_TYPES="cls"  SEEDS="42,43,44" \
#         sbatch --array=0-10 probing/slurm/fit_all.sh

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

MODELS_STR="${MODELS:-${MODELS_ALL[*]}}"
read -ra _models <<< "${MODELS_STR}"
MODEL_NAME="${_models[${SLURM_ARRAY_TASK_ID:-0}]}"

PROBE="${PROBE:-ridge}"
FEATURE_TYPES="${FEATURE_TYPES:-cls,spatial_mean}"
TARGETS="${TARGETS:-pressure,wind}"
SEEDS="${SEEDS:-42}"
DATA="${DATA:-balanced_980}"

banner "fit_all: ${MODEL_NAME}"

python -m probing.fit \
    --multirun \
    model="${MODEL_NAME}" \
    probe="${PROBE}" \
    data="${DATA}" \
    feature_type="${FEATURE_TYPES}" \
    target="${TARGETS}" \
    seed="${SEEDS}" \
    "$@"

echo "[done] fits for ${MODEL_NAME} → ${PROBE_DIR}/${MODEL_NAME}"
