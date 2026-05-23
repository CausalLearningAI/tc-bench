#!/bin/bash
#SBATCH --job-name=tc_geom
#SBATCH --output=logs/tc_geom_%A_%a.out
#SBATCH --error=logs/tc_geom_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# §4.2 intrinsic-geometry diagnostics, one task per model.
#
#     sbatch --array=0-10 probing/slurm/geometry_all.sh

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

MODELS_STR="${MODELS:-${MODELS_ALL[*]}}"
read -ra _models <<< "${MODELS_STR}"
MODEL_NAME="${_models[${SLURM_ARRAY_TASK_ID:-0}]}"

FEATURE_TYPES="${FEATURE_TYPES:-cls,spatial_mean}"
DATA="${DATA:-full}"

banner "geometry_all: ${MODEL_NAME}"

python -m probing.geometry \
    --multirun \
    model="${MODEL_NAME}" \
    data="${DATA}" \
    feature_type="${FEATURE_TYPES}" \
    "$@"

echo "[done] geometry for ${MODEL_NAME} → ${GEOM_DIR}/${MODEL_NAME}"
