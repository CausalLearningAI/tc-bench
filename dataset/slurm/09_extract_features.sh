#!/bin/bash
#SBATCH --job-name=tc_09_extract
#SBATCH --output=logs/tc_09_extract_%A_%a.out
#SBATCH --error=logs/tc_09_extract_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-10%8

# Stage 09: extract frozen-VFM features (CLS + spatial-mean) for every
# IR frame in $HF_DATASET, one job per model.
#
# Defaults to the 11 model variants reported in the paper. Override the
# model list (and array size) from the shell:
#
#     MODELS="dinov3-base mae-base" \
#         sbatch --array=0-1 dataset/slurm/09_extract_features.sh
#
# Input:  $HF_DATASET
# Output: $FEATURES_DIR/features_{model_type}

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

DEFAULT_MODELS=(
    dinov2-base dinov2-large
    dinov3-base dinov3-large dinov3-satellite
    clip-base clip-large
    siglip-base siglip2-base
    mae-base mae-large
)

if [[ -n "${MODELS:-}" ]]; then
    read -ra MODELS_ARR <<< "${MODELS}"
else
    MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

MODEL_TYPE="${MODELS_ARR[${SLURM_ARRAY_TASK_ID:-0}]}"
banner "09_extract_features: ${MODEL_TYPE}"

python "${REPO_ROOT}/dataset/09_extract_features.py" \
    --model_type "${MODEL_TYPE}" \
    --dataset_path "${HF_DATASET}" \
    --output_dir "${FEATURES_DIR}" \
    --batch_size "${BATCH_SIZE:-512}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --device "${DEVICE:-cuda}" \
    "$@"

echo "[done] features for ${MODEL_TYPE} in ${FEATURES_DIR}/features_${MODEL_TYPE}"
