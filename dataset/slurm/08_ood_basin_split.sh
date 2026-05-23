#!/bin/bash
#SBATCH --job-name=tc_08_ood_split
#SBATCH --output=logs/tc_08_ood_split_%A_%a.out
#SBATCH --error=logs/tc_08_ood_split_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Stage 08: regenerate train/val/test splits where the held-out basin
# never appears in train (cross-basin OOD evaluation, Fig. 1c).
#
# Run as an array, one task per VFM whose extracted features need a
# parallel OOD split:
#
#     MODELS="dinov3-base clip-base mae-base" \
#         sbatch --array=0-2 dataset/slurm/08_ood_basin_split.sh
#
# Or single-model:
#
#     sbatch --array=0 dataset/slurm/08_ood_basin_split.sh
#
# Inputs:  $FEATURES_DIR/features_{model}
# Outputs: $FEATURES_DIR/features_{model}_ood_splits

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

MODELS_STR="${MODELS:-dinov3-base}"
read -ra MODELS_ARR <<< "${MODELS_STR}"
MODEL_NAME="${MODELS_ARR[${SLURM_ARRAY_TASK_ID:-0}]}"

banner "08_ood_basin_split: ${MODEL_NAME}"

FEATURE_DATASET="${FEATURES_DIR}/features_${MODEL_NAME}"
OUTPUT_DIR="${FEATURES_DIR}/features_${MODEL_NAME}_ood_splits"
BASIN_FIELD="${BASIN_FIELD:-location}"
VAL_FRAC="${VAL_FRAC:-0.1}"

python "${REPO_ROOT}/dataset/08_ood_basin_split.py" \
    --feature_dataset "${FEATURE_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --basin_field "${BASIN_FIELD}" \
    --val_frac "${VAL_FRAC}" \
    "$@"

echo "[done] OOD splits in ${OUTPUT_DIR}"
