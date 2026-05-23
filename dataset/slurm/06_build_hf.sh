#!/bin/bash
#SBATCH --job-name=tc_06_build_hf
#SBATCH --output=logs/tc_06_build_hf_%j.out
#SBATCH --error=logs/tc_06_build_hf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00

# Stage 06: build the HuggingFace Arrow dataset with train/val/test splits
# at the cyclone (trajectory) level.
# Input:  $CONSOLIDATED_DIR
# Output: $HF_DATASET

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "06_build_hf"

mkdir -p "${HF_DATASET}"
python "${REPO_ROOT}/dataset/06_build_hf.py" \
    --consolidated_path "${CONSOLIDATED_DIR}" \
    --output_path "${HF_DATASET}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1 \
    --seed 42 \
    "$@"

echo "[done] HF dataset at ${HF_DATASET}"
