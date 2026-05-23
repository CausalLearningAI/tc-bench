#!/bin/bash
#SBATCH --job-name=tc_07_stats
#SBATCH --output=logs/tc_07_stats_%j.out
#SBATCH --error=logs/tc_07_stats_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=01:00:00

# Stage 07: compute per-channel normalization statistics on the train split.
# Input:  $HF_DATASET
# Output: $HF_DATASET/normalization_stats.json

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "07_normalize_stats"

python "${REPO_ROOT}/dataset/07_normalize_stats.py" \
    --dataset_path "${HF_DATASET}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --output "${HF_DATASET}/normalization_stats.json" \
    "$@"

echo "[done] stats at ${HF_DATASET}/normalization_stats.json"
