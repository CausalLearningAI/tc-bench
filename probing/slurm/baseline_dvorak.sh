#!/bin/bash
#SBATCH --job-name=tc_baseline_dvorak
#SBATCH --output=logs/tc_baseline_dvorak_%j.out
#SBATCH --error=logs/tc_baseline_dvorak_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Dvorak (1975) technique baseline — operates directly on the HF dataset
# (no VFM features required). Output is consumed by figures/.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "baseline_dvorak"

HF_DATASET="${HF_DATASET:-${DATA_ROOT}/dataset_hf}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/baselines/dvorak.json}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

python "${REPO_ROOT}/probing/baselines/dvorak.py" \
    --dataset_path "${HF_DATASET}" \
    --output_path "${OUTPUT_PATH}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --splits test validation train \
    "$@"

echo "[done] Dvorak baseline → ${OUTPUT_PATH}"
