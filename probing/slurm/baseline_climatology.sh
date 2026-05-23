#!/bin/bash
#SBATCH --job-name=tc_baseline_climatology
#SBATCH --output=logs/tc_baseline_climatology_%j.out
#SBATCH --error=logs/tc_baseline_climatology_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Per-basin / per-month climatology baseline (Fig. 1c reference).

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "baseline_climatology"

HF_DATASET="${HF_DATASET:-${DATA_ROOT}/dataset_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/baselines}"
mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/probing/baselines/climatology.py" \
    --dataset_path "${HF_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "[done] climatology baselines → ${OUTPUT_DIR}"
