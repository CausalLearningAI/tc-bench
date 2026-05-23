#!/bin/bash
#SBATCH --job-name=tc_05_consolidate
#SBATCH --output=logs/tc_05_consolidate_%j.out
#SBATCH --error=logs/tc_05_consolidate_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=02:00:00

# Stage 05: consolidate per-timestep netCDFs into one file per cyclone.
# Input:  $CROPPED_DIR
# Output: $CONSOLIDATED_DIR

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "05_consolidate_nc"

mkdir -p "${CONSOLIDATED_DIR}"
python "${REPO_ROOT}/dataset/05_consolidate_nc.py" \
    --input_dir "${CROPPED_DIR}" \
    --output_dir "${CONSOLIDATED_DIR}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "[done] consolidated netCDFs in ${CONSOLIDATED_DIR}"
