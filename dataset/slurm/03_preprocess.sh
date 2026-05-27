#!/bin/bash
#SBATCH --job-name=tc_03_preprocess
#SBATCH --output=logs/tc_03_preprocess_%j.out
#SBATCH --error=logs/tc_03_preprocess_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Stage 03: clean IbTRACS best tracks into per-agency tables on the 3-hour
# grid (the cropping stage joins these to GridSat frames).
# Inputs:  $IBTRACS_CSV
# Output:  $PREPROCESSED_DIR/dataset_ibtracs_basic_cols_{agency}.csv

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "03_preprocess"

mkdir -p "${PREPROCESSED_DIR}"
python "${REPO_ROOT}/dataset/03_preprocess.py" \
    --ibtracs_csv "${IBTRACS_CSV}" \
    --gridsat_dir "${GRIDSAT_DIR}" \
    --output_dir "${PREPROCESSED_DIR}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --years "${YEARS}" \
    --agencies "${AGENCIES}" \
    "$@"

touch "${PREPROCESSED_DIR}/.done"
echo "[done] preprocessed tracks in ${PREPROCESSED_DIR}"
