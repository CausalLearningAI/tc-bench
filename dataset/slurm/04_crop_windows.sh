#!/bin/bash
#SBATCH --job-name=tc_04_crop
#SBATCH --output=logs/tc_04_crop_%j.out
#SBATCH --error=logs/tc_04_crop_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Stage 04: crop ~1700 km x 1700 km windows centred on each best-track
# fix point, producing per-cyclone netCDF files.
# Inputs: $PREPROCESSED_DIR (tracks), $GRIDSAT_DIR (frames)
# Output: $CROPPED_DIR/{agency}/{year}_{name}.nc

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "04_crop_windows"

mkdir -p "${CROPPED_DIR}"
python "${REPO_ROOT}/dataset/04_crop_windows.py" \
    --input_dir "${PREPROCESSED_DIR}" \
    --gridsat_dir "${GRIDSAT_DIR}" \
    --output_dir "${CROPPED_DIR}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --years "${YEARS}" \
    --agencies "${AGENCIES}" \
    --only-cyclone "${ONLY_CYCLONE}" \
    "$@"

touch "${CROPPED_DIR}/.done"
echo "[done] cropped windows in ${CROPPED_DIR}"
