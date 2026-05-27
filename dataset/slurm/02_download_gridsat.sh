#!/bin/bash
#SBATCH --job-name=tc_02_gridsat
#SBATCH --output=logs/tc_02_gridsat_%j.out
#SBATCH --error=logs/tc_02_gridsat_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Stage 02: download GridSat-B1 brightness-temperature netCDF files.
# Output: $GRIDSAT_DIR/{YEAR}/*.nc
#
# Restrict the build with the YEARS / ONLY_CYCLONE env vars (see _env.sh);
# empty means the full 1980..present archive.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "02_download_gridsat"

mkdir -p "${GRIDSAT_DIR}"
python "${REPO_ROOT}/dataset/02_download_gridsat.py" \
    --output_dir "${GRIDSAT_DIR}" \
    --ibtracs_csv "${IBTRACS_CSV}" \
    --years "${YEARS}" \
    --only-cyclone "${ONLY_CYCLONE}" \
    "$@"

touch "${GRIDSAT_DIR}/.done"
echo "[done] netCDFs in ${GRIDSAT_DIR}"
