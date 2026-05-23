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
# Pass --start-year / --end-year through SBATCH_EXPORT if you only want
# a subset (default: 1980..present).

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "02_download_gridsat"

mkdir -p "${GRIDSAT_DIR}"
python "${REPO_ROOT}/dataset/02_download_gridsat.py" \
    --output_dir "${GRIDSAT_DIR}" \
    --ibtracs_csv "${IBTRACS_CSV}" \
    "$@"

echo "[done] netCDFs in ${GRIDSAT_DIR}"
