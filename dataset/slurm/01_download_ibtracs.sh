#!/bin/bash
#SBATCH --job-name=tc_01_ibtracs
#SBATCH --output=logs/tc_01_ibtracs_%j.out
#SBATCH --error=logs/tc_01_ibtracs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Stage 01: download the IbTRACS best-track CSV (since 1980).
# Output: $IBTRACS_CSV

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
banner "01_download_ibtracs"

mkdir -p "$(dirname "${IBTRACS_CSV}")"
python "${REPO_ROOT}/dataset/01_download_ibtracs.py" \
    --output "${IBTRACS_CSV}"

echo "[done] wrote ${IBTRACS_CSV}"
