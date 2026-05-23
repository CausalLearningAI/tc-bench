# Shared environment for TC-Bench dataset construction.
#
# Sourced by every dataset/slurm/*.sh wrapper and by run_all.sh.
# Override any variable from the shell that calls the wrapper, e.g.:
#
#     DATA_ROOT=/scratch/my_user/tcbench  sbatch dataset/slurm/06_build_hf.sh
#
# All paths are absolute. Outputs of stage N feed stage N+1, so changing
# DATA_ROOT in mid-pipeline is supported only if the prior stage outputs
# already exist there.

set -euo pipefail

# --- Repository root -----------------------------------------------------
# Resolves to the ClimGIVT/ root regardless of where the script is invoked.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# --- Data root -----------------------------------------------------------
# Top-level directory that holds every dataset artifact.
DATA_ROOT="${DATA_ROOT:-${HOME}/tcbench}"
export DATA_ROOT

# Stage outputs (override individually if needed).
IBTRACS_CSV="${IBTRACS_CSV:-${DATA_ROOT}/ibtracs/ibTRACS_since_1980.csv}"
GRIDSAT_DIR="${GRIDSAT_DIR:-${DATA_ROOT}/gridsat}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-${DATA_ROOT}/preprocessed}"
CROPPED_DIR="${CROPPED_DIR:-${DATA_ROOT}/cropped}"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-${DATA_ROOT}/consolidated}"
HF_DATASET="${HF_DATASET:-${DATA_ROOT}/dataset_hf}"
FEATURES_DIR="${FEATURES_DIR:-${DATA_ROOT}/image_features}"

export IBTRACS_CSV GRIDSAT_DIR PREPROCESSED_DIR CROPPED_DIR \
       CONSOLIDATED_DIR HF_DATASET FEATURES_DIR

# --- Python env activation ----------------------------------------------
# 1) Prefer the uv-managed .venv (README §1.2). 2) Fall back to conda only
#    when CONDA_ENV is explicitly set in the caller's shell.
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/.venv/bin/activate"
    elif [[ -n "${CONDA_ENV:-}" ]]; then
        for hook in \
            "${HOME}/miniforge3/etc/profile.d/conda.sh" \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"
        do
            if [[ -f "${hook}" ]]; then
                # shellcheck disable=SC1090
                source "${hook}"
                break
            fi
        done
        conda activate "${CONDA_ENV}"
    fi
fi

# --- Logging -------------------------------------------------------------
mkdir -p "${REPO_ROOT}/logs"

# --- Banner --------------------------------------------------------------
banner() {
    echo "============================================================"
    echo "Stage:     ${1:-?}"
    echo "Time:      $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "Host:      $(hostname)"
    echo "REPO_ROOT: ${REPO_ROOT}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "============================================================"
}
