# Shared environment for the probing protocol.
#
# Sourced by every probing/slurm/*.sh wrapper and by run_all.sh. Mirrors
# dataset/slurm/_env.sh so the same DATA_ROOT and CONDA_ENV work across
# both pipelines.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

DATA_ROOT="${DATA_ROOT:-${HOME}/tcbench}"
FEATURES_DIR="${FEATURES_DIR:-${DATA_ROOT}/image_features}"

# Probing artefact roots. `probing/run_all.sh` and the four Hydra entries
# read these via the `oc.env:` resolver.
PROBE_DIR="${PROBE_DIR:-${REPO_ROOT}/outputs/probes}"
DIAG_DIR="${DIAG_DIR:-${REPO_ROOT}/outputs/diagnostics}"
GEOM_DIR="${GEOM_DIR:-${REPO_ROOT}/outputs/geometry}"

export DATA_ROOT FEATURES_DIR PROBE_DIR DIAG_DIR GEOM_DIR

# Python env activation.
# 1) Prefer the uv-managed .venv (README §1.2). 2) Fall back to conda.
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

# figures/*.py do `from figures._style import ...`, which only resolves when the
# repo root is on sys.path. `python figures/foo.py` from REPO_ROOT does NOT add
# REPO_ROOT to sys.path on Python 3.11+, so export PYTHONPATH unconditionally.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${REPO_ROOT}/logs"

# Default sweep axes (override from the shell).
MODELS_ALL=(
    dinov2-base dinov2-large
    dinov3-base dinov3-large dinov3-satellite
    clip-base clip-large
    siglip-base siglip2-base
    mae-base mae-large
)
SEEDS_ALL=(42 43 44 45 46)
FEATURE_TYPES_ALL=(cls spatial_mean)
TARGETS_ALL=(pressure wind)

banner() {
    echo "============================================================"
    echo "Job:       ${1:-?}"
    echo "Time:      $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "Host:      $(hostname)"
    echo "REPO_ROOT: ${REPO_ROOT}"
    echo "PROBE_DIR: ${PROBE_DIR}"
    echo "============================================================"
}
