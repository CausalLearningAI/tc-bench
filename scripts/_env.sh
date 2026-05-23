# Shared environment for the per-figure reproducibility scripts.
#
# Sourced by every scripts/fig*.sh wrapper. Builds on top of
# probing/slurm/_env.sh so that DATA_ROOT / FEATURES_DIR / PROBE_DIR /
# DIAG_DIR / GEOM_DIR / MODELS_ALL / SEEDS_ALL stay consistent with
# probing/run_all.sh.

set -euo pipefail

SCRIPTS_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPTS_ENV_DIR}/.." && pwd)}"
export REPO_ROOT

# shellcheck source=../probing/slurm/_env.sh
source "${REPO_ROOT}/probing/slurm/_env.sh"

# Per-paper canonical knobs. Override from the shell when calling a script
# (e.g. SEEDS="42,43" bash scripts/fig2_qstat_cls.sh).
SEEDS_DEFAULT="${SEEDS_DEFAULT:-42,43,44,45,46}"   # 5 seeds for the main figures
SEEDS_NONLINEAR="${SEEDS_NONLINEAR:-42,43,44}"     # 3 seeds for App. E.2 Table 4
PROBE_DEFAULT="${PROBE_DEFAULT:-ridge}"
DATA_BALANCED="${DATA_BALANCED:-balanced_980}"
DATA_FULL="${DATA_FULL:-full}"
DATA_USONLY="${DATA_USONLY:-us_only}"
DATA_OOD="${DATA_OOD:-ood_basin}"
FIGS_DIR="${FIGS_DIR:-${REPO_ROOT}/figs}"
mkdir -p "${FIGS_DIR}"
export SEEDS_DEFAULT SEEDS_NONLINEAR PROBE_DEFAULT \
       DATA_BALANCED DATA_FULL DATA_USONLY DATA_OOD FIGS_DIR

# --slurm flag plumbing: dispatch the existing probing/slurm/*.sh
# wrappers via sbatch if requested, otherwise run them locally (looping
# over the array index).
USE_SLURM="${USE_SLURM:-0}"

parse_slurm_flag() {
    USE_SLURM=0
    local _rest=()
    for arg in "$@"; do
        case "${arg}" in
            --slurm) USE_SLURM=1 ;;
            -h|--help) print_help; exit 0 ;;
            *) _rest+=("${arg}") ;;
        esac
    done
    EXTRA_ARGS=("${_rest[@]}")
    export USE_SLURM
}

# Submit one of the probing/slurm/*.sh wrappers across a model-array.
submit_model_array() {
    local wrapper="$1"; shift
    local array_end="${1:-$((${#MODELS_ALL[@]} - 1))}"; shift || true
    if (( USE_SLURM )); then
        sbatch --array="0-${array_end}" "${wrapper}" "$@"
    else
        for i in $(seq 0 "${array_end}"); do
            SLURM_ARRAY_TASK_ID="${i}" bash "${wrapper}" "$@"
        done
    fi
}

# Submit a diagnostic by its name (q_stat / q_dyn / q_con).
submit_diagnostic() {
    local diag="$1"; shift
    local idx
    case "${diag}" in
        q_stat) idx=0 ;;
        q_dyn)  idx=1 ;;
        q_con)  idx=2 ;;
        *)      echo "unknown diagnostic ${diag}" >&2; return 2 ;;
    esac
    if (( USE_SLURM )); then
        sbatch --array="${idx}" "${REPO_ROOT}/probing/slurm/diagnose_all.sh" "$@"
    else
        SLURM_ARRAY_TASK_ID="${idx}" bash "${REPO_ROOT}/probing/slurm/diagnose_all.sh" "$@"
    fi
}

banner_fig() {
    echo "============================================================"
    echo "Figure script:  ${1:-?}"
    echo "Time:           $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "REPO_ROOT:      ${REPO_ROOT}"
    echo "USE_SLURM:      ${USE_SLURM}"
    echo "============================================================"
}
