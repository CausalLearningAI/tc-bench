#!/bin/bash
# =============================================================================
# Orchestrator for the TC-Bench probing protocol.
#
# Runs the canonical sweep used in the paper:
#
#     fit       — ridge probe × 11 VFMs × 2 feature_types × 2 targets × 5 seeds
#     diagnose  — produce Q_stat, Q_dyn, Q_con prediction CSVs
#     geometry  — §4.2 PCA / d_eff / feature_spread per model
#     aggregate — collect per-fit metrics into outputs/summary.csv
#
# Stages can be selected individually via flags.
#
# Usage:
#     bash probing/run_all.sh                    # local, all stages
#     bash probing/run_all.sh --slurm            # submit via sbatch
#     bash probing/run_all.sh --only fit         # one stage
#     bash probing/run_all.sh --only diagnose geometry
#
# Path/env overrides come from probing/slurm/_env.sh (PROBE_DIR,
# DIAG_DIR, GEOM_DIR, FEATURES_DIR, CONDA_ENV…).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Handle --help before sourcing the env, which activates conda and may
# fail under `set -u` on hosts whose conda hooks reference unbound vars.
for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0
    fi
done

source "${SCRIPT_DIR}/slurm/_env.sh"

USE_SLURM=0
ONLY_STAGES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)        USE_SLURM=1; shift ;;
        --only)         shift
                        while [[ $# -gt 0 && "$1" != --* ]]; do
                            ONLY_STAGES+=("$1"); shift
                        done ;;
        -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)              echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

want() {
    if (( ${#ONLY_STAGES[@]} == 0 )); then return 0; fi
    for s in "${ONLY_STAGES[@]}"; do [[ "${s}" == "${1}" ]] && return 0; done
    return 1
}

n_models() { echo $(( ${#MODELS_ALL[@]} - 1 )); }

submit() {
    local wrapper="$1"; shift
    local array="${1:-}"; [[ -n "${array}" ]] && shift
    if (( USE_SLURM )); then
        if [[ -n "${array}" ]]; then
            sbatch --array="${array}" "${wrapper}" "$@"
        else
            sbatch "${wrapper}" "$@"
        fi
    else
        if [[ -n "${array}" ]]; then
            # local fallback: loop over the array.
            local start end
            IFS=- read -r start end <<< "${array%%\%*}"
            for i in $(seq "${start}" "${end}"); do
                SLURM_ARRAY_TASK_ID="${i}" bash "${wrapper}" "$@"
            done
        else
            bash "${wrapper}" "$@"
        fi
    fi
}

want fit       && submit "${SCRIPT_DIR}/slurm/fit_all.sh"       "0-$(n_models)"
want diagnose  && submit "${SCRIPT_DIR}/slurm/diagnose_all.sh"  "0-2"
want geometry  && submit "${SCRIPT_DIR}/slurm/geometry_all.sh"  "0-$(n_models)"
want aggregate && python -m probing.aggregate

echo "[done] all requested probing stages complete."
