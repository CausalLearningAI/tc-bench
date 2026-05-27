#!/bin/bash
# =============================================================================
# Orchestrator for the TC-Bench dataset construction pipeline.
#
# Each stage is idempotent: it is skipped if its primary output already
# exists. Re-run individual stages with --force-N to override.
#
# Stages:
#     01  download IbTRACS CSV
#     02  download GridSat-B1 netCDFs (best-track timestamps only)
#     03  preprocess IbTRACS into per-agency tables on the 3-hour grid
#     04  crop fixed-size GridSat windows around each best-track fix
#     05  consolidate per-cyclone trajectories (NaN handling + validity masks)
#     06  build the HuggingFace Arrow dataset (train/val/test splits)
#     07  compute per-channel normalization statistics
#     08  generate cross-basin OOD splits  (optional, paper-only)
#     09  extract frozen-VFM features      (required for the probing protocol)
#
# Usage:
#     bash dataset/run_all.sh                # all stages, skip completed
#     bash dataset/run_all.sh --from 03      # resume from stage 03
#     bash dataset/run_all.sh --to   06      # stop after stage 06
#     bash dataset/run_all.sh --only 07 09   # just these stages
#     bash dataset/run_all.sh --slurm        # submit each stage via sbatch
#                                            #    (returns immediately)
#
# Path & env overrides come from dataset/slurm/_env.sh (see file for the
# full list of overridable variables, e.g. DATA_ROOT, CONDA_ENV, MODELS).
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
FROM_STAGE=1
TO_STAGE=9
ONLY_STAGES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)        USE_SLURM=1; shift ;;
        --from)         FROM_STAGE="$2"; shift 2 ;;
        --to)           TO_STAGE="$2"; shift 2 ;;
        --only)         shift
                        while [[ $# -gt 0 && "$1" != --* ]]; do
                            ONLY_STAGES+=("$1"); shift
                        done ;;
        -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)              echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Output marker for each stage; presence => "done".
stage_output() {
    case "$1" in
        01) echo "${IBTRACS_CSV}" ;;
        02) echo "${GRIDSAT_DIR}/.done" ;;
        03) echo "${PREPROCESSED_DIR}/.done" ;;
        04) echo "${CROPPED_DIR}/.done" ;;
        05) echo "${CONSOLIDATED_DIR}/.done" ;;
        06) echo "${HF_DATASET}/dataset_dict.json" ;;
        07) echo "${HF_DATASET}/normalization_stats.json" ;;
        08) echo "${FEATURES_DIR}/features_dinov3-base_ood_splits/.done" ;;
        09) echo "${FEATURES_DIR}/features_dinov3-base/dataset_info.json" ;;
    esac
}

want_stage() {
    local n="$1"
    if (( ${#ONLY_STAGES[@]} > 0 )); then
        for s in "${ONLY_STAGES[@]}"; do
            [[ "$(printf '%02d' "${s#0}")" == "${n}" ]] && return 0
        done
        return 1
    fi
    (( 10#${n} >= 10#${FROM_STAGE} && 10#${n} <= 10#${TO_STAGE} ))
}

run_stage() {
    local n="$1" desc="$2" tag="$3"
    local wrapper="${SCRIPT_DIR}/slurm/${n}_${tag}.sh"
    local marker; marker="$(stage_output "${n}")"
    if [[ -n "${marker}" && -e "${marker}" ]]; then
        echo "[skip] stage ${n} (${desc}) — output exists: ${marker}"
        return 0
    fi
    echo "[run ] stage ${n} (${desc})"
    if (( USE_SLURM )); then
        sbatch "${wrapper}"
    else
        bash "${wrapper}"
    fi
}

want_stage 01 && run_stage 01 "download IbTRACS"        download_ibtracs
want_stage 02 && run_stage 02 "download GridSat-B1"     download_gridsat
want_stage 03 && run_stage 03 "preprocess tracks"       preprocess
want_stage 04 && run_stage 04 "crop windows"            crop_windows
want_stage 05 && run_stage 05 "consolidate netCDFs"     consolidate_nc
want_stage 06 && run_stage 06 "build HF dataset"        build_hf
want_stage 07 && run_stage 07 "normalization stats"     normalize_stats
want_stage 08 && run_stage 08 "OOD basin split"         ood_basin_split
want_stage 09 && run_stage 09 "extract VFM features"    extract_features

echo "[done] all requested stages complete."
