#!/bin/bash
# =============================================================================
# Top-level reproducibility orchestrator: runs every per-figure script in
# the order required by data dependencies.
#
# Stages 1–4 share probe fits where possible (Fig. 3 / Fig. 8 / Q_con reuse
# probes from Fig. 2 / Fig. 7 / Fig. 1c respectively), so the natural order
# is: fits first → diagnostics → geometry → figures.
#
# Usage:
#     bash scripts/run_all.sh                # local
#     bash scripts/run_all.sh --slurm        # submit via sbatch
#     bash scripts/run_all.sh --only fig2 fig3 fig4
#
# Each --only token matches the stem of a scripts/fig*.sh / scripts/appE2*.sh
# file (e.g. "fig2" → scripts/fig2_qstat_cls.sh).
#
# Prerequisites (not run by this script):
#     bash dataset/run_all.sh          # build $DATA_ROOT and features_<model>/
# =============================================================================

set -euo pipefail

print_help() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }

# Handle --help before sourcing _env.sh (which activates conda under set -u).
for arg in "$@"; do
    [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }
done

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"

# Extract --only tokens (after stripping --slurm/-h, which _env already consumed).
ONLY_STAGES=()
i=0
while [[ ${i} -lt ${#EXTRA_ARGS[@]} ]]; do
    arg="${EXTRA_ARGS[${i}]}"
    if [[ "${arg}" == "--only" ]]; then
        i=$((i + 1))
        while [[ ${i} -lt ${#EXTRA_ARGS[@]} && "${EXTRA_ARGS[${i}]}" != --* ]]; do
            ONLY_STAGES+=("${EXTRA_ARGS[${i}]}")
            i=$((i + 1))
        done
    else
        i=$((i + 1))
    fi
done

want() {
    if (( ${#ONLY_STAGES[@]} == 0 )); then return 0; fi
    for s in "${ONLY_STAGES[@]}"; do [[ "${s}" == "${1}" ]] && return 0; done
    return 1
}

run_stage() {
    local stage="$1"
    local script="${REPO_ROOT}/scripts/${2}"
    if ! want "${stage}"; then return 0; fi
    echo
    echo ">>> Stage: ${stage}  ($(basename "${script}"))"
    if (( USE_SLURM )); then
        bash "${script}" --slurm
    else
        bash "${script}"
    fi
}

# Dataset / dataset-stat figures (no probe dependency).
run_stage fig5               fig5_dataset_dist.sh

# Main static-fidelity sweep (Fig. 2). Fig. 3 reuses these probes.
run_stage fig2               fig2_qstat_cls.sh
run_stage fig3               fig3_qdyn_cls.sh

# Spatial-mean ablation (App. E.3 — Figs. 7 + 8).
run_stage fig7               fig7_qstat_spatial_mean.sh
run_stage fig8               fig8_qdyn_spatial_mean.sh

# Manifold consistency (US-only subset; independent probe fits).
run_stage fig_qcon           fig_qcon.sh

# Cross-agency OOD (Fig. 1c) + Dvorak / climatology baselines.
run_stage fig1c              fig1c_ood.sh

# Intrinsic geometry on DINOv3-base (Fig. 4).
run_stage fig4               fig4_geometry.sh

# Pixel-supervision ablation (Fig. 6 / App. E.1).
run_stage fig6               fig6_pixelsup.sh

# Nonlinear probe controls (App. E.2 Table 4).
run_stage appE2              appE2_nonlinear_probes.sh

# NOTE: App. E.4 Table 5 (VideoMAE / V-JEPA2 / X-CLIP, 10 seeds) is NOT
# reproducible from HEAD — the video-feature extraction code was intentionally
# removed (see git log + CLAUDE.md). Re-introducing those models requires
# restoring the deleted scripts.

echo
echo "[done] all requested figure stages."
