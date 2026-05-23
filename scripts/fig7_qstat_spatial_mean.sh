#!/bin/bash
# =============================================================================
# Fig. 7 / App. E.3 — Static Fidelity Q_stat using the spatial-mean aggregation.
#
# Identical protocol to Fig. 2 but with feature_type=spatial_mean. Paper §E.3
# uses this to rule out the CLS token as the source of regime degradation.
#
# Pinned settings:
#   probe         ridge
#   feature_type  spatial_mean
#   target        pressure
#   data          balanced_980
#   seeds         42 43 44 45 46
# =============================================================================

print_help() { sed -n '2,13p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig7_qstat_spatial_mean"

PROBE="${PROBE_DEFAULT}"
FEATURE_TYPES="spatial_mean"
TARGETS="pressure"
SEEDS="${SEEDS_DEFAULT}"
DATA="${DATA_BALANCED}"
export PROBE FEATURE_TYPES TARGETS SEEDS DATA

submit_model_array "${REPO_ROOT}/probing/slurm/fit_all.sh"
submit_diagnostic q_stat

python "${REPO_ROOT}/figures/fig2_q_stat.py" \
    --predictions "${DIAG_DIR}/q_stat/predictions_spatial_mean_pressure_seed42.csv" \
    --output "${FIGS_DIR}/fig7_q_stat_spatial_mean.pdf"

echo "[done] Fig. 7 → ${FIGS_DIR}/fig7_q_stat_spatial_mean.pdf"
