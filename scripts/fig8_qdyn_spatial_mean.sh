#!/bin/bash
# =============================================================================
# Fig. 8 / App. E.3 — Dynamic Coherence Q_dyn using the spatial-mean aggregation.
#
# Same protocol as Fig. 3, swapping CLS for the spatial-mean token. Depends on
# the ridge probes fit by scripts/fig7_qstat_spatial_mean.sh.
#
# Pinned settings:
#   probe         ridge (loaded from disk)
#   feature_type  spatial_mean
#   target        pressure
#   data          balanced_980
#   seeds         42 43 44 45 46
# =============================================================================

print_help() { sed -n '2,13p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig8_qdyn_spatial_mean"

PROBE="${PROBE_DEFAULT}"
FEATURE_TYPES="spatial_mean"
TARGETS="pressure"
SEEDS="${SEEDS_DEFAULT}"
DATA="${DATA_BALANCED}"
export PROBE FEATURE_TYPES TARGETS SEEDS DATA

submit_diagnostic q_dyn

python "${REPO_ROOT}/figures/fig3_q_dyn.py" \
    --predictions "${DIAG_DIR}/q_dyn/predictions_spatial_mean_pressure_seed42.csv" \
    --output "${FIGS_DIR}/fig8_q_dyn_spatial_mean.pdf"

echo "[done] Fig. 8 → ${FIGS_DIR}/fig8_q_dyn_spatial_mean.pdf"
