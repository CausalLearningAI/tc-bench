#!/bin/bash
# =============================================================================
# Fig. 3 — Dynamic Coherence Q_dyn (CLS token).
#
# Paper §4: Q_dyn re-uses the *static* linear probe L fit for Q_stat and
# applies it to finite-difference feature deltas: ξ_dyn = ||L·Δz - Δy||.
# So this script depends on Fig. 2's ridge probes already living under
# ${PROBE_DIR}/<model>/ridge/cls_pressure_seed*.pkl — run scripts/fig2_qstat_cls.sh
# first (or scripts/run_all.sh).
#
# Pinned settings:
#   probe         ridge (loaded from disk)
#   feature_type  cls
#   target        pressure
#   data          balanced_980  (paper §4.1: same regime-balanced eval)
#   seeds         42 43 44 45 46
# =============================================================================

print_help() { sed -n '2,16p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig3_qdyn_cls"

PROBE="${PROBE_DEFAULT}"
FEATURE_TYPES="cls"
TARGETS="pressure"
SEEDS="${SEEDS_DEFAULT}"
DATA="${DATA_BALANCED}"
export PROBE FEATURE_TYPES TARGETS SEEDS DATA

# Diagnose (Q_dyn projects Δz with the cached ridge L).
submit_diagnostic q_dyn

# Render — fig3 reads the seed=42 CSV by convention.
python "${REPO_ROOT}/figures/fig3_q_dyn.py" \
    --predictions "${DIAG_DIR}/q_dyn/predictions_cls_pressure_seed42.csv" \
    --output "${FIGS_DIR}/fig3_q_dyn.pdf"

echo "[done] Fig. 3 → ${FIGS_DIR}/fig3_q_dyn.pdf"
