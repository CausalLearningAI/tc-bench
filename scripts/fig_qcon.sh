#!/bin/bash
# =============================================================================
# §4.1 / App. E.7 — Manifold Consistency Q_con (pressure–wind coupling).
#
# Q_con joins two Q_stat tables (one fit to pressure, one to wind) on a
# per-frame key and checks whether the recovered (P_c, V_m) respect the
# gradient-wind / latitude scaling Eq. E.1. The paper restricts this
# diagnostic to a single-agency subset (HURDAT, data=us_only) to neutralize
# the cross-agency wind-averaging inconsistency flagged in App. D.
#
# Pinned settings:
#   probe         ridge
#   feature_type  cls
#   targets       pressure AND wind  (both probes are required)
#   data          us_only
#   seeds         42 43 44 45 46
# =============================================================================

print_help() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig_qcon"

PROBE="${PROBE_DEFAULT}"
FEATURE_TYPES="cls"
TARGETS="pressure,wind"
SEEDS="${SEEDS_DEFAULT}"
DATA="${DATA_USONLY}"
export PROBE FEATURE_TYPES TARGETS SEEDS DATA

# --- 1. Fit ridge probes for both targets on the US-only subset --------------
submit_model_array "${REPO_ROOT}/probing/slurm/fit_all.sh"

# --- 2. Q_con diagnostic (q_con joins pressure + wind internally) ------------
submit_diagnostic q_con

# --- 3. Render the scatter + gap-evolution figure ----------------------------
python "${REPO_ROOT}/figures/fig_q_con.py" \
    --predictions "${DIAG_DIR}/q_con/predictions_cls_seed42.csv" \
    --output_dir "${FIGS_DIR}" \
    --model_family dinov3 --model_size base

echo "[done] Q_con → ${FIGS_DIR}/"
