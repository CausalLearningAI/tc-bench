#!/bin/bash
# =============================================================================
# Fig. 5 — App. D dataset-inspection histograms (per-agency P_c and V_m).
#
# Reads the two Q_stat prediction CSVs (cls × {pressure, wind} × seed=42).
# fig_data_distribution.py defaults to the modern outputs/diagnostics/q_stat/
# layout, so this wrapper just needs to make sure both CSVs exist — the Fig. 2
# sweep writes the pressure one, but no other wrapper writes the wind one.
#
# Pinned settings:
#   probe         ridge
#   feature_type  cls
#   target        pressure AND wind  (wind probe is fit lazily below)
#   data          balanced_980
# =============================================================================

print_help() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig5_dataset_dist"

# --- 1. Wind probe + Q_stat (only if the CSV is missing) ---------------------
WIND_CSV="${DIAG_DIR}/q_stat/predictions_cls_wind_seed42.csv"
if [[ ! -s "${WIND_CSV}" ]]; then
    PROBE="${PROBE_DEFAULT}"
    FEATURE_TYPES="cls"
    TARGETS="wind"
    SEEDS="${SEEDS_DEFAULT}"
    DATA="${DATA_BALANCED}"
    export PROBE FEATURE_TYPES TARGETS SEEDS DATA
    submit_model_array "${REPO_ROOT}/probing/slurm/fit_all.sh"
    submit_diagnostic q_stat
fi

# --- 2. Render the per-agency histograms -------------------------------------
python "${REPO_ROOT}/figures/fig_data_distribution.py" \
    --pressure_csv "${DIAG_DIR}/q_stat/predictions_cls_pressure_seed42.csv" \
    --wind_csv     "${WIND_CSV}" \
    --output_dir   "${FIGS_DIR}"

echo "[done] Fig. 5 → ${FIGS_DIR}/distribution_*"
