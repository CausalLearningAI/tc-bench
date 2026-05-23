#!/bin/bash
# =============================================================================
# Fig. 2 — Static Fidelity Q_stat boxplot (CLS token) for all 11 VFMs.
#
# Paper §4.1: trajectory-level split + regime-balanced eval, ridge probe with
# 5-fold CV over α ∈ {1e-3, …, 1e6}, normalized error ξ_stat = |y - Lz| / σ(P_c).
#
# Pinned settings:
#   probe         ridge
#   feature_type  cls
#   target        pressure  (paper's primary axis; wind is App. E auxiliary)
#   data          balanced_980
#   models        all 11 VFMs (MODELS_ALL in probing/slurm/_env.sh)
#   seeds         42 43 44 45 46
#
# Usage:
#   bash scripts/fig2_qstat_cls.sh                # local, all stages
#   bash scripts/fig2_qstat_cls.sh --slurm        # submit via sbatch
# =============================================================================

print_help() { sed -n '2,21p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig2_qstat_cls"

PROBE="${PROBE_DEFAULT}"
FEATURE_TYPES="cls"
TARGETS="pressure"
SEEDS="${SEEDS_DEFAULT}"
DATA="${DATA_BALANCED}"
export PROBE FEATURE_TYPES TARGETS SEEDS DATA

# --- 1. Fit ridge probes for every (model × seed) -----------------------------
submit_model_array "${REPO_ROOT}/probing/slurm/fit_all.sh"

# --- 2. Q_stat diagnostic (predictions CSV per seed) --------------------------
submit_diagnostic q_stat

# --- 3. Aggregate to summary.csv (cheap, run locally) -------------------------
python -m probing.aggregate

# --- 4. Render the figure -----------------------------------------------------
# fig2_q_stat.py plots one seed at a time; we use seed=42 for the headline PDF.
python "${REPO_ROOT}/figures/fig2_q_stat.py" \
    --predictions "${DIAG_DIR}/q_stat/predictions_cls_pressure_seed42.csv" \
    --output "${FIGS_DIR}/fig2_q_stat.pdf"

echo "[done] Fig. 2 → ${FIGS_DIR}/fig2_q_stat.pdf"
