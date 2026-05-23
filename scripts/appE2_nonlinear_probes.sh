#!/bin/bash
# =============================================================================
# App. E.2 / Table 4 — Nonlinear probe controls on DINOv3-base.
#
# Paper §E.2: re-runs Q_stat with higher-capacity readouts to rule out a
# probe-capacity explanation for the regime-dependent failure. Reports
# moderate / intense normalized MAE as mean ± std over THREE seeds.
#
# Pinned settings:
#   models        dinov3-base    (paper §E.2 restricts to this model)
#   feature_type  cls
#   target        pressure
#   data          balanced_980
#   seeds         42 43 44       (paper: "mean ± std over three independent runs")
#   probes        mlp,transformer (Table 4's two rows)
# =============================================================================

print_help() { sed -n '2,14p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "appE2_nonlinear_probes"

MODELS="dinov3-base"
PROBE="mlp,transformer"
FEATURE_TYPES="cls"
TARGETS="pressure"
SEEDS="${SEEDS_NONLINEAR}"     # 42,43,44
DATA="${DATA_BALANCED}"
export MODELS PROBE FEATURE_TYPES TARGETS SEEDS DATA

# Single-model "array" of length 1; the probe sweep is handled by Hydra
# multirun inside probing/slurm/fit_all.sh.
submit_model_array "${REPO_ROOT}/probing/slurm/fit_all.sh" 0

# Aggregate so summary.csv picks up the new rows (Table 4 reads from here).
python -m probing.aggregate

echo "[done] App. E.2 — MLP + Transformer probes on DINOv3-base (3 seeds each)"
echo "       Inspect outputs/summary.csv for mean ± std over seed."
