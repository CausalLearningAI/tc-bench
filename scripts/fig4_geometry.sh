#!/bin/bash
# =============================================================================
# Fig. 4 — §4.2 intrinsic-geometry diagnostics on DINOv3-base.
#
# Paper §4.2: PCA / effective dimensionality d_eff / feature spread within
# pressure bins. Analyses are restricted to bins with N ≥ 500, on the
# FULL dataset (not balanced), CLS token, for DINOv3-base specifically
# (the strongest-performing model under linear Q_stat).
#
# Pinned settings:
#   models        dinov3-base
#   feature_type  cls
#   data          full
#   (no seed sweep — geometry is deterministic per fixed feature set)
# =============================================================================

print_help() { sed -n '2,14p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig4_geometry"

MODELS="dinov3-base"          # paper §4.2 restricts to this model
FEATURE_TYPES="cls"
DATA="${DATA_FULL}"
export MODELS FEATURE_TYPES DATA

# Geometry diagnostics (single-model "array" of length 1).
submit_model_array "${REPO_ROOT}/probing/slurm/geometry_all.sh" 0

# --- Render the three-panel figure (PC1 vs P_c, d_eff bars, feature spread) --
python "${REPO_ROOT}/figures/fig4_geometry.py" \
    --feature_path "${FEATURES_DIR}/features_dinov3-base" \
    --split test \
    --output "${FIGS_DIR}/fig4_geometry.pdf"

# --- Optional sandbox panel: per-bin PC1 scatter -----------------------------
# fig4_pca1.py has no CLI; paths are hardcoded in the script.
python "${REPO_ROOT}/figures/fig4_pca1.py"

echo "[done] Fig. 4 → ${FIGS_DIR}/fig4_geometry.pdf"
