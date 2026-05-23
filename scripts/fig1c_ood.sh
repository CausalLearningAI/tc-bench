#!/bin/bash
# =============================================================================
# Fig. 1c — Cross-agency OOD MAE bar chart (DINOv3-base only).
#
# For each of nine agencies in
#   $FEATURES_DIR/features_<MODEL>_ood_splits/ood_basin_<basin>/
# we fit a ridge probe on that basin's re-carved train split and evaluate on
# its held-out test split. The "in distribution" reference is the regular
# Fig. 2 fit on `features_<MODEL>/`.
#
# Stage 08 only writes ood_splits for dinov3-base by default — override with
# MODEL=<other> if you have additional ood_splits trees on disk.
#
# Pinned settings:
#   probe         ridge
#   model         dinov3-base   (paper §E.4 / Fig. 1c restricts to this model)
#   feature_type  cls
#   target        pressure
#   data          full          (each ood split is itself a full train/val/test)
#   seed          42
# =============================================================================

print_help() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig1c_ood"

MODEL="${MODEL:-dinov3-base}"
BASINS=(atcf bom hurdat_atl hurdat_epa nadi newdelhi reunion tokyo wellington)
OOD_ROOT="${FEATURES_DIR}/features_${MODEL}_ood_splits"

if [[ ! -d "${OOD_ROOT}" ]]; then
    echo "[fig1c] Missing ${OOD_ROOT}" >&2
    echo "[fig1c] Run dataset stage 08 (08_ood_basin_split.py) for MODEL=${MODEL} first." >&2
    exit 1
fi

# --- 1. In-distribution baseline (writes outputs/probes/<MODEL>/ridge/...) ---
python -m probing.fit \
    probe=ridge model="${MODEL}" data=full \
    feature_type=cls target=pressure seed=42

# --- 2. One per-basin OOD fit (registered as model "<MODEL>_ood_<basin>") ---
# Each basin lives under .../features_<MODEL>_ood_splits/ood_basin_<basin>/.
# We symlink it under a fresh feature_dataset_root so the fit loader resolves
# the path via `features_<MODEL>_ood_<basin>`, which Hydra writes to a
# disjoint outputs/probes/ subdir.
WORKDIR="${REPO_ROOT}/outputs/_ood_workdir"
mkdir -p "${WORKDIR}"
for b in "${BASINS[@]}"; do
    ln -sfn "${OOD_ROOT}/ood_basin_${b}" "${WORKDIR}/features_${MODEL}_ood_${b}"
done

for b in "${BASINS[@]}"; do
    # Skip basins that already have a result on disk (idempotent re-runs).
    out_json="${PROBE_DIR}/${MODEL}_ood_${b}/ridge/cls_pressure_seed42.json"
    if [[ -s "${out_json}" ]]; then
        echo "[fig1c] ${b}: cached"
        continue
    fi
    python -m probing.fit \
        probe=ridge model="${MODEL}" data=full \
        feature_type=cls target=pressure seed=42 \
        model.name="${MODEL}_ood_${b}" \
        data.feature_dataset_root="${WORKDIR}"
done

# --- 3. Render the cross-agency bar chart ------------------------------------
python "${REPO_ROOT}/figures/fig_ood.py" \
    --probe_dir "${PROBE_DIR}" \
    --model "${MODEL}" \
    --output "${FIGS_DIR}/fig1c_ood.pdf"

echo "[done] Fig. 1c → ${FIGS_DIR}/fig1c_ood.pdf"
