#!/bin/bash
# =============================================================================
# Fig. 6 / App. E.1 — Supervised pixel-space baseline (ResNet-18 from scratch).
#
# The paper's §E.1 hyperparameters (arch=resnet18, base_channels=64,
# hidden_dim=128, dropout=0.3, AdamW lr=1e-4 wd=1e-2, cosine + 100-step
# warmup, batch_size=64, 200 epochs) live in
# configs/experiment/train_resnet.yaml — do not duplicate them here.
# Single seed (paper does not report multi-seed for this control).
#
# This is a sanity-check baseline, NOT a competitive benchmark — it
# demonstrates that intense-regime pressure remains statistically learnable
# from imagery, ruling out task difficulty as the cause of VFM degradation.
# =============================================================================

print_help() { sed -n '2,13p' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do [[ "${arg}" == "-h" || "${arg}" == "--help" ]] && { print_help; exit 0; }; done
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
parse_slurm_flag "$@"
banner_fig "fig6_pixelsup"

# --- 1. Train ResNet-18 from scratch -----------------------------------------
TRAIN_CMD=(python -m src.train experiment=train_resnet seed=42)

if (( USE_SLURM )); then
    sbatch --job-name=tc_pixelsup --gres=gpu:1 --partition=gpu100 \
           --cpus-per-task=8 --mem=64G --time=12:00:00 \
           --output="${REPO_ROOT}/logs/tc_pixelsup_%j.out" \
           --error="${REPO_ROOT}/logs/tc_pixelsup_%j.err" \
           --wrap="${TRAIN_CMD[*]}"
else
    "${TRAIN_CMD[@]}"
fi

# --- 2. Evaluate the best checkpoint to produce the predictions CSV ----------
# The eval entry point writes to its Hydra run dir; set CKPT_PATH from the
# shell when running this stage, or skip it and pass the CSV path to step 3
# manually.
if [[ -n "${CKPT_PATH:-}" ]]; then
    python -m src.eval ckpt_path="${CKPT_PATH}"
fi

# --- 3. Comparison figure (paper Fig. 6 + companion scatter) -----------------
# fig_baseline_vs_dinov3.py expects two CSVs: the pixel-baseline predictions
# and the corresponding DINOv3-base Q_stat predictions (from fig2 stage).
BASELINE_CSV="${BASELINE_CSV:-${REPO_ROOT}/outputs/eval/resnet18_predictions.csv}"
DINOV3_CSV="${DINOV3_CSV:-${DIAG_DIR}/q_stat/predictions_cls_pressure_seed42.csv}"

if [[ -f "${BASELINE_CSV}" && -f "${DINOV3_CSV}" ]]; then
    python "${REPO_ROOT}/figures/fig_baseline_vs_dinov3.py" \
        --baseline_csv "${BASELINE_CSV}" \
        --dinov3_csv "${DINOV3_CSV}" \
        --output_dir "${FIGS_DIR}"
else
    echo "[skip] Fig. 6 comparison plot — missing one of:"
    echo "       baseline:  ${BASELINE_CSV}"
    echo "       dinov3:    ${DINOV3_CSV}"
fi

echo "[done] Fig. 6 / App. E.1 — pixel-supervision baseline"
