#!/bin/bash
# Set up the uv-managed virtualenv used by every other script in this repo.
#
# Mirrors README §1: creates .venv at the repo root with Python 3.10,
# installs the `tcbench` package in editable mode, and adds the runtime
# dependency set (Hydra, Lightning, scikit-learn, xarray, …).
#
# Usage:
#     bash _install.sh              # auto-detect CUDA (nvidia-smi)
#     bash _install.sh --cpu        # force CPU wheels
#     bash _install.sh --cuda cu121 # pin a specific CUDA index
#
# Idempotent: re-running upgrades packages in place and skips venv creation
# when .venv/ already exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Resolve torch wheel index --------------------------------------------------
TORCH_INDEX=""
for arg in "$@"; do
    case "${arg}" in
        --cpu)    TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        --cuda)   shift ;; # value is consumed below
        cu*)      TORCH_INDEX="https://download.pytorch.org/whl/${arg}" ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    esac
done
if [[ -z "${TORCH_INDEX}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
fi

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# 2. venv --------------------------------------------------------------------
if [[ ! -d .venv ]]; then
    uv venv --python 3.10 .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# 3. PyTorch (CUDA-matched index) -------------------------------------------
echo "[install] torch from ${TORCH_INDEX}"
uv pip install "torch==2.4.*" torchvision --index-url "${TORCH_INDEX}"

# 4. tcbench (editable) + the rest of the runtime deps ----------------------
uv pip install -e .
uv pip install \
    "hydra-core>=1.3" hydra-colorlog hydra-optuna-sweeper \
    "lightning>=2.0" torchmetrics rootutils \
    datasets huggingface_hub transformers \
    scikit-learn scipy pandas numpy joblib jaxtyping \
    matplotlib seaborn colormaps tqdm rich \
    netCDF4 cfgrib xarray zarr \
    albumentations beautifulsoup4 requests wget \
    pytest

echo
echo "[done] .venv ready — activate with: source .venv/bin/activate"
echo "       sanity check: pytest"
