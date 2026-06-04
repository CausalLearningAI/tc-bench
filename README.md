<p align="center">
  <video src="assets/PerceptionPhysicsParadox.mp4" controls autoplay muted loop playsinline width="900">
    <a href="assets/PerceptionPhysicsParadox.mp4">Watch the Perception-Physics Paradox animation</a>
  </video>
</p>

<h1 align="center">TC-Bench</h1>

<p align="center">
  <strong>Probing scientific alignment in vision foundation models</strong><br>
  Companion code release for <strong>"The Perception–Physics Paradox: Probing Scientific Alignment with TC-Bench"</strong><br>
  Yao, Polesello, Pervez, Muller, Locatello — ICML 2026
</p>

<p align="center">
  <a href="#1-installation">Install</a> ·
  <a href="#2-data-construction-3">Build the dataset</a> ·
  <a href="#3-reproducing-the-paper">Reproduce figures</a> ·
  <a href="assets/PerceptionPhysicsParadox.mp4">Open the hero video</a>
</p>

TC-Bench is a global tropical-cyclone benchmark with a fully reproducible
construction pipeline (§3) and a structured **probing protocol** (§4) for
asking whether frozen Vision Foundation Models (VFMs) preserve the
physical degrees of freedom that scientific reasoning needs. The
headline finding: VFM representations stay perceptually robust but
**collapse along physically meaningful axes in intense regimes**
($P_c < 980$ hPa), so they sail through standard OOD tests yet fail
structural-alignment probes.

### Dataset example

<p align="center">
  <img src="assets/cyclone_animation.gif" alt="Animated infrared tropical cyclone sequence from TC-Bench" width="720">
</p>

TC-Bench turns temporal infrared cyclone imagery like this into paired
physical labels, frozen-model features, and regime-aware probes for
pressure, wind, and structural alignment.

This README is the single entry point for reproducing every paper
result and figure. It is organised as:

1. [Installation](#1-installation) — `uv`-based, no conda required.
2. [Data construction (§3)](#2-data-construction-3) — build TC-Bench and
   extract frozen VFM features.
3. [Reproducing the paper (§4 + figures)](#3-reproducing-the-paper) — one
   recipe per paper figure / table. The fastest path is the per-figure
   wrappers under [`scripts/`](scripts/) — see [§3.3](#33-per-figure-reproduction-recipes).

Subdirectory READMEs ([`dataset/`](dataset/), [`probing/`](probing/),
[`figures/`](figures/), [`scripts/`](scripts/)) carry the deeper,
file-level crosswalks. The paper PDF [`Cyclones.pdf`](Cyclones.pdf) is
the source of truth for what each component is supposed to do.

## Repository layout

```
ClimGIVT/
├── dataset/      §3 construction pipeline (9 stages + run_all.sh + SLURM)
├── probing/      §4 probes: fit / diagnose / geometry / aggregate (Hydra)
├── figures/      argparse-driven figure scripts + shared _style.py
├── scripts/      per-figure bash wrappers that pin the paper settings
├── src/          App. E.1 pixel-supervision ablation only
├── configs/      Hydra root for src/train + src/eval
├── notebooks/    exploratory analyses (NOT load-bearing)
├── tests/        unit + end-to-end smoke tests (56 tests, ~6 s, no GPU)
├── Cyclones.pdf  the paper
└── README.md     ← you are here
```

---

## 1. Installation

The dependency set is pure Python; we recommend [`uv`](https://docs.astral.sh/uv/)
because it resolves the lock in seconds and avoids the conda hooks that
broke the older `environment.yaml` flow on several cluster hosts.

### 1.1 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pipx install uv
```

### 1.2 One-shot setup with `_install.sh`

`_install.sh` at the repo root creates `.venv` (Python 3.10), installs
the `tcbench` package in editable mode, and pulls the full runtime
dependency set. It auto-detects CUDA via `nvidia-smi` and picks the
matching `torch==2.4.*` wheel index.

```bash
git clone https://github.com/CausalLearningAI/tc-bench.git
cd tc-bench
bash _install.sh                 # CUDA 12.1 when nvidia-smi is present, else CPU
bash _install.sh --cpu           # force CPU wheels
bash _install.sh --cuda cu118    # pin a different CUDA index
source .venv/bin/activate
```

The script is idempotent — re-running it upgrades packages in place
and skips venv creation when `.venv/` already exists.

### 1.3 Sanity check

```bash
pytest                                       # 56 tests, ~8 s, no GPU required
```

You should see `56 passed`. If you only changed `probing/` or `src/`,
`pytest tests/test_probing_core.py` (~3 s — includes one tiny PyTorch
TransformerProbe round-trip) is enough for the inner loop.

### 1.4 Optional: cluster (SLURM)

The dataset and probing pipelines auto-detect SLURM. `dataset/slurm/_env.sh`
and `probing/slurm/_env.sh` source `${REPO_ROOT}/.venv/bin/activate` when
present and fall back to `conda activate "${CONDA_ENV}"` only if you
explicitly set `CONDA_ENV` in your shell — so a fresh uv-only setup needs
no extra flag. See those two files for the full list of overridable
variables (`DATA_ROOT`, `FEATURES_DIR`, `PROBE_DIR`, `DIAG_DIR`, `GEOM_DIR`,
`MODELS`, …). `_env.sh` also exports `PYTHONPATH=$REPO_ROOT` so the
`figures/*.py` scripts (which `import figures._style`) resolve when run
via `python figures/foo.py`.

---

## 2. Data construction (§3)

§3 of the paper builds TC-Bench from two public sources:

1. **IbTRACS v4r01** — multi-agency best-track records (storm centre,
   minimum central pressure $P_c$, maximum sustained wind $V_m$),
2. **GridSat-B1** — three-hourly global infrared brightness-temperature
   netCDFs.

The pipeline lives entirely under [`dataset/`](dataset/) and is split
into nine numbered, idempotent stages. Each stage writes a fresh
artefact under `$DATA_ROOT` and is skipped on re-run if its output
already exists.

| #  | Script                            | Output (under `$DATA_ROOT`)                       | Notes |
|----|-----------------------------------|---------------------------------------------------|-------|
| 01 | `01_download_ibtracs.py`          | `ibtracs/ibTRACS_since_1980.csv`                  | ~50 MB |
| 02 | `02_download_gridsat.py`          | `gridsat/{year}/*.nc`                             | ~500 GB; the heavy step |
| 03 | `03_preprocess.py`                | `preprocessed/`                                   | merge tracks + IR onto a common 3-hourly grid |
| 04 | `04_crop_windows.py`              | `cropped/{cyclone_id}/{timestep}.nc`              | fixed-size windows around each best-track fix |
| 05 | `05_consolidate_nc.py`            | `consolidated/{cyclone_id}.nc`                    | one netCDF per cyclone |
| 06 | `06_build_hf.py`                  | `dataset_hf/` (HuggingFace Arrow)                 | trajectory-level 80/10/10 split |
| 07 | `07_normalize_stats.py`           | `dataset_hf/normalization_stats.json`             | per-channel mean/std |
| 08 | `08_ood_basin_split.py`           | `image_features/features_<model>_ood_splits/`     | App. E cross-basin OOD only |
| 09 | `09_extract_features.py`          | `image_features/features_<model>/`                | frozen-VFM features for §4 |

Final dataset after preprocessing: **2 813 train · 352 val · 352 test
cyclones** (trajectory-level splits — no temporal leakage).

### 2.1 One-shot build

```bash
# Local, full pipeline:
bash dataset/run_all.sh

# SLURM, one sbatch per stage:
bash dataset/run_all.sh --slurm

# Resume / run subset:
bash dataset/run_all.sh --from 06         # resume from stage 06
bash dataset/run_all.sh --only 09         # just stage 09
```

`DATA_ROOT` defaults to `${HOME}/tcbench`. Override from the shell:

```bash
DATA_ROOT=/scratch/$USER/tcbench bash dataset/run_all.sh
```

### 2.2 Feature extraction (stage 09)

Stage 09 runs frozen inference for each VFM and is the only stage that
needs a GPU. The paper uses 11 VFMs:

```text
dinov2-base   dinov2-large
dinov3-base   dinov3-large   dinov3-satellite
clip-base     clip-large
siglip-base   siglip2-base
mae-base      mae-large
```

The SLURM wrapper submits all 11 as an array; throttle with
`#SBATCH --array=0-10%N`. To restrict the set:

```bash
MODELS="dinov3-base dinov3-large" bash dataset/run_all.sh --only 09
```

Stage 09 writes per-frame **CLS** and **spatial-mean** features into
`$FEATURES_DIR/features_<model>/`. These are what `probing/` consumes.

See [`dataset/README.md`](dataset/README.md) for the full env-variable
contract.

---

## 3. Reproducing the paper

After §2 you have `$DATA_ROOT/image_features/features_<model>/` for
each VFM. Everything in §4 of the paper is reproduced from those
features plus the four Hydra entries in [`probing/`](probing/).

### 3.1 The probing protocol

The protocol composes four stages (`probing/run_all.sh` wires them
together):

```text
fit       — fit a ridge probe per (model, feature_type, target, seed)
            on a regime-balanced trajectory-level split.
diagnose  — re-use the fitted linear map to build prediction CSVs for
            the three diagnostics (Q_stat, Q_dyn, Q_con).
geometry  — §4.2 per-pressure-bin effective dimensionality, feature
            spread, PC1 means.
aggregate — collect every per-fit JSON into outputs/summary.csv.
```

#### Paper sweep

```bash
# Local end-to-end (≈ all four stages, single GPU):
bash probing/run_all.sh

# SLURM:
bash probing/run_all.sh --slurm

# Subset:
bash probing/run_all.sh --only fit
bash probing/run_all.sh --only diagnose geometry
```

The default sweep is the one used in the paper:

```text
11 VFMs × 2 feature_types (cls, spatial_mean) × 2 targets (pressure, wind)
       × 5 seeds (42–46) = 220 ridge fits
```

#### Single fits (handy for debugging)

```bash
python -m probing.fit probe=ridge model=dinov3-base feature_type=cls target=pressure seed=42
python -m probing.diagnose diagnostic=q_stat feature_type=cls target=pressure
python -m probing.geometry  model=dinov3-base feature_type=cls
python -m probing.aggregate
```

#### Probe-capacity sanity checks (App. E.2)

The protocol is **probe-agnostic**. To swap in a different probe
family, change `probe=`:

```bash
python -m probing.fit probe=lasso       model=dinov3-base ...
python -m probing.fit probe=mlp         model=dinov3-base ...   # sklearn MLPRegressor((2048,), max_iter=100) — paper App. E.2 default
python -m probing.fit probe=transformer model=dinov3-base ...   # 2-layer encoder, hidden=128, 4 heads, 4 learned tokens
```

`probe=mlp` and `probe=transformer` together reproduce the two rows of
Table 4.

#### Normalized error reporting (Eq. 4.1)

`probing.fit` records, for the overall test split and for each pressure
regime (intense / moderate, threshold 980 hPa):

* `rmse`, `mae` — raw values in hPa (or kt for wind),
* `sigma` — the global standard deviation of the test-split target,
* `normalized_rmse = rmse / sigma`, `normalized_mae = mae / sigma`.

The same `sigma` is used as the denominator for the overall row *and*
both regime rows, so the regime-level normalized errors are directly
comparable — a per-regime σ would erase the moderate-to-intense gap
reported in Fig. 2. The top-level JSON also carries `sigma_global` so
the aggregator surfaces it in `outputs/summary.csv`.

### 3.2 Outputs layout

```text
outputs/
├── probes/<model>/<probe>/<feature_type>_<target>_seed<seed>.{pkl,json}
├── diagnostics/{q_stat,q_dyn,q_con}/predictions_<feature_type>_<target>_seed<seed>.csv
├── geometry/<model>/<feature_type>.csv
└── summary.csv                                                 ← aggregate
```

### 3.3 Per-figure reproduction recipes

The fastest path is the thin bash wrappers under [`scripts/`](scripts/).
Each wrapper pins the exact paper settings (probe family, feature type,
data subset, seed list) for one figure and delegates the heavy work to
the `probing/slurm/*.sh` sweeps and the `figures/*.py` plot scripts —
so the experiment metadata lives in one place per figure.

```bash
# Everything, local:
bash scripts/run_all.sh

# Everything, via sbatch:
bash scripts/run_all.sh --slurm

# Just one or two figures:
bash scripts/run_all.sh --only fig2 fig3
bash scripts/fig4_geometry.sh
```

Per-figure wrappers (see [`scripts/README.md`](scripts/README.md) for
the full table):

| Paper figure / table                             | Wrapper                                                                                       |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Fig. 1c**   cross-agency OOD bar chart         | [`scripts/fig1c_ood.sh`](scripts/fig1c_ood.sh)                                                |
| **Fig. 2**    Q_stat CLS (5 seeds, 11 VFMs)      | [`scripts/fig2_qstat_cls.sh`](scripts/fig2_qstat_cls.sh)                                      |
| **Fig. 3**    Q_dyn CLS (reuses Fig. 2 probes)   | [`scripts/fig3_qdyn_cls.sh`](scripts/fig3_qdyn_cls.sh)                                        |
| **Fig. 4**    intrinsic geometry (DINOv3-base)   | [`scripts/fig4_geometry.sh`](scripts/fig4_geometry.sh)                                        |
| **Fig. 5 / App. D** dataset histograms           | [`scripts/fig5_dataset_dist.sh`](scripts/fig5_dataset_dist.sh)                                |
| **Fig. 6 / App. E.1** ResNet-18 pixel-sup        | [`scripts/fig6_pixelsup.sh`](scripts/fig6_pixelsup.sh)                                        |
| **Fig. 7 / App. E.3** Q_stat spatial_mean        | [`scripts/fig7_qstat_spatial_mean.sh`](scripts/fig7_qstat_spatial_mean.sh)                    |
| **Fig. 8 / App. E.3** Q_dyn spatial_mean         | [`scripts/fig8_qdyn_spatial_mean.sh`](scripts/fig8_qdyn_spatial_mean.sh)                      |
| **§4.1 Q_con** pressure–wind coupling            | [`scripts/fig_qcon.sh`](scripts/fig_qcon.sh)                                                  |
| **App. E.2 Table 4** MLP + Transformer probes    | [`scripts/appE2_nonlinear_probes.sh`](scripts/appE2_nonlinear_probes.sh)                      |

Not reproducible from HEAD: **App. E.4 Table 5** (VideoMAE / V-JEPA2 /
X-CLIP, 10 seeds) — the video-feature extraction code was intentionally
removed (see `git log` and `CLAUDE.md`).

If you'd rather drive the pipeline by hand, [`figures/`](figures/) holds
the argparse-driven plot scripts; the table below names the exact
artefact each one expects after the relevant `probing.*` stage.

| Paper figure / table                              | Script                                                                                  | Prerequisite stage |
|---------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------|
| **Fig. 1c**    cross-agency OOD bar chart         | [`figures/fig_ood.py`](figures/fig_ood.py)                                              | `fit data=ood_basin` (App. E) |
| **Fig. 2**     Q_stat boxplot                     | [`figures/fig2_q_stat.py`](figures/fig2_q_stat.py)                                      | `diagnose diagnostic=q_stat` |
| **Fig. 3**     Q_dyn coherence vs $P_c$           | [`figures/fig3_q_dyn.py`](figures/fig3_q_dyn.py)                                        | `diagnose diagnostic=q_dyn`  |
| **Fig. 4**     PC1 / d_eff / feature spread       | [`figures/fig4_geometry.py`](figures/fig4_geometry.py)                                  | features only (stage 09)     |
| **Fig. 4a (alt)** per-bin PC1 scatter             | [`figures/fig4_pca1.py`](figures/fig4_pca1.py)                                          | features only (stage 09)     |
| **§4.1 Q_con** scatter + gap evolution            | [`figures/fig_q_con.py`](figures/fig_q_con.py)                                          | `diagnose diagnostic=q_con`  |
| **Tab. 1**    probe values per model              | `outputs/summary.csv`                                                                   | `aggregate`                  |
| **App. D**    agency histograms                   | [`figures/fig_data_distribution.py`](figures/fig_data_distribution.py)                  | `diagnose diagnostic=q_stat` |
| **App. E.1**  supervised pixel-baseline vs probe  | [`figures/fig_baseline_vs_dinov3.py`](figures/fig_baseline_vs_dinov3.py)                | CNN predictions CSV (§3.4) + `diagnose q_stat` |
| **App. E.2**  capacity sanity feature scatter     | [`figures/fig_feature_analysis.py`](figures/fig_feature_analysis.py)                    | `diagnose diagnostic=q_stat` |

#### Recipes (copy/paste)

```bash
# --- Fig. 2: Static fidelity (Q_stat) -----------------------------------
python -m probing.diagnose diagnostic=q_stat feature_type=cls target=pressure
python figures/fig2_q_stat.py \
    --predictions outputs/diagnostics/q_stat/predictions_cls_pressure_seed42.csv \
    --output figs/fig2_q_stat.pdf

# --- Fig. 3: Dynamic coherence (Q_dyn) ----------------------------------
python -m probing.diagnose diagnostic=q_dyn feature_type=spatial_mean target=pressure
python figures/fig3_q_dyn.py \
    --predictions outputs/diagnostics/q_dyn/predictions_spatial_mean_pressure_seed42.csv \
    --output figs/fig3_q_dyn.pdf

# --- §4.1 Q_con: pressure–wind coupling ---------------------------------
python -m probing.diagnose diagnostic=q_con data=us_only feature_type=cls
python figures/fig_q_con.py \
    --predictions outputs/diagnostics/q_con/predictions_cls_seed42.csv \
    --output_dir figs/

# --- Fig. 4: §4.2 latent collapse ---------------------------------------
python figures/fig4_geometry.py \
    --feature_path "$DATA_ROOT/image_features/features_dinov3-base" \
    --split test \
    --output figs/fig4_geometry.pdf

# --- Tab. 1: collected probe metrics ------------------------------------
python -m probing.aggregate                # writes outputs/summary.csv

# --- App. D: agency histograms (defaults to outputs/diagnostics/q_stat) ----
# Prerequisite: q_stat diagnostic for BOTH targets on `balanced_980`.
# scripts/fig5_dataset_dist.sh does the missing wind fit lazily.
python figures/fig_data_distribution.py

# --- App. E.2: feature scatter -----------------------------------------
python figures/fig_feature_analysis.py \
    --results_dir outputs/diagnostics/q_stat
```

#### Known fragilities

* [`figures/fig4_pca1.py`](figures/fig4_pca1.py) — optional sandbox panel
  invoked at the tail of `scripts/fig4_geometry.sh`. Still hardcodes
  `feat_path` in `__main__`; edit to `$DATA_ROOT/image_features/features_dinov3-base`
  if you want the per-bin PC1 scatter. The wrapper continues even if this
  step is skipped, because the main `fig4_geometry.pdf` is already saved.

### 3.4 App. E.1: pixel-supervision ablation

The supervised pixel baseline lives under [`src/`](src/) and uses a
separate Hydra root ([`configs/train.yaml`](configs/train.yaml)). It
does *not* touch any VFM weights — it just trains a small CNN / ResNet
end-to-end on raw IR frames so the paper can rule out "the data lacks
the signal" as an explanation for the intense-regime collapse.

```bash
# SimpleCNN (~40 M params), 500 epochs, GPU:
python -m src.train experiment=simple_cnn

# ResNet-18 from scratch (paper §E.1 hyperparams pinned in the config):
python -m src.train experiment=train_resnet

# Quick smoke test (1 batch, no logger):
python -m src.train experiment=simple_cnn trainer.fast_dev_run=true logger=null

# Evaluate a checkpoint and dump per-sample predictions:
python -m src.eval ckpt_path=/path/to/last.ckpt
```

The eval entry writes a per-sample predictions CSV that feeds
`figures/fig_baseline_vs_dinov3.py` (App. E.1 Figure). The ResNet-18
hyperparameters in [`configs/experiment/train_resnet.yaml`](configs/experiment/train_resnet.yaml)
match paper §E.1 verbatim (AdamW lr=1e-4 / wd=1e-2, hidden_dim=128,
dropout=0.3, batch_size=64, cosine + 100-step warmup, 200 epochs) — do
not drift them; [`scripts/fig6_pixelsup.sh`](scripts/fig6_pixelsup.sh)
relies on the config being the source of truth.

### 3.5 Baselines (Dvorak, climatology)

```bash
# Dvorak (1975) operational baseline:
python -m probing.baselines.dvorak \
    --dataset_path "$DATA_ROOT/dataset_hf" \
    --output_path outputs/baselines/dvorak.json

# Per-basin / per-month climatology baseline:
python -m probing.baselines.climatology \
    --dataset_path "$DATA_ROOT/dataset_hf" \
    --output_dir outputs/baselines/

# SLURM wrappers:
sbatch probing/slurm/baseline_dvorak.sh
sbatch probing/slurm/baseline_climatology.sh
```

---

## Tips and known caveats

- **Hydra previews.** Every Hydra entry supports `--cfg job` to print
  the fully resolved config without running anything:
  `python -m probing.fit probe=ridge model=dinov3-base --cfg job`.
- **Cluster GPUs.** `MLPProbe` is sklearn-backed and runs on CPU.
  `TransformerProbe` defaults to `device=cuda`; if a login node is
  busy, fall back with `probe.params.device=cpu` or submit via
  `sbatch probing/slurm/fit_all.sh`. A reloaded `TransformerProbe`
  infers its device from the loaded model parameters, so a CPU host
  can read a probe trained on GPU without manual `.to()` calls.
- **Don't bring deleted files back.** The VFM Lightning regressors,
  Koopman scripts, `eval_old.py`, MNIST configs, and `fast_datamodule`
  were intentionally removed (`git log` carries the rationale). If you
  think you need them, check whether the probing protocol now covers
  the use case.
- **rtk caches tee output.** If `pytest` reports "No tests collected"
  but a stale rtk log claims failure, trust the live exit code.

## Citing

```bibtex
@inproceedings{yao2026perception,
    title     = {The Perception--Physics Paradox: Probing Scientific Alignment with TC-Bench},
    author    = {Yao, Dingling and Polesello, Andrea and Pervez, Adeel and
                 Muller, Caroline and Locatello, Francesco},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year      = {2026},
}
```

## License

MIT. See `LICENSE`.
