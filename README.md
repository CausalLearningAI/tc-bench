<p align="center">
  <a href="assets/PerceptionPhysicsParadox.mp4">
    <img src="assets/PerceptionPhysicsParadox.gif" alt="Perception-Physics Paradox animation" width="900">
  </a>
  <br>
  <a href="assets/PerceptionPhysicsParadox.mp4">Open the Perception-Physics Paradox MP4</a>
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
  <a href="#3-reproducing-the-paper">Reproduce results</a> ·
  <a href="#citing">Cite</a>
</p>

TC-Bench is a global tropical-cyclone benchmark with a fully reproducible
construction pipeline (§3) and a structured **probing protocol** (§4) for
asking whether frozen Vision Foundation Models (VFMs) preserve the
physical degrees of freedom that scientific reasoning needs. The
headline finding: VFM representations stay perceptually robust but
**collapse along physically meaningful axes in intense regimes**
($P_c < 980$ hPa), so they sail through standard OOD tests yet fail
structural-alignment probes.

## Dataset example

<p align="center">
  <img src="assets/cyclone_animation.gif" alt="Animated infrared tropical cyclone sequence from TC-Bench" width="214">
</p>

TC-Bench turns temporal infrared cyclone imagery like this into paired
physical labels, frozen-model features, and regime-aware probes for
pressure, wind, and structural alignment.

## What is included

```
tc-bench/
├── dataset/      §3 construction pipeline (9 stages + run_all.sh + SLURM)
├── probing/      §4 probes: fit / diagnose / geometry / aggregate (Hydra)
├── figures/      figure-generation scripts
├── scripts/      one-command wrappers for paper figures and tables
├── src/          supervised pixel-baseline experiments
├── configs/      Hydra root for src/train + src/eval
├── notebooks/    exploratory analyses
├── tests/        unit + smoke tests
└── assets/       README media
```

---

## 1. Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for a fast, reproducible
Python setup.

### 1.1 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pipx install uv
```

### 1.2 One-shot setup with `_install.sh`

`_install.sh` at the repo root creates `.venv` (Python 3.10), installs
the `tcbench` package in editable mode, and installs the runtime
dependencies. It auto-detects CUDA with `nvidia-smi`; use `--cpu` to
force CPU-only wheels.

```bash
git clone https://github.com/CausalLearningAI/tc-bench.git
cd tc-bench
bash _install.sh                 # CUDA 12.1 when nvidia-smi is present, else CPU
bash _install.sh --cpu           # force CPU wheels
bash _install.sh --cuda cu118    # pin a different CUDA index
source .venv/bin/activate
```

### 1.3 Sanity check

```bash
pytest
```

The test suite is CPU-only and should finish quickly.

### 1.4 Optional: cluster (SLURM)

The dataset and probing pipelines support SLURM:

```bash
bash dataset/run_all.sh --slurm
bash probing/run_all.sh --slurm
```

Environment variables such as `DATA_ROOT`, `FEATURES_DIR`, `PROBE_DIR`,
and `MODELS` can be set from the shell before running these scripts.

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
each VFM. The probing workflow then fits probes, runs diagnostics,
computes geometry summaries, and aggregates the results.

### 3.1 The probing protocol

Run the full paper sweep:

```bash
bash probing/run_all.sh

# Or submit through SLURM:
bash probing/run_all.sh --slurm

# Or run selected stages:
bash probing/run_all.sh --only fit
bash probing/run_all.sh --only diagnose geometry
```

The default configuration matches the paper sweep:

```text
11 VFMs × 2 feature_types (cls, spatial_mean) × 2 targets (pressure, wind)
       × 5 seeds (42–46) = 220 ridge fits
```

The four stages are:

```text
fit       Fit ridge probes on frozen VFM features.
diagnose  Build prediction CSVs for Q_stat, Q_dyn, and Q_con.
geometry  Compute per-pressure-bin geometry summaries.
aggregate Collect per-fit JSON files into outputs/summary.csv.
```

### 3.2 Outputs layout

```text
outputs/
├── probes/<model>/<probe>/<feature_type>_<target>_seed<seed>.{pkl,json}
├── diagnostics/{q_stat,q_dyn,q_con}/predictions_<feature_type>_<target>_seed<seed>.csv
├── geometry/<model>/<feature_type>.csv
└── summary.csv                                                 ← aggregate
```

### 3.3 Per-figure reproduction recipes

The fastest way to reproduce figures and tables is through the wrappers
under [`scripts/`](scripts/):

```bash
bash scripts/run_all.sh

# Or through SLURM:
bash scripts/run_all.sh --slurm

# Selected figures:
bash scripts/run_all.sh --only fig2 fig3
bash scripts/fig4_geometry.sh
```

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

You can also run the figure scripts directly once the corresponding
diagnostics exist:

```bash
# Fig. 2: static fidelity (Q_stat)
python -m probing.diagnose diagnostic=q_stat feature_type=cls target=pressure
python figures/fig2_q_stat.py \
    --predictions outputs/diagnostics/q_stat/predictions_cls_pressure_seed42.csv \
    --output figs/fig2_q_stat.pdf

# Fig. 3: dynamic coherence (Q_dyn)
python -m probing.diagnose diagnostic=q_dyn feature_type=spatial_mean target=pressure
python figures/fig3_q_dyn.py \
    --predictions outputs/diagnostics/q_dyn/predictions_spatial_mean_pressure_seed42.csv \
    --output figs/fig3_q_dyn.pdf

# Fig. 4: latent geometry
python figures/fig4_geometry.py \
    --feature_path "$DATA_ROOT/image_features/features_dinov3-base" \
    --split test \
    --output figs/fig4_geometry.pdf

# §4.1 Q_con: pressure-wind coupling
python -m probing.diagnose diagnostic=q_con data=us_only feature_type=cls
python figures/fig_q_con.py \
    --predictions outputs/diagnostics/q_con/predictions_cls_seed42.csv \
    --output_dir figs/

# Table 1: collected probe metrics
python -m probing.aggregate
```

### 3.4 Optional experiments

#### Pixel-supervision ablation

The supervised pixel baseline lives under [`src/`](src/) and uses a
separate Hydra root ([`configs/train.yaml`](configs/train.yaml)). It
trains small CNN / ResNet models end-to-end on raw IR frames.

```bash
python -m src.train experiment=simple_cnn
python -m src.train experiment=train_resnet
python -m src.eval ckpt_path=/path/to/last.ckpt
```

The evaluation CSV can be plotted with
[`figures/fig_baseline_vs_dinov3.py`](figures/fig_baseline_vs_dinov3.py).

#### Baselines

```bash
python -m probing.baselines.dvorak \
    --dataset_path "$DATA_ROOT/dataset_hf" \
    --output_path outputs/baselines/dvorak.json

python -m probing.baselines.climatology \
    --dataset_path "$DATA_ROOT/dataset_hf" \
    --output_dir outputs/baselines/
```

---

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
