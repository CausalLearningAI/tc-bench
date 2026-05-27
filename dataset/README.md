# `dataset/` — TC-Bench construction pipeline

Reproducible build of the **TC-Bench** tropical-cyclone benchmark from the
two upstream sources used in the paper (§3):

1. **IbTRACS v4r01** — multi-agency best-track records (storm centre,
   minimum central pressure $P_c$, maximum sustained wind $V_m$),
2. **GridSat-B1** — three-hourly global infrared brightness temperature
   netCDFs.

The pipeline is split into nine numbered stages. Each stage reads from
the previous stage's directory and writes a fresh artefact under
`$DATA_ROOT`. Every stage is runnable standalone and idempotent — re-runs
are safe.

## Stages

| #  | Script                      | Output (under `$DATA_ROOT`)                     |
|----|-----------------------------|-------------------------------------------------|
| 01 | `01_download_ibtracs.py`    | `ibtracs/ibTRACS_since_1980.csv`                |
| 02 | `02_download_gridsat.py`    | `gridsat/{year}/*.nc`                           |
| 03 | `03_preprocess.py`          | `preprocessed/dataset_ibtracs_basic_cols_{agency}.csv` |
| 04 | `04_crop_windows.py`        | `cropped/{agency}/{year}_{name}.nc`             |
| 05 | `05_consolidate_nc.py`      | `consolidated/{agency}/{year}_{name}.nc`        |
| 06 | `06_build_hf.py`            | `dataset_hf/` (HuggingFace Arrow)               |
| 07 | `07_normalize_stats.py`     | `dataset_hf/normalization_stats.json`           |
| 08 | `08_ood_basin_split.py`     | `image_features/features_{model}_ood_splits/`   |
| 09 | `09_extract_features.py`    | `image_features/features_{model}/`              |

Stages 01–07 produce the public **TC-Bench** dataset. Stage 09 produces
the frozen-VFM features consumed by [`probing/`](../probing/). Stage 08
is only needed for the cross-basin OOD evaluation in App. E.

Auxiliary: `split_by_agency.py` partitions the IbTRACS records by
reporting agency. It is not part of the main pipeline but is invoked by
some analyses in [`figures/`](../figures/) and [`notebooks/`](../notebooks/).

## Quick start

```bash
# Full pipeline, local execution:
bash dataset/run_all.sh

# Full pipeline on SLURM (one sbatch per stage, in order):
bash dataset/run_all.sh --slurm

# Resume from a specific stage:
bash dataset/run_all.sh --from 06

# Run a single stage:
bash dataset/run_all.sh --only 09

# One-year single-storm smoke test (stages 01-07, a few GB, CPU only):
DATA_ROOT=$PWD/smoke_data YEARS=2005 AGENCIES=hurdat_atl \
    ONLY_CYCLONE=KATRINA bash dataset/run_all.sh --to 07
```

Outputs are placed under `$DATA_ROOT` (default `${HOME}/tcbench`).
Override via the environment:

```bash
DATA_ROOT=/scratch/$USER/tcbench bash dataset/run_all.sh
```

## Configuration

All shared variables live in `slurm/_env.sh` and can be overridden from
the shell that calls a wrapper:

| Variable          | Default                                                | Meaning                              |
|-------------------|--------------------------------------------------------|--------------------------------------|
| `REPO_ROOT`       | parent of `dataset/`                                   | repository root                      |
| `CONDA_ENV`       | unset                                                  | conda env (fallback when `.venv` is absent) |
| `DATA_ROOT`       | `${HOME}/tcbench`                                      | top-level data directory             |
| `IBTRACS_CSV`     | `$DATA_ROOT/ibtracs/ibTRACS_since_1980.csv`            | stage 01 output                      |
| `GRIDSAT_DIR`     | `$DATA_ROOT/gridsat`                                   | stage 02 output                      |
| `PREPROCESSED_DIR`| `$DATA_ROOT/preprocessed`                              | stage 03 output                      |
| `CROPPED_DIR`     | `$DATA_ROOT/cropped`                                   | stage 04 output                      |
| `CONSOLIDATED_DIR`| `$DATA_ROOT/consolidated`                              | stage 05 output                      |
| `HF_DATASET`      | `$DATA_ROOT/dataset_hf`                                | stage 06–07 output                   |
| `FEATURES_DIR`    | `$DATA_ROOT/image_features`                            | stage 08–09 output                   |
| `MODELS`          | 11 paper models (stage 09) / `dinov3-base` (stage 08)  | space-separated VFM list             |
| `BATCH_SIZE`      | `512`                                                  | stage 09 batch size                  |
| `DEVICE`          | `cuda`                                                 | stage 09 inference device            |
| `YEARS`           | unset (all)                                            | restrict stages 02–04 to these years |
| `AGENCIES`        | unset (all 9)                                          | restrict stages 03–04 to these agencies |
| `ONLY_CYCLONE`    | unset (all)                                            | restrict stages 02/04 to NAME substring |

## Notes

- IbTRACS downloads are tiny (~50 MB). GridSat covers 1980–present and
  is the heavy part (~500 GB).
- Stage 09 runs once per VFM. The SLURM array submits all 11 paper
  models in parallel; throttle with `#SBATCH --array=0-10%N`.
- Cyclones with fewer than two valid timesteps after preprocessing are
  dropped. Final dataset size after the trajectory-level 80/10/10 split:
  **2 813 train · 352 val · 352 test cyclones**.
