# `scripts/` — paper reproduction wrappers

Thin bash wrappers that pin the exact paper settings (probe family, feature
type, data subset, seed list) for every figure in *The Perception–Physics
Paradox*. Each script delegates fitting and diagnostics to `probing/`, then
calls the corresponding plotting code in `figures/`.

## Layout

| Script                               | Paper artefact                                              | Pins                                                                 |
|--------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------|
| `_env.sh`                            | shared environment helper                                   | default seeds, model lists, output paths                            |
| `fig5_dataset_dist.sh`               | Fig. 5 / App. D per-agency histograms                       | reads q_stat predictions (lazily fits the missing wind probe)        |
| `fig2_qstat_cls.sh`                  | Fig. 2 Q_stat (CLS, all 11 VFMs)                            | ridge · cls · pressure · `balanced_980` · 5 seeds                    |
| `fig3_qdyn_cls.sh`                   | Fig. 3 Q_dyn (CLS)                                          | reuses Fig. 2 probes — diagnostic + plot only                        |
| `fig7_qstat_spatial_mean.sh`         | Fig. 7 / App. E.3 Q_stat (spatial mean)                     | identical to Fig. 2 with `feature_type=spatial_mean`                 |
| `fig8_qdyn_spatial_mean.sh`          | Fig. 8 / App. E.3 Q_dyn (spatial mean)                      | reuses Fig. 7 probes                                                 |
| `fig_qcon.sh`                        | §4.1 Q_con manifold consistency                             | ridge · cls · **pressure + wind** · `us_only` · 5 seeds              |
| `fig1c_ood.sh`                       | Fig. 1c cross-agency OOD bar chart                          | dinov3-base only · ridge · cls · pressure · 1 ID + 9 per-basin fits · seed=42 |
| `fig4_geometry.sh`                   | Fig. 4 intrinsic geometry on DINOv3-base                    | dinov3-base only · cls · `full`                                      |
| `fig6_pixelsup.sh`                   | Fig. 6 / App. E.1 ResNet-18 pixel-sup baseline              | paper hyperparams pinned via Hydra overrides (single seed=42)        |
| `appE2_nonlinear_probes.sh`          | App. E.2 nonlinear-probe sweep                              | dinov3-base · MLP probe · 3 seeds                                    |
| `run_all.sh`                         | top-level orchestrator                                      | `--slurm` and `--only <stage> [<stage>…]` flags                      |

## Invocation

```bash
# everything, local:
bash scripts/run_all.sh

# everything, through SLURM:
bash scripts/run_all.sh --slurm

# only Fig. 2 + Fig. 3:
bash scripts/run_all.sh --only fig2 fig3

# a single figure:
bash scripts/fig4_geometry.sh
```

## Before running

`dataset/run_all.sh` must have produced `$FEATURES_DIR/features_<model>/`
for each of the 11 VFMs in `MODELS_ALL` before any of these scripts run.
See [`dataset/README.md`](../dataset/README.md).
