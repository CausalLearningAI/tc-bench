# `probing/` — TC-Bench probing protocol

Implementation of the structural-alignment probing framework from §4 of
the paper. Three diagnostic probes, one §4.2 failure-mode analysis, and
all the plumbing required to reproduce Figures 2–4 from a directory of
extracted features.

## Concepts

The protocol assumes you have already run `dataset/run_all.sh` so that
`$FEATURES_DIR/features_<model>/` contains frozen-VFM features for each
model variant. From there:

1. **Fit** a *linear* (ridge) probe per `(model, feature_type, target, seed)`
   on a regime-balanced trajectory-level split.
2. **Diagnose** three probes that re-use the fitted linear map ``L``:
    * **Q_stat** — apply ``L`` to per-frame features → predictions CSV.
    * **Q_dyn**  — apply ``L`` to *feature deltas* `Δz_t` → compare with
      `Δy_t`. Same linear map; no separate fit.
    * **Q_con**  — pair pressure and wind ``L``s on the same frames →
      pressure–wind scatter.
3. **Geometry** — compute per-pressure-bin effective dimensionality,
   feature spread, and first-PC means (§4.2 / Fig. 4).
4. **Aggregate** — collect every per-fit metric JSON into one summary CSV.

The library is split as:

```
probing/
├── core/                      # protocol-agnostic primitives (no Hydra)
│   ├── data.py                # load_features, regime_balance, trajectory_split,
│   │                          # per_trajectory_finite_differences
│   ├── probes.py              # RidgeProbe, LassoProbe, MLPProbe (+ build_probe)
│   ├── metrics.py             # regression_metrics, regime_metrics, ξ_stat
│   ├── diagnostics.py         # q_stat / q_dyn / q_con  → DataFrame
│   └── geometry.py            # effective_dimensionality, pca_per_bin, feature_spread
├── fit.py                     # Hydra entry, fits + persists one probe
├── diagnose.py                # Hydra entry, builds one diagnostic CSV
├── geometry.py                # Hydra entry, §4.2 geometry tables
├── aggregate.py               # Hydra entry, collects per-fit JSONs
├── configs/                   # Hydra configs (composed by the four entries)
│   ├── fit.yaml  diagnose.yaml  geometry.yaml  aggregate.yaml
│   ├── probe/{ridge,lasso,mlp}.yaml
│   ├── diagnostic/{q_stat,q_dyn,q_con}.yaml
│   ├── data/{full,balanced_980,us_only,ood_basin}.yaml
│   └── model/{dinov2-base, …, mae-large}.yaml   # 11 VFM configs
├── slurm/                     # SLURM wrappers + shared _env.sh
│   ├── fit_all.sh  diagnose_all.sh  geometry_all.sh
└── run_all.sh                 # one-shot orchestrator (local or --slurm)
```

## Quick reference

Single fit:

```bash
python -m probing.fit \
    probe=ridge model=dinov3-base \
    feature_type=cls target=pressure seed=42
```

Full sweep (Hydra multirun):

```bash
python -m probing.fit -m \
    probe=ridge \
    model=dinov3-base,dinov3-large,clip-base,clip-large,siglip-base,siglip2-base,mae-base,mae-large,dinov2-base,dinov2-large,dinov3-satellite \
    feature_type=cls,spatial_mean \
    target=pressure,wind \
    seed=42,43,44,45,46
```

Diagnostic for every model:

```bash
python -m probing.diagnose diagnostic=q_stat feature_type=cls target=pressure
python -m probing.diagnose diagnostic=q_dyn  feature_type=spatial_mean target=pressure
python -m probing.diagnose diagnostic=q_con  data=us_only feature_type=cls
```

Geometry diagnostics:

```bash
python -m probing.geometry model=dinov3-base feature_type=cls
```

Aggregate per-fit metrics:

```bash
python -m probing.aggregate
```

End-to-end orchestrator:

```bash
bash probing/run_all.sh              # local
bash probing/run_all.sh --slurm      # one sbatch per stage
```

## Outputs

| Stage      | Path                                                          |
|------------|---------------------------------------------------------------|
| fit        | `${PROBE_DIR}/<model>/<probe>/<feature_type>_<target>_seed<seed>.{pkl,json}` |
| diagnose   | `${DIAG_DIR}/<diagnostic>/predictions_<feature_type>_<target>_seed<seed>.csv` |
| geometry   | `${GEOM_DIR}/<model>/<feature_type>.csv`                      |
| aggregate  | `outputs/summary.csv`                                         |

Default roots (overridable via env): `PROBE_DIR=outputs/probes`,
`DIAG_DIR=outputs/diagnostics`, `GEOM_DIR=outputs/geometry`.

## Mapping to paper figures

| Figure / Table                                | Driven by                                            |
|------------------------------------------------|------------------------------------------------------|
| Fig. 2 (Q_stat boxplot)                       | `diagnose diagnostic=q_stat`                         |
| Fig. 3 (Q_dyn coherence vs P_c)               | `diagnose diagnostic=q_dyn`                          |
| Q_con scatter (pressure–wind, §4.1)           | `diagnose diagnostic=q_con data=us_only`             |
| Fig. 4a (PC1 vs P_c)                          | `geometry`                                           |
| Fig. 4b (effective dimensionality)            | `geometry`                                           |
| Fig. 4c (feature spread)                      | `geometry`                                           |
| Tab. 1 (probe values per model)               | `aggregate`                                          |
