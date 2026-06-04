# `figures/` — paper figures

Each script reads artefacts from `outputs/` (produced by
`probing/run_all.sh`) and writes a PDF + PNG into `figs/`.

| Script                          | Paper figure / table                            | Driven by                                                  |
|---------------------------------|--------------------------------------------------|------------------------------------------------------------|
| `fig2_q_stat.py`                | Fig. 2 (Q_stat boxplot)                          | `outputs/diagnostics/q_stat/predictions_*.csv`             |
| `fig3_q_dyn.py`                 | Fig. 3 (Q_dyn coherence vs P_c)                  | `outputs/diagnostics/q_dyn/predictions_*.csv`              |
| `fig_q_con.py`                  | §4.1 Q_con scatter + gap evolution               | `outputs/diagnostics/q_con/predictions_*.csv`              |
| `fig4_geometry.py`              | Fig. 4 (PC1, d_eff, feature spread)              | `outputs/geometry/<model>/<feature_type>.csv`              |
| `fig4_pca1.py`                  | Fig. 4a per-bin PCA1 scatter                     | raw features                                               |
| `fig_data_distribution.py`      | App. D agency histograms                         | `outputs/diagnostics/q_stat/predictions_cls_{pressure,wind}_seed42.csv` |
| `fig_ood.py`                    | Fig. 1c (cross-agency OOD bar chart)             | `outputs/probes/<model>{,_ood_<basin>}/ridge/cls_pressure_seed42.json` |
| `fig_baseline_vs_dinov3.py`     | App. E.1 supervised pixel-baseline vs probe      | CNN predictions CSV + Q_stat CSV                           |
| `fig_feature_analysis.py`       | App. E.2 supplementary feature scatter           | `outputs/diagnostics/q_stat/predictions_*.csv`             |

## Shared style

```python
from figures._style import set_icml_style, lipari_family_palette, savefig
```

* `set_icml_style()` applies the journal `rcParams` (Times serif,
  fontsize 10, Type-1 PDF, no top/right spines, …).
* `lipari_family_palette()` returns a `FamilyPalette` keyed on
  `{dinov2, dinov3, clip, siglip, siglip2, mae}` — consistent across
  every figure.
* `savefig(fig, "figs/foo.pdf", formats=("pdf", "png"))` writes both
  formats and creates parent directories.

## Example commands

```bash
# Static fidelity:
python figures/fig2_q_stat.py \
    --predictions outputs/diagnostics/q_stat/predictions_cls_pressure_seed42.csv

# Dynamic coherence:
python figures/fig3_q_dyn.py \
    --predictions outputs/diagnostics/q_dyn/predictions_spatial_mean_pressure_seed42.csv

# Manifold consistency:
python figures/fig_q_con.py \
    --predictions outputs/diagnostics/q_con/predictions_cls_seed42.csv

# §4.2 geometry (per model):
python figures/fig4_geometry.py \
    --geometry_dir outputs/geometry/dinov3-base
```

Run the dataset + probing pipelines first (`bash dataset/run_all.sh`,
`bash probing/run_all.sh`).
