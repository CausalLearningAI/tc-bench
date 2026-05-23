#!/usr/bin/env python3
"""Figure 3 of the paper — Q_dyn coherence gap as a function of pressure.

Reads the diagnostic CSV written by
``python -m probing.diagnose diagnostic=q_dyn`` and plots the mean
absolute coherence gap ``|L Δz - Δy|`` per pressure bin, one line per
VFM family. The 980-hPa "structural inflection" threshold and the
catastrophic-regime band are annotated.

Usage::

    python figures/fig3_q_dyn.py \
        --predictions outputs/diagnostics/q_dyn/predictions_spatial_mean_pressure_seed42.csv \
        --output figs/fig3_q_dyn.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figures._style import lipari_family_palette, savefig, set_icml_style


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Q_dyn predictions CSV from probing.diagnose")
    ap.add_argument("--output", type=Path, default=Path("figs/fig3_q_dyn.pdf"),
                    help="Output PDF path")
    ap.add_argument("--inflection_hpa", type=float, default=980.0,
                    help="Structural-inflection threshold (annotated)")
    args = ap.parse_args()

    set_icml_style()

    df = pd.read_csv(args.predictions)
    df["nae"] = (df["proj_delta_features"] - df["delta_targets"]).abs()
    df["regime_bin"] = (df["targets"] // 5) * 5 + 2.5

    family_palette = lipari_family_palette().as_dict()

    fig, ax = plt.subplots(figsize=(3.75, 3.2))
    sns.lineplot(
        data=df, x="regime_bin", y="nae",
        hue="model_family", palette=family_palette,
        linewidth=1.2, marker="o", markersize=3,
        errorbar=("ci", 95), err_style="band", err_kws={"alpha": 0.1},
        alpha=0.9, ax=ax,
    )
    ax.invert_xaxis()
    ax.axvspan(895, 930, color="gray", alpha=0.08, label="_nolegend_")
    ax.axvline(args.inflection_hpa, color="black", linestyle="--",
               alpha=0.3, linewidth=0.8)

    y_top = ax.get_ylim()[1]
    ax.text(940, y_top * 0.82,
            f"Structural Inflection\n({int(args.inflection_hpa)}hPa Threshold)",
            color="gray", ha="right")

    ax.set_xlabel(r"Intensity (Pressure [hPa]) $\longleftarrow$ [Intense to Moderate]")
    ax.set_ylabel(r"Mean Coherence Gap $\xi_{\mathrm{dyn}}$")
    ax.legend(frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3,
              columnspacing=0.8, handletextpad=0.2)
    sns.despine(ax=ax)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.subplots_adjust(bottom=0.35, top=0.9, left=0.18, right=0.95)

    savefig(fig, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
