#!/usr/bin/env python3
"""Figure 2 of the paper — Q_stat normalized-error boxplot.

Reads the diagnostic CSV written by
``python -m probing.diagnose diagnostic=q_stat`` and produces one boxplot
per (model_family, model_size) split by intensity regime.

Usage::

    python figures/fig2_q_stat.py \
        --predictions outputs/diagnostics/q_stat/predictions_cls_pressure_seed42.csv \
        --output figs/fig2_q_stat.pdf \
        --threshold 980
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figures._style import intensity_palette, savefig, set_icml_style


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Q_stat predictions CSV from probing.diagnose")
    ap.add_argument("--output", type=Path, default=Path("figs/fig2_q_stat.pdf"),
                    help="Output PDF path")
    ap.add_argument("--threshold", type=float, default=980.0,
                    help="Pressure threshold separating moderate / intense regimes")
    ap.add_argument("--target", type=str, default="pressure",
                    help="Target variable used by the probe (for the y-axis label)")
    args = ap.parse_args()

    set_icml_style()
    df = pd.read_csv(args.predictions)

    # Tag regime + per-frame normalized absolute error (Eq. 4.1).
    df["Intensity"] = pd.cut(
        df["targets"], bins=[-np.inf, args.threshold, np.inf],
        labels=["Intense", "Moderate"],
    )
    df["Intensity"] = pd.Categorical(df["Intensity"],
                                     categories=["Moderate", "Intense"], ordered=True)
    global_std = float(df["targets"].std())
    df["residual"] = (df["targets"] - df["y_pred"]).abs()
    df["nae"] = df["residual"] / global_std

    # Single column of model_family-size for the boxplot x-axis.
    df["model"] = df["model_family"] + "-" + df["model_size"]

    fig, ax = plt.subplots(figsize=(6.75, 1.7))
    sns.boxplot(
        data=df, x="model", y="nae", hue="Intensity",
        dodge=True, palette=intensity_palette(),
        linewidth=1.0, width=0.5, gap=0.1, saturation=1.0,
        medianprops={"color": "black", "linewidth": 2.5, "solid_capstyle": "butt"},
        flierprops={"marker": "d", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=10)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_ylabel(r"norm. err. $\xi_{\mathrm{stat}}$")
    ax.set_xlabel("")
    ax.set_ylim(-0.1, 2.5)
    ax.legend(title="Intensity Regime", loc="upper right", ncol=2, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
