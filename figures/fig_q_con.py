#!/usr/bin/env python3
"""Q_con — manifold-consistency pressure–wind scatter and gap-evolution.

Reads the Q_con predictions CSV written by
``python -m probing.diagnose diagnostic=q_con`` (a per-frame join of the
pressure and wind static probes on the US-only split) and produces two
panels:

1. Pressure–wind scatter, ground truth vs model prediction, coloured by
   latitude regime.
2. Coriolis-separation gap as a function of pressure
   (:math:`\\Delta V_m^{\\mathrm{low-lat}} - \\Delta V_m^{\\mathrm{high-lat}}`)
   — the Q_con monotonic constraint of Eq. 4.4.

Usage::

    python figures/fig_q_con.py \
        --predictions outputs/diagnostics/q_con/predictions_cls_seed42.csv \
        --output_dir figs/

Bug fixes vs. the legacy script: ``seaborn.set_theme(style="whitegrid")``
(was ``"whitgrid"``) and the gap-evolution panel now uses
``ax.plot`` (``sns.plot`` does not exist).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figures._style import lipari_family_palette, savefig, set_icml_style


def _scatter_panel(ax, df, suffix, palette):
    sns.scatterplot(
        data=df, x=f"pressure_{suffix}", y=f"wind_{suffix}",
        hue="Lat_Regime", style="Lat_Regime",
        alpha=0.3, palette=palette, ax=ax,
    )
    for regime, color in palette.items():
        sub = df[df["Lat_Regime"] == regime]
        if len(sub) < 5:
            continue
        sns.regplot(
            data=sub, x=f"pressure_{suffix}", y=f"wind_{suffix}",
            scatter=False, color=color, ax=ax,
            label=f"{regime} trend",
        )


def _gap_evolution(ax, df, p_min=900, p_max=980, bin_size=10):
    bins = np.arange(p_min, p_max + bin_size, bin_size)
    df = df.copy()
    df["P_bin"] = pd.cut(df["pressure_true"], bins, labels=bins[:-1])
    stats_true = df.groupby(["P_bin", "Lat_Regime"], observed=False)["wind_true"].mean().unstack()
    stats_pred = df.groupby(["P_bin", "Lat_Regime"], observed=False)["wind_pred"].mean().unstack()
    gap_true = stats_true["Low (<15°)"] - stats_true["High (>25°)"]
    gap_pred = stats_pred["Low (<15°)"] - stats_pred["High (>25°)"]
    rel_err  = np.abs(gap_true - gap_pred) / gap_true
    idx_int  = gap_true.index.astype(int).to_numpy()
    ax.plot(idx_int, gap_true.values, color="gray", linewidth=2.5, linestyle="--",
            label="Ground truth gap")
    ax.plot(idx_int, gap_pred.values, color="black", linewidth=2.0,
            label="Predicted gap")
    ax.axhline(0, color="black", linewidth=1, alpha=0.3)
    ax.invert_xaxis()
    ax.set_xlabel("Central Pressure (hPa)")
    ax.set_ylabel(r"Coriolis Separation $\Delta V$ (kt)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return rel_err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Q_con predictions CSV from probing.diagnose")
    ap.add_argument("--output_dir", type=Path, default=Path("figs"),
                    help="Where to write fig_q_con.pdf and fig_q_con_gap.pdf")
    ap.add_argument("--model_family", type=str, default="dinov3",
                    help="Restrict the scatter panel to this family")
    ap.add_argument("--model_size", type=str, default="base",
                    help="Restrict the scatter panel to this size")
    args = ap.parse_args()

    set_icml_style()
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})

    df = pd.read_csv(args.predictions)

    df["Lat_Regime"] = pd.cut(
        df["latitudes_x"].abs() if "latitudes_x" in df.columns else df["latitudes"].abs(),
        bins=[0, 15, 25, 90], labels=["Low (<15°)", "Mid", "High (>25°)"],
    )
    fam = lipari_family_palette().as_dict()
    palette = {"Low (<15°)": fam["dinov3"], "Mid": fam["clip"], "High (>25°)": fam["mae"]}

    scatter_df = df[df["Lat_Regime"].isin(["Low (<15°)", "High (>25°)"])]
    scatter_df = scatter_df[
        (scatter_df["model_family"] == args.model_family)
        & (scatter_df["model_size"] == args.model_size)
    ]
    scatter_df = scatter_df.copy()
    scatter_df["Lat_Regime"] = scatter_df["Lat_Regime"].cat.remove_unused_categories()
    scatter_palette = {k: v for k, v in palette.items() if k != "Mid"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), sharey=True)
    _scatter_panel(axes[0], scatter_df, "true", scatter_palette)
    _scatter_panel(axes[1], scatter_df, "pred", scatter_palette)
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Model Predictions")
    axes[0].set_xlabel("Pressure (hPa)")
    axes[1].set_xlabel("Pressure (hPa)")
    axes[0].set_ylabel("Wind (kt)")
    axes[0].set_xlim(1020, 880)
    axes[1].set_xlim(1020, 880)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, bottom=0.2)
    savefig(fig, args.output_dir / "fig_q_con.pdf")

    fig2, ax = plt.subplots(figsize=(4.0, 3.0))
    rel_err = _gap_evolution(ax, scatter_df)
    print("Q_con relative error per pressure bin:")
    print(rel_err.to_string())
    savefig(fig2, args.output_dir / "fig_q_con_gap.pdf")


if __name__ == "__main__":
    main()
