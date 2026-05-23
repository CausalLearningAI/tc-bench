#!/usr/bin/env python3
"""
Visualize data distribution (histogram) of pressure and wind per agency.
Creates grouped histograms with mean and variance annotations.

Inputs default to the modern Hydra layout written by
``python -m probing.diagnose diagnostic=q_stat``; override with --pressure_csv
/ --wind_csv / --output_dir if your artefacts live elsewhere.
"""

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import colormaps as cmaps

REPO_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,  # TrueType (Type 1 compatible)
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'grid.linewidth': 0.5,
})



def extract_agency(cyclone_id: str) -> str:
    """
    Extract agency from cyclone_id.
    Format: $agency_$year_$name (e.g., hurdat_atl_2005_BETA -> hurdat_atl)
    """
    parts = cyclone_id.split("_")
    # Find the year part (4 digits) and take everything before it
    for i, part in enumerate(parts):
        if re.match(r"^\d{4}$", part):
            return "_".join(parts[:i])
    # Fallback: return first two parts
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def plot_grouped_histogram(
    df: pd.DataFrame,
    target_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 30,
):
    """
    Create grouped histogram by agency with mean and variance annotations.
    """
    agencies = sorted(df["agency"].unique())
    n_agencies = len(agencies)

    # Create color palette
    colors = cmaps.lipari(np.linspace(0.15, 0.85, n_agencies))

    # Compute stats per agency
    agency_stats = {}
    for agency in agencies:
        agency_df = df[df["agency"] == agency]
        agency_data = agency_df[target_col]
        n_cyclones = agency_df["cyclone_id"].nunique()
        agency_stats[agency] = {
            "data": agency_data,
            "n_samples": len(agency_data),
            "n_cyclones": n_cyclones,
            "mean": agency_data.mean(),
            "var": agency_data.var(),
            "std": agency_data.std(),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overlapping histograms
    ax1 = axes[0]
    stats_text = []

    for i, agency in enumerate(agencies):
        stats = agency_stats[agency]
        ax1.hist(
            stats["data"],
            bins=bins,
            alpha=0.5,
            label=f"{agency} (n={stats['n_samples']:,}, {stats['n_cyclones']} cyc)",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add vertical line for mean
        ax1.axvline(
            stats["mean"],
            color=colors[i],
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

        stats_text.append(
            f"{agency}: μ={stats['mean']:.1f}, σ={stats['std']:.1f} (n={stats['n_samples']:,}, {stats['n_cyclones']} cyclones)"
        )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"{title} - Overlapping Histograms", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: Side-by-side boxplots with stats
    ax2 = axes[1]

    # Prepare data and labels for boxplot (include sample size in label)
    agency_data_list = [agency_stats[agency]["data"].values for agency in agencies]
    boxplot_labels = [
        f"{agency}\n(n={agency_stats[agency]['n_samples']:,})"
        for agency in agencies
    ]

    bp = ax2.boxplot(
        agency_data_list,
        tick_labels=boxplot_labels,
        patch_artist=True,
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel(xlabel, fontsize=12)
    ax2.set_title(f"{title} - Distribution by Agency", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels if needed
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats annotation
    # stats_str = "\n".join(stats_text)
    # fig.text(
    #     0.02,
    #     0.02,
    #     stats_str,
    #     fontsize=9,
    #     family="monospace",
    #     verticalalignment="bottom",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    # )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Statistics for {title}:")
    for line in stats_text:
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_diag = REPO_ROOT / "outputs" / "diagnostics" / "q_stat"
    ap.add_argument("--pressure_csv", type=Path,
                    default=default_diag / "predictions_cls_pressure_seed42.csv")
    ap.add_argument("--wind_csv", type=Path,
                    default=default_diag / "predictions_cls_wind_seed42.csv")
    ap.add_argument("--output_dir", type=Path, default=REPO_ROOT / "figs")
    args = ap.parse_args()

    pressure_csv = args.pressure_csv
    wind_csv = args.wind_csv
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading pressure data...")
    df_pressure = pd.read_csv(pressure_csv)
    df_pressure["agency"] = df_pressure["cyclone_id"].apply(extract_agency)

    print("Loading wind data...")
    df_wind = pd.read_csv(wind_csv)
    df_wind["agency"] = df_wind["cyclone_id"].apply(extract_agency)

    # Print data summary
    print(f"\nPressure data: {len(df_pressure)} samples")
    print(f"Wind data: {len(df_wind)} samples")
    print(f"Agencies found: {sorted(df_pressure['agency'].unique())}")

    # Plot pressure distribution
    print("\n--- Pressure Distribution ---")
    plot_grouped_histogram(
        df=df_pressure,
        target_col="targets",
        title="Pressure Distribution",
        xlabel="Pressure (mb)",
        output_path=output_dir / "distribution_pressure_by_agency.png",
        bins=40,
    )

    # Plot wind distribution
    print("\n--- Wind Distribution ---")
    plot_grouped_histogram(
        df=df_wind,
        target_col="targets",
        title="Wind Speed Distribution",
        xlabel="Wind Speed (kts)",
        output_path=output_dir / "distribution_wind_by_agency.png",
        bins=40,
    )

    # Create combined figure
    print("\n--- Creating Combined Figure ---")
    create_combined_figure(df_pressure, df_wind, output_dir)


def create_combined_figure(
    df_pressure: pd.DataFrame,
    df_wind: pd.DataFrame,
    output_dir: Path,
):
    """Create a combined figure with both pressure and wind distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agencies = sorted(df_pressure["agency"].unique())
    n_agencies = len(agencies)
    colors = cmaps.lipari(np.linspace(0.15, 0.85, n_agencies))

    # Compute stats per agency
    agency_stats = {}
    for agency in agencies:
        p_df = df_pressure[df_pressure["agency"] == agency]
        w_df = df_wind[df_wind["agency"] == agency]
        agency_stats[agency] = {
            "n_samples": len(p_df),
            "n_cyclones": p_df["cyclone_id"].nunique(),
            "pressure_data": p_df["targets"],
            "wind_data": w_df["targets"],
            "pressure_mean": p_df["targets"].mean(),
            "pressure_std": p_df["targets"].std(),
            "wind_mean": w_df["targets"].mean(),
            "wind_std": w_df["targets"].std(),
        }

    # Create boxplot labels with sample sizes
    boxplot_labels = [
        f"{agency}\n(n={agency_stats[agency]['n_samples']:,})"
        for agency in agencies
    ]

    # Top row: Pressure
    ax1, ax2 = axes[0]

    for i, agency in enumerate(agencies):
        stats = agency_stats[agency]
        ax1.hist(
            stats["pressure_data"],
            bins=40,
            alpha=0.5,
            label=f"{agency} (n={stats['n_samples']:,}, {stats['n_cyclones']} cyc)",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.axvline(stats["pressure_mean"], color=colors[i], linestyle="--", linewidth=2, alpha=0.8)

    ax1.set_xlabel("Pressure (mb)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Pressure Distribution by Agency", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Pressure boxplot
    pressure_data_list = [agency_stats[agency]["pressure_data"].values for agency in agencies]
    bp1 = ax2.boxplot(pressure_data_list, tick_labels=boxplot_labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Pressure (mb)", fontsize=11)
    ax2.set_title("Pressure Boxplot by Agency", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Bottom row: Wind
    ax3, ax4 = axes[1]

    for i, agency in enumerate(agencies):
        stats = agency_stats[agency]
        ax3.hist(
            stats["wind_data"],
            bins=40,
            alpha=0.5,
            label=f"{agency} (n={stats['n_samples']:,}, {stats['n_cyclones']} cyc)",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        ax3.axvline(stats["wind_mean"], color=colors[i], linestyle="--", linewidth=2, alpha=0.8)

    ax3.set_xlabel("Wind Speed (kts)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Wind Speed Distribution by Agency", fontsize=12)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # Wind boxplot
    wind_data_list = [agency_stats[agency]["wind_data"].values for agency in agencies]
    bp2 = ax4.boxplot(wind_data_list, tick_labels=boxplot_labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel("Wind Speed (kts)", fontsize=11)
    ax4.set_title("Wind Speed Boxplot by Agency", fontsize=12)
    ax4.grid(axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats text box
    # all_stats = ["Pressure:"] + pressure_stats + ["", "Wind:"] + wind_stats
    # stats_str = "\n".join(all_stats)
    # fig.text(
    #     0.02,
    #     0.02,
    #     stats_str,
    #     fontsize=8,
    #     family="monospace",
    #     verticalalignment="bottom",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    # )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    output_path = output_dir / "distribution_combined_by_agency.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
