#!/usr/bin/env python3
"""
plot_latent_collapse_figures.py

Create ICML-26 style figures for latent collapse analysis:
(A) Effective dimensionality vs pressure
(B) Regime-conditioned eigenspectrum (developing vs intense)

Inputs (from failure_mode_latent_collapse_fast.py):
- panelA_deff_vs_pressure.csv
- panelB_spectrum_developing.npy
- panelB_spectrum_intense.npy

Outputs:
- fig_deff_vs_pressure.pdf
- fig_eigenspectrum_regimes.pdf
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# ICML 2026 style configuration
# ============================================================

def set_icml_style():
    mpl.rcParams.update({
        # Font / text
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        # Lines / markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        # Axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # PDF / vector safety
        "pdf.fonttype": 42,   # TrueType
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


# ============================================================
# Figure A: Effective dimension vs pressure
# ============================================================

def plot_deff_vs_pressure(csv_path, out_path, threshold):
    df = pd.read_csv(csv_path)
    df = df.sort_values("pressure_mid")

    fig, ax = plt.subplots(figsize=(3.2, 2.2))  # single-column ICML

    ax.plot(
        df["pressure_mid"],
        df["deff"],
        marker="o",
        color="black",
        label="Effective dimension",
    )

    # Regime boundary
    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=0.8,
        color="gray",
        label=f"{int(threshold)} hPa",
    )

    ax.set_xlabel("Central pressure (hPa)")
    ax.set_ylabel("Effective dimension")

    ax.set_xlim(df["pressure_mid"].min() - 5, df["pressure_mid"].max() + 5)
    ax.set_ylim(bottom=0)

    ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Figure B: Regime-conditioned eigenspectrum
# ============================================================

def plot_eigenspectrum(dev_path, int_path, out_path, max_k=50):
    spec_dev = np.load(dev_path)
    spec_int = np.load(int_path)

    K = min(len(spec_dev), len(spec_int), max_k)
    x = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(3.2, 2.2))  # single-column ICML

    ax.plot(
        x,
        spec_dev[:K],
        color="black",
        linestyle="-",
        label="Developing",
    )

    ax.plot(
        x,
        spec_int[:K],
        color="black",
        linestyle="--",
        label="Intense",
    )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Normalized variance")

    ax.set_yscale("log")
    ax.set_xlim(1, K)

    ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing analysis outputs")
    parser.add_argument("--threshold", type=float, default=980.0)
    parser.add_argument("--max_k", type=int, default=50)
    args = parser.parse_args()

    set_icml_style()

    # Paths
    panelA_csv = os.path.join(args.results_dir, "panelA_deff_vs_pressure.csv")
    spec_dev = os.path.join(args.results_dir, "panelB_spectrum_developing.npy")
    spec_int = os.path.join(args.results_dir, "panelB_spectrum_intense.npy")

    figA_path = os.path.join(args.results_dir, "fig_deff_vs_pressure.pdf")
    figB_path = os.path.join(args.results_dir, "fig_eigenspectrum_regimes.pdf")

    # Plot
    plot_deff_vs_pressure(panelA_csv, figA_path, args.threshold)
    plot_eigenspectrum(spec_dev, spec_int, figB_path, args.max_k)

    print("Figures saved:")
    print(" ", figA_path)
    print(" ", figB_path)


if __name__ == "__main__":
    main()
