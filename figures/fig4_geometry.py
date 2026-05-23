#!/usr/bin/env python3
"""
latent_collapse.py

Visualizes the feature dimension collapse along physical axes in intense regimes
(pressure < 980 hPa) for DINOv3-base model.

This figure demonstrates the core finding of the Perception-Physics Paradox:
- In moderate regimes, pressure variations map to distinct latent representations
- In intense regimes, the latent space collapses along the physical axis

Output: ICML 2026 style figure (Type 1 fonts, fontsize=10)
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets import load_from_disk
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ICML 2026 Style Settings (Type 1 fonts, fontsize=10)
# ==============================================================================

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

# Color scheme
# COLORS = {
#     'intense': '#5ab4ac',      # Teal for intense regime
#     'moderate': '#d8b365',     # Gold/tan for moderate regime
#     'threshold': '#2c3e50',    # Dark blue-gray for threshold line
#     'collapse_zone': '#e74c3c', # Red for collapse annotation
# }

COLORS = {
    'intense': '#a59ead',      # Teal for intense regime
    'moderate': '#E6E1D7',     # Gold/tan for moderate regime
    'threshold': '#2c3e50',    # Dark blue-gray for threshold line
    'collapse_zone': '#f4f4f4', # Red for collapse annotation
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_features_and_pressure(
    feature_path: str,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CLS features and pressure from HuggingFace dataset."""

    ds = load_from_disk(feature_path)[split]

    features_list = []
    pressure_list = []

    for ex in ds:
        mask = np.array(ex['frame_valid_mask']).astype(bool)
        feats = np.array(ex['features']['cls'])[mask]
        press = np.array(ex['pressure'])[mask]

        # Filter valid pressure values
        valid = ~np.isnan(press)
        features_list.extend(feats[valid])
        pressure_list.extend(press[valid])

    return np.array(features_list), np.array(pressure_list)


# ==============================================================================
# Analysis Functions
# ==============================================================================

def compute_effective_dimensionality(X: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute effective dimensionality using participation ratio.

    d_eff = (sum(lambda_i))^2 / sum(lambda_i^2)

    This measures how many dimensions effectively contribute to variance.
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    N = X_centered.shape[0]

    # Gram matrix approach for efficiency
    G = (X_centered @ X_centered.T) / (N - 1)
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.clip(eigvals, eps, None)

    return (eigvals.sum() ** 2) / (eigvals ** 2).sum()


def compute_within_bin_spread(
    features: np.ndarray,
    pressure: np.ndarray,
    bins: List[Tuple[float, float]],
    n_samples: int = 500,
    seed: int = 42
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute within-bin feature spread (standard deviation of pairwise distances).

    Returns: (bin_centers, mean_spreads, std_spreads)
    """
    rng = np.random.default_rng(seed)

    bin_centers = []
    mean_spreads = []
    std_spreads = []

    for lo, hi in bins:
        idx = np.where((pressure >= lo) & (pressure < hi))[0]

        if len(idx) < 50:
            continue

        # Subsample if necessary
        if len(idx) > n_samples:
            idx = rng.choice(idx, n_samples, replace=False)

        X_bin = features[idx]
        X_bin_centered = X_bin - X_bin.mean(axis=0, keepdims=True)

        # Compute pairwise distances (sample for efficiency)
        n_pairs = min(len(idx) * (len(idx) - 1) // 2, 5000)
        dists = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(X_bin_centered), 2, replace=False)
            dists.append(np.linalg.norm(X_bin_centered[i] - X_bin_centered[j]))

        bin_centers.append((lo + hi) / 2)
        mean_spreads.append(np.mean(dists))
        std_spreads.append(np.std(dists))

    return bin_centers, mean_spreads, std_spreads


def compute_binned_effective_dim(
    features: np.ndarray,
    pressure: np.ndarray,
    bins: List[Tuple[float, float]],
    n_samples: int = 384,
    seed: int = 42
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute effective dimensionality for each pressure bin.

    Returns: (bin_centers, d_effs, counts)
    """
    rng = np.random.default_rng(seed)

    bin_centers = []
    d_effs = []
    counts = []

    for lo, hi in bins:
        idx = np.where((pressure >= lo) & (pressure < hi))[0]

        if len(idx) < n_samples:
            continue

        sel = rng.choice(idx, n_samples, replace=False)
        X_bin = features[sel]

        d_eff = compute_effective_dimensionality(X_bin)

        bin_centers.append((lo + hi) / 2)
        d_effs.append(d_eff)
        counts.append(len(idx))

    return bin_centers, d_effs, counts


# ==============================================================================
# Plotting Functions
# ==============================================================================

def create_latent_collapse_figure(
    features: np.ndarray,
    pressure: np.ndarray,
    output_path: str,
    threshold: float = 980.0
):
    """
    Create a multi-panel figure showing latent collapse in intense regimes.

    Panel A: Latent-Physical space scatter (PC1 vs Pressure)
    Panel B: Effective dimensionality vs pressure bins
    Panel C: Within-bin feature spread (normalized)
    """

    # ==== Data Preparation ====
    # Standardize and compute PCA
    scaler = StandardScaler()
    z_norm = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_norm)

    # Define regimes
    mask_intense = pressure < threshold
    mask_moderate = pressure >= threshold

    # Pressure bins
    bins = [
        (870, 920), (920, 940), (940, 960), (960, 980),
        (980, 1000),
        (1000, 1020)
    ]

    # Compute metrics
    bin_centers_dim, d_effs, _ = compute_binned_effective_dim(
        features, pressure, bins, n_samples=384, seed=42
    )
    bin_centers_spread, spreads, spread_stds = compute_within_bin_spread(
        features, pressure, bins, n_samples=500, seed=42
    )

    # Normalize spread to show relative collapse
    spread_normalized = np.array(spreads) / max(spreads)

    # ==== Create Figure ====
    fig = plt.figure(figsize=(7.0, 1.4))  # ICML double-column width

    # Grid spec for three panels
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ==== Panel A: Latent-Physical Space ====
    # Create a custom colormap for pressure
    cmap = LinearSegmentedColormap.from_list(
        'pressure',
        [COLORS['intense'], '#FFFFFF', COLORS['moderate']],
        N=256
    )

    # Scatter plot with pressure-based coloring
    # sample a balanced subset for visibility
    rng = np.random.default_rng(42)
    scatter = ax_a.scatter(
        pressure, z_pca[:, 0],
        c=pressure, cmap=cmap,
        alpha=0.25, s=3,
        vmin=870, vmax=1020,
        rasterized=True
    )

    # Add regime-specific overlay with higher visibility
    ax_a.scatter(
        pressure[mask_intense], z_pca[mask_intense, 0],
        c=COLORS['intense'], alpha=0.35, s=4,
        label=f'Intense ($<$ {int(threshold)} hPa)',
        rasterized=True
    )

    # Vertical threshold line
    ax_a.axvline(
        x=threshold, color=COLORS['threshold'],
        linestyle='--', linewidth=1.2, alpha=0.8
    )

    # Annotate collapse zone
    ylim = ax_a.get_ylim()
    ax_a.fill_betweenx(
        ylim, 870, threshold,
        alpha=0.08, color=COLORS['collapse_zone'],
        zorder=0
    )
    ax_a.text(
        900, ylim[1] * 0.15, 'Collapse\nZone',
        fontsize=8, fontweight='bold',
        ha='center', va='top',
        color='#636363'
    )

    ax_a.set_xlabel('Minimum Central Pressure $P_c$ (hPa)')
    ax_a.set_ylabel('Latent PC1')
    ax_a.set_xlim(1025, 865)  # Inverted (intense = lower pressure)
    ax_a.set_ylim(ylim)
    ax_a.set_title('(a) Latent-Physical Relationship', fontsize=10, fontweight='bold', pad=8)

    # Legend
    ax_a.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=7)

    # ==== Panel B: Effective Dimensionality ====
    colors_dim = [COLORS['intense'] if c < threshold else COLORS['moderate']
                  for c in bin_centers_dim]

    bars = ax_b.bar(
        range(len(bin_centers_dim)), d_effs,
        color=colors_dim, edgecolor='black', linewidth=0.5,
        alpha=0.85
    )

    # Add threshold annotation
    threshold_idx = None
    for i, c in enumerate(bin_centers_dim):
        if c >= threshold:
            threshold_idx = i
            break

    if threshold_idx is not None and threshold_idx > 0:
        ax_b.axvline(
            x=threshold_idx - 0.5, color=COLORS['threshold'],
            linestyle='--', linewidth=1.2
        )

    # Labels
    bin_labels = [f'{int(lo)}-{int(hi)}' for lo, hi in bins if len(np.where((pressure >= lo) & (pressure < hi))[0]) >= 384]
    ax_b.set_xticks(range(len(bin_labels)))
    ax_b.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
    ax_b.set_xlabel('Pressure Bin (hPa)')
    ax_b.set_ylabel('Effective Dim. $d_{\\mathrm{eff}}$')
    ax_b.set_title('(b) Dimensionality Collapse', fontsize=10, fontweight='bold', pad=8)

    # Annotate the collapse
    if len(d_effs) >= 2:
        collapse_ratio = d_effs[0] / d_effs[-1] if d_effs[-1] > 0 else 0
        ax_b.text(
            0.95, 0.95, f'Collapse: {collapse_ratio:.1f}x',
            transform=ax_b.transAxes, fontsize=7,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )

    # ==== Panel C: Within-bin Spread ====
    colors_spread = [COLORS['intense'] if c < threshold else COLORS['moderate']
                     for c in bin_centers_spread]

    ax_c.bar(
        range(len(bin_centers_spread)), spread_normalized,
        color=colors_spread, edgecolor='black', linewidth=0.5,
        alpha=0.85, yerr=np.array(spread_stds) / max(spreads) * 0.5  # Scaled error bars
    )
    
    for i, c in enumerate(bin_centers_spread):
        if c >= threshold:
            threshold_idx = i
            break

    # Threshold line
    if threshold_idx is not None and threshold_idx > 0:
        ax_c.axvline(
            x=threshold_idx - 0.5, color=COLORS['threshold'],
            linestyle='--', linewidth=1.2
        )

    # Labels
    spread_labels = [f'{int(lo)}-{int(hi)}' for lo, hi in bins if len(np.where((pressure >= lo) & (pressure < hi))[0]) >= 50]
    ax_c.set_xticks(range(len(spread_labels)))
    ax_c.set_xticklabels(spread_labels, rotation=45, ha='right', fontsize=7)
    ax_c.set_xlabel('Pressure Bin (hPa)')
    ax_c.set_ylabel('Normalized Spread')
    ax_c.set_ylim(0, 1.15)
    ax_c.set_title('(c) Feature Spread Reduction', fontsize=10, fontweight='bold', pad=8)

    # Add legend for regime colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['intense'], edgecolor='black', label='Intense'),
        Patch(facecolor=COLORS['moderate'], edgecolor='black', label='Moderate')
    ]
    ax_c.legend(handles=legend_elements, loc='upper right', fontsize=7, 
                framealpha=0.9, bbox_to_anchor=(1, 0.35))

    # ==== Save ====
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")

    return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate latent collapse visualization for DINOv3-base'
    )
    _data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    _features_dir = os.environ.get("FEATURES_DIR", os.path.join(_data_root, "image_features"))
    parser.add_argument(
        '--feature_path', type=str,
        default=os.path.join(_features_dir, "features_dinov3-base"),
        help='Path to DINOv3-base features (HuggingFace format)'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        help='Dataset split to use'
    )
    parser.add_argument(
        '--output', type=str,
        default='scripts/99_plots/latent_collapse.pdf',
        help='Output path for the figure'
    )
    parser.add_argument(
        '--threshold', type=float, default=980.0,
        help='Pressure threshold for intense regime (hPa)'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.feature_path}")
    features, pressure = load_features_and_pressure(args.feature_path, args.split)
    print(f"Loaded {len(features)} samples, feature dim = {features.shape[1]}")
    print(f"Pressure range: [{pressure.min():.0f}, {pressure.max():.0f}] hPa")

    n_intense = (pressure < args.threshold).sum()
    n_moderate = (pressure >= args.threshold).sum()
    print(f"Intense regime (< {args.threshold} hPa): {n_intense} samples")
    print(f"Moderate regime (>= {args.threshold} hPa): {n_moderate} samples")

    print("\nGenerating latent collapse figure...")
    create_latent_collapse_figure(
        features, pressure,
        str(output_path),
        threshold=args.threshold
    )

    print("Done!")


if __name__ == '__main__':
    main()
