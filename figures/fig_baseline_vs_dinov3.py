"""
Compare baseline baseline predictions with DINOv3 linear probe results.

This script visualizes:
1. Scatter plots of predictions vs targets for both models
2. Error analysis by pressure regime (showing failure in intense regime for DINOv3)
3. Demonstrates that pressure information is preserved in raw images but not in
   pretrained representations for intense cyclones (< 980 mb)

Usage:
    python scripts/99_plots/compare_baseline_vs_dinov3.py \
        --baseline_csv analysis_results/baseline_baseline/predictions_baseline_pressure.csv \
        --dinov3_csv analysis_results/Qstat/balanced_980/predictions_cls_pressure_seed42.csv \
        --output_dir analysis_results/comparison_plots
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import colormaps as cmaps

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

colors = cmaps.lipari(np.linspace(0.15, 0.85, 6))


def load_predictions(csv_path: str) -> pd.DataFrame:
    """Load predictions CSV and ensure consistent column names."""
    df = pd.read_csv(csv_path)
    return df


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute regression metrics."""
    mask = ~(np.isnan(targets) | np.isnan(predictions))
    targets = targets[mask]
    predictions = predictions[mask]

    mae = mean_absolute_error(targets, predictions)
    median_ae = np.median(np.abs(targets - predictions))
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    return {
        'MAE': mae,
        'Median_AE': median_ae,
        'RMSE': rmse,
        'R2': r2,
        'N': len(targets)
    }


def compute_metrics_by_regime(df: pd.DataFrame, threshold: float = 980) -> dict:
    """Compute metrics separately for intense (< threshold) and moderate (>= threshold) regimes."""
    intense_mask = df['targets'] < threshold
    moderate_mask = df['targets'] >= threshold

    results = {
        'all': compute_metrics(df['targets'].values, df['y_pred'].values),
        'intense': compute_metrics(
            df.loc[intense_mask, 'targets'].values,
            df.loc[intense_mask, 'y_pred'].values
        ) if intense_mask.sum() > 0 else None,
        'moderate': compute_metrics(
            df.loc[moderate_mask, 'targets'].values,
            df.loc[moderate_mask, 'y_pred'].values
        ) if moderate_mask.sum() > 0 else None,
    }

    return results


def plot_scatter_baseline(baseline_df: pd.DataFrame,
                           output_path: Path, threshold: float = 980):
    """Create side-by-side scatter plots comparing baseline and DINOv3 predictions."""
    fig, ax = plt.subplots(figsize=(3.75, 2.2))

    datasets = [
        ('ResNet18 (from scratch)', baseline_df, ax),
        # ('DINOv3-Base Probe', dinov3_df, axes[1]),
    ]

    for title, df, ax in datasets:
        targets = df['targets'].values
        predictions = df['y_pred'].values
        
        std = np.float64(19.117932021855125)# global std for test set, same as used in vfm eval

        # ax.scatter(targets[moderate_mask], predictions[moderate_mask],
        #           alpha=0.3, s=10, c='#2196F3', label=f'Moderate (>= {threshold} mb)')
        ax.scatter(targets, predictions,
                  alpha=0.5, s=15, c=colors[-1])

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')

        # Add threshold line
        # ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
        # ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)

        # Compute and display metrics
        metrics = compute_metrics_by_regime(df, threshold)

        metrics_text = f"norm. Mean AE={metrics['all']['MAE']/std:.1f}\nnorm. Median AE={metrics['all']['Median_AE']/std:.1f}"
        # if metrics['intense']:
        #     metrics_text += f"Intense: RMSE={metrics['intense']['RMSE']:.1f} mb (N={metrics['intense']['N']})\n"
        # if metrics['moderate']:
        #     metrics_text += f"moderate: RMSE={metrics['moderate']['RMSE']:.1f} mb (N={metrics['moderate']['N']})"

        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Target Pressure (hPa)')
        ax.set_ylabel('Predicted Pressure (hPa)')
        ax.set_title(title, fontweight='bold')
        # ax.legend(loc='lower right', fontsize=9)
        # ax.set_aspect('equal', adjustable='box')

        # Set same limits for both plots
        ax.set_xlim(880, 980)
        ax.set_ylim(880, 980)

    plt.tight_layout()
    plt.savefig(output_path / 'resnet18_scatter_mae.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved scatter comparison to {output_path / 'resnet18_scatter.pdf'}")
    


def plot_scatter_comparison(baseline_df: pd.DataFrame, dinov3_df: pd.DataFrame,
                           output_path: Path, threshold: float = 980):
    """Create side-by-side scatter plots comparing baseline and DINOv3 predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [
        ('ResNet18 (from scratch)', baseline_df, axes[0]),
        # ('DINOv3-Base Probe', dinov3_df, axes[1]),
    ]

    for title, df, ax in datasets:
        targets = df['targets'].values
        predictions = df['y_pred'].values

        # Color by regime
        intense_mask = targets < threshold
        moderate_mask = targets >= threshold

        # ax.scatter(targets[moderate_mask], predictions[moderate_mask],
        #           alpha=0.3, s=10, c='#2196F3', label=f'Moderate (>= {threshold} mb)')
        ax.scatter(targets[intense_mask], predictions[intense_mask],
                  alpha=0.5, s=15, c='#F44336', label=f'Intense (< {threshold} mb)')

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')

        # Add threshold line
        ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)

        # Compute and display metrics
        metrics = compute_metrics_by_regime(df, threshold)

        metrics_text = f"Overall: RMSE={metrics['all']['RMSE']:.1f} mb\n"
        if metrics['intense']:
            metrics_text += f"Intense: RMSE={metrics['intense']['RMSE']:.1f} mb (N={metrics['intense']['N']})\n"
        if metrics['moderate']:
            metrics_text += f"moderate: RMSE={metrics['moderate']['RMSE']:.1f} mb (N={metrics['moderate']['N']})"

        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Target Pressure (mb)', fontsize=12)
        ax.set_ylabel('Predicted Pressure (mb)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal', adjustable='box')

        # Set same limits for both plots
        ax.set_xlim(880, 1020)
        ax.set_ylim(880, 1020)

    plt.tight_layout()
    plt.savefig(output_path / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'scatter_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved scatter comparison to {output_path / 'scatter_comparison.png'}")


def plot_error_by_pressure(baseline_df: pd.DataFrame, dinov3_df: pd.DataFrame,
                          output_path: Path, threshold: float = 980):
    """Plot prediction error as a function of target pressure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute errors
    baseline_df = baseline_df.copy()
    dinov3_df = dinov3_df.copy()
    baseline_df['error'] = baseline_df['y_pred'] - baseline_df['targets']
    dinov3_df['error'] = dinov3_df['y_pred'] - dinov3_df['targets']

    # Binned error analysis
    pressure_bins = np.arange(880, 1020, 10)
    bin_centers = (pressure_bins[:-1] + pressure_bins[1:]) / 2

    for df, name, color, ax_idx in [
        (baseline_df, 'baseline', '#4CAF50', 0),
        (dinov3_df, 'DINOv3-Base', '#FF9800', 1)
    ]:
        # Left plot: individual errors
        ax = axes[ax_idx]
        targets = df['targets'].values
        errors = df['error'].values

        intense_mask = targets < threshold

        ax.scatter(targets[~intense_mask], errors[~intense_mask],
                  alpha=0.2, s=8, c='#2196F3', label=f'Moderate (>= {threshold} mb)')
        ax.scatter(targets[intense_mask], errors[intense_mask],
                  alpha=0.4, s=12, c='#F44336', label=f'Intense (< {threshold} mb)')

        # Compute binned statistics
        bin_means = []
        bin_stds = []
        for i in range(len(pressure_bins) - 1):
            mask = (targets >= pressure_bins[i]) & (targets < pressure_bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(errors[mask].mean())
                bin_stds.append(errors[mask].std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        # Plot binned statistics
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                   fmt='o-', color='black', markersize=6, linewidth=2,
                   capsize=3, label='Binned mean +/- std')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5, label=f'Threshold ({threshold} mb)')

        ax.set_xlabel('Target Pressure (mb)', fontsize=12)
        ax.set_ylabel('Prediction Error (mb)', fontsize=12)
        ax.set_title(f'{name}: Error vs Target Pressure', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(880, 1020)
        ax.set_ylim(-60, 60)

    plt.tight_layout()
    # plt.savefig(output_path / 'error_by_pressure.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'error_by_pressure.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved error analysis to {output_path / 'error_by_pressure.png'}")


def plot_rmse_by_regime(baseline_df: pd.DataFrame, dinov3_df: pd.DataFrame,
                       output_path: Path, threshold: float = 980):
    """Bar plot comparing RMSE in intense vs moderate regimes for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_metrics = compute_metrics_by_regime(baseline_df, threshold)
    dinov3_metrics = compute_metrics_by_regime(dinov3_df, threshold)

    # Prepare data
    models = ['baseline\n(from scratch)', 'DINOv2-Base\n(linear probe)']
    x = np.arange(len(models))
    width = 0.35

    # Extract RMSE values
    baseline_intense = baseline_metrics['intense']['RMSE'] if baseline_metrics['intense'] else 0
    baseline_moderate = baseline_metrics['moderate']['RMSE'] if baseline_metrics['moderate'] else 0
    dinov3_intense = dinov3_metrics['intense']['RMSE'] if dinov3_metrics['intense'] else 0
    dinov3_moderate = dinov3_metrics['moderate']['RMSE'] if dinov3_metrics['moderate'] else 0

    intense_rmse = [baseline_intense, dinov3_intense]
    moderate_rmse = [baseline_moderate, dinov3_moderate]

    # Create bars
    bars1 = ax.bar(x - width/2, intense_rmse, width, label=f'Intense (< {threshold} mb)',
                   color='#F44336', alpha=0.8)
    bars2 = ax.bar(x + width/2, moderate_rmse, width, label=f'moderate (>= {threshold} mb)',
                   color='#2196F3', alpha=0.8)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    ax.set_ylabel('RMSE (mb)', fontsize=12)
    ax.set_title('Prediction Error by Intensity Regime', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, max(max(intense_rmse), max(moderate_rmse)) * 1.3)

    # Add annotation explaining the key finding
    ax.annotate('Representation\ncollapse in\nintense regime',
               xy=(1 - width/2, dinov3_intense),
               xytext=(1.3, dinov3_intense + 5),
               fontsize=10, ha='left',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               color='red')

    plt.tight_layout()
    plt.savefig(output_path / 'rmse_by_regime.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'rmse_by_regime.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved RMSE comparison to {output_path / 'rmse_by_regime.png'}")


def plot_combined_analysis(baseline_df: pd.DataFrame, dinov3_df: pd.DataFrame,
                          output_path: Path, threshold: float = 980):
    """Create a combined figure for publication."""
    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: Scatter plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Row 1, Col 3: RMSE bar plot
    ax3 = fig.add_subplot(gs[0, 2])

    # Row 2: Error analysis (spanning 2 columns) and histogram
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])

    # --- Scatter plots ---
    for title, df, ax in [('baseline (from scratch)', baseline_df, ax1),
                          ('DINOv2-Base Probe', dinov3_df, ax2)]:
        targets = df['targets'].values
        predictions = df['y_pred'].values
        intense_mask = targets < threshold

        ax.scatter(targets[~intense_mask], predictions[~intense_mask],
                  alpha=0.3, s=10, c='#2196F3', label='moderate')
        ax.scatter(targets[intense_mask], predictions[intense_mask],
                  alpha=0.5, s=15, c='#F44336', label='Intense')

        min_val, max_val = 880, 1020
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)

        metrics = compute_metrics_by_regime(df, threshold)
        ax.text(0.02, 0.98, f"RMSE: {metrics['all']['RMSE']:.1f} mb",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Target (mb)')
        ax.set_ylabel('Predicted (mb)')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8)

    # --- RMSE bar plot ---
    baseline_metrics = compute_metrics_by_regime(baseline_df, threshold)
    dinov3_metrics = compute_metrics_by_regime(dinov3_df, threshold)

    models = ['baseline', 'DINOv2']
    x = np.arange(len(models))
    width = 0.35

    baseline_intense = baseline_metrics['intense']['RMSE'] if baseline_metrics['intense'] else 0
    baseline_moderate = baseline_metrics['moderate']['RMSE'] if baseline_metrics['moderate'] else 0
    dinov3_intense = dinov3_metrics['intense']['RMSE'] if dinov3_metrics['intense'] else 0
    dinov3_moderate = dinov3_metrics['moderate']['RMSE'] if dinov3_metrics['moderate'] else 0

    bars1 = ax3.bar(x - width/2, [baseline_intense, dinov3_intense], width,
                    label='Intense', color='#F44336', alpha=0.8)
    bars2 = ax3.bar(x + width/2, [baseline_moderate, dinov3_moderate], width,
                    label='moderate', color='#2196F3', alpha=0.8)

    for bar in bars1:
        ax3.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        ax3.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    ax3.set_ylabel('RMSE (mb)')
    ax3.set_title('Error by Regime', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(fontsize=8)

    # --- Error vs pressure ---
    baseline_df_copy = baseline_df.copy()
    dinov3_df_copy = dinov3_df.copy()
    baseline_df_copy['error'] = baseline_df_copy['y_pred'] - baseline_df_copy['targets']
    dinov3_df_copy['error'] = dinov3_df_copy['y_pred'] - dinov3_df_copy['targets']

    pressure_bins = np.arange(880, 1020, 10)
    bin_centers = (pressure_bins[:-1] + pressure_bins[1:]) / 2

    for df, name, color in [(baseline_df_copy, 'baseline', '#4CAF50'),
                            (dinov3_df_copy, 'DINOv2', '#FF9800')]:
        targets = df['targets'].values
        errors = np.abs(df['error'].values)

        bin_means = []
        for i in range(len(pressure_bins) - 1):
            mask = (targets >= pressure_bins[i]) & (targets < pressure_bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(errors[mask].mean())
            else:
                bin_means.append(np.nan)

        ax4.plot(bin_centers, bin_means, 'o-', color=color, label=name,
                linewidth=2, markersize=6)

    ax4.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5,
                label=f'Threshold ({threshold} mb)')
    ax4.set_xlabel('Target Pressure (mb)')
    ax4.set_ylabel('Mean Absolute Error (mb)')
    ax4.set_title('Prediction Error by Pressure Regime', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_xlim(880, 1020)

    # Shade intense region
    ax4.axvspan(880, threshold, alpha=0.1, color='red', label='Intense regime')

    # --- Error distribution histogram ---
    baseline_intense_errors = baseline_df_copy.loc[baseline_df_copy['targets'] < threshold, 'error'].values
    dinov3_intense_errors = dinov3_df_copy.loc[dinov3_df_copy['targets'] < threshold, 'error'].values

    ax5.hist(baseline_intense_errors, bins=30, alpha=0.6, label='baseline', color='#4CAF50', density=True)
    ax5.hist(dinov3_intense_errors, bins=30, alpha=0.6, label='DINOv2', color='#FF9800', density=True)
    ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Prediction Error (mb)')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Error Distribution\n(Intense: < {threshold} mb)', fontweight='bold')
    ax5.legend(fontsize=9)

    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')

    plt.savefig(output_path / 'combined_analysis.png', dpi=200, bbox_inches='tight')
    plt.savefig(output_path / 'combined_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved combined analysis to {output_path / 'combined_analysis.png'}")


def print_summary_table(baseline_df: pd.DataFrame, dinov3_df: pd.DataFrame, threshold: float = 980):
    """Print summary statistics table."""
    baseline_metrics = compute_metrics_by_regime(baseline_df, threshold)
    dinov3_metrics = compute_metrics_by_regime(dinov3_df, threshold)

    print("\n" + "="*70)
    print("SUMMARY: baseline vs DINOv2 Linear Probe Performance")
    print("="*70)
    print(f"\nThreshold for intense regime: < {threshold} mb")
    print("\n{:<25} {:>15} {:>15}".format("Metric", "baseline", "DINOv2-Base"))
    print("-"*55)

    # Overall
    print("{:<25} {:>15.2f} {:>15.2f}".format(
        "Overall RMSE (mb)",
        baseline_metrics['all']['RMSE'],
        dinov3_metrics['all']['RMSE']
    ))

    # Intense regime
    if baseline_metrics['intense'] and dinov3_metrics['intense']:
        print("{:<25} {:>15.2f} {:>15.2f}".format(
            f"Intense RMSE (< {threshold} mb)",
            baseline_metrics['intense']['RMSE'],
            dinov3_metrics['intense']['RMSE']
        ))
        print("{:<25} {:>15d} {:>15d}".format(
            "Intense N samples",
            baseline_metrics['intense']['N'],
            dinov3_metrics['intense']['N']
        ))

    # moderate regime
    if baseline_metrics['moderate'] and dinov3_metrics['moderate']:
        print("{:<25} {:>15.2f} {:>15.2f}".format(
            f"moderate RMSE (>= {threshold} mb)",
            baseline_metrics['moderate']['RMSE'],
            dinov3_metrics['moderate']['RMSE']
        ))

    # Ratio
    if baseline_metrics['intense'] and dinov3_metrics['intense']:
        ratio = dinov3_metrics['intense']['RMSE'] / baseline_metrics['intense']['RMSE']
        print("\n" + "-"*55)
        print(f"DINOv2/baseline RMSE ratio (intense): {ratio:.2f}x")
        print("-"*55)

    print("\nKey finding: baseline trained from scratch achieves better performance")
    print(f"in the intense regime (< {threshold} mb), demonstrating that:")
    print("  1. Pressure signal IS present in raw satellite imagery")
    print("  2. Pretrained VFM representations FAIL to preserve this signal")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs DINOv3 predictions')
    parser.add_argument('--baseline_csv', type=str, required=True,
                       help='Path to baseline predictions CSV')
    parser.add_argument('--dinov3_csv', type=str, required=True,
                       help='Path to DINOv3 predictions CSV')
    parser.add_argument('--output_dir', type=str, default='analysis_results/comparison_plots',
                       help='Output directory for plots')
    parser.add_argument('--threshold', type=float, default=980,
                       help='Pressure threshold for intense regime (default: 980 mb)')

    args = parser.parse_args()

    # Load data
    print(f"Loading baseline predictions from {args.baseline_csv}")
    baseline_df = load_predictions(args.baseline_csv)
    print(f"  Loaded {len(baseline_df)} predictions")

    print(f"Loading DINOv3 predictions from {args.dinov3_csv}")
    dinov3_df = load_predictions(args.dinov3_csv)
    print(f"  Loaded {len(dinov3_df)} predictions")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_scatter_baseline(baseline_df, output_path, args.threshold)
    exit(0)
    plot_scatter_comparison(baseline_df, dinov3_df, output_path, args.threshold)
    plot_error_by_pressure(baseline_df, dinov3_df, output_path, args.threshold)
    plot_rmse_by_regime(baseline_df, dinov3_df, output_path, args.threshold)
    plot_combined_analysis(baseline_df, dinov3_df, output_path, args.threshold)

    # Print summary
    print_summary_table(baseline_df, dinov3_df, args.threshold)

    print(f"All plots saved to {output_path}")


if __name__ == '__main__':
    main()
