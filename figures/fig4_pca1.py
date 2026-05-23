import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets import load_from_disk

# ICML Style Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def generate_latent_collapse_plot(feature_path):
    # Load features and physical labels
    # Expected: data['features'] (N, 768), data['pressure'] (N,)
    data = load_from_disk(feature_path)['test']
    
    z = []
    p_true = []
    for ex in data:
        mask = np.array(ex['frame_valid_mask']).astype(bool)
        features = np.array(ex["features"]["cls"])[mask]
        pressures = np.array(ex['pressure'])[mask]
        z.extend(features)
        p_true.extend(pressures)
        
    z = np.array(z)
    p_true = np.array(p_true)
    
    # 1. Dimensionality Reduction to capture the main axis of variation
    z_norm = StandardScaler().fit_transform(z)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_norm)
    latent_axis = z_pca[:, 0]  # Primary latent variation
    
    # 2. Define Regimes
    threshold = 980
    mask_intense = p_true < threshold
    mask_moderate = p_true >= threshold
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(3.25, 2.5)) # Single column width
    
    # Scatter points with transparency to show density collapse
    ax.scatter(p_true[mask_moderate], latent_axis[mask_moderate], 
               c='#d8b365', alpha=0.15, s=2, label='Moderate ($\geq$ 980 hPa)', rasterized=True)
    ax.scatter(p_true[mask_intense], latent_axis[mask_intense], 
               c='#5ab4ac', alpha=0.4, s=2, label='Intense ($<$ 980 hPa)', rasterized=True)
    
    # Vertical line at structural inflection point
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(threshold-2, ax.get_ylim()[1]*0.9, 'Latent Collapse Zone', 
            rotation=90, verticalalignment='top', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Minimum Central Pressure $P_c$ (hPa)')
    ax.set_ylabel('Latent PC1 (DINOv3-Base)')
    ax.invert_xaxis() # Pressure decreases to the right (intensification)
    
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('scripts/99_plots/latent_collapse_dinov3.pdf', bbox_inches='tight', dpi=300)
    print("Plot saved to scripts/99_plots/latent_collapse_dinov3.pdf")

if __name__ == "__main__":
    _data_root = os.environ.get("DATA_ROOT", os.path.join(os.environ["HOME"], "tcbench"))
    _features_dir = os.environ.get("FEATURES_DIR", os.path.join(_data_root, "image_features"))
    feat_path = os.path.join(_features_dir, "features_dinov3-base")
    generate_latent_collapse_plot(feat_path)