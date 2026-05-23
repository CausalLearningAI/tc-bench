
from __future__ import annotations

import argparse
import json
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
import json
import pandas as pd

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


BASINS = ['in distr.', 'atcf', 'bom', 'hurdat_atl', 'hurdat_epa',
          'nadi', 'newdelhi', 'reunion', 'tokyo', 'wellington']


def _load_results(probe_dir: Path, model_name: str, err_name: str = 'mae') -> pd.DataFrame:
    """Read MAE from the modern ``probing.fit`` JSON layout.

    Expects:
      ``<probe_dir>/<model_name>/ridge/cls_pressure_seed42.json``        (in-dist)
      ``<probe_dir>/<model_name>_ood_<basin>/ridge/cls_pressure_seed42.json`` (per-basin)
    """
    id_json = probe_dir / model_name / 'ridge' / 'cls_pressure_seed42.json'
    with open(id_json) as f:
        errs = [json.load(f)['overall'][err_name]]
    for b in BASINS[1:]:
        ood_json = probe_dir / f'{model_name}_ood_{b}' / 'ridge' / 'cls_pressure_seed42.json'
        with open(ood_json) as f:
            errs.append(json.load(f)['overall'][err_name])
    return pd.DataFrame({'basin': BASINS, 'err': np.asarray(errs)})


def plot(df: pd.DataFrame, output_path: Path) -> None:
    print(df)
    fig, ax = plt.subplots(figsize=(3.75, 2.2))
    basin_indices = np.arange(len(BASINS))
    ax.bar(basin_indices, df['err'], color=COLORS['moderate'], edgecolor='black')
    ax.set_xticks(basin_indices)
    ax.set_xticklabels(BASINS, rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.patches[0].set_color(COLORS['intense'])
    ax.patches[0].set_edgecolor('black')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['intense'], lw=4, label='In Distribution'),
        Line2D([0], [0], color=COLORS['moderate'], lw=4, label='Out of Distribution'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f'Wrote {output_path}')


if __name__ == "__main__":
    import argparse
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe_dir', type=Path,
                    default=repo_root / 'outputs' / 'probes')
    ap.add_argument('--model', type=str, default='dinov3-base')
    ap.add_argument('--err_name', type=str, default='mae')
    ap.add_argument('--output', type=Path,
                    default=repo_root / 'figs' / 'fig1c_ood.pdf')
    args = ap.parse_args()

    df = _load_results(args.probe_dir, args.model, args.err_name)
    plot(df, args.output)