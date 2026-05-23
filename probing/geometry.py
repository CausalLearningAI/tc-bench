"""Hydra entry: intrinsic-geometry diagnostics for §4.2 of the paper.

For a given model and feature_type, computes three quantities per
pressure bin and writes them to a single CSV that the Fig. 4 script
consumes:

* ``d_eff``         — participation-ratio effective dimensionality
* ``feature_spread``— mean pairwise Euclidean distance
* ``pc1_mean``      — within-bin mean of the first PCA score

Usage::

    python -m probing.geometry \
        model=dinov3-base feature_type=cls

Output:

    ${output_dir}/${model}/${feature_type}.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from probing.core.data import load_features
from probing.core.geometry import (
    effective_dimensionality,
    feature_spread,
    make_pressure_bins,
    pca_per_bin,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="geometry")
def main(cfg: DictConfig) -> str:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    feature_path = Path(cfg.data.feature_dataset_root) / f"features_{cfg.model.name}"
    features = load_features(feature_path, splits=tuple(cfg.data.splits))
    X = features.features(cfg.feature_type)
    p = features.pressure

    bins = make_pressure_bins(tuple(cfg.bins.edges))

    rows = []
    spreads = feature_spread(X, p, bins, max_pairs=int(cfg.max_pairs))
    pcs = pca_per_bin(X, p, bins, n_components=1, standardize=True)
    for b in bins:
        mask = b.mask(p)
        rows.append({
            "pressure_low":   b.low,
            "pressure_high":  b.high,
            "label":          b.label,
            "n":              int(mask.sum()),
            "d_eff":          effective_dimensionality(X[mask]),
            "feature_spread": spreads[b.label],
            "pc1_mean":       float(pcs[b.label].mean()) if pcs[b.label].size else float("nan"),
        })

    out_dir = Path(cfg.output_dir) / cfg.model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.feature_type}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Wrote %s.", out_path)
    return str(out_path)


if __name__ == "__main__":
    main()
