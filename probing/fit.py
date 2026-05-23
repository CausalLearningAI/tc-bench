"""Hydra entry: fit a probe and persist it to disk.

Usage::

    python -m probing.fit \
        probe=ridge model=dinov3-base \
        feature_type=cls target=pressure seed=42

Sweep::

    python -m probing.fit -m \
        probe=ridge,lasso \
        model=dinov3-base,dinov3-large,clip-base \
        feature_type=cls,spatial_mean \
        target=pressure,wind \
        seed=42,43,44

Outputs:

    ${output_dir}/${model}/${probe}/${feature_type}_${target}_seed${seed}.pkl
    ${output_dir}/${model}/${probe}/${feature_type}_${target}_seed${seed}.json   (metrics)

The same per-seed trajectory split is used by the static, dynamic, and
manifold-consistency diagnostics — see :mod:`probing.diagnose`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probing.core.data import (
    load_features,
    regime_balance,
    trajectory_split,
)
from probing.core.metrics import regime_metrics, regression_metrics
from probing.core.probes import build_probe

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="fit")
def main(cfg: DictConfig) -> dict:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- 1. Load features ------------------------------------------------
    feature_path = Path(cfg.data.feature_dataset_root) / f"features_{cfg.model.name}"
    features = load_features(
        feature_path,
        splits=tuple(cfg.data.splits),
        us_only=bool(cfg.data.us_only),
    )

    # --- 2. Optional regime balancing -----------------------------------
    if cfg.data.balance_threshold is not None:
        features = regime_balance(features, float(cfg.data.balance_threshold))
        log.info("Regime-balanced to %d frames (%d trajectories).",
                 features.cls_features.shape[0], features.num_trajectories)

    # --- 3. Trajectory-level split --------------------------------------
    train_mask, test_mask = trajectory_split(
        features, test_size=cfg.data.test_size, seed=int(cfg.seed),
    )
    X_train = features.features(cfg.feature_type)[train_mask]
    y_train = features.target(cfg.target)[train_mask]
    X_test  = features.features(cfg.feature_type)[test_mask]
    y_test  = features.target(cfg.target)[test_mask]

    # --- 4. Fit the probe -----------------------------------------------
    probe_kwargs = OmegaConf.to_container(cfg.probe.params, resolve=True) or {}
    probe = build_probe(cfg.probe.name, **probe_kwargs)
    probe.fit(X_train, y_train)

    # --- 5. Evaluate on test split (overall + per-regime) ---------------
    # Normalization follows Eq. 4.1: σ is the *global* std of the target
    # on the held-out test split, shared across overall + both regimes
    # so the regime-level normalized errors are comparable to each other
    # (a per-regime σ would erase the degradation Fig. 2 reports).
    y_pred = probe.predict(X_test)
    finite_test = np.isfinite(y_test)
    sigma_global = float(np.std(y_test[finite_test])) if finite_test.any() else float("nan")
    metrics = {
        "overall":  regression_metrics(y_test, y_pred, sigma=sigma_global),
        "regimes":  regime_metrics(
            y_test, y_pred,
            threshold_hpa=float(cfg.data.regime_threshold_hpa),
            intense_below=(cfg.target == "pressure"),
            sigma=sigma_global,
        ),
        "sigma_global":     sigma_global,
        "seed":             int(cfg.seed),
        "probe":            cfg.probe.name,
        "model":            cfg.model.name,
        "feature_type":     cfg.feature_type,
        "target":           cfg.target,
        "n_train":          int(train_mask.sum()),
        "n_test":           int(test_mask.sum()),
        "balance_threshold": cfg.data.balance_threshold,
        "us_only":          bool(cfg.data.us_only),
    }
    if hasattr(probe, "selected_alpha_") and probe.selected_alpha_ is not None:
        metrics["selected_alpha"] = float(probe.selected_alpha_)
    log.info("Test metrics: %s", json.dumps(metrics["overall"], indent=2))

    # --- 6. Persist probe + metrics -------------------------------------
    out_dir = (Path(cfg.output_dir) / cfg.model.name / cfg.probe.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.feature_type}_{cfg.target}_seed{int(cfg.seed)}"
    probe.save(out_dir / f"{stem}.pkl")
    with (out_dir / f"{stem}.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved probe + metrics under %s", out_dir)
    return metrics


if __name__ == "__main__":
    main()
