"""Hydra entry: load fitted probes and produce a diagnostic DataFrame.

Each invocation processes one (``diagnostic``, ``feature_type``,
``target``, ``seed``) tuple sweeping over every model. The diagnostic
DataFrame schema is stable across the three probes (Q_stat, Q_dyn, Q_con)
so that figures and aggregations are uniform downstream.

Usage::

    python -m probing.diagnose \
        diagnostic=q_stat feature_type=cls target=pressure seed=42

Outputs:

    ${output_dir}/${diagnostic}/predictions_${feature_type}_${target}_seed${seed}.csv

Implementation note: Q_dyn re-uses the *static* linear probe directly
(its coefficients are applied to feature deltas). Q_con merges two Q_stat
DataFrames (pressure and wind) on a per-frame key. This matches the
paper's protocol — no separate probe is fit for the dynamic or
consistency probes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from probing.core import diagnostics
from probing.core.data import MODEL_FAMILIES
from probing.core.probes import RidgeProbe

log = logging.getLogger(__name__)


def _probe_path(cfg: DictConfig, model_name: str, target: str) -> Path:
    return (
        Path(cfg.probe_dir) / model_name / cfg.probe.name
        / f"{cfg.feature_type}_{target}_seed{int(cfg.seed)}.pkl"
    )


def _feature_dataset_path(cfg: DictConfig, model_name: str) -> Path:
    return Path(cfg.data.feature_dataset_root) / f"features_{model_name}"


def _load_probe(cfg: DictConfig, model_name: str, target: str):
    """Load a probe of the configured family. Defaults to RidgeProbe."""
    path = _probe_path(cfg, model_name, target)
    if cfg.probe.name == "ridge":
        return RidgeProbe.load(path)
    if cfg.probe.name == "lasso":
        from probing.core.probes import LassoProbe
        return LassoProbe.load(path)
    if cfg.probe.name == "mlp":
        from probing.core.probes import MLPProbe
        return MLPProbe.load(path)
    if cfg.probe.name == "transformer":
        from probing.core.probes import TransformerProbe
        return TransformerProbe.load(path)
    raise ValueError(f"unknown probe family {cfg.probe.name!r}")


def _run_q_stat(cfg: DictConfig, models: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name in models:
        probe = _load_probe(cfg, model_name, target=cfg.target)
        frames.append(diagnostics.q_stat(
            probe,
            feature_dataset_path=_feature_dataset_path(cfg, model_name),
            model_name=model_name,
            feature_type=cfg.feature_type,
            target=cfg.target,
            seed=int(cfg.seed),
            splits=tuple(cfg.data.test_splits),
            us_only=bool(cfg.data.us_only),
        ))
    return pd.concat(frames, ignore_index=True)


def _run_q_dyn(cfg: DictConfig, models: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name in models:
        probe = _load_probe(cfg, model_name, target=cfg.target)
        frames.append(diagnostics.q_dyn(
            probe,
            feature_dataset_path=_feature_dataset_path(cfg, model_name),
            model_name=model_name,
            feature_type=cfg.feature_type,
            target=cfg.target,
            seed=int(cfg.seed),
            splits=tuple(cfg.data.test_splits),
        ))
    return pd.concat(frames, ignore_index=True)


def _run_q_con(cfg: DictConfig, models: list[str]) -> pd.DataFrame:
    # We need Q_stat tables for both pressure and wind, then merge.
    cfg_pressure = OmegaConf.merge(cfg, OmegaConf.create({"target": "pressure"}))
    cfg_wind     = OmegaConf.merge(cfg, OmegaConf.create({"target": "wind"}))
    df_pressure = _run_q_stat(cfg_pressure, models)
    df_wind     = _run_q_stat(cfg_wind,     models)
    return diagnostics.q_con(df_pressure, df_wind)


@hydra.main(version_base=None, config_path="configs", config_name="diagnose")
def main(cfg: DictConfig) -> str:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    models = list(cfg.data.models) if cfg.data.models else list(MODEL_FAMILIES)
    if cfg.diagnostic.name == "q_stat":
        df = _run_q_stat(cfg, models)
    elif cfg.diagnostic.name == "q_dyn":
        df = _run_q_dyn(cfg, models)
    elif cfg.diagnostic.name == "q_con":
        df = _run_q_con(cfg, models)
    else:
        raise ValueError(f"unknown diagnostic {cfg.diagnostic.name!r}")

    if cfg.diagnostic.name == "q_con":
        # Q_con drops 'target' since it joins both — write to a target-agnostic file.
        stem = f"predictions_{cfg.feature_type}_seed{int(cfg.seed)}"
    else:
        stem = f"predictions_{cfg.feature_type}_{cfg.target}_seed{int(cfg.seed)}"
    out_dir = Path(cfg.output_dir) / cfg.diagnostic.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows).", out_path, len(df))
    return str(out_path)


if __name__ == "__main__":
    main()
