"""Collect per-seed / per-model fit results into one summary table.

Reads every ``${probe_dir}/<model>/<probe>/*.json`` written by
:mod:`probing.fit` and emits a long-format CSV indexed by
``(probe, model, feature_type, target, seed)`` with the overall and
per-regime metrics flattened into columns.

Usage::

    python -m probing.aggregate \
        probe_dir=outputs/probes \
        output_path=outputs/summary.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _flatten(metrics: dict) -> dict:
    """Flatten the {overall, regimes:{moderate, intense}, …} tree into columns."""
    flat = {f"overall.{k}": v for k, v in metrics["overall"].items()}
    for regime, sub in metrics.get("regimes", {}).items():
        for k, v in sub.items():
            flat[f"{regime}.{k}"] = v
    scalar_keys = ("seed", "probe", "model", "feature_type", "target",
                   "n_train", "n_test", "balance_threshold", "us_only",
                   "selected_alpha", "sigma_global")
    for k in scalar_keys:
        if k in metrics:
            flat[k] = metrics[k]
    return flat


@hydra.main(version_base=None, config_path="configs", config_name="aggregate")
def main(cfg: DictConfig) -> str:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    probe_dir = Path(cfg.probe_dir)
    rows: list[dict] = []
    for json_path in sorted(probe_dir.rglob("*.json")):
        with json_path.open() as f:
            metrics = json.load(f)
        rows.append(_flatten(metrics))
    if not rows:
        log.warning("No probe metric files found under %s", probe_dir)
        return ""
    df = pd.DataFrame(rows)
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(df), out_path)
    return str(out_path)


if __name__ == "__main__":
    main()
