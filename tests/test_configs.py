"""Hydra-composition smoke tests for every top-level config root.

A config that fails to compose is a footgun (it surfaces only on a
launcher run). We compose every entry point's root config plus every
experiment config so CI fails fast on a broken `defaults:` chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

PROBING_CONFIG_DIR = str(
    (Path(__file__).resolve().parent.parent / "probing" / "configs").resolve()
)
TRAIN_CONFIG_DIR = str(
    (Path(__file__).resolve().parent.parent / "configs").resolve()
)


# ------------------------------------------------------------------------ #
# probing/ Hydra roots                                                      #
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("config_name", ["fit", "diagnose", "geometry", "aggregate"])
def test_probing_root_composes(config_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=PROBING_CONFIG_DIR):
        cfg = compose(config_name=config_name)
    assert cfg is not None


@pytest.mark.parametrize("probe", ["ridge", "lasso", "mlp", "transformer"])
def test_fit_with_each_probe(probe: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=PROBING_CONFIG_DIR):
        cfg = compose(config_name="fit", overrides=[f"probe={probe}"])
    assert cfg.probe.name == probe


@pytest.mark.parametrize("model", [
    "dinov2-base", "dinov2-large", "dinov3-base", "dinov3-large",
    "dinov3-satellite", "clip-base", "clip-large",
    "siglip-base", "siglip2-base", "mae-base", "mae-large",
])
def test_every_paper_model_config_composes(model: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=PROBING_CONFIG_DIR):
        cfg = compose(config_name="fit", overrides=[f"model={model}"])
    assert cfg.model.name == model


@pytest.mark.parametrize("diagnostic", ["q_stat", "q_dyn", "q_con"])
def test_diagnose_with_each_diagnostic(diagnostic: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=PROBING_CONFIG_DIR):
        cfg = compose(config_name="diagnose", overrides=[f"diagnostic={diagnostic}"])
    assert cfg.diagnostic.name == diagnostic


@pytest.mark.parametrize("data", ["full", "balanced_980", "us_only", "ood_basin"])
def test_each_data_config_composes(data: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=PROBING_CONFIG_DIR):
        cfg = compose(config_name="fit", overrides=[f"data={data}"])
    assert "feature_dataset_root" in cfg.data


# ------------------------------------------------------------------------ #
# pixel-sup ablation                                                        #
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("config_name", ["train", "eval"])
def test_pixel_sup_root_composes(config_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=TRAIN_CONFIG_DIR):
        cfg = compose(config_name=config_name)
    assert cfg is not None


@pytest.mark.parametrize("experiment", ["simple_cnn", "train_resnet"])
def test_pixel_sup_experiments_compose(experiment: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=TRAIN_CONFIG_DIR):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
    assert "_target_" in cfg.data
    assert "_target_" in cfg.model
