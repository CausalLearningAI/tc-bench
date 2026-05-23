"""End-to-end smoke tests for the Hydra-driven probing pipeline.

These tests run the entire `fit → diagnose → aggregate` chain on the
synthetic feature dataset from :mod:`tests.conftest`. They don't assert
anything about model accuracy; they assert that the wiring is intact
and that every artefact lands at its expected path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import probing.aggregate as aggregate_mod
import probing.diagnose as diagnose_mod
import probing.fit as fit_mod

CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "probing" / "configs").resolve())


def _compose(name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=name, overrides=overrides)


def test_fit_diagnose_aggregate_end_to_end(synthetic_feature_dataset, tmp_path):
    features_root = synthetic_feature_dataset.parent
    # The model config keys off features_<model.name>/, so rename the
    # synthetic features dir to that convention.
    target_features_dir = features_root / "features_dinov3-base"
    if not target_features_dir.exists():
        synthetic_feature_dataset.rename(target_features_dir)

    probe_dir = tmp_path / "probes"
    diag_dir = tmp_path / "diagnostics"

    common_overrides = [
        f"data.feature_dataset_root={features_root}",
        f"output_dir={probe_dir}",
        "data.balance_threshold=null",          # too few samples for the 980 split
        "data.test_size=0.3",
        "probe.params.alphas=[0.1, 1.0, 10.0]",
        "probe.params.cv_folds=2",
        "seed=42",
    ]

    # --- fit (pressure + wind so q_con can join) ---------------------
    for target in ("pressure", "wind"):
        cfg = _compose("fit", common_overrides + [f"target={target}"])
        fit_mod.main(cfg)
        pkl = probe_dir / "dinov3-base" / "ridge" / f"cls_{target}_seed42.pkl"
        json_meta = pkl.with_suffix(".json")
        assert pkl.exists(), f"probe pickle missing for target={target}"
        assert json_meta.exists()
        with json_meta.open() as f:
            meta = json.load(f)
        assert meta["probe"] == "ridge"
        assert meta["target"] == target
        assert meta["overall"]["n"] > 0

    # --- diagnose Q_stat / Q_dyn / Q_con -----------------------------
    diag_overrides = [
        f"data.feature_dataset_root={features_root}",
        f"probe_dir={probe_dir}",
        f"output_dir={diag_dir}",
        "data.balance_threshold=null",
        "data.models=[dinov3-base]",
        "seed=42",
    ]
    for diagnostic, expected_stem in [
        ("q_stat", "predictions_cls_pressure_seed42.csv"),
        ("q_dyn",  "predictions_cls_pressure_seed42.csv"),
        ("q_con",  "predictions_cls_seed42.csv"),
    ]:
        cfg = _compose("diagnose", diag_overrides + [f"diagnostic={diagnostic}"])
        diagnose_mod.main(cfg)
        out_csv = diag_dir / diagnostic / expected_stem
        assert out_csv.exists(), f"{diagnostic} CSV missing"
        df = pd.read_csv(out_csv)
        assert "model_family" in df.columns
        assert len(df) > 0

    # --- aggregate ---------------------------------------------------
    agg_csv = tmp_path / "summary.csv"
    cfg = _compose("aggregate",
                   [f"probe_dir={probe_dir}", f"output_path={agg_csv}"])
    aggregate_mod.main(cfg)
    assert agg_csv.exists()
    df = pd.read_csv(agg_csv)
    # Two probes were fit (pressure + wind), one model, one seed → 2 rows.
    assert len(df) == 2
    assert set(df["target"]) == {"pressure", "wind"}
