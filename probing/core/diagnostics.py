"""Q_stat, Q_dyn, Q_con — the three structural-alignment diagnostics.

Each function takes the artefacts produced by :mod:`probing.fit` (one
fitted probe per ``(model, feature_type, target, seed)``) plus the
test-split features, and emits a tidy :class:`pandas.DataFrame` that
:mod:`figures` consumes to produce the paper plots.

The CSV schema is shared across diagnostics so that the figure scripts
can be uniform — see each function's docstring for the column list.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from probing.core.data import (
    FeatureType,
    TargetName,
    per_trajectory_finite_differences,
    split_model_name,
)
from probing.core.probes import Probe

__all__ = ["q_stat", "q_dyn", "q_con"]


# --------------------------------------------------------------------- #
# Q_stat — static fidelity.                                              #
# --------------------------------------------------------------------- #


def q_stat(
    probe: Probe,
    feature_dataset_path: str | Path,
    *,
    model_name: str,
    feature_type: FeatureType,
    target: TargetName,
    seed: int,
    splits: tuple[str, ...] = ("test",),
    us_only: bool = False,
) -> pd.DataFrame:
    """Static fidelity probe (paper Eq. 4.1).

    For every valid frame of ``splits``, applies the fitted probe and
    emits a row::

        model_family, model_size, cyclone_id, cyclone_idx, frame_idx,
        feature_type, target, seed, latitudes, longitudes,
        targets, y_pred
    """
    from probing.core.data import load_features

    features = load_features(feature_dataset_path, splits=splits, us_only=us_only)
    X = features.features(feature_type)
    y_true = features.target(target)
    y_pred = probe.predict(X)

    family, size = split_model_name(model_name)
    n = X.shape[0]
    return pd.DataFrame({
        "model_family":  np.full(n, family, dtype=object),
        "model_size":    np.full(n, size, dtype=object),
        "cyclone_id":    features.cyclone_ids,
        "cyclone_idx":   features.trajectory_ids,
        "frame_idx":     np.arange(n),
        "feature_type":  np.full(n, feature_type, dtype=object),
        "target":        np.full(n, target, dtype=object),
        "seed":          np.full(n, seed, dtype=np.int64),
        "latitudes":     features.latitudes,
        "longitudes":    features.longitudes,
        "targets":       y_true,
        "y_pred":        y_pred,
    })


# --------------------------------------------------------------------- #
# Q_dyn — dynamic coherence.                                             #
# --------------------------------------------------------------------- #


def q_dyn(
    probe: Probe,
    feature_dataset_path: str | Path,
    *,
    model_name: str,
    feature_type: FeatureType,
    target: TargetName,
    seed: int,
    splits: tuple[str, ...] = ("test",),
) -> pd.DataFrame:
    """Dynamic coherence probe (paper Eq. 4.2).

    Re-uses the *static* linear probe ``L`` and applies it to
    finite-difference feature deltas: ``L · Δz`` is compared against the
    physical delta ``Δy``. Non-linear probes raise.

    Emits one row per valid transition::

        model_family, model_size, cyclone_id, cyclone_idx, frame_idx,
        feature_type, target, seed, latitudes, longitudes,
        targets,            (the pressure at frame t)
        delta_targets,      (Δy = y_{t+1} - y_t)
        proj_delta_features (L · Δz)
    """
    try:
        coef = np.asarray(probe.coef_, dtype=np.float64)
    except AttributeError as e:
        raise ValueError(
            f"Q_dyn requires a linear probe (probe={probe.name!r} is non-linear)"
        ) from e

    diffs = per_trajectory_finite_differences(
        feature_dataset_path,
        feature_type=feature_type,
        target=target,
        splits=splits,
    )

    proj_delta = diffs["delta_features"] @ coef
    family, size = split_model_name(model_name)
    n = proj_delta.shape[0]
    return pd.DataFrame({
        "model_family":         np.full(n, family, dtype=object),
        "model_size":           np.full(n, size, dtype=object),
        "cyclone_id":           diffs["cyclone_id"],
        "cyclone_idx":          diffs["cyclone_idx"],
        "frame_idx":            diffs["frame_idx"],
        "feature_type":         np.full(n, feature_type, dtype=object),
        "target":               np.full(n, target, dtype=object),
        "seed":                 np.full(n, seed, dtype=np.int64),
        "latitudes":            diffs["latitudes"],
        "longitudes":           diffs["longitudes"],
        "targets":              diffs["targets"],
        "delta_targets":        diffs["delta_targets"],
        "proj_delta_features":  proj_delta,
    })


# --------------------------------------------------------------------- #
# Q_con — manifold consistency.                                          #
# --------------------------------------------------------------------- #


def q_con(
    q_stat_pressure: pd.DataFrame,
    q_stat_wind: pd.DataFrame,
) -> pd.DataFrame:
    """Manifold consistency probe (paper Eq. 4.4).

    Joins two :func:`q_stat` outputs (one fitted to pressure, one to
    wind) on (``model_family``, ``model_size``, ``cyclone_id``,
    ``cyclone_idx``, ``frame_idx``) and renames the columns so the
    figure script can produce the pressure–wind scatter directly::

        pressure_true, pressure_pred, wind_true, wind_pred

    Pairs are dropped if either probe's row is missing for that frame.
    """
    key = ["model_family", "model_size", "cyclone_id",
           "cyclone_idx", "frame_idx", "latitudes", "longitudes", "seed"]
    pressure = q_stat_pressure[key + ["targets", "y_pred"]].rename(
        columns={"targets": "pressure_true", "y_pred": "pressure_pred"}
    )
    wind = q_stat_wind[key + ["targets", "y_pred"]].rename(
        columns={"targets": "wind_true", "y_pred": "wind_pred"}
    )
    return pressure.merge(wind, on=key, how="inner")
