"""Feature loading, trajectory-level splitting, and regime balancing.

Probes operate on the per-frame features produced by
:mod:`dataset.09_extract_features`, packaged as a HuggingFace
:class:`datasets.Dataset` with one row per *cyclone trajectory* and the
following columns:

================  =================================================
``cyclone_id``    multi-agency cyclone identifier (e.g. ``hurdat_…``)
``features``      ``{"cls": (T, D), "spatial_mean": (T, D)}``
``pressure``      ``(T,)`` minimum central pressure in hPa
``wind``          ``(T,)`` maximum sustained wind in kt
``center``        ``(T, 2)`` (lat, lon) at each frame
``frame_valid_mask`` ``(T,)`` bool mask of usable frames
================  =================================================

This module provides the boring-but-load-bearing data plumbing used by
every probe: collapsing valid frames across splits, keeping trajectory
identifiers around for leakage-free splitting, and the regime-balancing
protocol from the paper's §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm

FeatureType = Literal["cls", "spatial_mean"]
TargetName = Literal["pressure", "wind"]


@dataclass
class FlattenedFeatures:
    """Flattened (frame-level) feature tensors with trajectory bookkeeping.

    Each ``(N, ...)`` array enumerates *valid* frames (those passing the
    per-trajectory ``frame_valid_mask``). ``trajectory_ids`` maps each
    frame back to its trajectory index, enabling
    :func:`trajectory_split` to produce leakage-free splits.
    """

    cls_features: np.ndarray          # (N, D)
    spatial_mean_features: np.ndarray # (N, D)
    pressure: np.ndarray              # (N,)
    wind: np.ndarray                  # (N,)
    latitudes: np.ndarray             # (N,)
    longitudes: np.ndarray            # (N,)
    cyclone_ids: np.ndarray           # (N,) object array of strings
    trajectory_ids: np.ndarray        # (N,) int array

    @property
    def feature_dim(self) -> int:
        return int(self.cls_features.shape[1])

    @property
    def num_trajectories(self) -> int:
        return int(np.unique(self.trajectory_ids).size)

    def select(self, mask: np.ndarray) -> "FlattenedFeatures":
        """Index every field with the same boolean / integer mask."""
        return FlattenedFeatures(
            cls_features=self.cls_features[mask],
            spatial_mean_features=self.spatial_mean_features[mask],
            pressure=self.pressure[mask],
            wind=self.wind[mask],
            latitudes=self.latitudes[mask],
            longitudes=self.longitudes[mask],
            cyclone_ids=self.cyclone_ids[mask],
            trajectory_ids=self.trajectory_ids[mask],
        )

    def features(self, feature_type: FeatureType) -> np.ndarray:
        if feature_type == "cls":
            return self.cls_features
        if feature_type == "spatial_mean":
            return self.spatial_mean_features
        raise ValueError(f"unknown feature_type: {feature_type!r}")

    def target(self, name: TargetName) -> np.ndarray:
        if name == "pressure":
            return self.pressure
        if name == "wind":
            return self.wind
        raise ValueError(f"unknown target: {name!r}")


def load_features(
    dataset_path: str | Path,
    splits: Iterable[str] = ("train", "validation", "test"),
    *,
    us_only: bool = False,
) -> FlattenedFeatures:
    """Load and flatten features from a feature-dataset directory.

    Args:
        dataset_path: directory written by :mod:`dataset.09_extract_features`.
        splits: which HF splits to concatenate. The paper protocol pools
            all three and then re-splits per seed at trajectory level
            (see :func:`trajectory_split`).
        us_only: if true, keep only HURDAT-agency cyclones — used for
            the wind-coupling diagnostic (Q_con) where wind labels are
            most consistent.

    Returns:
        :class:`FlattenedFeatures` with valid frames stacked along axis 0.
    """
    dataset_path = Path(dataset_path)
    dataset = load_from_disk(str(dataset_path))

    cls_features: list[np.ndarray] = []
    spatial_mean_features: list[np.ndarray] = []
    pressure: list[float] = []
    wind: list[float] = []
    latitudes: list[float] = []
    longitudes: list[float] = []
    cyclone_ids: list[str] = []
    trajectory_ids: list[int] = []
    traj_idx = 0

    for split_name in splits:
        if split_name not in dataset:
            continue
        split_data = dataset[split_name]
        for example in tqdm(split_data, desc=f"loading {split_name}"):
            cyclone_id = example.get("cyclone_id", f"cyclone_{traj_idx}")
            if us_only and "hurdat" not in cyclone_id:
                continue

            mask = np.array(example["frame_valid_mask"], dtype=bool)
            if not np.any(mask):
                continue

            cls_fts = np.array(example["features"]["cls"])[mask]
            spm_fts = np.array(example["features"]["spatial_mean"])[mask]
            p_arr   = np.array(example["pressure"])[mask]
            w_arr   = np.array(example["wind"])[mask]
            center  = np.array(example["center"])[mask]
            n_valid = int(mask.sum())

            cls_features.append(cls_fts)
            spatial_mean_features.append(spm_fts)
            pressure.extend(p_arr.tolist())
            wind.extend(w_arr.tolist())
            latitudes.extend(center[:, 0].tolist())
            longitudes.extend(center[:, 1].tolist())
            cyclone_ids.extend([cyclone_id] * n_valid)
            trajectory_ids.extend([traj_idx] * n_valid)
            traj_idx += 1

    return FlattenedFeatures(
        cls_features=np.concatenate(cls_features, axis=0),
        spatial_mean_features=np.concatenate(spatial_mean_features, axis=0),
        pressure=np.asarray(pressure),
        wind=np.asarray(wind),
        latitudes=np.asarray(latitudes),
        longitudes=np.asarray(longitudes),
        cyclone_ids=np.asarray(cyclone_ids, dtype=object),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
    )


def regime_balance(
    features: FlattenedFeatures,
    threshold_hpa: float,
    *,
    rng_seed: int = 42,
) -> FlattenedFeatures:
    """Downsample frames so that the two pressure regimes are equal-sized.

    The probing protocol partitions data into a *moderate* (``P_c >=
    threshold``) and *intense* (``P_c < threshold``) regime, then trains
    the linear probe on a balanced subset to prevent the imbalance from
    masking degradation in the high-intensity tail (Fig. 2).

    Args:
        features: flattened features with valid frames.
        threshold_hpa: the regime boundary (980 hPa in the paper).
        rng_seed: seeded so that the same balanced subset is used across
            probe families.

    Returns:
        a new :class:`FlattenedFeatures` containing
        ``min(|moderate|, |intense|)`` frames from each regime.
    """
    bins = [-np.inf, threshold_hpa, np.inf]
    assignment = np.digitize(features.pressure, bins)
    unique_bins, counts = np.unique(assignment, return_counts=True)
    min_samples = int(counts.min())

    rng = np.random.default_rng(rng_seed)
    keep: list[np.ndarray] = []
    for b in unique_bins:
        idx_in_bin = np.flatnonzero(assignment == b)
        keep.append(rng.choice(idx_in_bin, size=min_samples, replace=False))
    keep_idx = np.sort(np.concatenate(keep))
    return features.select(keep_idx)


def trajectory_split(
    features: FlattenedFeatures,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_mask, test_mask)`` over frames split by trajectory.

    Splitting at the trajectory (cyclone) level is required to avoid
    temporal leakage: frames within the same cyclone are correlated and
    a random frame-level split would let the probe memorise the trend
    of a held-out trajectory it has already seen part of.
    """
    unique_trajs = np.unique(features.trajectory_ids)
    train_trajs, test_trajs = train_test_split(
        unique_trajs, test_size=test_size, random_state=seed
    )
    train_mask = np.isin(features.trajectory_ids, train_trajs)
    test_mask = np.isin(features.trajectory_ids, test_trajs)
    return train_mask, test_mask


def per_trajectory_finite_differences(
    feature_dataset_path: str | Path,
    *,
    feature_type: FeatureType,
    target: TargetName,
    splits: Iterable[str] = ("test",),
) -> dict[str, np.ndarray]:
    """Compute per-trajectory finite differences for the Q_dyn diagnostic.

    Returns a dict with keys::

        cyclone_idx, frame_idx, cyclone_id,
        delta_features (N, D), delta_targets (N,), targets (N,),
        latitudes (N,), longitudes (N,)

    where ``N`` is the number of valid transitions (frames whose own
    target and the next frame's target are both finite and where both
    feature vectors contain no NaNs).
    """
    dataset = load_from_disk(str(feature_dataset_path))
    rows: dict[str, list] = {
        "cyclone_idx": [],
        "frame_idx": [],
        "cyclone_id": [],
        "delta_features": [],
        "delta_targets": [],
        "targets": [],
        "latitudes": [],
        "longitudes": [],
    }
    for split_name in splits:
        split = dataset[split_name]
        for ci, example in enumerate(tqdm(split, desc=f"Δ {split_name}")):
            mask = np.array(example["frame_valid_mask"], dtype=bool)
            feats = np.array(example["features"][feature_type])
            targs = np.array(example[target])
            center = np.array(example["center"])
            valid_t  = mask[:-1]
            valid_t1 = mask[1:]
            ok_target = np.isfinite(targs[:-1]) & np.isfinite(targs[1:])
            ok_feat_t  = np.isfinite(feats[:-1]).all(axis=1)
            ok_feat_t1 = np.isfinite(feats[1:]).all(axis=1)
            trans_mask = valid_t & valid_t1 & ok_target & ok_feat_t & ok_feat_t1
            if not np.any(trans_mask):
                continue
            dfeat = (feats[1:] - feats[:-1])[trans_mask]
            dtarg = (targs[1:] - targs[:-1])[trans_mask]
            t_targ = targs[:-1][trans_mask]
            t_lat  = center[:-1, 0][trans_mask]
            t_lon  = center[:-1, 1][trans_mask]
            n = trans_mask.sum()
            rows["cyclone_idx"].extend([ci] * n)
            rows["frame_idx"].extend(np.flatnonzero(trans_mask).tolist())
            rows["cyclone_id"].extend([example.get("cyclone_id", f"cyclone_{ci}")] * n)
            rows["delta_features"].append(dfeat)
            rows["delta_targets"].extend(dtarg.tolist())
            rows["targets"].extend(t_targ.tolist())
            rows["latitudes"].extend(t_lat.tolist())
            rows["longitudes"].extend(t_lon.tolist())
    return {
        "cyclone_idx":   np.asarray(rows["cyclone_idx"]),
        "frame_idx":     np.asarray(rows["frame_idx"]),
        "cyclone_id":    np.asarray(rows["cyclone_id"], dtype=object),
        "delta_features": np.concatenate(rows["delta_features"], axis=0),
        "delta_targets": np.asarray(rows["delta_targets"]),
        "targets":       np.asarray(rows["targets"]),
        "latitudes":     np.asarray(rows["latitudes"]),
        "longitudes":    np.asarray(rows["longitudes"]),
    }


# Canonical model registry used by Hydra configs to expand sweeps.
MODEL_FAMILIES: dict[str, dict[str, str]] = {
    "dinov2-base":      {"family": "dinov2",   "size": "base"},
    "dinov2-large":     {"family": "dinov2",   "size": "large"},
    "dinov3-base":      {"family": "dinov3",   "size": "base"},
    "dinov3-large":     {"family": "dinov3",   "size": "large"},
    "dinov3-satellite": {"family": "dinov3",   "size": "satellite"},
    "clip-base":        {"family": "clip",     "size": "base"},
    "clip-large":       {"family": "clip",     "size": "large"},
    "siglip-base":      {"family": "siglip",   "size": "base"},
    "siglip2-base":     {"family": "siglip2",  "size": "base"},
    "mae-base":         {"family": "mae",      "size": "base"},
    "mae-large":        {"family": "mae",      "size": "large"},
}


def split_model_name(name: str) -> tuple[str, str]:
    """Return ``(family, size)`` for a registered VFM name (e.g. ``dinov3-base``)."""
    meta = MODEL_FAMILIES.get(name)
    if meta is None:
        family, _, size = name.partition("-")
        return family, size or "base"
    return meta["family"], meta["size"]
