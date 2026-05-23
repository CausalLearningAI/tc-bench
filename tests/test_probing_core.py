"""Unit tests for :mod:`probing.core`."""

from __future__ import annotations

import numpy as np
import pytest

from probing.core.data import (
    MODEL_FAMILIES,
    load_features,
    regime_balance,
    split_model_name,
    trajectory_split,
)
from probing.core.geometry import (
    effective_dimensionality,
    feature_spread,
    make_pressure_bins,
    pca_per_bin,
)
from probing.core.metrics import (
    normalized_absolute_error,
    regime_metrics,
    regression_metrics,
)
from probing.core.probes import LassoProbe, RidgeProbe, TransformerProbe, build_probe


# --------------------------------------------------------------------- #
# Data primitives                                                       #
# --------------------------------------------------------------------- #


def test_load_features_returns_consistent_shapes(synthetic_feature_dataset):
    feats = load_features(synthetic_feature_dataset)
    n = feats.cls_features.shape[0]
    assert feats.spatial_mean_features.shape == feats.cls_features.shape
    assert feats.pressure.shape == (n,)
    assert feats.wind.shape == (n,)
    assert feats.latitudes.shape == (n,)
    assert feats.cyclone_ids.shape == (n,)
    assert feats.trajectory_ids.shape == (n,)
    # 30 trajectories (20+5+5), each with 16 frames => 480 valid frames.
    assert n == 30 * 16
    assert feats.num_trajectories == 30
    assert feats.feature_dim == 32


def test_load_features_us_only_filter(synthetic_feature_dataset):
    feats = load_features(synthetic_feature_dataset, us_only=True)
    # Our synthetic ids all start with `hurdat_` so us_only should keep all.
    assert feats.num_trajectories == 30
    # Make sure every retained id contains "hurdat".
    assert all("hurdat" in cid for cid in feats.cyclone_ids)


def test_regime_balance_equalises_bin_counts(synthetic_feature_dataset):
    feats = load_features(synthetic_feature_dataset)
    bal = regime_balance(feats, threshold_hpa=980.0)
    n_intense  = int((bal.pressure < 980).sum())
    n_moderate = int((bal.pressure >= 980).sum())
    assert n_intense == n_moderate, "regime_balance must yield equal-sized bins"


def test_trajectory_split_is_leakage_free(synthetic_feature_dataset):
    feats = load_features(synthetic_feature_dataset)
    train_mask, test_mask = trajectory_split(feats, test_size=0.25, seed=0)
    train_traj = set(feats.trajectory_ids[train_mask].tolist())
    test_traj  = set(feats.trajectory_ids[test_mask].tolist())
    assert train_traj.isdisjoint(test_traj)
    assert train_traj.union(test_traj) == set(np.unique(feats.trajectory_ids).tolist())


def test_model_registry_completeness():
    assert "dinov3-base" in MODEL_FAMILIES
    family, size = split_model_name("dinov3-base")
    assert (family, size) == ("dinov3", "base")
    family, size = split_model_name("unknown-large")
    assert (family, size) == ("unknown", "large")
    assert len(MODEL_FAMILIES) == 11, "the paper benchmarks 11 VFM variants"


# --------------------------------------------------------------------- #
# Probes                                                                #
# --------------------------------------------------------------------- #


def test_build_probe_registry():
    ridge = build_probe("ridge")
    assert isinstance(ridge, RidgeProbe)
    lasso = build_probe("lasso")
    assert isinstance(lasso, LassoProbe)
    transformer = build_probe("transformer")
    assert isinstance(transformer, TransformerProbe)
    with pytest.raises(KeyError):
        build_probe("nope")


def test_transformer_probe_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(80, 16)).astype(np.float32), rng.normal(size=(20, 16)).astype(np.float32)
    w_true = rng.normal(size=16)
    y_train = X_train @ w_true + rng.normal(scale=0.01, size=80)
    probe = build_probe("transformer", device="cpu", max_epochs=3, batch_size=16,
                        num_layers=2, num_heads=4, hidden_dim=128, num_tokens=4)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    assert y_pred.shape == (20,)
    path = tmp_path / "tprobe.pt"
    probe.save(path)
    reloaded = TransformerProbe.load(path)
    np.testing.assert_allclose(reloaded.predict(X_test), y_pred, atol=1e-5)
    with pytest.raises(AttributeError):
        _ = probe.coef_


def test_ridge_probe_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(80, 10)), rng.normal(size=(20, 10))
    w_true = rng.normal(size=10)
    y_train = X_train @ w_true + rng.normal(scale=0.01, size=80)
    y_test  = X_test  @ w_true
    probe = build_probe("ridge")
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    assert y_pred.shape == (20,)
    metrics = regression_metrics(y_test, y_pred)
    assert metrics["pearson"] > 0.95
    # save / load round-trip:
    path = tmp_path / "probe.pkl"
    probe.save(path)
    reloaded = RidgeProbe.load(path)
    np.testing.assert_allclose(reloaded.predict(X_test), y_pred, atol=1e-6)
    # coef_ is exposed for Q_dyn:
    assert probe.coef_.shape == (10,)


# --------------------------------------------------------------------- #
# Metrics                                                               #
# --------------------------------------------------------------------- #


def test_metrics_nan_safe():
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.1, np.nan, 3.0, 4.2])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["n"] == 2  # only positions 0 and 3 are jointly finite
    assert np.isfinite(metrics["mae"])


def test_regression_metrics_reports_normalized_with_sigma():
    rng = np.random.default_rng(0)
    y_true = rng.normal(loc=970.0, scale=15.0, size=200)
    y_pred = y_true + rng.normal(scale=2.0, size=200)
    m = regression_metrics(y_true, y_pred, sigma=15.0)
    for k in ("rmse", "mae", "sigma", "normalized_rmse", "normalized_mae"):
        assert k in m
    assert m["sigma"] == 15.0
    assert abs(m["normalized_rmse"] - m["rmse"] / 15.0) < 1e-9
    assert abs(m["normalized_mae"]  - m["mae"]  / 15.0) < 1e-9


def test_regime_metrics_partition_and_shared_sigma():
    y_true = np.array([990.0, 950.0, 970.0, 1010.0])
    y_pred = np.array([991.0, 948.0, 969.0, 1011.0])
    out = regime_metrics(y_true, y_pred, threshold_hpa=980.0,
                         intense_below=True, sigma=20.0)
    assert {"intense", "moderate"} <= out.keys()
    # 2 samples below 980 (intense), 2 at or above (moderate).
    assert out["intense"]["n"] == 2
    assert out["moderate"]["n"] == 2
    # The denominator must be the global sigma, not each regime's local std.
    assert out["intense"]["sigma"] == 20.0
    assert out["moderate"]["sigma"] == 20.0


def test_normalized_absolute_error_baseline_at_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    # naive mean estimator → expected ξ_stat ≈ 1.
    nae = normalized_absolute_error(y, np.full_like(y, y.mean()))
    assert abs(nae.mean() - np.mean(np.abs(y - y.mean())) / y.std()) < 1e-9


# --------------------------------------------------------------------- #
# Geometry                                                              #
# --------------------------------------------------------------------- #


def test_effective_dimensionality_bounds():
    rng = np.random.default_rng(0)
    isotropic = rng.normal(size=(500, 16))
    rank1 = rng.normal(size=(500, 1)) @ rng.normal(size=(1, 16))
    d_iso = effective_dimensionality(isotropic)
    d_r1  = effective_dimensionality(rank1)
    assert d_iso > 4 * d_r1  # isotropic spread vs rank-1 collapse


def test_geometry_pipeline(synthetic_feature_dataset):
    feats = load_features(synthetic_feature_dataset)
    bins = make_pressure_bins()
    spread = feature_spread(feats.cls_features, feats.pressure, bins)
    pcs    = pca_per_bin(feats.cls_features, feats.pressure, bins)
    # spread is finite (NaN only when a bin has < 2 samples).
    assert any(np.isfinite(v) for v in spread.values())
    # PCA returns one column per bin.
    for label, scores in pcs.items():
        assert scores.ndim == 2 and scores.shape[1] == 1
