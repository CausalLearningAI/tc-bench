"""Intrinsic-geometry diagnostics for §4.2 of the paper.

Three quantities, each computed inside narrow pressure bins so that the
moderate vs intense regime contrast in Fig. 4 falls out directly:

* :func:`pca_per_bin` — top principal components of the within-bin
  feature covariance, used for the Latent–Physical relationship
  visualisation (Fig. 4a).
* :func:`effective_dimensionality` — *participation ratio* of the
  within-bin covariance eigenspectrum
  (Fig. 4b; Eq. 4.5 of the paper).
* :func:`feature_spread` — mean pairwise Euclidean distance within a
  bin, capped at ``max_pairs`` to keep memory bounded (Fig. 4c).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PressureBin",
    "make_pressure_bins",
    "effective_dimensionality",
    "pca_per_bin",
    "feature_spread",
]


@dataclass(frozen=True)
class PressureBin:
    """Half-open interval ``[low, high)`` plus a human-readable label."""

    low: float
    high: float
    label: str

    def mask(self, pressure: np.ndarray) -> np.ndarray:
        return (pressure >= self.low) & (pressure < self.high)


def make_pressure_bins(
    edges: tuple[float, ...] = (875, 920, 960, 980, 990, 1000, 1020),
) -> list[PressureBin]:
    """Build contiguous bins from a list of ``edges`` (paper default)."""
    bins: list[PressureBin] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bins.append(PressureBin(low=lo, high=hi, label=f"{int(lo)}–{int(hi)}"))
    return bins


def effective_dimensionality(X: np.ndarray, *, eps: float = 1e-12) -> float:
    r"""Participation ratio :math:`d_{\text{eff}} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2`.

    This is the *effective number of latent directions* used by the
    representation inside the bin. Higher = more spread, lower = the
    features collapse along fewer directions.
    """
    if X.shape[0] < 2:
        return float("nan")
    X = X - X.mean(axis=0, keepdims=True)
    eigvals = np.linalg.eigvalsh(np.cov(X.T))
    eigvals = np.clip(eigvals, 0, None)
    num = float(eigvals.sum()) ** 2
    den = float(np.square(eigvals).sum()) + eps
    return num / den


def pca_per_bin(
    features: np.ndarray,
    pressure: np.ndarray,
    bins: list[PressureBin],
    *,
    n_components: int = 1,
    standardize: bool = True,
) -> dict[str, np.ndarray]:
    """Run PCA inside each bin. Returns ``{bin_label: scores (n_i, k)}``.

    ``features`` must be ``(N, D)``, ``pressure`` must be ``(N,)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out: dict[str, np.ndarray] = {}
    for b in bins:
        mask = b.mask(pressure)
        if mask.sum() < max(n_components + 1, 2):
            out[b.label] = np.empty((0, n_components))
            continue
        X = features[mask]
        if standardize:
            X = StandardScaler().fit_transform(X)
        out[b.label] = PCA(n_components=n_components).fit_transform(X)
    return out


def feature_spread(
    features: np.ndarray,
    pressure: np.ndarray,
    bins: list[PressureBin],
    *,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean pairwise Euclidean distance inside each bin (Fig. 4c).

    For bins with $n_i (n_i-1)/2 > $``max_pairs`` we sample uniformly to
    keep the computation cheap. Returns ``{bin_label: mean_distance}``.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for b in bins:
        mask = b.mask(pressure)
        n = int(mask.sum())
        if n < 2:
            out[b.label] = float("nan")
            continue
        X = features[mask]
        n_pairs = n * (n - 1) // 2
        if n_pairs > max_pairs:
            ii = rng.integers(0, n, size=max_pairs)
            jj = rng.integers(0, n, size=max_pairs)
            ok = ii != jj
            ii, jj = ii[ok], jj[ok]
        else:
            ii, jj = np.triu_indices(n, k=1)
        diffs = X[ii] - X[jj]
        out[b.label] = float(np.linalg.norm(diffs, axis=1).mean())
    return out
