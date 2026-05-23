"""Regression metrics shared by every probe.

Returning a single :class:`dict[str, float]` lets us serialise consistent
JSON across probe types and across the static / dynamic / consistency
diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = ["regression_metrics", "regime_metrics", "normalized_absolute_error"]


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sigma: float | None = None,
) -> dict[str, float]:
    """Return a flat dictionary of scalar regression metrics.

    Keys: ``rmse``, ``mae``, ``r2``, ``pearson``, ``spearman``,
    ``sigma``, ``normalized_rmse``, ``normalized_mae``, ``n``.

    The paper (§4, Eq. 4.1) reports static error normalized by the
    target's standard deviation so the value 1 corresponds to the naive
    mean estimator. To compare regimes against the *same* baseline pass
    a precomputed ``sigma`` (computed once from the training target);
    otherwise the local ``std(y_true)`` is used.

    NaN-safe via a finite-value mask: rows where either target or
    prediction is non-finite are excluded.
    """
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[finite]
    y_p = y_pred[finite]
    if y_t.size < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"),
                "sigma": float(sigma) if sigma is not None else float("nan"),
                "normalized_rmse": float("nan"), "normalized_mae": float("nan"),
                "n": int(y_t.size)}
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    pearson_r, _ = pearsonr(y_t, y_p)
    spearman_r, _ = spearmanr(y_t, y_p)
    sigma_used = float(np.std(y_t)) if sigma is None else float(sigma)
    denom = max(sigma_used, 1e-12)
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "pearson": float(pearson_r), "spearman": float(spearman_r),
        "sigma": sigma_used,
        "normalized_rmse": rmse / denom,
        "normalized_mae": mae / denom,
        "n": int(y_t.size),
    }


def regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold_hpa: float = 980.0,
    intense_below: bool = True,
    sigma: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compute :func:`regression_metrics` for each pressure regime.

    Returns a dict ``{"moderate": {...}, "intense": {...}}``. The
    moderate / intense partition uses ``threshold_hpa`` on ``y_true``
    when ``y_true`` is pressure (the default in the paper); pass an
    explicit precomputed mask via the *frame*-level diagnostics module
    if you need a regime carved on a different variable.

    ``sigma`` is forwarded to :func:`regression_metrics` for *both*
    sub-regimes so that the normalization denominator is global (the
    paper's convention) — using each regime's local std would erase the
    very degradation Fig. 2 reports.
    """
    if sigma is None:
        finite = np.isfinite(y_true)
        sigma = float(np.std(y_true[finite])) if finite.any() else None
    intense_mask = (y_true < threshold_hpa) if intense_below else (y_true > threshold_hpa)
    return {
        "intense":  regression_metrics(y_true[intense_mask],  y_pred[intense_mask],
                                       sigma=sigma),
        "moderate": regression_metrics(y_true[~intense_mask], y_pred[~intense_mask],
                                       sigma=sigma),
    }


def normalized_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    r"""Per-sample normalized absolute error :math:`\xi_{\text{stat}}` (Eq. 4.1).

    ``sigma`` defaults to ``std(y_true)`` so that the value 1 corresponds
    to the performance of the naive mean estimator. Pass a precomputed
    ``sigma`` to reuse the same denominator across regimes.
    """
    if sigma is None:
        sigma = float(np.std(y_true))
    return np.abs(y_true - y_pred) / max(sigma, 1e-12)
