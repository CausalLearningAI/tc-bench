"""Probe families used to operationalize the structural-alignment probes.

The protocol from the paper (§4) uses **linear** probes (ridge with CV
on $\\alpha$) as the headline diagnostic. The other probe families
provide:

* ``LassoProbe`` — sanity check on coefficient sparsity (App. E.2).
* ``MLPProbe``, ``TransformerProbe`` — nonlinear-capacity sanity checks
  (App. E.2): they rule out a "the probe is too weak" explanation for
  the regime-dependent degradation reported in Fig. 2. ``MLPProbe``
  matches the paper's setup (sklearn ``MLPRegressor`` with a single
  2048-unit hidden layer, ReLU, Adam, ``max_iter=100``).
* ``DvorakProbe`` — the standard meteorological baseline (Dvorak, 1975).
* ``ClimatologyProbe`` — per-basin / per-month mean baseline that
  bounds the contribution of trivial covariates (Fig. 1c).

Every probe implements the same minimal interface so that
:mod:`probing.fit` and :mod:`probing.diagnose` are probe-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import joblib
import numpy as np


__all__ = [
    "Probe",
    "RidgeProbe",
    "LassoProbe",
    "MLPProbe",
    "TransformerProbe",
    "build_probe",
]


class Probe(Protocol):
    """Minimal interface every probe family implements.

    The protocol is intentionally narrow: linear and non-linear probes
    differ enormously in implementation but converge on this surface.
    """

    name: ClassVar[str]
    fitted_: bool

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Probe": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "Probe": ...

    @property
    def coef_(self) -> np.ndarray:
        """Linear coefficients, if defined. Used by Q_dyn.

        Non-linear probes raise :class:`AttributeError`.
        """
        ...


# --------------------------------------------------------------------- #
# Linear probes (ridge / lasso) wrap sklearn with the paper's protocol  #
# of cross-validated $\alpha$ selection on the train fold.              #
# --------------------------------------------------------------------- #


_DEFAULT_ALPHAS: tuple[float, ...] = (
    1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6,
)


@dataclass
class RidgeProbe:
    """Ridge regression with cross-validated $\\alpha$ on the train fold."""

    name: ClassVar[str] = "ridge"
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS
    cv_folds: int = 5
    selected_alpha_: float | None = None
    model_: Any = None
    fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        from sklearn.linear_model import Ridge, RidgeCV

        cv = RidgeCV(alphas=list(self.alphas), cv=self.cv_folds,
                     scoring="neg_mean_squared_error")
        cv.fit(X, y)
        self.selected_alpha_ = float(cv.alpha_)
        self.model_ = Ridge(alpha=self.selected_alpha_).fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "RidgeProbe.fit must be called first"
        return self.model_.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model_.coef_

    @property
    def intercept_(self) -> float:
        return float(self.model_.intercept_)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model_, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        from sklearn.linear_model import Ridge
        model = joblib.load(Path(path))
        assert isinstance(model, Ridge)
        return cls(alphas=(model.alpha,), selected_alpha_=float(model.alpha),
                   model_=model, fitted_=True)


@dataclass
class LassoProbe:
    """L1-regularized linear probe — App. E.2 sparsity sanity check."""

    name: ClassVar[str] = "lasso"
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS
    cv_folds: int = 5
    max_iter: int = 10000
    seed: int = 0
    selected_alpha_: float | None = None
    model_: Any = None
    fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoProbe":
        from sklearn.linear_model import Lasso, LassoCV

        cv = LassoCV(alphas=list(self.alphas), cv=self.cv_folds,
                     random_state=self.seed, max_iter=self.max_iter)
        cv.fit(X, y)
        self.selected_alpha_ = float(cv.alpha_)
        self.model_ = Lasso(alpha=self.selected_alpha_,
                            max_iter=self.max_iter, random_state=self.seed).fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "LassoProbe.fit must be called first"
        return self.model_.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model_.coef_

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model_, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "LassoProbe":
        from sklearn.linear_model import Lasso
        model = joblib.load(Path(path))
        assert isinstance(model, Lasso)
        return cls(alphas=(model.alpha,), selected_alpha_=float(model.alpha),
                   model_=model, fitted_=True)


# --------------------------------------------------------------------- #
# Non-linear MLP probe — App. E.2 Table 4 capacity sanity check.        #
# sklearn ``MLPRegressor`` with a single 2048-unit hidden layer, ReLU,  #
# Adam, ``max_iter=100``. Matches the paper protocol verbatim.          #
# --------------------------------------------------------------------- #


@dataclass
class MLPProbe:
    """sklearn ``MLPRegressor`` probe — paper App. E.2 default."""

    name: ClassVar[str] = "mlp"
    hidden_layer_sizes: tuple[int, ...] = (2048,)
    max_iter: int = 100
    seed: int = 0
    model_: Any = None
    selected_alpha_: float | None = None  # unused — kept for interface parity
    fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbe":
        from sklearn.neural_network import MLPRegressor

        self.model_ = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            random_state=self.seed,
            max_iter=self.max_iter,
        ).fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "MLPProbe.fit must be called first"
        return self.model_.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        raise AttributeError("MLPProbe is non-linear; coef_ is undefined")

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model_, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "MLPProbe":
        from sklearn.neural_network import MLPRegressor
        model = joblib.load(Path(path))
        assert isinstance(model, MLPRegressor)
        probe = cls(hidden_layer_sizes=tuple(model.hidden_layer_sizes),
                    max_iter=int(model.max_iter))
        probe.model_ = model
        probe.fitted_ = True
        return probe


# --------------------------------------------------------------------- #
# Transformer probe — App. E.2 capacity sanity check.                   #
# Lightweight 2-layer encoder (hidden=128, 4 heads). The flat CLS / mean
# feature is split into a small token sequence so self-attention has
# something to mix; pooled output is projected to the scalar target.
# --------------------------------------------------------------------- #


@dataclass
class TransformerProbe:
    """Two-layer Transformer encoder probe (App. E.2, Table 4)."""

    name: ClassVar[str] = "transformer"
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_tokens: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "cuda"
    model_: Any = None
    feature_dim_: int | None = None
    selected_alpha_: float | None = None  # unused — kept for interface parity
    fitted_: bool = False

    def _build(self, input_dim: int):
        import torch
        from torch import nn

        hidden, n_tok = self.hidden_dim, self.num_tokens

        class _Module(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.n_tok = n_tok
                self_inner.hidden = hidden
                self_inner.token_proj = nn.Linear(input_dim, n_tok * hidden)
                self_inner.pos_emb = nn.Parameter(torch.zeros(1, n_tok, hidden))
                nn.init.trunc_normal_(self_inner.pos_emb, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=self.num_heads,
                    dim_feedforward=4 * hidden,
                    dropout=self.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self_inner.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
                self_inner.head = nn.Linear(hidden, 1)

            def forward(self_inner, x):
                h = self_inner.token_proj(x).view(x.shape[0], self_inner.n_tok, self_inner.hidden)
                h = h + self_inner.pos_emb
                h = self_inner.encoder(h)
                return self_inner.head(h.mean(dim=1))

        return _Module()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerProbe":
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        self.feature_dim_ = int(X.shape[1])
        model = self._build(self.feature_dim_).to(self.device)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(X.shape[0])
        n_val = max(1, int(self.val_fraction * X.shape[0]))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        x_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        x_val   = torch.tensor(X[val_idx],   dtype=torch.float32).to(self.device)
        y_val   = torch.tensor(y[val_idx],   dtype=torch.float32).unsqueeze(1).to(self.device)
        loader = DataLoader(TensorDataset(x_train, y_train),
                            batch_size=self.batch_size, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        best_val, best_state, since_best = float("inf"), None, 0
        for _ in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(x_val), y_val).item()
            if val_loss + 1e-6 < best_val:
                best_val, since_best = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        assert self.fitted_, "TransformerProbe.fit must be called first"
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            preds = self.model_(torch.tensor(X, dtype=torch.float32).to(device))
        return preds.squeeze(1).cpu().numpy()

    @property
    def coef_(self) -> np.ndarray:
        raise AttributeError("TransformerProbe is non-linear; coef_ is undefined")

    def save(self, path: str | Path) -> None:
        import torch
        state = {
            "hidden_dim":  self.hidden_dim,
            "num_layers":  self.num_layers,
            "num_heads":   self.num_heads,
            "num_tokens":  self.num_tokens,
            "dropout":     self.dropout,
            "feature_dim": self.feature_dim_,
            "state_dict":  self.model_.state_dict(),
        }
        torch.save(state, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "TransformerProbe":
        import torch
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            hidden_dim=int(state["hidden_dim"]),
            num_layers=int(state["num_layers"]),
            num_heads=int(state["num_heads"]),
            num_tokens=int(state["num_tokens"]),
            dropout=float(state["dropout"]),
        )
        probe.feature_dim_ = int(state["feature_dim"])
        probe.model_ = probe._build(probe.feature_dim_)
        probe.model_.load_state_dict(state["state_dict"])
        probe.fitted_ = True
        return probe


# --------------------------------------------------------------------- #
# Probe factory                                                         #
# --------------------------------------------------------------------- #


_REGISTRY: dict[str, type] = {
    RidgeProbe.name:       RidgeProbe,
    LassoProbe.name:       LassoProbe,
    MLPProbe.name:         MLPProbe,
    TransformerProbe.name: TransformerProbe,
}


def build_probe(name: str, **kwargs: Any) -> Probe:
    """Instantiate a probe by name. ``kwargs`` flow through to the dataclass."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown probe {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
