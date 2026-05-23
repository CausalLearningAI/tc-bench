"""Tensor-level regression metrics used by :mod:`src.models.base_regressor`.

These are the train/val metrics logged by the pixel-supervision
ablation (RMSE, MAE, per-target normalized error, etc.). They are kept
torch-native so they can be computed inside the training loop without
detaching to numpy.
"""
from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics import Metric

def _flatten(x: Tensor) -> Tensor:
    # Accept (B,C,H,W), (B,H,W), (C,H,W), (H,W)
    return x.reshape(-1)

class RMSE(Metric):
    """Streaming, exact root mean squared error over the whole epoch."""
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("squared_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",       default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        err = preds - target
        self.squared_sum += (_flatten(err) ** 2).sum().to(self.squared_sum.dtype)
        self.count       += torch.tensor(err.numel(), device=self.count.device, dtype=self.count.dtype)

    def compute(self) -> Tensor:
        return torch.sqrt(self.squared_sum / self.count.clamp_min(1))

class AverageMAE(Metric):
    """Streaming, exact mean absolute error over the whole epoch."""
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("abs_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        err = torch.abs(preds - target)
        self.abs_sum += _flatten(err).sum().to(self.abs_sum.dtype)
        self.count   += torch.tensor(err.numel(), device=self.count.device, dtype=self.count.dtype)

    def compute(self) -> Tensor:
        return self.abs_sum / self.count.clamp_min(1)

class P95AbsErrorExact(Metric):
    """
    Exact 95th percentile of |pred - target|.
    Stores all errors on CPU RAM -> simple & exact but can be memory heavy.
    """
    full_state_update: bool = True  # gather all shards in DDP for exact percentile

    def __init__(self, keep_on_cpu: bool = True):
        super().__init__()
        self.keep_on_cpu = keep_on_cpu
        self.add_state("errors", default=[], dist_reduce_fx="cat")  # torchmetrics will cat tensors

    def update(self, preds: Tensor, target: Tensor):
        e = torch.abs(preds - target).detach()
        e = _flatten(e)
        if self.keep_on_cpu:
            e = e.to("cpu", non_blocking=True)
        self.errors.append(e)

    def compute(self) -> Tensor:
        if len(self.errors) == 0:
            return torch.tensor(0.0)
        errs = torch.cat(self.errors, dim=0)
        return torch.quantile(errs, q=0.95)

class P95AbsErrorHistogram(Metric):
    """
    Approximate 95th percentile of |pred - target| with a fixed histogram (memory-light).
    Choose 'vmax' generously or set adapt_minmax=True to expand on the fly.
    """
    full_state_update: bool = False

    def __init__(self, bins: int = 2048, vmax: float = 200.0, adapt_minmax: bool = True):
        super().__init__()
        self.bins = bins
        self.add_state("hist", default=torch.zeros(bins, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("minv", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="min")
        self.add_state("maxv", default=torch.tensor(vmax, dtype=torch.float64), dist_reduce_fx="max")
        self.adapt = adapt_minmax

    def update(self, preds: Tensor, target: Tensor):
        e = torch.abs(preds - target).detach().to(torch.float32)
        e = _flatten(e)

        # Optionally adapt histogram range to data (one-sided expansion)
        if self.adapt:
            self.maxv = torch.maximum(self.maxv, e.max().to(self.maxv.dtype))
            self.minv = torch.minimum(self.minv, e.min().to(self.minv.dtype))

        # Bin
        minv = float(self.minv)
        maxv = float(self.maxv) + 1e-12
        idx = torch.clamp(((e - minv) / (maxv - minv) * self.bins).floor().long(), 0, self.bins - 1)
        # Count with scatter_add
        counts = torch.bincount(idx, minlength=self.bins).to(self.hist.dtype)
        self.hist += counts

    def compute(self) -> Tensor:
        total = self.hist.sum().clamp_min(1.0)
        cdf = torch.cumsum(self.hist, dim=0) / total
        k = torch.searchsorted(cdf, torch.tensor(0.95).type_as(cdf))
        # Map bin index back to value
        width = (self.maxv - self.minv) / self.bins
        return (self.minv + (k.to(self.minv.dtype) + 0.5) * width).to(torch.float32)

class TailMAE(Metric):
    """
    MAE on the top-5% of TARGET values. Memory- and DDP-safe:
    - keeps a fixed-size GPU reservoir to avoid gigantic quantiles
    - no CPU collectives (works with NCCL)
    - exact on the reservoir (approx overall if dataset >> reservoir)
    """
    full_state_update: bool = False  # we manage state manually

    def __init__(self, tail_p: float = 0.95, buffer_cap: int = 2_000_000, **kwargs):
        super().__init__(**kwargs)
        self.tail_p = float(tail_p)
        self.buffer_cap = int(buffer_cap)

        # Flat reservoirs on device; start empty
        self.add_state("targets_buf", default=torch.empty(0), dist_reduce_fx=None)
        self.add_state("preds_buf",   default=torch.empty(0), dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor):
        # flatten on the same device as preds/metric (GPU under NCCL)
        device = preds.device
        y = target.detach().reshape(-1).to(device)
        x = preds.detach().reshape(-1).to(device)

        # append then reservoir-trim if over capacity
        self.targets_buf = torch.cat([self.targets_buf.to(device), y], dim=0)
        self.preds_buf   = torch.cat([self.preds_buf.to(device),   x], dim=0)

        n = self.targets_buf.numel()
        if n > self.buffer_cap:
            # uniform subsample to buffer_cap (in-GPU to avoid CPU collectives)
            idx = torch.randperm(n, device=device)[: self.buffer_cap]
            self.targets_buf = self.targets_buf.index_select(0, idx)
            self.preds_buf   = self.preds_buf.index_select(0, idx)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.targets_buf.numel() == 0:
            return torch.tensor(0.0, device=self.targets_buf.device, dtype=torch.float32)

        y = self.targets_buf
        x = self.preds_buf

        # robust q95 on manageable buffer size
        q95 = torch.quantile(y, self.tail_p)
        mask = y >= q95
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y.device, dtype=torch.float32)

        return torch.mean(torch.abs(x[mask] - y[mask])).to(torch.float32)

