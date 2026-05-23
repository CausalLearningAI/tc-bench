"""Pixel-supervision ablation for the TC-Bench paper (App. E.1).

Everything under :mod:`src` exists to support the supervised baseline
described in App. E.1: train a small architecture from random
initialization on raw IR satellite frames, *without* any pretraining or
frozen-VFM features. The result is used in the paper to establish that
intensity-relevant signal *is* statistically present in the imagery —
so the regime-dependent failure of frozen-VFM probes (§4) is a
representational issue, not a data issue.

This package is intentionally minimal:

* :mod:`src.train`     — Hydra entry, ``python -m src.train experiment=…``
* :mod:`src.eval`      — Hydra entry, ``python -m src.eval ckpt_path=…``
* :mod:`src.models`    — Lightning modules (SimpleCNN, ResNetRegressor)
* :mod:`src.data`      — the single canonical datamodule (``CycloneDataModule``)
* :mod:`src.losses`    — shared per-target regression metrics
* :mod:`src.utils`     — Hydra / logging / instantiation helpers

The probing protocol from the paper (§4) lives in a separate package,
:mod:`probing`, and uses an entirely separate Hydra root.
"""
