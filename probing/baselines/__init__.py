"""Reference baselines used in the paper to anchor the probe values.

These are *not* VFM probes — they operate directly on the HF dataset
without any frozen-VFM features. They are kept here so the probing
protocol is self-contained.

* :mod:`probing.baselines.dvorak` — Dvorak (1975) intensity estimate
  from infrared cloud-top patterns (heuristic, no learning).
* :mod:`probing.baselines.climatology` — per-basin and per-month mean
  predictors (bounds the contribution of trivial covariates).
"""
