"""TC-Bench probing protocol.

This package implements the structural-alignment probing framework from
the paper (§4): three diagnostic probes — ``Q_stat``, ``Q_dyn``, ``Q_con``
— applied to *frozen* VFM features extracted under
:mod:`dataset.09_extract_features`.

Sub-packages and modules:

* :mod:`probing.core` — the protocol-agnostic library
    * :mod:`probing.core.data`        — feature loading, trajectory splits, regime balancing
    * :mod:`probing.core.probes`      — probe families (Ridge, Lasso, MLP, Dvorak, climatology)
    * :mod:`probing.core.metrics`     — regression metrics
    * :mod:`probing.core.diagnostics` — Q_stat / Q_dyn / Q_con DataFrame construction
    * :mod:`probing.core.geometry`    — §4.2 intrinsic-geometry diagnostics

* ``probing.fit``       — Hydra entry, trains and persists a probe (.pkl)
* ``probing.diagnose``  — Hydra entry, turns probes + features into a predictions CSV
* ``probing.geometry``  — Hydra entry, runs §4.2 PCA / effective-dim / feature-spread
* ``probing.aggregate`` — Hydra entry, collects per-seed / per-model artefacts
"""
