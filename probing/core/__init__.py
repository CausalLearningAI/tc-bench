"""Protocol-agnostic primitives shared by every probing entry point.

The split between this package and the top-level Hydra entries
(``probing.fit``, ``probing.diagnose``, ``probing.geometry``,
``probing.aggregate``) is deliberate: anything in :mod:`probing.core`
must be importable and usable without Hydra, so it can be reused from
notebooks, ad-hoc scripts, or future probes.
"""

from probing.core import data, diagnostics, geometry, metrics, probes

__all__ = ["data", "diagnostics", "geometry", "metrics", "probes"]
