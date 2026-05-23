"""Shared matplotlib styling for every ICML camera-ready figure.

Call :func:`set_icml_style` at the top of any figure script. It applies
the journal-style :mod:`matplotlib.rcParams` (serif font, fontsize 10,
Type-1 compatible PDF embedding, etc.) so the entire ``figures/``
directory produces visually consistent PDFs without repeating the
boilerplate 8 times.

The colour palette is taken from `colormaps.lipari`, sampled at fixed
positions so model families map to the same hue across every figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


_ICML_RCPARAMS: dict[str, object] = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,     # TrueType embedding (Type-1 compatible)
    "ps.fonttype":       42,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "grid.linewidth":     0.5,
}


def set_icml_style() -> None:
    """Apply ICML rcParams to the global matplotlib config."""
    mpl.rcParams.update(_ICML_RCPARAMS)


# --------------------------------------------------------------------- #
# Family-stable colour palette                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class FamilyPalette:
    """Mapping from VFM family name to a stable hex/RGB triple."""

    dinov2:   tuple[float, float, float]
    dinov3:   tuple[float, float, float]
    clip:     tuple[float, float, float]
    siglip:   tuple[float, float, float]
    siglip2:  tuple[float, float, float]
    mae:      tuple[float, float, float]

    def as_dict(self) -> dict[str, tuple[float, float, float]]:
        return {
            "dinov2":  self.dinov2,
            "dinov3":  self.dinov3,
            "clip":    self.clip,
            "siglip":  self.siglip,
            "siglip2": self.siglip2,
            "mae":     self.mae,
        }


def lipari_family_palette() -> FamilyPalette:
    """Sample ``colormaps.lipari`` at 6 equally-spaced positions.

    Falls back to a manual palette if the ``colormaps`` package is not
    installed (CI environments without the optional dep).
    """
    try:
        import numpy as np
        import colormaps as cmaps  # type: ignore[import-not-found]
        c = cmaps.lipari(np.linspace(0.15, 0.85, 6))
        return FamilyPalette(
            dinov2=tuple(c[0]),  dinov3=tuple(c[1]),  clip=tuple(c[2]),
            siglip=tuple(c[3]), siglip2=tuple(c[4]), mae=tuple(c[5]),
        )
    except Exception:
        # ColorBrewer Dark2 fallback.
        return FamilyPalette(
            dinov2=(0.10, 0.62, 0.47), dinov3=(0.85, 0.37, 0.01),
            clip=(0.46, 0.44, 0.70),  siglip=(0.91, 0.16, 0.54),
            siglip2=(0.40, 0.65, 0.12), mae=(0.90, 0.67, 0.01),
        )


def intensity_palette() -> dict[str, tuple[float, float, float]]:
    """Two-colour palette for moderate vs intense regimes."""
    fam = lipari_family_palette()
    return {"Intense": fam.dinov3, "Moderate": fam.mae}


def figsize(
    width: float = 3.25,
    height: float = 2.5,
) -> tuple[float, float]:
    """Default ICML single-column size (3.25 in wide)."""
    return (width, height)


def savefig(
    fig: plt.Figure,
    path,
    *,
    formats: Iterable[str] = ("pdf", "png"),
) -> None:
    """Save ``fig`` in every requested format (path's extension is replaced)."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(p.with_suffix(f".{ext}"))
