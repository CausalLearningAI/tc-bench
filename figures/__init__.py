"""Paper figures.

Every script in this package is a self-contained, argparse-driven entry
point that consumes artefacts from :mod:`probing` (predictions CSVs,
geometry tables, baselines) and writes a PDF + PNG into ``figs/``.

The shared :mod:`figures._style` module applies the ICML camera-ready
:mod:`matplotlib.rcParams` so every figure embeds Type-1-compatible
fonts and uses the same model-family colour palette.
"""
