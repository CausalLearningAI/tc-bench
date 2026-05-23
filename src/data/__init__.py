"""Lightning datamodules for the pixel-supervision ablation.

Only one canonical datamodule is kept: :class:`CycloneDataModule` reads
the HuggingFace Arrow dataset produced by ``dataset/06_build_hf.py`` and
exposes per-frame batches (one IR image + scalar pressure / wind).

Earlier branches of this code carried three near-duplicate datamodules
(``cyclone``, ``cat5``, ``fast``); they have been merged into this one
configurable class.
"""

from .cyclone_datamodule import CycloneDataModule, CycloneFrameDataset

__all__ = ["CycloneDataModule", "CycloneFrameDataset"]
