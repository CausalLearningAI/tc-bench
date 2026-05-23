"""Models for the pixel-supervision ablation (App. E.1 of the paper).

These are deliberately trained end-to-end from raw IR satellite frames, in
contrast to the probing protocol (under ``probing/``) which operates on frozen
VFM features. Only two architectures are kept here:

* :class:`SimpleCNN` — a from-scratch convolutional baseline.
* :class:`ResNetRegressor` — ImageNet-style ResNet backbone with a regression head.

Both inherit from :class:`BaseIntensityRegressor` which defines the shared
optimizer / scheduler / loss / metric machinery.
"""

from .base_regressor import BaseIntensityRegressor
from .resnet_regressor import ResNetRegressor
from .simple_cnn import SimpleCNN

__all__ = [
    "BaseIntensityRegressor",
    "ResNetRegressor",
    "SimpleCNN",
]
