#!/usr/bin/env python
"""Editable install for the TC-Bench code release.

Packages exposed:
  - src/        the pixel-supervision ablation (App. E.1)
  - dataset/    the construction pipeline (also exposed as a package
                so that probing/* can import dataset.<stage> if needed)
  - probing/    the structural-alignment probing protocol (§4)
"""

from setuptools import find_packages, setup

setup(
    name="tcbench",
    version="0.1.0",
    description="TC-Bench: probing scientific alignment in vision foundation models.",
    url="https://github.com/CausalLearningAI/tc-bench",
    install_requires=[
        "lightning",
        "hydra-core",
    ],
    packages=find_packages(include=["src*", "probing*", "dataset*"]),
    entry_points={
        "console_scripts": [
            # Pixel-supervision ablation.
            "tcbench-train      = src.train:main",
            "tcbench-eval       = src.eval:main",
            # Probing protocol.
            "tcbench-fit        = probing.fit:main",
            "tcbench-diagnose   = probing.diagnose:main",
            "tcbench-geometry   = probing.geometry:main",
            "tcbench-aggregate  = probing.aggregate:main",
        ]
    },
)
