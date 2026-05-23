"""Shared pytest fixtures.

The headline fixture is :func:`synthetic_feature_dataset` which writes a
tiny HuggingFace Arrow dataset compatible with
:func:`probing.core.data.load_features`. Tests that exercise the probing
pipeline use it as a stand-in for the multi-GB feature dataset produced
by ``dataset/09_extract_features.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict


def _make_split(n_cyclones: int, *, seed: int, feature_dim: int = 32,
                frames_per_cyclone: int = 16) -> Dataset:
    """Build a synthetic split with ``n_cyclones`` trajectories."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cyclones):
        T = frames_per_cyclone
        # Pressure walks downward then back up (a synthetic intensification curve).
        base = rng.uniform(940, 1010)
        pressure = base + np.cumsum(rng.normal(0, 2, T))
        wind = -0.7 * (pressure - 1000) + rng.normal(0, 5, T)
        cls = rng.normal(0, 1, (T, feature_dim)) + 0.02 * pressure[:, None]
        spm = rng.normal(0, 1, (T, feature_dim)) + 0.01 * pressure[:, None]
        center = np.stack([rng.uniform(-30, 30, T), rng.uniform(-180, 180, T)], axis=1)
        rows.append({
            "cyclone_id": f"hurdat_{seed}_{i:04d}",
            "features": {"cls": cls.tolist(), "spatial_mean": spm.tolist()},
            "pressure": pressure.tolist(),
            "wind": wind.tolist(),
            "center": center.tolist(),
            "frame_valid_mask": [True] * T,
            "timestamps": [f"1980-01-01T{h:02d}:00:00" for h in range(T)],
            "location": "NA",
        })
    return Dataset.from_list(rows)


@pytest.fixture(scope="session")
def synthetic_feature_dataset(tmp_path_factory) -> Path:
    """Materialise a tiny feature dataset on disk and return its path."""
    out = tmp_path_factory.mktemp("features_synth")
    dataset = DatasetDict({
        "train":      _make_split(n_cyclones=20, seed=1),
        "validation": _make_split(n_cyclones=5,  seed=2),
        "test":       _make_split(n_cyclones=5,  seed=3),
    })
    dataset.save_to_disk(str(out))
    return out
