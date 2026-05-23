"""Structural tests for every numbered dataset stage.

We don't actually run a stage (each pulls multi-GB upstream data) and
we don't try to import them either — some stages have heavy upstream
dependencies (``wget``, ``xarray``, …) that are not part of the test
extras. We only verify:

1. Every stage's Python script exists at the expected numbered path.
2. The corresponding SLURM wrapper exists, is executable, and points
   at the right Python file via the stage script name.
3. The SLURM wrapper sources ``_env.sh`` (centralised env contract).
4. The orchestrator's ``--help`` runs successfully.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

STAGES = [
    ("01_download_ibtracs.py",   "01_download_ibtracs.sh"),
    ("02_download_gridsat.py",   "02_download_gridsat.sh"),
    ("03_preprocess.py",         "03_preprocess.sh"),
    ("04_crop_windows.py",       "04_crop_windows.sh"),
    ("05_consolidate_nc.py",     "05_consolidate_nc.sh"),
    ("06_build_hf.py",           "06_build_hf.sh"),
    ("07_normalize_stats.py",    "07_normalize_stats.sh"),
    ("08_ood_basin_split.py",    "08_ood_basin_split.sh"),
    ("09_extract_features.py",   "09_extract_features.sh"),
]


@pytest.mark.parametrize("py,sh", STAGES)
def test_stage_wiring(py: str, sh: str) -> None:
    py_path = DATASET_DIR / py
    sh_path = DATASET_DIR / "slurm" / sh
    assert py_path.exists(),  f"missing stage script: {py_path}"
    assert sh_path.exists(),  f"missing SLURM wrapper:  {sh_path}"
    assert os.access(sh_path, os.X_OK), f"SLURM wrapper not executable: {sh_path}"
    contents = sh_path.read_text()
    assert "source" in contents and "_env.sh" in contents, (
        f"SLURM wrapper {sh} must source _env.sh for shared env contract"
    )
    assert py in contents, (
        f"SLURM wrapper {sh} should reference its python script {py}"
    )


def test_orchestrator_help_runs() -> None:
    res = subprocess.run(
        ["bash", str(DATASET_DIR / "run_all.sh"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
