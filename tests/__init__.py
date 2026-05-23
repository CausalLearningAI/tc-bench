"""Smoke tests for the TC-Bench camera-ready release.

These tests do not require GPUs, the real dataset, or any extracted VFM
features. They cover:

* ``test_probing_core.py`` — primitives (load/balance/split, probes,
  metrics, geometry).
* ``test_probing_pipeline.py`` — end-to-end fit → diagnose → aggregate
  on a synthetic feature dataset written to a tmp directory.
* ``test_configs.py`` — Hydra composition for every top-level entry
  point + every experiment config.
* ``test_dataset_stages.py`` — each dataset stage script has a
  well-formed argparse interface.

Run with ``pytest``.
"""
