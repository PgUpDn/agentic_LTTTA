"""Centralised filesystem paths and RealPDEBench ``sys.path`` wiring.

We deliberately do **not** ``pip install`` the RealPDEBench package (it pulls a
large set of model dependencies). Instead we add the local checkout to
``sys.path`` and import only the two lightweight modules we reuse:

* ``realpdebench.model.fno`` -> the ``FNO3d`` surrogate
* ``realpdebench.utils.metrics`` -> ``kinetic_energy`` / ``mse_loss`` / ``eval_metrics``
"""

from __future__ import annotations

import os
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PKG_DIR)

REALPDEBENCH_DIR = os.path.join(WORKSPACE_ROOT, "RealPDEBench")
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "dataset_RealPDE")

RESULTS_DIR = os.path.join(PKG_DIR, "results")
CONFIG_DIR = os.path.join(PKG_DIR, "config")
CKPT_DIR = os.path.join(PKG_DIR, "checkpoints")

# Where downloaded foil real Arrow shards live.
FOIL_REAL_DIR = os.path.join(DATASET_ROOT, "foil", "hf_dataset", "real")


def ensure_realpdebench_on_path() -> str:
    """Add the local RealPDEBench checkout to ``sys.path`` (idempotent)."""
    if REALPDEBENCH_DIR not in sys.path:
        sys.path.insert(0, REALPDEBENCH_DIR)
    return REALPDEBENCH_DIR


def make_dirs() -> None:
    for d in (RESULTS_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)
