"""A tiny regime-keyed memory bank for the ``retrieve_memory`` action.

Stores the best-performing affine calibration seen for each flow regime
(keyed by ``Re``/``AoA`` parsed from ``sim_id``). When the controller revisits a
known regime it can retrieve the stored correction instead of re-deriving it.
"""

from __future__ import annotations

from typing import Dict, Optional

from .calibration import Calibration


class RegimeMemory:
    def __init__(self) -> None:
        self.bank: Dict[str, dict] = {}  # regime_key -> {"state": ..., "score": float}

    def has(self, key: str) -> bool:
        return key in self.bank

    def get(self, key: str) -> Optional[dict]:
        entry = self.bank.get(key)
        return entry["state"] if entry else None

    def update(self, key: str, calibration: Calibration, score: float) -> bool:
        """Store the calibration if it improves on what we have (lower score=err)."""
        cur = self.bank.get(key)
        if cur is None or score < cur["score"]:
            self.bank[key] = {"state": calibration.state(), "score": float(score)}
            return True
        return False

    def keys(self):
        return list(self.bank.keys())

    def summary(self) -> dict:
        return {k: round(v["score"], 4) for k, v in self.bank.items()}
