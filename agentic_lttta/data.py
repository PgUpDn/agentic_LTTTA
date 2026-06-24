"""Foil real-data loading from the downloaded Hugging Face Arrow shards.

Each Arrow shard under ``dataset_RealPDE/foil/hf_dataset/real/`` holds exactly
one trajectory row with columns:

* ``sim_id``  -> e.g. ``"10000_0.0.h5"`` == ``{Reynolds}_{AoA}.h5``
* ``u``,``v`` -> raw little-endian ``float32`` blobs of shape (T, H, W)
* ``shape_t``, ``shape_h``, ``shape_w`` -> the (T, H, W) dims

Real PIV data has no pressure channel, so we stack ``[u, v, 0]`` -> (T, H, W, 3)
to stay compatible with the RealPDEBench 3-channel model interface. Spatial
sub-sampling (default factor 4 -> 32x64) keeps the CPU prototype fast.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .paths import FOIL_REAL_DIR

_SIM_ID_RE = re.compile(r"(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)")


def parse_sim_id(sim_id: str) -> Tuple[float, float]:
    """``"10000_0.0.h5"`` -> ``(Reynolds=10000.0, AoA=0.0)``."""
    m = _SIM_ID_RE.search(sim_id)
    if not m:
        return (float("nan"), float("nan"))
    return float(m.group(1)), float(m.group(2))


@dataclass
class FoilTrajectory:
    sim_id: str
    reynolds: float
    aoa: float
    data: np.ndarray  # float32 [T, H, W, 3] with p-channel == 0

    @property
    def regime_key(self) -> str:
        return f"Re{int(round(self.reynolds))}_AoA{self.aoa:g}"

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def hw(self) -> Tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]


def _decode_field(blob: bytes, t: int, h: int, w: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != t * h * w:
        raise ValueError(f"blob size {arr.size} != T*H*W={t*h*w}")
    return arr.reshape(t, h, w)


def load_trajectory_from_arrow(path: str, sub_s: int = 4) -> FoilTrajectory:
    from datasets import Dataset  # local import (heavy)

    ds = Dataset.from_file(path)
    row = ds[0]
    t, h, w = int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"])
    u = _decode_field(row["u"], t, h, w)[:, ::sub_s, ::sub_s]
    v = _decode_field(row["v"], t, h, w)[:, ::sub_s, ::sub_s]
    p = np.zeros_like(u)
    data = np.ascontiguousarray(np.stack([u, v, p], axis=-1), dtype=np.float32)
    re_, aoa = parse_sim_id(row["sim_id"])
    return FoilTrajectory(sim_id=row["sim_id"], reynolds=re_, aoa=aoa, data=data)


def list_real_shards(real_dir: str = FOIL_REAL_DIR) -> List[str]:
    return sorted(glob.glob(os.path.join(real_dir, "*.arrow")))


def load_foil_trajectories(
    real_dir: str = FOIL_REAL_DIR,
    max_traj: Optional[int] = None,
    sub_s: int = 4,
) -> List[FoilTrajectory]:
    """Load up to ``max_traj`` downloaded real trajectories (subsampled)."""
    shards = list_real_shards(real_dir)
    if not shards:
        raise FileNotFoundError(
            f"No foil real Arrow shards found in {real_dir}. "
            "Run: python -m agentic_lttta.scripts.download_data"
        )
    if max_traj is not None:
        shards = shards[:max_traj]
    return [load_trajectory_from_arrow(p, sub_s=sub_s) for p in shards]


class TrainWindowSampler:
    """Samples ``(input, target)`` windows of length ``in_step`` for surrogate
    training, drawn from the *training* portion of one or more trajectories.

    Returns float32 arrays shaped ``[B, in_step, H, W, C]``.
    """

    def __init__(
        self,
        trajectories: List[FoilTrajectory],
        in_step: int,
        train_stop_frac: float = 0.6,
        seed: int = 0,
    ):
        self.in_step = in_step
        self.rng = np.random.default_rng(seed)
        self.pool: List[Tuple[np.ndarray, int]] = []
        for tr in trajectories:
            stop = int(tr.n_frames * train_stop_frac)
            # need 2 * in_step contiguous frames (input + target)
            last_start = stop - 2 * in_step
            if last_start <= 0:
                continue
            self.pool.append((tr.data[:stop], last_start))
        if not self.pool:
            raise ValueError("No trajectory long enough for the requested in_step.")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _ in range(batch_size):
            data, last_start = self.pool[self.rng.integers(len(self.pool))]
            s = int(self.rng.integers(0, last_start + 1))
            xs.append(data[s : s + self.in_step])
            ys.append(data[s + self.in_step : s + 2 * self.in_step])
        return (
            np.ascontiguousarray(np.stack(xs), dtype=np.float32),
            np.ascontiguousarray(np.stack(ys), dtype=np.float32),
        )
