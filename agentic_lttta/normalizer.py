"""Per-channel z-score normaliser (self-contained).

We use our own normaliser instead of RealPDEBench's ``GaussianNormalizer`` so the
surrogate and its statistics stay self-consistent without depending on the
benchmark's cached ``mean_std.pt`` (which is tied to the numerical split used to
train the released checkpoints).

Statistics are per-channel (u, v, p). Real foil data has no pressure, so its
``p`` channel is all-zeros; the std guard maps a zero std to 1.0 so the channel
passes through unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class ZNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cpu"):
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.mean = mean.to(device).float()
        self.std = std.to(device).float()
        self.device = device

    @classmethod
    def fit(cls, frames: np.ndarray, device: str = "cpu") -> "ZNormalizer":
        """Fit on frames shaped ``[N, H, W, C]`` (statistics taken over N,H,W)."""
        arr = np.asarray(frames, dtype=np.float32)
        c = arr.shape[-1]
        flat = arr.reshape(-1, c)
        mean = torch.from_numpy(flat.mean(axis=0))
        std = torch.from_numpy(flat.std(axis=0))
        return cls(mean, std, device=device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        return (x.to(self.device) - self.mean[..., :c]) / self.std[..., :c]

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        return x.to(self.device) * self.std[..., :c] + self.mean[..., :c]

    # --- (de)serialisation -------------------------------------------------
    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> "ZNormalizer":
        return cls(state["mean"], state["std"], device=device)
