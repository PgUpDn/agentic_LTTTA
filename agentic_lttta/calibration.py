"""Persistent affine calibration + uncertainty estimate.

A cheap, gradient-free correction applied to the surrogate's raw output:
``calibrated = gain * raw + bias`` (per channel). Updating also refreshes a
per-channel residual ``sigma`` used to build predictive intervals for the
Safe-Prediction-Score (SPS).
"""

from __future__ import annotations

import torch


class Calibration:
    def __init__(self, c: int = 2, device: str = "cpu", default_sigma: float = 0.05):
        self.c = c
        self.device = device
        self.gain = torch.ones(c, device=device)
        self.bias = torch.zeros(c, device=device)
        self.sigma = torch.full((c,), float(default_sigma), device=device)
        self.fitted = False

    def apply(self, pred: torch.Tensor) -> torch.Tensor:
        c = self.c
        out = pred.clone()
        shape = [1] * (pred.dim() - 1) + [c]
        out[..., :c] = pred[..., :c] * self.gain.view(*shape) + self.bias.view(*shape)
        return out

    def update(self, raw_pred: torch.Tensor, real: torch.Tensor, momentum: float = 0.5) -> None:
        c = self.c
        p = raw_pred[..., :c].reshape(-1, c)
        t = real[..., :c].reshape(-1, c)
        ps = p.std(dim=0).clamp_min(1e-6)
        ts = t.std(dim=0).clamp_min(1e-6)
        gain_new = ts / ps
        bias_new = t.mean(dim=0) - gain_new * p.mean(dim=0)
        self.gain = (1 - momentum) * self.gain + momentum * gain_new
        self.bias = (1 - momentum) * self.bias + momentum * bias_new
        cal = p * self.gain + self.bias
        sig_new = (t - cal).std(dim=0).clamp_min(1e-6)
        self.sigma = (1 - momentum) * self.sigma + momentum * sig_new
        self.fitted = True

    # --- (de)serialise -----------------------------------------------------
    def state(self) -> dict:
        return {
            "gain": self.gain.clone(),
            "bias": self.bias.clone(),
            "sigma": self.sigma.clone(),
            "fitted": self.fitted,
        }

    def load(self, st: dict) -> None:
        self.gain = st["gain"].to(self.device).clone()
        self.bias = st["bias"].to(self.device).clone()
        self.sigma = st["sigma"].to(self.device).clone()
        self.fitted = bool(st.get("fitted", True))
