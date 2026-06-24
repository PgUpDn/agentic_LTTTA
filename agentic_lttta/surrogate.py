"""The forecasting surrogate: a small FNO-3D wrapper with test-time-adaptation hooks.

We reuse RealPDEBench's ``FNO3d`` (imported via ``sys.path``) and wrap it with:

* our self-contained :class:`~agentic_lttta.normalizer.ZNormalizer`
* a fast ``predict_block`` (normalise -> forward -> denormalise, no grad)
* ``adapt_parameters`` exposing a *small* subset of weights (BatchNorm affine
  and/or the output head) for cheap test-time gradient updates
* ``save`` / ``load`` / ``clone`` for the expert pool

The unit step maps ``in_step`` frames -> the next ``in_step`` frames, i.e.
input ``[B, L, H, W, C]`` -> output ``[B, L, H, W, C]``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch

from .normalizer import ZNormalizer
from .paths import ensure_realpdebench_on_path


@dataclass
class SurrogateConfig:
    in_step: int = 10
    height: int = 32
    width: int = 64
    channels: int = 3
    modes1: int = 4
    modes2: int = 8
    modes3: int = 8
    fno_width: int = 20
    n_layers: int = 3


def build_fno(cfg: SurrogateConfig, device: str = "cpu"):
    ensure_realpdebench_on_path()
    from realpdebench.model.fno import FNO3d

    shape_in = (cfg.in_step, cfg.height, cfg.width, cfg.channels)
    shape_out = (cfg.in_step, cfg.height, cfg.width, cfg.channels)
    model = FNO3d(
        modes1=cfg.modes1,
        modes2=cfg.modes2,
        modes3=cfg.modes3,
        n_layers=cfg.n_layers,
        width=cfg.fno_width,
        shape_in=shape_in,
        shape_out=shape_out,
    ).to(device)
    return model


class Surrogate:
    def __init__(
        self,
        model,
        normalizer: ZNormalizer,
        cfg: SurrogateConfig,
        device: str = "cpu",
        name: str = "base",
    ):
        self.model = model
        self.normalizer = normalizer
        self.cfg = cfg
        self.device = device
        self.name = name
        self.model.eval()

    # --- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict_block(self, window_phys: torch.Tensor) -> torch.Tensor:
        """``[B, L, H, W, C]`` physical -> next-block ``[B, L, H, W, C]`` physical."""
        self.model.eval()
        x = self.normalizer.preprocess(window_phys.to(self.device).float())
        y = self.model(x)
        return self.normalizer.postprocess(y)

    def forward_norm(self, window_norm: torch.Tensor) -> torch.Tensor:
        """Differentiable forward in *normalised* space (for adaptation)."""
        return self.model(window_norm.to(self.device).float())

    # --- adaptation hooks --------------------------------------------------
    def adapt_parameters(self, scope: str = "bn") -> List[torch.nn.Parameter]:
        """Return the small parameter subset to adapt at test time.

        ``scope`` in {``"bn"``, ``"head"``, ``"bn+head"``}.
        """
        params: List[torch.nn.Parameter] = []
        if "bn" in scope:
            for bn in self.model.bns:
                params += [p for p in bn.parameters() if p.requires_grad]
        if "head" in scope:
            params += [p for p in self.model.fc2.parameters() if p.requires_grad]
        return params

    # --- (de)serialisation / cloning --------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "cfg": asdict(self.cfg),
                "name": self.name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Surrogate":
        ckpt = torch.load(path, map_location=device)
        cfg = SurrogateConfig(**ckpt["cfg"])
        model = build_fno(cfg, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        normalizer = ZNormalizer.from_state_dict(ckpt["normalizer"], device=device)
        return cls(model, normalizer, cfg, device=device, name=ckpt.get("name", "base"))

    def clone(self, name: Optional[str] = None) -> "Surrogate":
        model = build_fno(self.cfg, device=self.device)
        model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        norm = ZNormalizer(
            self.normalizer.mean.clone(), self.normalizer.std.clone(), self.device
        )
        return Surrogate(model, norm, self.cfg, self.device, name or f"{self.name}_clone")


def train_surrogate(
    sampler,
    cfg: SurrogateConfig,
    normalizer: ZNormalizer,
    n_iters: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 25,
    seed: int = 0,
) -> Surrogate:
    """Train the tiny FNO surrogate on next-block prediction (MSE in norm space)."""
    torch.manual_seed(seed)
    model = build_fno(cfg, device=device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for it in range(1, n_iters + 1):
        x_np, y_np = sampler.sample(batch_size)
        x = normalizer.preprocess(torch.from_numpy(x_np))
        y = normalizer.preprocess(torch.from_numpy(y_np))
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_every and (it % log_every == 0 or it == 1):
            print(f"[train] iter {it}/{n_iters} loss={loss.item():.5f}")
    sur = Surrogate(model, normalizer, cfg, device=device, name="base")
    sur.train_history = history  # type: ignore[attr-defined]
    return sur
