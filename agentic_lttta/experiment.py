"""Central experiment setup: load foil data + train/cache the surrogate ONCE.

Both the CLI scripts and the Google-ADK design tools call :func:`get_experiment`,
which caches the (data, surrogate, experts, eval-kwargs) bundle in-process so the
design agent can evaluate many policies without re-training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .data import FoilTrajectory, load_foil_trajectories
from .normalizer import ZNormalizer
from .paths import CKPT_DIR, make_dirs
from .surrogate import Surrogate, SurrogateConfig, train_surrogate
from .data import TrainWindowSampler


@dataclass
class ExperimentSettings:
    max_traj: int = 1
    sub_s: int = 4
    in_step: int = 10
    fno_width: int = 20
    n_layers: int = 3
    modes1: int = 4
    modes2: int = 8
    modes3: int = 8
    train_iters: int = 200
    batch_size: int = 4
    lr: float = 1.0e-3
    train_stop_frac: float = 0.6
    n_blocks: int = 60
    seed: int = 0
    device: str = "cpu"


_CACHE: Dict[str, Dict[str, Any]] = {}


def _ckpt_path(settings: ExperimentSettings) -> str:
    return os.path.join(CKPT_DIR, f"base_subs{settings.sub_s}_in{settings.in_step}.pt")


def get_experiment(
    settings: Optional[ExperimentSettings] = None, force_retrain: bool = False
) -> Dict[str, Any]:
    settings = settings or ExperimentSettings()
    key = repr(asdict(settings))
    if key in _CACHE and not force_retrain:
        return _CACHE[key]

    make_dirs()
    trajectories: List[FoilTrajectory] = load_foil_trajectories(
        max_traj=settings.max_traj, sub_s=settings.sub_s
    )
    h, w = trajectories[0].hw
    cfg = SurrogateConfig(
        in_step=settings.in_step,
        height=h,
        width=w,
        channels=3,
        modes1=settings.modes1,
        modes2=settings.modes2,
        modes3=settings.modes3,
        fno_width=settings.fno_width,
        n_layers=settings.n_layers,
    )

    ckpt = _ckpt_path(settings)
    if os.path.exists(ckpt) and not force_retrain:
        base = Surrogate.load(ckpt, device=settings.device)
    else:
        # Fit normaliser on the training portion of all trajectories.
        train_frames = np.concatenate(
            [tr.data[: int(tr.n_frames * settings.train_stop_frac)] for tr in trajectories],
            axis=0,
        )
        normalizer = ZNormalizer.fit(train_frames, device=settings.device)
        del train_frames
        sampler = TrainWindowSampler(
            trajectories,
            in_step=settings.in_step,
            train_stop_frac=settings.train_stop_frac,
            seed=settings.seed,
        )
        base = train_surrogate(
            sampler,
            cfg,
            normalizer,
            n_iters=settings.train_iters,
            batch_size=settings.batch_size,
            lr=settings.lr,
            device=settings.device,
            seed=settings.seed,
        )
        base.save(ckpt)

    min_len = min(tr.n_frames for tr in trajectories)
    start_frame = int(min_len * settings.train_stop_frac)
    experts = {"base": base}  # single expert for now; pool is extensible

    bundle = {
        "settings": settings,
        "trajectories": trajectories,
        "base_surrogate": base,
        "experts": experts,
        "eval_kwargs": {
            "in_step": settings.in_step,
            "n_blocks": settings.n_blocks,
            "start_frame": start_frame,
        },
        "cfg": cfg,
        "checkpoint": ckpt,
    }
    _CACHE[key] = bundle
    return bundle
