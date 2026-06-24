"""Bounded action space: context, result, and registry.

Each action is a small function ``action(ctx) -> ActionResult`` that may mutate
the rollout's persistent state (surrogate weights, calibration, memory, budget)
and returns the input window used to predict the *next* block. The set of action
names is the controller's bounded action space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class BlockContext:
    # active model + expert pool (select_expert may swap ``active_surrogate``)
    active_surrogate: Any
    experts: Dict[str, Any]
    # persistent corrective state
    calibration: Any            # agentic_lttta.calibration.Calibration
    memory: Any                 # agentic_lttta.regime_memory.RegimeMemory
    budget: Any                 # agentic_lttta.budget.BudgetGuard
    # this block's tensors  ([1, L, H, W, C], physical units)
    prev_window: torch.Tensor   # input that produced this block
    raw_pred: torch.Tensor      # uncalibrated model output for this block
    pred_block: torch.Tensor    # calibrated prediction (already scored)
    real_block: torch.Tensor    # revealed ground-truth (streaming observation)
    # regime
    regime_key: str
    reynolds: float
    aoa: float
    # misc
    device: str = "cpu"
    policy: Dict[str, Any] = field(default_factory=dict)
    n_channels: int = 2

    def free_run_window(self) -> torch.Tensor:
        """Default next input: our own (calibrated) prediction."""
        return self.pred_block


@dataclass
class ActionResult:
    next_window: torch.Tensor
    info: Dict[str, Any] = field(default_factory=dict)


ACTION_REGISTRY: Dict[str, Callable[[BlockContext], ActionResult]] = {}


def register(name: str):
    def deco(fn: Callable[[BlockContext], ActionResult]):
        ACTION_REGISTRY[name] = fn
        return fn

    return deco


def action_names() -> List[str]:
    return list(ACTION_REGISTRY.keys())
