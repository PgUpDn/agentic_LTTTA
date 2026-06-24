"""The causal streaming LTTTA loop: Observe -> Predict -> Adapt -> Repeat.

For each block we (1) predict the next ``in_step`` frames from the current window,
(2) score them against the now-revealed real block, (3) summarise a compact state,
(4) let the controller pick one bounded action, and (5) execute it to set the
input window for the next block. Real observations are only ever used *after* the
block they belong to has been predicted (delayed by one block) -- i.e. causally.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .actions import ACTION_REGISTRY, BlockContext
from .budget import BudgetGuard
from .calibration import Calibration
from .controller import BoundedController
from .data import FoilTrajectory
from .metrics_track2 import (
    CompositeConfig,
    compute_composite,
    mvpe,
    rel_l2,
    rmse,
    safe_prediction_score,
    time_score,
    tke_error,
)
from .regime_memory import RegimeMemory
from .state import CompactState

_EMA_BETA = 0.3


def _block(data: np.ndarray, a: int, b: int) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(data[a:b])).unsqueeze(0).float()


def run_streaming_eval(
    active_surrogate,
    trajectory: FoilTrajectory,
    controller: BoundedController,
    *,
    in_step: int,
    n_blocks: int,
    start_frame: int = 0,
    experts: Optional[Dict[str, Any]] = None,
    memory: Optional[RegimeMemory] = None,
    composite_cfg: Optional[CompositeConfig] = None,
    collect_logs: bool = True,
) -> Dict[str, Any]:
    experts = experts or {}
    memory = memory if memory is not None else RegimeMemory()
    composite_cfg = composite_cfg or CompositeConfig()
    policy = controller.policy
    device = getattr(active_surrogate, "device", "cpu")
    C = 2  # measured channels (u, v); p is unmeasured/zero

    data = trajectory.data
    L = in_step
    T = trajectory.n_frames
    n_blocks = max(1, min(n_blocks, (T - start_frame) // L - 1))

    budget = BudgetGuard(
        time_budget_s=float(policy["time_budget_s"]),
        max_adapt_steps=int(policy["max_adapt_steps"]),
        max_obs=int(policy["max_obs"]),
    )
    calibration = Calibration(c=C, device=device)
    cur_window = _block(data, start_frame, start_frame + L)

    pred_blocks: List[torch.Tensor] = []
    real_blocks: List[torch.Tensor] = []
    sigmas: List[torch.Tensor] = []
    block_logs: List[dict] = []
    action_counts: Dict[str, int] = {}

    ema = 0.0
    prev_err = 0.0
    steps_since_adapt = 999
    steps_since_obs = 999

    wall0 = time.perf_counter()
    for k in range(n_blocks):
        s_idx = start_frame + (k + 1) * L
        real_block = _block(data, s_idx, s_idx + L)

        raw_pred = active_surrogate.predict_block(cur_window)
        pred = calibration.apply(raw_pred)

        err = rel_l2(pred, real_block, c=C)
        pred_blocks.append(pred.detach())
        real_blocks.append(real_block)
        sigmas.append(calibration.sigma.detach().clone())

        ema = err if k == 0 else (1 - _EMA_BETA) * ema + _EMA_BETA * err
        slope = 0.0 if k == 0 else err - prev_err
        prev_err = err

        state = CompactState(
            block_idx=k,
            n_blocks=n_blocks,
            last_rel_l2=err,
            ema_rel_l2=ema,
            err_slope=slope,
            uncertainty=float(calibration.sigma.mean()),
            time_frac_left=budget.time_frac_left(),
            adapt_frac_left=budget.adapt_frac_left(),
            obs_frac_left=budget.obs_frac_left(),
            steps_since_adapt=steps_since_adapt,
            steps_since_obs=steps_since_obs,
            reynolds=trajectory.reynolds,
            aoa=trajectory.aoa,
            has_memory=int(memory.has(trajectory.regime_key)),
        )

        action = controller.decide(state)
        decision_info = dict(getattr(controller, "last_decision_info", {}) or {})
        if decision_info.get("llm_used"):
            budget.spend_time(float(decision_info.get("llm_latency_s", 0.0)))
        ctx = BlockContext(
            active_surrogate=active_surrogate,
            experts=experts,
            calibration=calibration,
            memory=memory,
            budget=budget,
            prev_window=cur_window,
            raw_pred=raw_pred,
            pred_block=pred,
            real_block=real_block,
            regime_key=trajectory.regime_key,
            reynolds=trajectory.reynolds,
            aoa=trajectory.aoa,
            device=device,
            policy=policy,
            n_channels=C,
        )
        result = ACTION_REGISTRY[action](ctx)
        budget.tick_action()
        active_surrogate = ctx.active_surrogate
        cur_window = result.next_window.detach()

        steps_since_adapt = 0 if action == "update_adapter" else steps_since_adapt + 1
        steps_since_obs = 0 if action == "observe" else steps_since_obs + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        if collect_logs:
            block_logs.append(
                {
                    "block": k,
                    "rel_l2": round(err, 5),
                    "decision": decision_info,
                    **result.info,
                }
            )

    wall = time.perf_counter() - wall0

    pred_full = torch.cat(pred_blocks, dim=1)
    target_full = torch.cat(real_blocks, dim=1)
    sigma_mean = torch.stack(sigmas, dim=0).mean(dim=0)

    m_rel_l2 = rel_l2(pred_full, target_full, c=C)
    m_rmse = rmse(pred_full, target_full, c=C)
    m_tke = tke_error(pred_full, target_full)
    m_mvpe = mvpe(pred_full, target_full)
    m_time = time_score(wall, float(policy["time_budget_s"]))
    sps = safe_prediction_score(pred_full, target_full, sigma_mean, z=float(policy["sigma_z"]), c=C)
    per_block = [bl["rel_l2"] for bl in block_logs] if collect_logs else []

    metrics = {
        "rel_l2": m_rel_l2,
        "rmse": m_rmse,
        "tke": m_tke,
        "mvpe": m_mvpe,
        "time_score": m_time,
        "sps": sps["sps"],
        "coverage": sps["coverage"],
        "wall_s": round(wall, 4),
        "first_block_rel_l2": per_block[0] if per_block else None,
        "last_block_rel_l2": per_block[-1] if per_block else None,
    }
    metrics.update(compute_composite(metrics, composite_cfg))

    return {
        "metrics": metrics,
        "regime": trajectory.regime_key,
        "n_blocks": n_blocks,
        "action_counts": action_counts,
        "budget": budget.summary(),
        "block_logs": block_logs,
    }


def evaluate_policy(
    policy: Dict[str, Any],
    trajectories: List[FoilTrajectory],
    base_surrogate,
    *,
    in_step: int,
    n_blocks: int,
    start_frame: int,
    experts: Optional[Dict[str, Any]] = None,
    composite_cfg: Optional[CompositeConfig] = None,
    collect_logs: bool = False,
) -> Dict[str, Any]:
    """Run a policy over one or more trajectories; returns aggregate + per-traj.

    Each trajectory is a fresh online episode (cloned weights, fresh budget) but
    they share one regime-memory bank so ``retrieve_memory`` can transfer across
    trajectories of the same regime.
    """
    controller = BoundedController(policy)
    memory = RegimeMemory()
    composite_cfg = composite_cfg or CompositeConfig()
    per_traj = []
    for tr in trajectories:
        episode = base_surrogate.clone()
        ep_experts = {n: e.clone() for n, e in (experts or {}).items()}
        res = run_streaming_eval(
            episode,
            tr,
            controller,
            in_step=in_step,
            n_blocks=n_blocks,
            start_frame=start_frame,
            experts=ep_experts,
            memory=memory,
            composite_cfg=composite_cfg,
            collect_logs=collect_logs,
        )
        per_traj.append(res)

    keys = ["rel_l2", "rmse", "tke", "mvpe", "time_score", "sps", "composite"]
    agg = {k: float(np.mean([r["metrics"][k] for r in per_traj])) for k in keys}
    merged_actions: Dict[str, int] = {}
    for r in per_traj:
        for a, c in r["action_counts"].items():
            merged_actions[a] = merged_actions.get(a, 0) + c
    return {
        "aggregate": agg,
        "action_counts": merged_actions,
        "memory": memory.summary(),
        "per_trajectory": [
            {
                "regime": r["regime"],
                "metrics": r["metrics"],
                "actions": r["action_counts"],
                "budget": r["budget"],
                **({"block_logs": r["block_logs"]} if collect_logs else {}),
            }
            for r in per_traj
        ],
    }
