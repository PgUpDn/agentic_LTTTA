"""``update_adapter``: TENT-style few-step gradient adaptation.

Takes a few SGD steps on a *small* parameter subset (BatchNorm affine and/or the
output head) so that ``model(prev_window) -> real_block`` (a supervised pair from
the just-revealed observation). Counts gradient steps and wall-clock time against
the budget. Improves all subsequent free-running predictions.
"""

from __future__ import annotations

import time

import torch

from .base import ActionResult, BlockContext, register


@register("update_adapter")
def update_adapter(ctx: BlockContext) -> ActionResult:
    k = int(ctx.policy.get("adapt_steps", 3))
    scope = ctx.policy.get("adapt_scope", "bn")
    lr = float(ctx.policy.get("adapt_lr", 1e-3))

    if not ctx.budget.can_adapt(k):
        return ActionResult(
            next_window=ctx.free_run_window(),
            info={"action": "update_adapter", "applied": False, "note": "budget"},
        )

    sur = ctx.active_surrogate
    params = sur.adapt_parameters(scope)
    if not params:
        return ActionResult(
            next_window=ctx.free_run_window(),
            info={"action": "update_adapter", "applied": False, "note": "no params"},
        )

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    xn = sur.normalizer.preprocess(ctx.prev_window.to(ctx.device).float())
    yn = sur.normalizer.preprocess(ctx.real_block.to(ctx.device).float())
    c = ctx.n_channels

    t0 = time.time()
    sur.model.train()
    last_loss = 0.0
    for _ in range(k):
        opt.zero_grad()
        pred = sur.model(xn)
        loss = torch.mean((pred[..., :c] - yn[..., :c]) ** 2)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    sur.model.eval()
    dt = time.time() - t0

    ctx.budget.spend_adapt(k)
    ctx.budget.spend_time(dt)

    new_raw = sur.predict_block(ctx.prev_window)
    new_pred = ctx.calibration.apply(new_raw)
    return ActionResult(
        next_window=new_pred,
        info={"action": "update_adapter", "applied": True, "k": k, "loss": last_loss, "dt": round(dt, 4)},
    )
