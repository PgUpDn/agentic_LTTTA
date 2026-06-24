"""``observe``: integrate the streaming observation by re-seeding the rollout.

The strongest correction (eliminates accumulated drift) but it consumes the
limited *observation budget*. Also opportunistically refreshes the affine
calibration from the revealed block.
"""

from __future__ import annotations

from .base import ActionResult, BlockContext, register


@register("observe")
def observe(ctx: BlockContext) -> ActionResult:
    if ctx.budget.can_observe():
        ctx.budget.spend_obs()
        ctx.calibration.update(
            ctx.raw_pred, ctx.real_block, momentum=ctx.policy.get("calib_momentum", 0.5)
        )
        return ActionResult(
            next_window=ctx.real_block, info={"action": "observe", "reseed": True}
        )
    # budget exhausted -> degrade gracefully to free-run
    return ActionResult(
        next_window=ctx.free_run_window(),
        info={"action": "observe", "reseed": False, "note": "obs budget exhausted"},
    )
