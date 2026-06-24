"""``retrieve_memory``: load a stored calibration for the current flow regime."""

from __future__ import annotations

from .base import ActionResult, BlockContext, register


@register("retrieve_memory")
def retrieve_memory(ctx: BlockContext) -> ActionResult:
    st = ctx.memory.get(ctx.regime_key)
    if st is None:
        return ActionResult(
            next_window=ctx.free_run_window(),
            info={"action": "retrieve_memory", "hit": False},
        )
    ctx.calibration.load(st)
    corrected = ctx.calibration.apply(ctx.raw_pred)
    return ActionResult(
        next_window=corrected,
        info={"action": "retrieve_memory", "hit": True, "regime": ctx.regime_key},
    )
