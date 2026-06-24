"""``skip_update``: cheapest action -- free-run on our own prediction."""

from __future__ import annotations

from .base import ActionResult, BlockContext, register


@register("skip_update")
def skip_update(ctx: BlockContext) -> ActionResult:
    return ActionResult(next_window=ctx.free_run_window(), info={"action": "skip_update"})
