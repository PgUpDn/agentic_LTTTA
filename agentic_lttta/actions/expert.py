"""``select_expert``: route to the best surrogate in the pool for this regime.

Evaluates each expert on the just-revealed block (``prev_window -> real_block``)
and switches the active surrogate to the lowest-error one. Compute counts against
the time budget.
"""

from __future__ import annotations

import time

from .base import ActionResult, BlockContext, register


@register("select_expert")
def select_expert(ctx: BlockContext) -> ActionResult:
    if not ctx.experts:
        return ActionResult(
            next_window=ctx.free_run_window(),
            info={"action": "select_expert", "switched": False, "note": "no experts"},
        )

    from ..metrics_track2 import rel_l2

    t0 = time.time()
    best_name, best_err, best_pred = None, None, None
    for name, ex in ctx.experts.items():
        pr = ex.predict_block(ctx.prev_window)
        e = rel_l2(pr, ctx.real_block, c=ctx.n_channels)
        if best_err is None or e < best_err:
            best_name, best_err, best_pred = name, e, pr
    ctx.budget.spend_time(time.time() - t0)

    ctx.active_surrogate = ctx.experts[best_name]
    new_pred = ctx.calibration.apply(best_pred)
    return ActionResult(
        next_window=new_pred,
        info={"action": "select_expert", "expert": best_name, "rel_l2": best_err},
    )
