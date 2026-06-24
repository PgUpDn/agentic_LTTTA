"""``recalibrate``: cheap, gradient-free affine correction + uncertainty update.

Uses only per-channel *summary statistics* of the revealed block (not a full
field re-seed), so it does **not** spend the observation budget. Also records the
resulting calibration into the regime memory bank.
"""

from __future__ import annotations

from .base import ActionResult, BlockContext, register


@register("recalibrate")
def recalibrate(ctx: BlockContext) -> ActionResult:
    ctx.calibration.update(
        ctx.raw_pred, ctx.real_block, momentum=ctx.policy.get("calib_momentum", 0.5)
    )
    corrected = ctx.calibration.apply(ctx.raw_pred)

    from ..metrics_track2 import rel_l2

    err = rel_l2(corrected, ctx.real_block, c=ctx.n_channels)
    ctx.memory.update(ctx.regime_key, ctx.calibration, err)
    return ActionResult(
        next_window=corrected, info={"action": "recalibrate", "rel_l2_after": err}
    )
