"""The compact state the controller "reads" before choosing an action.

This mirrors the organizers' reference Agentic-Controller blueprint:
"Read state" -> recent observations, prediction error, uncertainty level,
budget remaining, retrieved memory. We keep it deliberately small (a handful of
scalars) so the online decision is cheap and so an LLM design-agent can reason
about it offline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CompactState:
    block_idx: int            # which rollout block we are about to predict
    n_blocks: int             # total blocks in the rollout
    last_rel_l2: float        # realised rel-L2 of the *previous* (now revealed) block
    ema_rel_l2: float         # exponential moving average of rel-L2 (drift signal)
    err_slope: float          # short-term trend of rel-L2 (>0 => error growing)
    uncertainty: float        # current per-channel residual sigma estimate (SPS)
    time_frac_left: float     # budget remaining (0..1)
    adapt_frac_left: float    # gradient-step budget remaining (0..1)
    obs_frac_left: float      # observation budget remaining (0..1)
    steps_since_adapt: int    # blocks since the last weight update
    steps_since_obs: int      # blocks since the last observation re-seed
    reynolds: float           # regime descriptor (from sim_id)
    aoa: float                # angle of attack (from sim_id)
    has_memory: int           # 1 if a stored correction exists for this regime

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"blk {self.block_idx}/{self.n_blocks} relL2={self.last_rel_l2:.3f} "
            f"ema={self.ema_rel_l2:.3f} slope={self.err_slope:+.3f} "
            f"sigma={self.uncertainty:.3f} t={self.time_frac_left:.2f} "
            f"a={self.adapt_frac_left:.2f} o={self.obs_frac_left:.2f}"
        )
