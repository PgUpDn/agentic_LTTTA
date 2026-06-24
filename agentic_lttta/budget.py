"""Budget accounting for the bounded online controller.

The competition counts **all** agent/adaptation computation against a runtime
("Time Score") budget, and observation integration is a limited resource. The
``BudgetGuard`` enforces three caps and exposes the remaining fractions so the
controller can make budget-aware decisions:

* ``time_budget_s``     -- wall-clock seconds for the whole online rollout
* ``max_adapt_steps``   -- total gradient steps allowed across all blocks
* ``max_obs``           -- how many blocks may re-seed from real observations
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetGuard:
    time_budget_s: float
    max_adapt_steps: int
    max_obs: int

    used_time_s: float = 0.0
    used_adapt_steps: int = 0
    used_obs: int = 0
    n_actions: int = field(default=0)

    # --- queries -----------------------------------------------------------
    def can_adapt(self, k_steps: int = 1) -> bool:
        return (
            self.used_adapt_steps + k_steps <= self.max_adapt_steps
            and self.time_frac_left() > 0.0
        )

    def can_observe(self) -> bool:
        return self.used_obs < self.max_obs

    def time_frac_left(self) -> float:
        if self.time_budget_s <= 0:
            return 0.0
        return max(0.0, 1.0 - self.used_time_s / self.time_budget_s)

    def adapt_frac_left(self) -> float:
        if self.max_adapt_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - self.used_adapt_steps / self.max_adapt_steps)

    def obs_frac_left(self) -> float:
        if self.max_obs <= 0:
            return 0.0
        return max(0.0, 1.0 - self.used_obs / self.max_obs)

    # --- mutations ---------------------------------------------------------
    def spend_time(self, seconds: float) -> None:
        self.used_time_s += max(0.0, float(seconds))

    def spend_adapt(self, k_steps: int) -> None:
        self.used_adapt_steps += int(k_steps)

    def spend_obs(self, n: int = 1) -> None:
        self.used_obs += int(n)

    def tick_action(self) -> None:
        self.n_actions += 1

    # --- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        return {
            "time_budget_s": self.time_budget_s,
            "used_time_s": round(self.used_time_s, 4),
            "time_frac_left": round(self.time_frac_left(), 4),
            "max_adapt_steps": self.max_adapt_steps,
            "used_adapt_steps": self.used_adapt_steps,
            "max_obs": self.max_obs,
            "used_obs": self.used_obs,
            "n_actions": self.n_actions,
        }
