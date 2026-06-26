"""The bounded controller: a transparent policy mapping compact state to action.

The policy is fully parameterised (thresholds, budget reserves, action toggles,
and per-action knobs); the offline Google-ADK design team tunes these parameters.
The ``"fixed"`` mode is used for ablations (e.g., always ``skip_update`` == the
no-adaptation baseline). The optional ``"llm_gateway"`` mode demonstrates online
LLM action selection through an organizer-gateway interface while preserving the
same bounded action space.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from .actions import BOUNDED_ACTIONS
from .llm_gateway import build_gateway
from .state import CompactState


DEFAULT_POLICY: Dict[str, Any] = {
    "mode": "rule",
    "fixed_action": "skip_update",
    "err_low": 0.08,
    "err_high": 0.20,
    "slope_adapt": 0.02,
    "obs_reserve": 0.0,
    "adapt_reserve": 0.0,
    "time_reserve": 0.05,
    "use_memory": True,
    "use_observe": True,
    "use_adapter": True,
    "use_expert": False,
    "expert_every": 0,
    "adapt_steps": 3,
    "adapt_scope": "bn",
    "adapt_lr": 0.001,
    "calib_momentum": 0.5,
    "sigma_z": 1.96,
    "time_budget_s": 60.0,
    "max_adapt_steps": 60,
    "max_obs": 8,
    "llm_gateway_provider": "mock",
    "llm_gateway_max_calls": 8,
    "llm_gateway_min_interval": 1,
    "llm_gateway_timeout_s": 10.0,
    "mock_llm_latency_s": 0.02,
    "use_physics_advisor": False,
    "advisor_every": 1,
    "advisor_allowed_actions": BOUNDED_ACTIONS,
}


class BoundedController:
    def __init__(self, policy: Dict[str, Any] | None = None):
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.llm_calls = 0
        self.last_decision_info: Dict[str, Any] = {}
        self.gateway = (
            build_gateway(self.policy)
            if self.policy.get("mode") == "llm_gateway"
            else None
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BoundedController":
        with open(path, "r") as f:
            return cls(yaml.safe_load(f) or {})

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.policy, f, sort_keys=False)

    # --- the decision ------------------------------------------------------
    def decide(self, s: CompactState) -> str:
        self.last_decision_info = {"controller_mode": self.policy.get("mode", "rule")}
        p = self.policy
        if p.get("mode") == "fixed":
            return p.get("fixed_action", "skip_update")
        if p.get("mode") == "llm_gateway":
            return self._decide_llm_gateway(s)
        return self._decide_rule(s)

    def _decide_rule(self, s: CompactState) -> str:
        p = self.policy
        e = s.last_rel_l2
        can_obs = p["use_observe"] and s.obs_frac_left > p["obs_reserve"]
        can_adapt = (
            p["use_adapter"]
            and s.adapt_frac_left > p["adapt_reserve"]
            and s.time_frac_left > p["time_reserve"]
        )

        # 1) very low error -> stay cheap
        if e <= p["err_low"]:
            return "skip_update"

        # 2) known regime shortcut: reuse a stored calibration
        if p["use_memory"] and s.has_memory and e <= p["err_high"]:
            return "retrieve_memory"

        # 3) high error -> strongest affordable correction
        if e >= p["err_high"]:
            if can_obs:
                return "observe"
            if can_adapt:
                return "update_adapter"
            return "recalibrate"

        # 4) moderate error trending up -> adapt weights if affordable
        if s.err_slope > p["slope_adapt"] and can_adapt:
            return "update_adapter"

        # 5) optional periodic expert routing
        if (
            p["use_expert"]
            and p["expert_every"] > 0
            and s.block_idx % int(p["expert_every"]) == 0
        ):
            return "select_expert"

        # 6) default cheap statistical correction
        return "recalibrate"

    def _decide_llm_gateway(self, s: CompactState) -> str:
        p = self.policy
        min_interval = max(1, int(p.get("llm_gateway_min_interval", 1)))
        max_calls = max(0, int(p.get("llm_gateway_max_calls", 0)))
        if (
            self.gateway is None
            or self.llm_calls >= max_calls
            or s.block_idx % min_interval != 0
            or s.time_frac_left <= float(p.get("time_reserve", 0.05))
        ):
            action = self._decide_rule(s)
            self.last_decision_info.update(
                {
                    "llm_used": False,
                    "fallback_action": action,
                    "fallback_reason": "gateway unavailable, interval, calls, or time budget",
                    "llm_calls": self.llm_calls,
                }
            )
            return action

        try:
            decision = self.gateway.decide(s, p, BOUNDED_ACTIONS)
        except Exception as exc:  # noqa: BLE001 - degrade safely to the rule policy
            action = self._decide_rule(s)
            self.last_decision_info.update(
                {
                    "llm_used": False,
                    "fallback_action": action,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                    "llm_calls": self.llm_calls,
                }
            )
            return action

        self.llm_calls += 1
        self.last_decision_info.update(
            {
                "llm_used": True,
                "llm_calls": self.llm_calls,
                "llm_action": decision.action,
                "llm_rationale": decision.rationale,
                "llm_latency_s": round(decision.latency_s, 4),
                "llm_usage": decision.usage,
            }
        )
        return decision.action

    def validate(self) -> None:
        bad = [a for a in [self.policy.get("fixed_action")] if a and a not in BOUNDED_ACTIONS]
        if bad:
            raise ValueError(f"Unknown action(s) in policy: {bad}")
