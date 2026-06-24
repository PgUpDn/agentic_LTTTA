"""Organizer-gateway interface for online LLM action selection.

The online LLM agent is intentionally narrow: it reads the compact rollout
state and returns one action from the bounded action space. It does not replace
the physical predictor or bypass budgets.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from .state import CompactState


@dataclass
class GatewayDecision:
    action: str
    rationale: str = ""
    latency_s: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class MockOrganizerGateway:
    """Deterministic local stand-in for the official organizer gateway.

    Use this in demos and CI to exercise the online-LLM control path without a
    participant-owned API key or network dependency.
    """

    def __init__(self, latency_s: float = 0.02):
        self.latency_s = max(0.0, float(latency_s))

    def decide(
        self,
        state: CompactState,
        policy: Dict[str, Any],
        allowed_actions: Sequence[str],
    ) -> GatewayDecision:
        t0 = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)

        e = state.last_rel_l2
        err_low = float(policy.get("err_low", 0.08))
        err_high = float(policy.get("err_high", 0.20))
        action = "recalibrate"
        rationale = "moderate error; apply cheap statistical correction"

        if e <= err_low:
            action = "skip_update"
            rationale = "error is low; preserve budget"
        elif state.has_memory and e <= err_high and "retrieve_memory" in allowed_actions:
            action = "retrieve_memory"
            rationale = "known regime with reusable calibration memory"
        elif e >= err_high:
            if state.obs_frac_left > float(policy.get("obs_reserve", 0.0)):
                action = "observe"
                rationale = "high error; use streaming observation budget"
            elif (
                state.adapt_frac_left > float(policy.get("adapt_reserve", 0.0))
                and state.time_frac_left > float(policy.get("time_reserve", 0.05))
            ):
                action = "update_adapter"
                rationale = "high error; adapt lightweight parameters"
        elif (
            state.err_slope > float(policy.get("slope_adapt", 0.02))
            and state.adapt_frac_left > float(policy.get("adapt_reserve", 0.0))
            and state.time_frac_left > float(policy.get("time_reserve", 0.05))
        ):
            action = "update_adapter"
            rationale = "error is trending up; adapt before drift compounds"

        if action not in allowed_actions:
            action = "recalibrate" if "recalibrate" in allowed_actions else allowed_actions[0]
            rationale = "fallback to a valid bounded action"

        latency = time.perf_counter() - t0
        return GatewayDecision(
            action=action,
            rationale=rationale,
            latency_s=latency,
            usage={"mock": True, "prompt_tokens": 0, "completion_tokens": 0},
        )


class HTTPOrganizerGateway:
    """Thin adapter for a future official HTTP gateway.

    Expected response JSON:
    ``{"action": "observe", "rationale": "...", "usage": {...}}``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        token_env: str = "LTTTA_GATEWAY_TOKEN",
        timeout_s: float = 10.0,
    ):
        self.endpoint = endpoint or os.environ.get("LTTTA_GATEWAY_URL", "")
        self.token = os.environ.get(token_env, "")
        self.timeout_s = float(timeout_s)
        if not self.endpoint:
            raise RuntimeError("Set LTTTA_GATEWAY_URL for organizer_http mode.")

    def decide(
        self,
        state: CompactState,
        policy: Dict[str, Any],
        allowed_actions: Sequence[str],
    ) -> GatewayDecision:
        payload = {
            "task": "realpde_lttta_action_selection",
            "state": state.to_dict(),
            "allowed_actions": list(allowed_actions),
            "policy": {
                k: policy[k]
                for k in (
                    "err_low",
                    "err_high",
                    "slope_adapt",
                    "time_reserve",
                    "adapt_reserve",
                    "obs_reserve",
                    "adapt_steps",
                    "adapt_scope",
                    "max_obs",
                    "max_adapt_steps",
                )
                if k in policy
            },
            "response_schema": {"action": list(allowed_actions), "rationale": "string"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"organizer gateway request failed: {exc}") from exc
        latency = time.perf_counter() - t0

        action = str(raw.get("action", "")).strip()
        if action not in allowed_actions:
            raise ValueError(f"organizer gateway returned invalid action: {action!r}")
        return GatewayDecision(
            action=action,
            rationale=str(raw.get("rationale", "")),
            latency_s=latency,
            usage=dict(raw.get("usage") or {}),
            raw=raw,
        )


def build_gateway(policy: Dict[str, Any]):
    provider = str(policy.get("llm_gateway_provider", "mock")).lower()
    if provider in {"mock", "mock_organizer_gateway"}:
        return MockOrganizerGateway(latency_s=float(policy.get("mock_llm_latency_s", 0.02)))
    if provider in {"organizer_http", "official_http"}:
        return HTTPOrganizerGateway(
            endpoint=policy.get("llm_gateway_url"),
            timeout_s=float(policy.get("llm_gateway_timeout_s", 10.0)),
        )
    raise ValueError(f"Unknown llm_gateway_provider: {provider}")
