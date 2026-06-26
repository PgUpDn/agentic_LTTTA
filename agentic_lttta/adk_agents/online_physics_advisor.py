"""Online Google-ADK physics advisor for experimental LTTTA rollouts.

The advisor is intentionally optional. Normal policy evaluation imports no ADK
objects and behaves exactly as before; ADK is imported only when this class is
constructed. The advisor reads only causal information from the just-revealed
block and recommends one existing bounded action. Python-side validation keeps
the execution surface limited to the registered local actions.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch

from ..actions import BOUNDED_ACTIONS
from ..llm_config import configure_env
from ..state import CompactState

_ADAPT_SCOPES = {"bn", "head", "bn+head"}
_MAX_RESEARCH_CHARS = 2600


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _round(value: Any, ndigits: int = 5) -> float:
    return round(_finite(value), ndigits)


def _run_coro_sync(coro):
    """Run an async ADK session call from sync code, even inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - cross-thread relay
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _with_timeout(fn, timeout_s: Optional[float]):
    if timeout_s is None or timeout_s <= 0:
        return fn()
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - cross-thread relay
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(float(timeout_s))
    if thread.is_alive():
        raise TimeoutError(f"advisor call exceeded {timeout_s:.1f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in advisor response")


def _channel_diagnostics(pred: torch.Tensor, target: torch.Tensor, c: int) -> Dict[str, Any]:
    p = pred[..., :c].detach().float().reshape(-1, c)
    t = target[..., :c].detach().float().reshape(-1, c)
    p_std = p.std(dim=0).clamp_min(1e-6)
    t_std = t.std(dim=0).clamp_min(1e-6)
    return {
        "bias": [_round(v) for v in (p.mean(dim=0) - t.mean(dim=0)).tolist()],
        "std_ratio": [_round(v) for v in (p_std / t_std).tolist()],
        "target_std": [_round(v) for v in t_std.tolist()],
    }


def _energy_drift(pred: torch.Tensor, target: torch.Tensor, c: int) -> Dict[str, float]:
    p = pred[..., :c].detach().float()
    t = target[..., :c].detach().float()
    p_ke = 0.5 * torch.mean(torch.sum(p * p, dim=-1))
    t_ke = 0.5 * torch.mean(torch.sum(t * t, dim=-1)).clamp_min(1e-8)
    return {
        "pred_ke": _round(p_ke),
        "target_ke": _round(t_ke),
        "rel_ke_drift": _round((p_ke - t_ke) / t_ke),
    }


def _wake_profile_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    stations: Iterable[float] = (0.40, 0.55, 0.70, 0.85),
) -> Dict[str, float]:
    p = pred.detach().float()
    t = target.detach().float()
    _b, _steps, _h, w, _c = p.shape
    out: Dict[str, float] = {}
    for station in stations:
        x = min(int(float(station) * w), w - 1)
        pp = p[:, :, :, x, 0].mean(dim=1)
        tt = t[:, :, :, x, 0].mean(dim=1)
        rng = (tt.amax(dim=1) - tt.amin(dim=1)).clamp_min(1e-6)
        err = ((pp - tt).abs().mean(dim=1) / rng).mean()
        out[f"x_{station:.2f}"] = _round(err)
    return out


def _divergence_proxy(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """A grid-spacing-free incompressibility proxy from centered differences."""
    def div_abs(x: torch.Tensor) -> torch.Tensor:
        u = x[..., 0]
        v = x[..., 1]
        if u.shape[-1] < 3 or v.shape[-2] < 3:
            return torch.tensor(0.0, device=x.device)
        du_dx = u[..., 2:] - u[..., :-2]
        dv_dy = v[..., 2:, :] - v[..., :-2, :]
        core = du_dx[..., 1:-1, :] + dv_dy[..., :, 1:-1]
        return core.abs().mean()

    p_div = div_abs(pred.detach().float())
    t_div = div_abs(target.detach().float()).clamp_min(1e-8)
    return {"pred_abs": _round(p_div), "target_abs": _round(t_div), "ratio": _round(p_div / t_div)}


def build_physics_diagnostics(
    *,
    state: CompactState,
    raw_pred: torch.Tensor,
    pred_block: torch.Tensor,
    real_block: torch.Tensor,
    n_channels: int,
    recent_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Summarise causal tensor information into a compact JSON-safe payload."""
    from ..metrics_track2 import mvpe, rel_l2, rmse

    raw_err = rel_l2(raw_pred, real_block, c=n_channels)
    cal_err = rel_l2(pred_block, real_block, c=n_channels)
    return {
        "state": state.to_dict(),
        "block": {"idx": state.block_idx, "n_blocks": state.n_blocks},
        "regime": {
            "reynolds": _round(state.reynolds, 2),
            "aoa": _round(state.aoa, 3),
            "key": f"Re{int(round(state.reynolds))}_AoA{state.aoa:g}",
        },
        "error": {
            "raw_rel_l2": _round(raw_err),
            "calibrated_rel_l2": _round(cal_err),
            "rmse": _round(rmse(pred_block, real_block, c=n_channels)),
            "mvpe": _round(mvpe(pred_block, real_block)),
            "ema_rel_l2": _round(state.ema_rel_l2),
            "slope": _round(state.err_slope),
        },
        "channel": _channel_diagnostics(pred_block, real_block, n_channels),
        "energy": _energy_drift(pred_block, real_block, n_channels),
        "wake_profile_error": _wake_profile_errors(pred_block, real_block),
        "divergence_proxy": _divergence_proxy(pred_block, real_block),
        "uncertainty": {"sigma_mean": _round(state.uncertainty)},
        "budget": {
            "time_frac_left": _round(state.time_frac_left),
            "adapt_frac_left": _round(state.adapt_frac_left),
            "obs_frac_left": _round(state.obs_frac_left),
        },
        "recent_actions": list(recent_actions or [])[-8:],
    }


@dataclass
class OnlinePhysicsAdvisor:
    """Small synchronous wrapper around ADK LlmAgents for online guidance."""

    model: str = "gemini-2.5-flash"
    timeout_s: Optional[float] = 90.0
    allowed_actions: List[str] = field(default_factory=lambda: list(BOUNDED_ACTIONS))

    def __post_init__(self) -> None:
        configure_env()
        if not self.allowed_actions:
            self.allowed_actions = list(BOUNDED_ACTIONS)
        self.research_cache: Dict[str, str] = {}
        self.recent_actions: List[str] = []
        self._decision_agent = None
        self._research_agent = None

    def start_episode(self, _regime_key: str) -> None:
        self.recent_actions = []

    def record_action(self, action: str) -> None:
        self.recent_actions.append(str(action))
        if len(self.recent_actions) > 16:
            self.recent_actions = self.recent_actions[-16:]

    def _make_research_agent(self):
        if self._research_agent is not None:
            return self._research_agent
        from google.adk.agents import LlmAgent
        from google.adk.tools.google_search_tool import GoogleSearchTool

        search = GoogleSearchTool(bypass_multi_tools_limit=True, model=self.model)
        self._research_agent = LlmAgent(
            name="online_physics_researcher",
            model=self.model,
            description="Researches airfoil wake physics priors for LTTTA control.",
            instruction=(
                "You research fluid-dynamics priors for online control of airfoil "
                "wake forecasting. Use Google Search when helpful. Return a compact "
                "actionable summary focused on wake drift, Reynolds/AoA effects, "
                "energy/profile errors, and when to re-seed observations versus "
                "adapt local model parameters."
            ),
            tools=[search],
        )
        return self._research_agent

    def _make_decision_agent(self):
        if self._decision_agent is not None:
            return self._decision_agent
        from google.adk.agents import LlmAgent

        self._decision_agent = LlmAgent(
            name="online_physics_advisor",
            model=self.model,
            description="Recommends one bounded LTTTA action from causal diagnostics.",
            instruction=(
                "You are an online physics advisor for a causal Long-Term Test-Time "
                "Adaptation controller. Recommend exactly one existing bounded "
                "action. Never invent actions. Return ONLY a JSON object with keys: "
                "action, reason, confidence, and optional adapt_steps, adapt_scope, "
                "calib_momentum. confidence must be 0..1. Keep reason under 30 words."
            ),
            tools=[],
        )
        return self._decision_agent

    def _run_agent_text(self, agent, prompt: str, app_name: str) -> str:
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        def run() -> str:
            runner = InMemoryRunner(agent=agent, app_name=app_name)
            session = _run_coro_sync(
                runner.session_service.create_session(app_name=app_name, user_id="online")
            )
            msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            texts: List[str] = []
            for ev in runner.run(user_id="online", session_id=session.id, new_message=msg):
                content = getattr(ev, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    text = getattr(part, "text", None)
                    if text and text.strip():
                        texts.append(text.strip())
            return "\n".join(texts).strip()

        return _with_timeout(run, self.timeout_s)

    def research_key(self, diagnostics: Dict[str, Any]) -> str:
        return str(diagnostics.get("regime", {}).get("key") or "unknown_regime")

    def ensure_research(self, diagnostics: Dict[str, Any]) -> str:
        key = self.research_key(diagnostics)
        if key in self.research_cache:
            return self.research_cache[key]
        regime = diagnostics.get("regime", {})
        prompt = (
            "Research airfoil wake forecasting control priors for this regime and "
            "return a compact summary with 4-6 bullets.\n"
            f"Regime JSON: {json.dumps(regime, sort_keys=True)}\n"
            "Focus on: wake profile bias, vortex shedding/energy drift, Reynolds "
            "and angle-of-attack effects, and practical online correction choices."
        )
        try:
            summary = self._run_agent_text(
                self._make_research_agent(), prompt, "lttta_online_physics_research"
            )
        except Exception as exc:  # noqa: BLE001
            summary = (
                "Fallback physics prior: preserve wake profile and kinetic energy; "
                "use observe for large drift, update_adapter for rising moderate "
                f"error, recalibrate for channel bias. Research error: {type(exc).__name__}: {exc}"
            )
        summary = re.sub(r"\s+", " ", summary).strip()[:_MAX_RESEARCH_CHARS]
        self.research_cache[key] = summary
        return summary

    def decide(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        key = self.research_key(diagnostics)
        try:
            research = self.ensure_research(diagnostics)
            prompt = (
                "Choose the next bounded action from these allowed actions only: "
                f"{', '.join(self.allowed_actions)}.\n"
                "Physics research summary:\n"
                f"{research}\n\n"
                "Causal online diagnostics JSON:\n"
                f"{json.dumps(diagnostics, sort_keys=True)}\n\n"
                "Return ONLY JSON, for example: "
                '{"action":"recalibrate","reason":"channel bias dominates",'
                '"confidence":0.72}'
            )
            raw = self._run_agent_text(
                self._make_decision_agent(), prompt, "lttta_online_physics_advisor"
            )
            obj = _extract_json_object(raw)
            return {
                "action": obj.get("action"),
                "reason": str(obj.get("reason", ""))[:240],
                "confidence": max(0.0, min(1.0, _finite(obj.get("confidence"), 0.0))),
                "adapt_steps": obj.get("adapt_steps"),
                "adapt_scope": obj.get("adapt_scope"),
                "calib_momentum": obj.get("calib_momentum"),
                "research_key": key,
                "raw_response": raw[:1000],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action": None,
                "reason": "",
                "confidence": 0.0,
                "research_key": key,
                "error": f"{type(exc).__name__}: {exc}",
            }


def clean_advisor_overrides(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Return validated optional policy overrides from an advisor decision."""
    overrides: Dict[str, Any] = {}
    if decision.get("adapt_steps") is not None:
        try:
            overrides["adapt_steps"] = max(1, min(12, int(decision["adapt_steps"])))
        except Exception:
            pass
    if decision.get("adapt_scope") in _ADAPT_SCOPES:
        overrides["adapt_scope"] = str(decision["adapt_scope"])
    if decision.get("calib_momentum") is not None:
        try:
            overrides["calib_momentum"] = max(0.0, min(1.0, float(decision["calib_momentum"])))
        except Exception:
            pass
    return overrides
