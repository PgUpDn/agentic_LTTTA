"""ADK tools for testing the online physics advisor from any agent."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import yaml

from ..actions import BOUNDED_ACTIONS
from ..controller import DEFAULT_POLICY
from ..experiment import ExperimentSettings, get_experiment
from ..paths import CONFIG_DIR, RESULTS_DIR, WORKSPACE_ROOT
from ..rollout import evaluate_policy
from .online_physics_advisor import OnlinePhysicsAdvisor


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(WORKSPACE_ROOT, path)


def _load_policy(policy_path: str) -> Dict[str, Any]:
    path = _resolve_path(policy_path)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _physics_advisor_model() -> str:
    path = os.path.join(CONFIG_DIR, "models.yaml")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("physics_advisor_model") or "gemini-2.5-flash")
    except Exception:
        return "gemini-2.5-flash"


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 4) for k, v in metrics.items()}


def _compact_logs(result: Dict[str, Any], max_log_blocks: int) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for traj in result.get("per_trajectory", []):
        for block in traj.get("block_logs", [])[: max(0, int(max_log_blocks))]:
            logs.append({
                "regime": traj.get("regime"),
                "block": block.get("block"),
                "rel_l2": block.get("rel_l2"),
                "executed_action": block.get("action"),
                "advisor_action": block.get("advisor_action"),
                "advisor_accepted": block.get("advisor_accepted"),
                "advisor_fallback_action": block.get("advisor_fallback_action"),
                "advisor_fallback_reason": block.get("advisor_fallback_reason"),
                "advisor_reason": block.get("advisor_reason"),
                "advisor_confidence": block.get("advisor_confidence"),
                "advisor_overrides": block.get("advisor_overrides", {}),
            })
    return logs


def _budget_summaries(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"regime": traj.get("regime"), "budget": traj.get("budget", {})}
        for traj in result.get("per_trajectory", [])
    ]


def _save_json(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_eval(
    *,
    policy: Dict[str, Any],
    n_blocks: int,
    advisor_every: int,
    use_advisor: bool,
    advisor_timeout_s: float = 90.0,
) -> Dict[str, Any]:
    bundle = get_experiment(ExperimentSettings(n_blocks=int(n_blocks)))
    advisor = None
    if use_advisor:
        advisor = OnlinePhysicsAdvisor(
            model=_physics_advisor_model(),
            timeout_s=float(advisor_timeout_s),
            allowed_actions=policy.get("advisor_allowed_actions") or BOUNDED_ACTIONS,
        )
    return evaluate_policy(
        policy,
        bundle["trajectories"],
        bundle["base_surrogate"],
        experts=bundle["experts"],
        collect_logs=True,
        advisor=advisor,
        advisor_every=int(advisor_every),
        advisor_timeout_s=float(advisor_timeout_s),
        **bundle["eval_kwargs"],
    )


def get_online_advisor_setup_info() -> dict:
    """Return action space, current policy, eval defaults, and advisor options."""
    policy_path = os.path.join(CONFIG_DIR, "policy.yaml")
    policy = _load_policy(policy_path) if os.path.exists(policy_path) else DEFAULT_POLICY
    return {
        "purpose": "Test an experimental online LLM physics advisor during rollout.",
        "bounded_actions": BOUNDED_ACTIONS,
        "policy_path": policy_path,
        "current_policy": policy,
        "eval_defaults": {
            "n_blocks": 6,
            "advisor_every": int(policy.get("advisor_every", 1)),
            "max_log_blocks": 8,
        },
        "advisor": {
            "model": _physics_advisor_model(),
            "uses_google_search": True,
            "time_score_note": "LLM calls happen online; this is for accuracy exploration.",
        },
    }


def run_online_physics_advisor_eval(
    n_blocks: int = 6,
    advisor_every: int = 1,
    policy_path: str = "agentic_lttta/config/policy.yaml",
    max_log_blocks: int = 8,
    advisor_timeout_s: float = 90.0,
) -> dict:
    """Run a short advisor-enabled online rollout and return compact logs."""
    policy = _load_policy(policy_path)
    result = _run_eval(
        policy=policy,
        n_blocks=int(n_blocks),
        advisor_every=int(advisor_every),
        use_advisor=True,
        advisor_timeout_s=float(advisor_timeout_s),
    )
    payload = {
        "policy_path": _resolve_path(policy_path),
        "advisor": {
            "enabled": True,
            "model": _physics_advisor_model(),
            "advisor_every": int(advisor_every),
        },
        "metrics": _round_metrics(result["aggregate"]),
        "actions": result["action_counts"],
        "budget": _budget_summaries(result),
        "advisor_logs": _compact_logs(result, int(max_log_blocks)),
    }
    payload["saved_to"] = _save_json("online_physics_advisor_web_eval.json", payload)
    return payload


def compare_online_with_without_advisor(
    n_blocks: int = 6,
    advisor_every: int = 1,
    advisor_timeout_s: float = 90.0,
) -> dict:
    """Compare the current policy with and without the online physics advisor."""
    policy_path = os.path.join(CONFIG_DIR, "policy.yaml")
    policy = _load_policy(policy_path)
    baseline = _run_eval(
        policy=policy,
        n_blocks=int(n_blocks),
        advisor_every=int(advisor_every),
        use_advisor=False,
        advisor_timeout_s=float(advisor_timeout_s),
    )
    advised = _run_eval(
        policy=policy,
        n_blocks=int(n_blocks),
        advisor_every=int(advisor_every),
        use_advisor=True,
        advisor_timeout_s=float(advisor_timeout_s),
    )
    payload = {
        "policy_path": policy_path,
        "n_blocks": int(n_blocks),
        "advisor_every": int(advisor_every),
        "without_advisor": {
            "metrics": _round_metrics(baseline["aggregate"]),
            "actions": baseline["action_counts"],
            "budget": _budget_summaries(baseline),
        },
        "with_advisor": {
            "metrics": _round_metrics(advised["aggregate"]),
            "actions": advised["action_counts"],
            "budget": _budget_summaries(advised),
            "advisor_logs": _compact_logs(advised, 8),
        },
    }
    payload["saved_to"] = _save_json("online_physics_advisor_web_compare.json", payload)
    return payload


ONLINE_ADVISOR_TOOLS = [
    get_online_advisor_setup_info,
    run_online_physics_advisor_eval,
    compare_online_with_without_advisor,
]
