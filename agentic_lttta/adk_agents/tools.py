"""Python *tools* the Google-ADK design agents call.

These wrap the heavy compute (surrogate training + streaming evaluation) so the
LLM agents only ever exchange small JSON payloads. The experiment bundle (data +
trained surrogate) is initialised once via :func:`init_tools` and cached, so each
``evaluate_policy`` call is just a streaming rollout (no re-training).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..actions import BOUNDED_ACTIONS
from ..controller import DEFAULT_POLICY
from ..experiment import ExperimentSettings, get_experiment
from ..paths import CONFIG_DIR, RESULTS_DIR
from ..rollout import evaluate_policy

_BUNDLE: Optional[Dict[str, Any]] = None
_TUNABLE_KEYS = [
    "err_low", "err_high", "slope_adapt",
    "use_observe", "use_adapter", "use_memory", "use_expert", "expert_every",
    "adapt_steps", "adapt_scope", "adapt_lr", "calib_momentum",
    "max_obs", "max_adapt_steps", "time_budget_s",
]


def init_tools(settings: Optional[ExperimentSettings] = None) -> Dict[str, Any]:
    """Load data + train/cache the surrogate; must be called before the agents run."""
    global _BUNDLE
    _BUNDLE = get_experiment(settings or ExperimentSettings())
    return _BUNDLE


def _ensure() -> Dict[str, Any]:
    if _BUNDLE is None:
        init_tools()
    assert _BUNDLE is not None
    return _BUNDLE


def _run(policy: Dict[str, Any]) -> Dict[str, Any]:
    b = _ensure()
    res = evaluate_policy(
        policy, b["trajectories"], b["base_surrogate"], experts=b["experts"], **b["eval_kwargs"]
    )
    agg = {k: round(v, 4) for k, v in res["aggregate"].items()}
    return {"metrics": agg, "actions": res["action_counts"]}


# --- tools -----------------------------------------------------------------
def get_setup_info() -> dict:
    """Return the LTTTA design problem context: the bounded action space, the
    tunable policy knobs with their current defaults, the flow regimes, and the
    no-adaptation baseline metrics. Call this first to understand the task."""
    b = _ensure()
    baseline_policy = {**DEFAULT_POLICY, "mode": "fixed", "fixed_action": "skip_update"}
    baseline = _run(baseline_policy)
    return {
        "objective": "Maximize 'composite' (0..1, higher=better). It rewards low "
        "rel_l2/mvpe/tke errors, high time_score (runtime efficiency), and high "
        "SPS (uncertainty calibration). All adaptation compute is budgeted.",
        "bounded_actions": BOUNDED_ACTIONS,
        "tunable_knobs": {k: DEFAULT_POLICY[k] for k in _TUNABLE_KEYS},
        "regimes": [tr.regime_key for tr in b["trajectories"]],
        "eval_kwargs": b["eval_kwargs"],
        "no_adaptation_baseline": baseline,
        "hint": "err_low/err_high are thresholds on the previous block's rel_l2. "
        "observe re-seeds from real data (uses obs budget); update_adapter does a "
        "few gradient steps (uses adapt+time budget); recalibrate is a cheap affine "
        "fix; retrieve_memory reuses a stored regime calibration.",
    }


def evaluate_policy_tool(
    err_low: float = 0.08,
    err_high: float = 0.20,
    slope_adapt: float = 0.02,
    use_observe: bool = True,
    use_adapter: bool = True,
    use_memory: bool = True,
    adapt_steps: int = 3,
    adapt_scope: str = "bn",
    adapt_lr: float = 0.001,
    max_obs: int = 8,
    max_adapt_steps: int = 60,
) -> dict:
    """Evaluate one bounded-controller policy on the foil streaming benchmark and
    return aggregate Track-2 metrics. Lower rel_l2/mvpe/tke is better; higher
    composite/time_score/sps is better. ``adapt_scope`` is one of 'bn','head','bn+head'.
    Use this to search for the policy with the highest 'composite'."""
    policy = {
        **DEFAULT_POLICY,
        "mode": "rule",
        "err_low": float(err_low),
        "err_high": float(err_high),
        "slope_adapt": float(slope_adapt),
        "use_observe": bool(use_observe),
        "use_adapter": bool(use_adapter),
        "use_memory": bool(use_memory),
        "adapt_steps": int(adapt_steps),
        "adapt_scope": str(adapt_scope),
        "adapt_lr": float(adapt_lr),
        "max_obs": int(max_obs),
        "max_adapt_steps": int(max_adapt_steps),
    }
    return _run(policy)


def save_policy(
    note: str,
    err_low: float = 0.08,
    err_high: float = 0.20,
    slope_adapt: float = 0.02,
    use_observe: bool = True,
    use_adapter: bool = True,
    use_memory: bool = True,
    adapt_steps: int = 3,
    adapt_scope: str = "bn",
    adapt_lr: float = 0.001,
    max_obs: int = 8,
    max_adapt_steps: int = 60,
) -> dict:
    """Persist the chosen policy to agentic_lttta/config/policy.yaml (the design
    artifact). Call this once at the end with the BEST policy you found. ``note``
    should briefly justify the choice. Returns the saved path and its metrics."""
    import yaml

    policy = {
        "mode": "rule", "fixed_action": "skip_update",
        "err_low": float(err_low), "err_high": float(err_high),
        "slope_adapt": float(slope_adapt),
        "obs_reserve": 0.0, "adapt_reserve": 0.0, "time_reserve": 0.05,
        "use_memory": bool(use_memory), "use_observe": bool(use_observe),
        "use_adapter": bool(use_adapter), "use_expert": False, "expert_every": 0,
        "adapt_steps": int(adapt_steps), "adapt_scope": str(adapt_scope),
        "adapt_lr": float(adapt_lr), "calib_momentum": 0.5, "sigma_z": 1.96,
        "time_budget_s": float(DEFAULT_POLICY["time_budget_s"]),
        "max_adapt_steps": int(max_adapt_steps), "max_obs": int(max_obs),
    }
    metrics = _run(policy)
    path = os.path.join(CONFIG_DIR, "policy.yaml")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(policy, f, sort_keys=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tuned_policy_result.json"), "w") as f:
        json.dump({"note": note, "policy": policy, "metrics": metrics}, f, indent=2)
    return {"saved_to": path, "note": note, "metrics": metrics}


ALL_TOOLS = [get_setup_info, evaluate_policy_tool, save_policy]
ANALYST_TOOLS = [get_setup_info, evaluate_policy_tool]
TUNER_TOOLS = [evaluate_policy_tool, save_policy]
