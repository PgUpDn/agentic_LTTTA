"""Demonstrate online LLM-agent action selection with a mock organizer gateway."""

from __future__ import annotations

import argparse
import json
import os

from ..paths import CONFIG_DIR


def _summarize(name: str, result: dict) -> dict:
    logs = result["per_trajectory"][0].get("block_logs", [])
    llm_logs = [row for row in logs if row.get("decision", {}).get("llm_used")]
    return {
        "name": name,
        "aggregate": {k: round(v, 4) for k, v in result["aggregate"].items()},
        "actions": result["action_counts"],
        "llm_calls": len(llm_logs),
        "budget": result["per_trajectory"][0]["budget"],
        "first_llm_decisions": [
            {
                "block": row["block"],
                "action": row["decision"].get("llm_action"),
                "rationale": row["decision"].get("llm_rationale"),
                "latency_s": row["decision"].get("llm_latency_s"),
            }
            for row in llm_logs[:5]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=os.path.join(CONFIG_DIR, "policy_llm_demo.yaml"))
    ap.add_argument("--n-blocks", type=int, default=20)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--mock-latency-s", type=float, default=None)
    args = ap.parse_args()

    import yaml

    from ..controller import DEFAULT_POLICY
    from ..experiment import ExperimentSettings, get_experiment
    from ..rollout import evaluate_policy

    with open(args.policy, "r") as f:
        llm_policy = yaml.safe_load(f) or {}
    if args.mock_latency_s is not None:
        llm_policy["mock_llm_latency_s"] = args.mock_latency_s

    bundle = get_experiment(
        ExperimentSettings(n_blocks=args.n_blocks, train_iters=args.train_iters)
    )

    no_adapt = {**DEFAULT_POLICY, "mode": "fixed", "fixed_action": "skip_update"}
    rule_policy = {**DEFAULT_POLICY, **{k: v for k, v in llm_policy.items() if not k.startswith("llm_")}}
    rule_policy["mode"] = "rule"

    common = {
        "trajectories": bundle["trajectories"],
        "base_surrogate": bundle["base_surrogate"],
        "experts": bundle["experts"],
        **bundle["eval_kwargs"],
    }
    results = {
        "no_adaptation": evaluate_policy(no_adapt, collect_logs=False, **common),
        "rule_controller": evaluate_policy(rule_policy, collect_logs=False, **common),
        "online_llm_agent_mock_gateway": evaluate_policy(llm_policy, collect_logs=True, **common),
    }
    print(json.dumps([_summarize(k, v) for k, v in results.items()], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
