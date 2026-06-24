"""Evaluate a single policy YAML with the streaming LTTTA loop; print + save JSON."""

from __future__ import annotations

import argparse
import json
import os

import yaml

from ..experiment import ExperimentSettings, get_experiment
from ..paths import RESULTS_DIR
from ..rollout import evaluate_policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="agentic_lttta/config/policy.yaml")
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "online_eval.json"))
    args = ap.parse_args()

    with open(args.policy, "r") as f:
        policy = yaml.safe_load(f) or {}

    bundle = get_experiment(
        ExperimentSettings(n_blocks=args.n_blocks, train_iters=args.train_iters)
    )
    res = evaluate_policy(
        policy,
        bundle["trajectories"],
        bundle["base_surrogate"],
        experts=bundle["experts"],
        collect_logs=True,
        **bundle["eval_kwargs"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"policy": policy, "result": res}, f, indent=2)
    print(json.dumps({"aggregate": {k: round(v, 4) for k, v in res["aggregate"].items()},
                      "actions": res["action_counts"]}, indent=2))
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
