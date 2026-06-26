"""Evaluate a single policy YAML with the streaming LTTTA loop; print + save JSON."""

from __future__ import annotations

import argparse
import json
import os

from ..paths import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="agentic_lttta/config/policy.yaml")
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "online_eval.json"))
    ap.add_argument("--use-physics-advisor", action="store_true")
    ap.add_argument("--advisor-model", default=None)
    ap.add_argument("--advisor-every", type=int, default=None)
    ap.add_argument("--advisor-timeout-s", type=float, default=90.0)
    ap.add_argument(
        "--advisor-out",
        default=os.path.join(RESULTS_DIR, "online_physics_advisor_eval.json"),
    )
    args = ap.parse_args()

    import yaml

    from ..experiment import ExperimentSettings, get_experiment
    from ..rollout import evaluate_policy

    with open(args.policy, "r") as f:
        policy = yaml.safe_load(f) or {}

    use_advisor = bool(args.use_physics_advisor or policy.get("use_physics_advisor", False))
    advisor = None
    advisor_every = int(
        args.advisor_every if args.advisor_every is not None else policy.get("advisor_every", 1)
    )
    advisor_model = args.advisor_model
    if use_advisor:
        from ..adk_agents.model_config import load_model_config
        from ..adk_agents.online_physics_advisor import OnlinePhysicsAdvisor

        cfg = load_model_config()
        advisor_model = advisor_model or cfg.get("physics_advisor_model", "gemini-2.5-flash")
        advisor = OnlinePhysicsAdvisor(
            model=advisor_model,
            timeout_s=args.advisor_timeout_s,
            allowed_actions=policy.get("advisor_allowed_actions") or None,
        )

    bundle = get_experiment(
        ExperimentSettings(n_blocks=args.n_blocks, train_iters=args.train_iters)
    )
    res = evaluate_policy(
        policy,
        bundle["trajectories"],
        bundle["base_surrogate"],
        experts=bundle["experts"],
        collect_logs=True,
        advisor=advisor,
        advisor_every=advisor_every,
        advisor_timeout_s=args.advisor_timeout_s,
        **bundle["eval_kwargs"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "policy": policy,
                "advisor": {
                    "enabled": use_advisor,
                    "model": advisor_model,
                    "advisor_every": advisor_every,
                },
                "result": res,
            },
            f,
            indent=2,
        )
    if use_advisor:
        os.makedirs(os.path.dirname(os.path.abspath(args.advisor_out)), exist_ok=True)
        with open(args.advisor_out, "w") as f:
            json.dump(
                {
                    "policy": policy,
                    "advisor": {
                        "enabled": True,
                        "model": advisor_model,
                        "advisor_every": advisor_every,
                    },
                    "result": res,
                },
                f,
                indent=2,
            )
    print(json.dumps({"aggregate": {k: round(v, 4) for k, v in res["aggregate"].items()},
                      "actions": res["action_counts"],
                      "advisor": {"enabled": use_advisor, "model": advisor_model}}, indent=2))
    print(f"\nSaved -> {args.out}")
    if use_advisor:
        print(f"Advisor run saved -> {args.advisor_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
