"""Baseline comparison: no-adaptation (free-run) vs the default hand-tuned policy.

This is the headline sanity check for the prototype: a good adaptive policy
should lower long-horizon Relative-L2 / MVPE versus pure free-running, while the
budget guard keeps the runtime in check.
"""

from __future__ import annotations

import argparse
import json

from ..controller import DEFAULT_POLICY
from ..experiment import ExperimentSettings, get_experiment
from ..rollout import evaluate_policy


def _fmt(agg: dict) -> dict:
    return {k: round(v, 4) for k, v in agg.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--train-iters", type=int, default=200)
    args = ap.parse_args()

    bundle = get_experiment(
        ExperimentSettings(n_blocks=args.n_blocks, train_iters=args.train_iters)
    )
    trs, base, experts, ek = (
        bundle["trajectories"], bundle["base_surrogate"],
        bundle["experts"], bundle["eval_kwargs"],
    )

    baseline_policy = {**DEFAULT_POLICY, "mode": "fixed", "fixed_action": "skip_update"}
    res_base = evaluate_policy(baseline_policy, trs, base, experts=experts, **ek)
    res_adapt = evaluate_policy(DEFAULT_POLICY, trs, base, experts=experts, **ek)

    out = {
        "eval_kwargs": ek,
        "baseline_no_adapt": {
            "aggregate": _fmt(res_base["aggregate"]),
            "actions": res_base["action_counts"],
        },
        "default_adaptive": {
            "aggregate": _fmt(res_adapt["aggregate"]),
            "actions": res_adapt["action_counts"],
        },
    }
    print(json.dumps(out, indent=2))
    b, a = res_base["aggregate"], res_adapt["aggregate"]
    print("\n=== Summary ===")
    print(f"rel_l2   : {b['rel_l2']:.4f} (no-adapt) -> {a['rel_l2']:.4f} (adaptive)")
    print(f"mvpe     : {b['mvpe']:.4f} (no-adapt) -> {a['mvpe']:.4f} (adaptive)")
    print(f"composite: {b['composite']:.4f} (no-adapt) -> {a['composite']:.4f} (adaptive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
