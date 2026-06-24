"""Train (and cache) the tiny FNO surrogate on the downloaded foil real data."""

from __future__ import annotations

import argparse
import json

from ..experiment import ExperimentSettings, get_experiment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-traj", type=int, default=1)
    ap.add_argument("--sub-s", type=int, default=4)
    ap.add_argument("--in-step", type=int, default=10)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    args = ap.parse_args()

    settings = ExperimentSettings(
        max_traj=args.max_traj,
        sub_s=args.sub_s,
        in_step=args.in_step,
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        n_blocks=args.n_blocks,
    )
    bundle = get_experiment(settings, force_retrain=args.force)
    base = bundle["base_surrogate"]
    trs = bundle["trajectories"]
    print(json.dumps({
        "checkpoint": bundle["checkpoint"],
        "n_trajectories": len(trs),
        "regimes": [tr.regime_key for tr in trs],
        "resolution": list(trs[0].hw),
        "n_params": int(sum(p.numel() for p in base.model.parameters())),
        "eval_kwargs": bundle["eval_kwargs"],
        "final_train_loss": getattr(base, "train_history", [None])[-1],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
