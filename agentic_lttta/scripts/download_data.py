"""Download foil **real** Arrow shards + metadata into ``dataset_RealPDE/``.

Each real shard is one full trajectory (~1 GB). For the CPU prototype 1-2 shards
are plenty. Mirrors the approach in the reference ``analysis`` repo but uses the
``huggingface_hub`` Python API (no aria2c dependency).
"""

from __future__ import annotations

import argparse
import os
import time

from ..paths import DATASET_ROOT

REPO = "AI4Science-WestlakeU/RealPDEBench"
META_FILES = [
    "foil/channels.json",
    "foil/hf_dataset/real/dataset_info.json",
    "foil/hf_dataset/real/state.json",
    "foil/hf_dataset/train_index_real.json",
    "foil/hf_dataset/val_index_real.json",
    "foil/hf_dataset/test_index_real.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-shards", type=int, default=1, help="number of real trajectory shards")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"), help="e.g. https://hf-mirror.com")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(endpoint=args.endpoint)
    files = api.list_repo_files(REPO, repo_type="dataset", token=args.token)
    shards = sorted(
        f for f in files
        if f.startswith("foil/hf_dataset/real/") and f.endswith(".arrow")
    )[: args.n_shards]

    print(f"Downloading metadata + {len(shards)} shard(s) -> {args.dataset_root}")
    t0 = time.time()
    for f in META_FILES + shards:
        path = hf_hub_download(
            repo_id=REPO, filename=f, repo_type="dataset",
            local_dir=args.dataset_root, endpoint=args.endpoint, token=args.token,
        )
        print(f"  ok  {f}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
