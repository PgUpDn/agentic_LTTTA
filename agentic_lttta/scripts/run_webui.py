"""Launch the ADK web UI for the LTTTA design team.

    python -m agentic_lttta.scripts.run_webui --port 8000

This wraps ``python -m google.adk.cli web agentic_lttta/web`` (there is no ``adk``
console script in this env). The web UI lists three apps you can run and observe
node-by-node: ``lttta_designer`` (full team), ``rollout_analyst``, ``policy_tuner``.
On a remote VS Code server the port is auto-forwarded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from ..adk_agents.model_config import load_model_config
from ..llm_config import configure_env
from ..paths import PKG_DIR, WORKSPACE_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    configure_env()
    web_dir = os.path.join(PKG_DIR, "web")
    print("Per-agent models:", load_model_config())
    print(f"Serving ADK web UI from: {web_dir}")
    print(f"Open http://{args.host}:{args.port}/  (VS Code will offer to forward the port)")
    cmd = [
        sys.executable, "-m", "google.adk.cli", "web", web_dir,
        "--host", args.host, "--port", str(args.port),
    ]
    print("Launching:", " ".join(cmd))
    return subprocess.run(cmd, cwd=WORKSPACE_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
