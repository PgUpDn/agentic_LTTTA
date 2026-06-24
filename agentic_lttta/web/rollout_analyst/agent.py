"""ADK web app — the rollout_analyst node, standalone.

Type a request like:
    "Characterise the benchmark and the no-adaptation baseline, and tell me which actions help most."
and watch it call get_setup_info + evaluate_policy_tool.
"""

from __future__ import annotations

import pathlib
import sys

_WS = pathlib.Path(__file__).resolve().parents[3]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from agentic_lttta.adk_agents import model_config as M  # noqa: E402
from agentic_lttta.adk_agents.tools import init_tools  # noqa: E402
from agentic_lttta.llm_config import configure_env  # noqa: E402

configure_env()
try:
    init_tools()
except Exception as exc:  # noqa: BLE001
    print(f"[web] init_tools deferred: {exc}")

_cfg = M.load_model_config()
root_agent = M.make_analyst(_cfg["analyst_model"])
