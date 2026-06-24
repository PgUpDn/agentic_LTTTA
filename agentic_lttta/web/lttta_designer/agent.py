"""ADK web app — the FULL LTTTA design team (lead + analyst + tuner).

Type a request like:
    "Design the best LTTTA controller policy for the foil wake benchmark and save it."
and watch the trace: lttta_design_lead -> rollout_analyst -> policy_tuner -> save_policy.
"""

from __future__ import annotations

import pathlib
import sys

# Make the agentic_lttta package importable (workspace root), regardless of CWD.
_WS = pathlib.Path(__file__).resolve().parents[3]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from agentic_lttta.adk_agents import model_config as M  # noqa: E402
from agentic_lttta.adk_agents.tools import init_tools  # noqa: E402
from agentic_lttta.llm_config import configure_env  # noqa: E402

configure_env()
try:
    init_tools()  # load the cached surrogate so the tools are ready
except Exception as exc:  # noqa: BLE001
    print(f"[web] init_tools deferred: {exc}")

_cfg = M.load_model_config()
root_agent = M.build_team(_cfg)[0]
