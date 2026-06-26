"""ADK web app -- online physics advisor experiment."""

from __future__ import annotations

import pathlib
import sys

_WS = pathlib.Path(__file__).resolve().parents[3]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from google.adk.agents import LlmAgent  # noqa: E402

from agentic_lttta.adk_agents import model_config as M  # noqa: E402
from agentic_lttta.adk_agents.online_advisor_tools import (  # noqa: E402
    compare_online_with_without_advisor,
    get_online_advisor_setup_info,
    run_online_physics_advisor_eval,
)
from agentic_lttta.llm_config import configure_env  # noqa: E402

configure_env()
_CFG = M.load_model_config()

INSTRUCTION = """\
You let the user test an experimental online physics advisor for LTTTA.
Call get_online_advisor_setup_info first if the user asks what is available.
For evaluation, prefer compare_online_with_without_advisor with n_blocks=6 unless
the user requests a different short horizon. Report metrics, action histograms,
budget summaries, and a few advisor log entries. Be explicit that the online LLM
advisor is not intended for competitive Time Score.
"""

root_agent = LlmAgent(
    name="online_physics_advisor",
    model=_CFG.get("physics_advisor_model", "gemini-2.5-flash"),
    description="Runs and compares online LTTTA rollouts with a physics research advisor.",
    instruction=INSTRUCTION,
    tools=[
        get_online_advisor_setup_info,
        run_online_physics_advisor_eval,
        compare_online_with_without_advisor,
    ],
)
