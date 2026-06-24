"""The Google-ADK multi-agent design team.

A coordinator (`lttta_design_lead`) delegates to two specialist LLM agents,
wrapped as tools for deterministic, synchronous delegation:

* **rollout_analyst** -- characterises the benchmark, the no-adaptation baseline,
  and which actions/thresholds reduce long-horizon error (read-only probing).
* **policy_tuner** -- searches the bounded-controller knobs to maximise the
  composite score and saves the winning policy to ``config/policy.yaml``.

The online controller itself uses **no LLM** -- these agents run *offline* to
compose the fast online policy.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from ..llm_config import DEFAULT_MODEL, configure_env
from .tools import ANALYST_TOOLS, TUNER_TOOLS


ANALYST_INSTRUCTION = """\
You are the Rollout Analyst for a Long-Term Test-Time Adaptation (LTTTA) study on
airfoil wake forecasting.
1. Call `get_setup_info` to learn the bounded action space, the tunable policy
   knobs (with defaults), the flow regimes, and the no-adaptation baseline.
2. Probe 2-3 contrasting policies with `evaluate_policy_tool` (e.g. recalibrate-
   only by setting use_observe=false/use_adapter=false; an adapter-heavy policy;
   an observe-heavy policy). Observe how rel_l2, mvpe, tke, time_score and
   composite change.
3. Report a SHORT findings summary: the baseline composite, which actions help
   most, and recommended ranges for err_low, err_high, adapt_steps and max_obs.
Do NOT save anything. Keep it under ~150 words.
"""

TUNER_INSTRUCTION = """\
You are the Policy Tuner. Your goal: maximise the 'composite' score.
1. Try ~6-8 candidate policies with `evaluate_policy_tool`, varying err_low
   (0.04-0.12), err_high (0.14-0.30), slope_adapt, use_observe/use_adapter,
   adapt_steps (1-6), adapt_scope ('bn' | 'head' | 'bn+head') and max_obs.
2. Track the best composite seen.
3. Call `save_policy` EXACTLY ONCE with the best parameters and a one-line note.
Report the final composite and the parameters you saved. Be concise.
"""

LEAD_INSTRUCTION = """\
You coordinate the design of a fast, bounded LTTTA controller policy.
Step 1: call the `rollout_analyst` tool to characterise the problem and baseline.
Step 2: call the `policy_tuner` tool to search the policy space and SAVE the best
        policy (it returns the saved metrics).
Step 3: write a concise final report: the tuned policy parameters, its composite
        score, and the improvement over the no-adaptation baseline.
Do not invent numbers -- rely on the tool results.
"""


def build_agents(model: str = DEFAULT_MODEL):
    """Construct (lead, analyst, tuner). Credentials come from the environment."""
    configure_env()

    analyst = LlmAgent(
        name="rollout_analyst",
        model=model,
        description="Characterises the LTTTA benchmark, baseline, and useful actions.",
        instruction=ANALYST_INSTRUCTION,
        tools=ANALYST_TOOLS,
    )
    tuner = LlmAgent(
        name="policy_tuner",
        model=model,
        description="Searches controller knobs to maximise composite and saves the best policy.",
        instruction=TUNER_INSTRUCTION,
        tools=TUNER_TOOLS,
    )
    lead = LlmAgent(
        name="lttta_design_lead",
        model=model,
        description="Coordinates LTTTA policy design across specialist agents.",
        instruction=LEAD_INSTRUCTION,
        tools=[AgentTool(agent=analyst), AgentTool(agent=tuner)],
    )
    return lead, analyst, tuner


def build_design_lead(model: str = DEFAULT_MODEL):
    return build_agents(model)[0]


# ADK convention: expose a module-level ``root_agent`` for ``adk run`` / ``adk web``.
root_agent = None
try:  # best-effort so plain ``import`` never fails offline
    root_agent = build_design_lead()
except Exception:  # noqa: BLE001
    root_agent = None
